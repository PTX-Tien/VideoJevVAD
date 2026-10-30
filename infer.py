#!/usr/bin/env python3
"""VideoJev Global-Local VAD (clean), GPU inference. VideoJev-4B frozen, zero-shot, no GT, no event bank.

  window  local windows: 24 s / 12 s stride / 4 fps (<= 96 frames, 8192 context). One sequence per window:
          [tg_generic, tg_ordinary, qa_generic]  ->  g(t) = sigmoid grounding prob of tg_generic per frame-pair
          candidate, q_window = P(A) of qa_generic.                                   (1 GPU)
  global  whole video at 0.5 fps, <= 96 frames (uniform over the video if longer), one question in its own sequence
          -> q_global = P(A).                                                          (1 GPU)
  full    whole video at 0.5 fps, native resolution, no frame cap, model split over 2 GPUs; videos above
          FULL_MAX_TOKENS visual tokens are cut into the fewest equal non-overlapping segments; tg_generic alone
          -> g_full(t) = sigmoid(logits) per candidate.                                 (2 GPU)

Output: outputs/<dataset>/raw_<stage>/<video>.json (resumable: done videos are skipped unless --redo).
"""
from __future__ import annotations

import argparse
import importlib
import math
import time
import traceback
from pathlib import Path

import torch  # noqa: F401  (torch before decord: decord-first segfaults at CUDA init)
import numpy as np

import vj


def main() -> int:
    import os
    os.umask(0)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['window', 'global', 'full'])
    ap.add_argument('--dataset', choices=list(vj.DATA), required=True)
    ap.add_argument('--videos-file', type=Path, help='default: every test video of --dataset')
    ap.add_argument('--only-video', action='append', default=[])
    ap.add_argument('--out', type=Path, help='default: outputs/<dataset>/raw_<stage>')
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--max-windows', type=int, help='smoke only (window stage); needs its own --out')
    ap.add_argument('--redo', action='store_true')
    args = ap.parse_args()
    if args.max_windows and not args.out:
        raise SystemExit('--max-windows needs its own --out (never mix partial videos into the main outputs)')
    stems = list(args.only_video) + ([s.strip() for s in args.videos_file.read_text().splitlines() if s.strip()]
                                     if args.videos_file else [])
    stems = sorted(dict.fromkeys(stems or vj.test_videos(args.dataset)))[args.shard_index::args.num_shards]
    paths = vj.index_videos(args.dataset)
    if missing := [s for s in stems if s not in paths]:
        raise SystemExit(f'videos not found: {missing[:5]}')
    out = args.out or vj.OUT / args.dataset / f'raw_{args.stage}'
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stems if args.redo or not (out / f'{s}.json').is_file()]
    print(f'{args.stage} {args.dataset}: {len(todo)}/{len(stems)} videos to do -> {out}', flush=True)
    if not todo:
        return 0

    n_gpus = 2 if args.stage == 'full' else 1
    if torch.cuda.device_count() < n_gpus:
        raise SystemExit(f'{args.stage} needs {n_gpus} visible GPU(s), see {torch.cuda.device_count()}')
    t0 = time.time()
    model, processor, sha = vj.load_model(n_gpus)
    mod = importlib.import_module(type(model).__module__)
    dev = next(model.backbone.visual.parameters()).device
    print(f'VideoJev-4B loaded on {n_gpus} GPU(s) in {time.time() - t0:.0f}s (patched modeling sha256 {sha[:12]})', flush=True)

    @torch.inference_mode()
    def ask(frames, times, eff, d, questions, context):
        pixels, rec = vj.prepare_window(processor, frames, times, eff, d, context=context)
        feats = model.encode_video(pixels.to(dev), torch.tensor([rec['grid_thw']]))
        groups = list(mod.prepare_query_groups(processor, rec, questions, context))
        if len(groups) != 1:
            raise RuntimeError('all questions of a call must fit one sequence')
        batch, rows = groups[0]
        o = model(**batch, visual_features=feats)
        ans = {}
        for i, row in enumerate(rows):
            n = int(o.score_mask[i].sum())
            lg = o.logits[i, :n].float().cpu().numpy()
            p = o.probabilities[i, :n].float().cpu().numpy()
            if not np.all(np.isfinite(lg)):
                raise FloatingPointError(f'non-finite logits for {row["id"]}')
            if row['task'] == 'grounding' and n != len(rec['score_times']):
                raise RuntimeError('candidate count != pair timeline')
            ans[row['id']] = dict(logits=lg.tolist(), p=p.tolist())
        return rec, ans, int(batch['input_ids'].shape[1])

    def peaks():
        return [round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(n_gpus)]

    for vi, stem in enumerate(todo, 1):
        for i in range(n_gpus):
            torch.cuda.reset_peak_memory_stats(i)
        s = time.time()
        res = {'video': stem, 'dataset': args.dataset, 'stage': args.stage, 'gt_used': False}
        try:
            video = vj.Video(paths[stem])
            res.update(duration=video.duration, source_fps=video.fps, source_frames=video.frames)
            if args.stage == 'window':
                wins = []
                for a, b in vj.fixed_windows(video.duration)[:args.max_windows]:
                    fr, tm, eff, d = video.window(a, b, vj.WINDOW_FPS)
                    rec, ans, _ = ask(fr, tm, eff, d, vj.WINDOW_QUERIES, vj.CONTEXT)
                    wins.append(dict(window=[a, b], n_frames=len(tm), grid_thw=rec['grid_thw'],
                                     score_times_abs=(np.asarray(rec['score_times']) + a).tolist(),
                                     score_bounds_abs=(np.asarray(rec['score_bounds']) + a).tolist(),
                                     g=ans['tg_generic']['p'], logits=ans['tg_generic']['logits'],
                                     q_window=ans['qa_generic']['p'][0], qa_probs=ans['qa_generic']['p']))
                res.update(fps=vj.WINDOW_FPS, windows=wins, complete=args.max_windows is None)
                msg = f'{len(wins)} windows, max q_window {max(w["q_window"] for w in wins):.2f}'
            elif args.stage == 'global':
                fr, tm, eff, d = video.window(0., video.duration, vj.GLOBAL_FPS)
                rec, ans, _ = ask(fr, tm, eff, d, [vj.QA_GLOBAL], vj.CONTEXT)
                p = ans['q_global']['p']
                res.update(n_frames=len(tm), effective_fps=eff, grid_thw=rec['grid_thw'], q_global=p[0], probs=p)
                msg = f'{len(tm)} frames, q_global {p[0]:.3f}'
            else:
                tpp = vj.native_tokens_per_pair(processor, *video.vr[0].shape[:2])
                pairs = (int(np.ceil(video.duration * vj.GLOBAL_FPS)) + 1) // 2
                k = max(1, math.ceil(pairs * tpp / vj.FULL_MAX_TOKENS))
                edges = np.linspace(0, video.duration, k + 1)
                wins = []
                for a, b in zip(edges[:-1], edges[1:]):
                    a, b = float(a), float(b)
                    fr, tm, eff, d = video.window(a, b, vj.GLOBAL_FPS, max_frames=None)
                    rec, ans, n_in = ask(fr, tm, eff, d, [vj.TG_GENERIC], vj.full_context(len(tm), tpp))
                    wins.append(dict(window=[a, b], n_frames=len(tm), grid_thw=rec['grid_thw'], visual_tokens=rec['tokens'],
                                     input_tokens=n_in, frame_px=[rec['grid_thw'][1] * 16, rec['grid_thw'][2] * 16],
                                     score_times_abs=(np.asarray(rec['score_times']) + a).tolist(),
                                     score_bounds_abs=(np.asarray(rec['score_bounds']) + a).tolist(),
                                     g=ans['tg_generic']['p'], logits=ans['tg_generic']['logits']))
                res.update(fps=vj.GLOBAL_FPS, max_tokens=vj.FULL_MAX_TOKENS, segments=k, windows=wins)
                msg = f'{k} segment(s) x {wins[0]["n_frames"]} frames @ {wins[0]["frame_px"]} px'
            res['ok'] = True
        except Exception as e:  # noqa: BLE001  (OOM etc. are recorded; never replaced by another method)
            res.update(ok=False, error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc(limit=4))
            msg = 'ERROR ' + res['error'][:200]
            torch.cuda.empty_cache()
        res.update(total_s=time.time() - s, peak_gib=peaks())
        vj.write_json(out / f'{stem}.json', res)
        print(f'[{vi}/{len(todo)}] {stem}: {res.get("duration", 0):.0f}s, {msg}, {res["total_s"]:.1f}s, peak {res["peak_gib"]} GiB', flush=True)
    print('[DONE]', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
