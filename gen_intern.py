#!/usr/bin/env python3
"""InternVL3.5-2B writes, once per video, the queries VideoJev will answer in every window (GPU, 1 card, no GT).

Frames shown to InternVL (16, time order, 448x448): 8 uniform over the whole video + 4 uniform inside each of the two
highest-ranked gl_v10 candidates (20 % budget of c(t) = q_window * g(t); no GT). Greedy generation, JSON output:
  tg_query  one sentence: the single most unusual / harmful action visible   -> VideoJev grounding query
  question  one question about what happens in a short segment of this scene -> VideoJev choice question
  option_a  the unusual action          option_b  the ordinary activity of this scene
  option_c  people before / after an incident (reacting, gathering, arriving, leaving)
  option_d  fixed: "None of these is clearly visible."  (written by this script, not generated)
No class list, no event bank, no label is given to InternVL. A reply that is not valid JSON is recorded as an error
(no fallback text). Output: outputs/<ds>/gen_intern/<video>.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from pathlib import Path

import torch  # noqa: F401  (torch before decord)
import numpy as np

import vj
import score

IV_PATH = Path('/mmlab_students/storageStudents/nguyenvd/tienptx/vjepa2/checkpoints/hf_cache/'
               'models--OpenGVLab--InternVL3_5-2B-HF/snapshots/3f301ffcf3dcbb47893afae6650ea3e78d96fb6d')
IV_SIZE = {'height': 448, 'width': 448}      # vision image_size (the shipped video config's 384 breaks the reshape)
N_UNIFORM, N_PER_CAND, N_CANDS, MIN_CAND_S = 8, 4, 2, 4.0
MAX_NEW_TOKENS = 220
OPTION_D = 'None of these is clearly visible.'
PROMPT = (
    'These frames are sampled in time order from one video. Write text for another video model that will inspect '
    'short segments of this same video.\n'
    'Rules: describe only what is visible; describe actions, not crime categories; present tense; '
    'each value at most 20 words.\n'
    'Return only a JSON object with these keys:\n'
    '"tg_query": one sentence describing the single most unusual, dangerous or harmful action in the video '
    '(if nothing unusual is visible, describe the most unusual-looking moment);\n'
    '"question": one question asking what is happening in a short segment of this scene;\n'
    '"option_a": the action of tg_query, written as an answer;\n'
    '"option_b": the ordinary activity of this scene when nothing unusual happens;\n'
    '"option_c": what the people in this scene do just before or just after that action '
    '(for example waiting, reacting, gathering, arriving or leaving).\n'
    'Every value must be a non-empty sentence.')


def frame_times(dataset, video, duration, source):
    """8 uniform + 4 inside each of the top-2 gl_v10 candidates (widened to MIN_CAND_S), sorted."""
    raw, _ = score.read('window', dataset, video, source)
    secs = score.grid(duration)
    c = score.windowed_qg(secs, raw['windows'])
    t = list((np.arange(N_UNIFORM) + .5) * duration / N_UNIFORM)
    for cand in score.propose(secs, c, duration)[:N_CANDS]:
        a, b = cand['start'], cand['end']
        if b - a < MIN_CAND_S:
            m = (a + b) / 2
            a = max(0., min(m - MIN_CAND_S / 2, duration - MIN_CAND_S))
            b = min(duration, a + MIN_CAND_S)
        t += list(a + (np.arange(N_PER_CAND) + .5) * (b - a) / N_PER_CAND)
    return np.sort(np.asarray(t))


def parse(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise ValueError('no JSON object in the reply')
    d = json.loads(m.group(0))
    out = {}
    for k in ('tg_query', 'question', 'option_a', 'option_b', 'option_c'):
        v = d.get(k)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f'missing or empty "{k}"')
        out[k] = re.sub(r'^\s*[A-Da-d][.)]\s*', '', v.strip())     # letters are added by this script
    return out


def videojev_queries(gen):
    """The two VideoJev queries of one window (one causal sequence, grounding first)."""
    return [{'id': 'tg_gen', 'task': 'grounding', 'query': gen['tg_query']},
            {'id': 'qa_gen', 'task': 'choice', 'query': gen['question'],
             'options': [f'A. {gen["option_a"]}', f'B. {gen["option_b"]}', f'C. {gen["option_c"]}', f'D. {OPTION_D}']}]


def main() -> int:
    import os
    os.umask(0)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=list(vj.DATA), required=True)
    ap.add_argument('--videos-file', type=Path)
    ap.add_argument('--only-video', action='append', default=[])
    ap.add_argument('--out', type=Path, help='default: outputs/<dataset>/gen_intern')
    ap.add_argument('--source', choices=['auto', 'clean', 'cache'], default='auto', help='where the gl_v10 windows come from')
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--redo', action='store_true')
    args = ap.parse_args()
    stems = list(args.only_video) + ([s.strip() for s in args.videos_file.read_text().splitlines() if s.strip()]
                                     if args.videos_file else [])
    stems = sorted(dict.fromkeys(stems or vj.test_videos(args.dataset)))[args.shard_index::args.num_shards]
    out = args.out or vj.OUT / args.dataset / 'gen_intern'
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in stems if args.redo or not (out / f'{s}.json').is_file()]
    print(f'gen_intern {args.dataset}: {len(todo)}/{len(stems)} videos -> {out}', flush=True)
    if not todo:
        return 0
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.integrations import sdpa_attention
    sdpa_attention.use_gqa_in_sdpa = lambda *a, **k: False
    processor = AutoProcessor.from_pretrained(str(IV_PATH))
    model = AutoModelForImageTextToText.from_pretrained(str(IV_PATH), dtype=torch.float16, device_map={'': 'cuda:0'},
                                                        attn_implementation='sdpa').eval()
    prompt = processor.apply_chat_template([{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': PROMPT}]}],
                                           add_generation_prompt=True, tokenize=False)
    paths = vj.index_videos(args.dataset)
    for vi, stem in enumerate(todo, 1):
        torch.cuda.reset_peak_memory_stats()
        s = time.time()
        res = {'video': stem, 'dataset': args.dataset, 'model': 'InternVL3.5-2B-HF fp16', 'prompt': PROMPT, 'gt_used': False}
        try:
            video = vj.Video(paths[stem])
            t = frame_times(args.dataset, stem, video.duration, args.source)
            idx = np.clip(np.rint(t * video.fps).astype(int), 0, video.frames - 1)
            frames = video.vr.get_batch(idx.tolist()).asnumpy()
            inp = processor(text=[prompt], videos=[frames], size=IV_SIZE, return_tensors='pt').to('cuda:0')
            inp = {k: (v.half() if v.is_floating_point() else v) for k, v in inp.items()}
            with torch.inference_mode():
                ids = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            reply = processor.tokenizer.decode(ids[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)
            res.update(frame_times=t.tolist(), reply=reply)
            gen = parse(reply)
            res.update(ok=True, gen=gen, videojev_queries=videojev_queries(gen))
        except Exception as e:  # noqa: BLE001
            res.update(ok=False, error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc(limit=3))
        res.update(time_s=time.time() - s, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
        vj.write_json(out / f'{stem}.json', res)
        if res['ok']:
            g = res['gen']
            print(f'[{vi}/{len(todo)}] {stem}: TG "{g["tg_query"]}" | Q "{g["question"]}" | A "{g["option_a"]}" | '
                  f'B "{g["option_b"]}" | C "{g["option_c"]}" ({res["time_s"]:.1f}s)', flush=True)
        else:
            print(f'[{vi}/{len(todo)}] {stem}: ERROR {res["error"]} | reply {res.get("reply", "")[:200]!r}', flush=True)
    print('[DONE]', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
