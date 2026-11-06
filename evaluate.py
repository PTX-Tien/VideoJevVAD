#!/usr/bin/env python3
"""VideoJev Global-Local VAD (clean), CPU evaluation of outputs/<ds>/<mode>/timeline. GT is read ONLY here, after every
score and candidate is fixed; nothing is tuned (0.5 operating point and 20 % budget are fixed in advance).

GT convention (as V10): abnormal span [a/fps, (b+1)/fps) in source frames; scores evaluated at every source-frame centre
(linear interpolation of the 0.5 s grid). Writes outputs/<ds>/<mode>/metrics.md and metrics.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as APS, roc_auc_score as AUC

import vj


def overlap(a, b):
    return sum(max(0., min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


def main() -> int:
    os.umask(0)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', choices=list(vj.DATA), required=True)
    ap.add_argument('--mode', default='gl_v10')
    ap.add_argument('--dir', type=Path, help='default: outputs/<dataset>/<mode>')
    ap.add_argument('--videos-file', type=Path, help='evaluate only these videos (pilot)')
    args = ap.parse_args()
    d = args.dir or vj.OUT / args.dataset / args.mode
    ann = vj.load_annotations(args.dataset)
    if args.videos_file:
        keep = {x.strip() for x in args.videos_file.read_text().splitlines() if x.strip()}
        ann = {k: v for k, v in ann.items() if k in keep}
    rows, missing = [], []
    for v, spans_f in sorted(ann.items()):
        p = d / 'timeline' / f'{v}.json'
        if not p.is_file():
            missing.append(v)
            continue
        r = json.loads(p.read_text())
        fps, dur = float(r['source_fps']), float(r['duration'])
        ft = (np.arange(int(r['source_frames'])) + .5) / fps
        ft = ft[ft < dur]
        gt = [(s / fps, (e + 1) / fps) for s, e in spans_f]
        y = np.zeros(len(ft), bool)
        for s, e in gt:
            y |= (ft >= s) & (ft < e)
        s = np.interp(ft, np.asarray(r['secs']), np.asarray(r['c']))
        cand = [(c['start'], c['end']) for c in r['candidates']]
        gsec = sum(e - s0 for s0, e in gt)
        rows.append(dict(v=v, y=y, s=s, abn=bool(gt), vs=r['video_score'],
                         rec=overlap(cand, gt) / gsec if gsec else None,
                         prec=overlap(cand, gt) / max(1e-9, sum(e - s0 for s0, e in cand)) if gsec else None,
                         hit=any(overlap([c], gt) > 0 for c in cand) if gsec else None))
    y = np.concatenate([r['y'] for r in rows])
    s = np.concatenate([r['s'] for r in rows])
    nrm = np.concatenate([np.full(len(r['y']), not r['abn']) for r in rows])
    ab = [r for r in rows if 0 < r['y'].sum() < len(r['y'])]
    p, neg = s >= .5, ~y
    lab = np.array([r['abn'] for r in rows])
    vs = np.array([r['vs'] for r in rows])
    m = dict(n_videos=len(rows), n_abnormal=int(lab.sum()),
             micro_AUC=100 * AUC(y, s), micro_AP=100 * APS(y, s),
             macro_AUC=100 * np.mean([AUC(r['y'], r['s']) for r in ab]), macro_AP=100 * np.mean([APS(r['y'], r['s']) for r in ab]),
             TPR=100 * p[y].mean(), precision=100 * y[p].mean(), FPR_all=100 * p[neg].mean(), FPR_normal=100 * p[nrm].mean(),
             FPR_abn_outside=100 * p[neg & ~nrm].mean(), FP_share_abnormal_videos=100 * (p & neg & ~nrm).sum() / max(1, (p & neg).sum()),
             cand_recall=100 * np.mean([r['rec'] for r in rows if r['abn']]),
             cand_precision=100 * np.mean([r['prec'] for r in rows if r['abn']]),
             cand_hit=100 * np.mean([r['hit'] for r in rows if r['abn']]),
             video_AUC=100 * AUC(lab, vs), video_AP=100 * APS(lab, vs))
    L = [f'# {args.mode} on {args.dataset.upper()} ({len(rows)}/{len(ann)} videos, {int(lab.sum())} abnormal)', '',
         'Offline, zero-shot VideoJev-4B, no event bank, no GT in any score; GT used only here.', '',
         '| micro AUC | micro AP | macro AUC | macro AP | TPR@0.5 | Prec@0.5 | FPR all neg | FPR normal videos | FPR abnormal outside GT | FP share in abnormal videos |',
         '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
         f'| {m["micro_AUC"]:.2f} | {m["micro_AP"]:.2f} | {m["macro_AUC"]:.2f} | {m["macro_AP"]:.2f} | {m["TPR"]:.1f}% | {m["precision"]:.1f}% | '
         f'{m["FPR_all"]:.2f}% | {m["FPR_normal"]:.2f}% | {m["FPR_abn_outside"]:.2f}% | {m["FP_share_abnormal_videos"]:.1f}% |', '',
         '| 20 % candidates: GT recall | precision | hit (videos) | video-level AUC (max c) | video-level AP |', '|---:|---:|---:|---:|---:|',
         f'| {m["cand_recall"]:.1f}% | {m["cand_precision"]:.1f}% | {m["cand_hit"]:.1f}% | {m["video_AUC"]:.2f} | {m["video_AP"]:.2f} |', '']
    if missing:
        L += [f'Missing timelines: {len(missing)} ({", ".join(missing[:10])}{" ..." if len(missing) > 10 else ""})', '']
    (d / 'metrics.md').write_text('\n'.join(L) + '\n')
    (d / 'metrics.json').write_text(json.dumps(m, indent=1))
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
