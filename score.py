#!/usr/bin/env python3
"""VideoJev Global-Local VAD (clean), CPU scoring: anomaly timeline c(t) on a 0.5 s grid + 20 % candidates. No GT.

Modes (formulas fixed in advance, no weights, no thresholds):
  gl_v10       c(t) = mean over the 24 s windows covering t of  q_window * g_window(t)          [MAIN RESULT]
  gl_v12       c(t) = q_global * g_full(t)            (g_full: whole-video grounding at 0.5 fps, 2 GPU)
  gl_v12_qwin  c(t) = q_window(t) * g_full(t)         (q_window(t): mean q_window of the windows covering t)
g on the grid: linear interpolation between candidate centres inside each call (window / segment), held constant
beyond the first / last centre; calls covering the same grid point are averaged.
Candidates: the top ceil(20 %) grid cells by c(t) (ties: earlier first); each contiguous run is one segment.

Inputs (--source): clean = outputs/<ds>/raw_{window,global,full} written by infer.py;
                   cache = the earlier runs (read-only, same models / prompts / settings):
                           window  ../outputs/ucf_fps4_TG_first/raw, ../XD/outputs/xd_fps4_TG_first/raw (V10; only
                                   tg_generic and qa_generic are read, never the event bank)
                           global  ../GlobalLocal/outputs/full_<ds>/global_raw
                           full    ../V12/outputs/full_<ds>/fullmax/raw
                   auto  = clean if the file exists, else cache.
Output: outputs/<ds>/<mode>/timeline/<video>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

import vj

CACHE = {
    'window': {'ucf': vj.ROOT / 'outputs/ucf_fps4_TG_first/raw', 'xd': vj.ROOT / 'XD/outputs/xd_fps4_TG_first/raw'},
    'global': {d: vj.ROOT / f'GlobalLocal/outputs/full_{d}/global_raw' for d in vj.DATA},
    'full': {d: vj.ROOT / f'V12/outputs/full_{d}/fullmax/raw' for d in vj.DATA},
}
NEEDS = {'gl_v10': ('window',), 'gl_v12': ('global', 'full'), 'gl_v12_qwin': ('window', 'full')}


def read(stage, dataset, video, source):
    clean = vj.OUT / dataset / f'raw_{stage}' / f'{video}.json'
    if source == 'clean' or (source == 'auto' and clean.is_file()):
        return json.loads(clean.read_text()), 'clean'
    r = json.loads((CACHE[stage][dataset] / f'{video}.json').read_text())
    if stage == 'window':      # V10 cache -> clean format (generic outputs only)
        r['windows'] = [dict(window=w['window'], score_times_abs=w['score_times_abs'], score_bounds_abs=w['score_bounds_abs'],
                             g=w['orders']['TG_first']['raw']['tg_generic']['probabilities'],
                             q_window=w['orders']['TG_first']['raw']['qa_generic']['probabilities'][0]) for w in r['windows']]
        r['ok'] = True
    return r, 'cache'


def grid(duration):
    secs = np.arange(0, duration, vj.GRID) + vj.GRID / 2
    return secs[secs < duration]


def on_grid(secs, calls, key):
    """Mean over covering calls of the call's curve (key='g': interpolated candidates, key='q_window': constant)."""
    acc, cnt = np.zeros(len(secs)), np.zeros(len(secs))
    for w in calls:
        a, b = w['window']
        m = (secs >= a) & (secs < b)
        if key == 'g':
            t, g = np.asarray(w['score_times_abs'], float), np.asarray(w['g'], float)
            acc[m] += np.interp(secs[m], t, g, left=g[0], right=g[-1])
        else:
            acc[m] += float(w[key])
        cnt[m] += 1
    if not (cnt > 0).all():
        raise RuntimeError('grid not fully covered')
    return acc / cnt


def propose(secs, score, duration):
    k = max(1, math.ceil(vj.BUDGET * len(secs)))
    keep = np.zeros(len(secs), bool)
    keep[np.lexsort((secs, -score))[:k]] = True
    segs, i = [], 0
    while i < len(secs):
        if keep[i]:
            j = i
            while j + 1 < len(secs) and keep[j + 1]:
                j += 1
            segs.append({'start': float(secs[i] - vj.GRID / 2), 'end': float(min(duration, secs[j] + vj.GRID / 2)),
                         'mean_score': float(score[i:j + 1].mean()), 'peak_score': float(score[i:j + 1].max())})
            i = j + 1
        else:
            i += 1
    return sorted(segs, key=lambda s: -s['mean_score'])


def main() -> int:
    os.umask(0)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=list(vj.DATA), required=True)
    ap.add_argument('--mode', choices=list(NEEDS), default='gl_v10')
    ap.add_argument('--source', choices=['auto', 'clean', 'cache'], default='auto')
    ap.add_argument('--out', type=Path, help='default: outputs/<dataset>/<mode>')
    args = ap.parse_args()
    out = (args.out or vj.OUT / args.dataset / args.mode) / 'timeline'
    out.mkdir(parents=True, exist_ok=True)
    used, errors = {}, []
    for v in vj.test_videos(args.dataset):
        try:
            raw = {}
            for st in NEEDS[args.mode]:
                raw[st], src = read(st, args.dataset, v, args.source)
                used[f'{st}:{src}'] = used.get(f'{st}:{src}', 0) + 1
                if not raw[st].get('ok', True):
                    raise RuntimeError(f'{st} failed: {raw[st].get("error")}')
            base = raw['window'] if 'window' in raw else raw['full']
            dur = float(base['duration'])
            secs = grid(dur)
            if args.mode == 'gl_v10':
                acc, cnt = np.zeros(len(secs)), np.zeros(len(secs))
                for w in raw['window']['windows']:       # q * g per window, then mean over covering windows
                    a, b = w['window']
                    m = (secs >= a) & (secs < b)
                    t, g = np.asarray(w['score_times_abs'], float), np.asarray(w['g'], float)
                    acc[m] += w['q_window'] * np.interp(secs[m], t, g, left=g[0], right=g[-1])
                    cnt[m] += 1
                c = acc / cnt
                g, q = on_grid(secs, raw['window']['windows'], 'g'), on_grid(secs, raw['window']['windows'], 'q_window')
            else:
                g = on_grid(secs, raw['full']['windows'], 'g')
                q = (np.full(len(secs), float(raw['global']['q_global'])) if args.mode == 'gl_v12'
                     else on_grid(secs, raw['window']['windows'], 'q_window'))
                c = q * g
            vj.write_json(out / f'{v}.json', {
                'video': v, 'dataset': args.dataset, 'mode': args.mode, 'duration': dur, 'source_fps': base['source_fps'],
                'source_frames': base['source_frames'], 'grid_step': vj.GRID, 'secs': secs.tolist(), 'c': c.tolist(),
                'g': g.tolist(), 'q': q.tolist(), 'video_score': float(c.max()), 'candidates': propose(secs, c, dur),
                'budget': vj.BUDGET, 'gt_used': False})
        except Exception as e:  # noqa: BLE001
            errors.append(f'{v}: {type(e).__name__}: {e}')
    print(f'{args.mode} {args.dataset}: {len(vj.test_videos(args.dataset)) - len(errors)} timelines -> {out}; inputs {used}', flush=True)
    for e in errors[:20]:
        print('  ERROR', e)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
