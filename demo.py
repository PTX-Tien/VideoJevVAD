#!/usr/bin/env python3
"""Animated score plot of one video (CPU): c(t), g(t), q_window(t) and the ground truth, drawn over time.
Scores only: no video frame is read or written, so the GIF can be shared without the dataset.

  python demo.py --dataset ucf --video Shooting008_x264 --out my_scores.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score

import score
import vj

W, H, L, R, T, B = 900, 360, 60, 24, 74, 58
INK, MUTED, GRID, GT = (33, 37, 41), (108, 117, 125), (233, 236, 239), (255, 214, 214)
SERIES = [('c', 'c(t) = q_window x g(t)', (217, 72, 15), 4), ('g', 'g(t) grounding', (25, 113, 194), 2),
          ('q', 'q_window', (12, 166, 120), 2)]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=list(vj.DATA), required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--title', help='default: the video name')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--frames', type=int, default=70)
    args = ap.parse_args()
    raw, _ = score.read('window', args.dataset, args.video, 'auto')
    dur, fps, ws = float(raw['duration']), float(raw['source_fps']), raw['windows']
    for w in ws:
        w['q'] = float(w['q_window'])
    secs = score.grid(dur)
    curves = {'c': score.windowed_qg(secs, ws), 'g': score.on_grid(secs, ws, 'g'), 'q': score.on_grid(secs, ws, 'q')}
    spans = [(s / fps, (e + 1) / fps) for s, e in vj.load_annotations(args.dataset)[args.video]]
    y = np.zeros(len(secs), bool)
    for a, b in spans:
        y |= (secs >= a) & (secs < b)
    auc = 100 * roc_auc_score(y, curves['c'])
    small, big = ImageFont.truetype(FONT, 13), ImageFont.truetype(FONT, 16)
    x_of = lambda t: L + (W - L - R) * t / dur                    # noqa: E731
    y_of = lambda v: H - B - (H - B - T) * v                      # noqa: E731
    frames = []
    for i in range(1, args.frames + 1):
        tau = dur * i / args.frames
        im = Image.new('RGB', (W, H), (255, 255, 255))
        d = ImageDraw.Draw(im)
        d.text((L, 10), f'{args.title or args.video}  |  AUC on this video {auc:.1f}', fill=INK, font=big)
        x = L
        for _, name, col, _ in SERIES:
            d.rounded_rectangle((x, 42, x + 18, 50), 3, fill=col)
            d.text((x + 24, 38), name, fill=INK, font=small)
            x += 40 + int(d.textlength(name, font=small))
        d.rectangle((x, 40, x + 18, 52), fill=GT)
        d.text((x + 24, 38), 'ground truth', fill=INK, font=small)
        for a, b in spans:
            d.rectangle((x_of(a), T, x_of(min(b, dur)), H - B), fill=GT)
        for v in (0, .5, 1):
            d.line((L, y_of(v), W - R, y_of(v)), fill=GRID, width=1)
            d.text((L - 30, y_of(v) - 8), f'{v:g}', fill=MUTED, font=small)
        for t in np.linspace(0, dur, 7):
            d.text((x_of(t) - 12, H - B + 6), f'{t:.0f}s', fill=MUTED, font=small)
        m = secs <= tau
        for key, _, col, width in reversed(SERIES):
            if m.sum() > 1:
                d.line([(x_of(t), y_of(v)) for t, v in zip(secs[m], curves[key][m])], fill=col, width=width, joint='curve')
        d.line((x_of(tau), T, x_of(tau), H - B), fill=MUTED, width=1)
        d.text((L, H - 24), 'Scores only. No dataset frames are included.', fill=MUTED, font=small)
        frames.append(im.quantize(colors=32, dither=Image.Dither.NONE))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames += [frames[-1]] * 20
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    print(f'{args.out}: AUC {auc:.2f}, {args.out.stat().st_size / 1024:.0f} KB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
