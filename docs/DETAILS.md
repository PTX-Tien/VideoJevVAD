# VideoJev Global-Local VAD (clean)

Offline, zero-shot video anomaly detection with **VideoJev-4B** (frozen, training-free) on UCF-Crime and XD-Violence.

- One generic question and one generic grounding query. No class names and no event bank.
- No GT in any score, setting or prompt. GT is read only by `evaluate.py`.
- Offline, not causal: a frame's score can use later frames of the same window or video.

**Main result: `gl_v10` (GL c(t) = q_window × g(t)).**

## 1. Pipeline

| Part | What it computes | Setting |
|---|---|---|
| `window` (1 GPU) | Local windows. For each window: g(t) per frame-pair candidate and q_window | 24 s windows, 12 s stride, 4 fps (≤ 96 frames), context 8192 |
| `global` (1 GPU) | q_global, one look at the whole video | 0.5 fps, ≤ 96 frames (spread uniformly over longer videos) |
| `full` (2 GPU) | g_full(t): grounding over the whole video | 0.5 fps, native resolution (cap 640×640), no frame cap. A video above 26,880 visual tokens is cut into the fewest equal, non-overlapping segments |

Score modes (`score.py`). All formulas were fixed in advance, with no weights and no thresholds:

| Mode | c(t) | Inputs |
|---|---|---|
| **`gl_v10` (main)** | mean over the 24 s windows covering t of **q_window × g_window(t)** | window |
| `gl_v12` | **q_global × g_full(t)** | global + full |
| `gl_v12_qwin` | **q_window(t) × g_full(t)**, where q_window(t) is the mean q_window of the windows covering t | window + full |

How the scores are built:
- **g:** the sigmoid grounding probability of each candidate. There is one candidate per pair of sampled frames: centre = mean of the two timestamps, bounds halfway between pairs, first bound 0, last bound = duration. On the 0.5 s grid, g is linearly interpolated between candidate centres inside each call, and calls that cover the same point are averaged.
- **q:** P(A) of a 3-option question. The options are scored with a softmax, so A + B + C = 1.
- **Candidates:** the top 20% of 0.5 s grid cells by c(t). Each contiguous run of selected cells is one segment.
- **Video-level score:** max c(t).

## 2. Prompt templates (`vj.py`, the only prompts)

**Window sequence:** `[tg_generic, tg_ordinary, qa_generic]` in one causal sequence per window.
- This is the V10 TG_first order with the event bank removed.
- `tg_ordinary` is kept only so that q_window and g are bit-for-bit the V10 values. Its output is not used.

```text
tg_generic  (grounding) : "A clearly unusual or dangerous incident occurs."
tg_ordinary (grounding) : "People or vehicles perform ordinary activities without any visible accident or attack."
qa_generic  (choice)    : "Does this video segment contain a clearly visible unusual, dangerous or harmful incident
                           rather than ordinary activity?"
                           A. Yes, a clearly unusual or harmful event is visible.
                           B. No, only ordinary activity is visible.
                           C. Insufficient evidence to determine.          -> q_window = P(A)
```

**Global question:** asked alone in its own sequence.

```text
q_global (choice) : "Does this entire video contain a clearly visible unusual, dangerous, or harmful incident?"
                     A. Yes   B. No   C. Insufficient evidence            -> q_global = P(A)
```

**Full-video grounding:** `tg_generic` alone, with the same text as above. g_full = sigmoid(logits).

## 3. How to run

Use only free GPUs and check `nvidia-smi` first. The `window` and `global` stages need about 10.3 GB on one card; `full` needs two cards with about 10 GB free each.

```bash

# A. Reproduce every number below from the existing caches (CPU only, a few minutes)
bash run.sh reproduce

# B. Run from scratch
bash run.sh smoke 6 7                 # tiny run of each stage (window/global on GPU 6, full on GPUs 6+7) -> outputs/smoke
bash run.sh window ucf 6 7            # one background shard per GPU; resumable
bash run.sh global ucf 6
bash run.sh full   ucf 6 7            # 2 GPU (model split 3.6 / 6.5 GiB of weights)
bash run.sh status
bash run.sh score  ucf gl_v10         # timeline c(t) + candidates -> outputs/ucf/gl_v10/timeline/
bash run.sh eval   ucf gl_v10         # -> outputs/ucf/gl_v10/metrics.md
bash run.sh score  ucf gl_v12 && bash run.sh eval ucf gl_v12
bash run.sh score  ucf gl_v12_qwin && bash run.sh eval ucf gl_v12_qwin
# same with xd
```

- `score --source auto` (the default) reads `outputs/<ds>/raw_*` when present, otherwise the earlier read-only caches:
  - V10 windows: `../outputs/ucf_fps4_TG_first/raw` and `../XD/outputs/xd_fps4_TG_first/raw`. Only `tg_generic` and `qa_generic` are read.
  - Global: `../GlobalLocal/outputs/full_<ds>/global_raw`.
  - Full-video: `../V12/outputs/full_<ds>/fullmax/raw`.
- Output layout:
  - `outputs/<ds>/raw_{window,global,full}/<video>.json` holds the raw model outputs: g, logits, q, real timestamps and bounds, frames, grid, time, peak VRAM.
  - `outputs/<ds>/<mode>/timeline/<video>.json` holds secs, c, g, q, candidates and the video score.

Checks done on 2026-10-30:
- **`reproduce` from the caches** gives exactly the numbers in section 4.
- **GPU smoke of `infer.py` against the caches:**
  - window: |Δg| = 0 and |Δq| = 0 (Abuse028, 2 windows).
  - global: q_global identical (XD 1383 s video).
  - full: |Δg| ≤ 2e-3 (Shooting032, RoadAccidents021; fp16 on 2 GPUs).

Environment:
- Extra kernels from `../pydeps` (fla 0.2.2).
- Checkpoint revision `1c1a253`, loaded through `../checkpoint_fp16`, which patches BF16 to FP16 autocast for the RTX 2080 Ti.
- `use_gqa_in_sdpa=False`. Vision tower and decision head in fp32.

## 4. Results (full test sets: UCF 290 videos / 140 abnormal, XD 800 / 500)

Frame level: micro = all frames pooled; macro = mean over abnormal videos. The 0.5 operating point is fixed and was not tuned.

| Dataset | Mode | micro AUC | micro AP | macro AUC | macro AP | TPR@0.5 | Prec@0.5 | FPR normal videos | FPR abnormal outside GT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UCF | **gl_v10 (main)** | 89.57 | **46.23** | **82.06** | **62.51** | 63.7% | 35.5% | 0.52% | 24.86% |
| UCF | gl_v12 | 90.27 | 45.83 | 78.11 | 55.22 | 63.7% | 39.7% | 0.20% | 21.20% |
| UCF | gl_v12_qwin | **90.79** | 45.82 | 79.48 | 56.56 | 62.8% | 42.2% | 0.09% | 19.01% |
| XD | **gl_v10 (main)** | **92.11** | **74.94** | **85.18** | **77.82** | 70.0% | 71.3% | 1.05% | 25.59% |
| XD | gl_v12 | 87.07 | 62.36 | 78.65 | 69.54 | 47.2% | 63.1% | 0.39% | 26.60% |
| XD | gl_v12_qwin | 87.26 | 63.20 | 79.00 | 69.97 | 47.5% | 63.3% | 0.51% | 26.21% |

20% candidates and video level:

| Dataset | Mode | GT recall | Precision | Hit (videos) | Video AUC (max c) | Video AP |
|---|---|---:|---:|---:|---:|---:|
| UCF | gl_v10 | 57.3% | 46.3% | 92.1% | 96.33 | 96.86 |
| UCF | gl_v12 | 50.5% | 42.4% | 80.7% | 95.28 | 95.77 |
| UCF | gl_v12_qwin | 53.0% | 43.8% | 80.7% | 96.84 | 97.36 |
| XD | gl_v10 | 45.3% | 73.8% | 97.8% | 96.93 | 97.84 |
| XD | gl_v12 | 37.7% | 67.4% | 92.6% | 97.92 | 98.35 |
| XD | gl_v12_qwin | 38.4% | 68.1% | 93.0% | 96.70 | 97.20 |

### Reading

- **gl_v10 is the main result.** It is best on macro AUC/AP on both datasets and on every metric on XD.
- **The gl_v12 modes have fewer FP on normal videos, but miss short events.**
  - The whole-video context keeps the background low: FPR on normal videos is 0.09–0.5%.
  - At 0.5 fps one candidate is about 4 s wide. Hit@20% for events ≤ 5 s falls from 81% to 60% on UCF and from 58% to 32% on XD; XD has 560 of its 1,226 events at ≤ 5 s.
  - fps (0.5 vs 4) and context (whole video vs 24 s) change together in this comparison, so their effects are confounded.
- **The remaining weak point is FP inside abnormal videos.** 91–99% of all FP fall in abnormal videos, outside GT, mostly just before or after the event.
- **The V12 modes on the same 0.5 fps pilot:** see `../V12/V12_FULL_VIDEO_GROUNDING_REPORT.md`.

### Rules respected

- Nothing was selected on UCF or XD test scores. The modes and formulas were declared before evaluation.
- No event bank (the old V10 13-class bank is not read).
- No training.
- Offline is not causal.
- Comparisons with other projects (e.g. Causal PH) are references only, under a different protocol.

## 5. Files

| File | Role |
|---|---|
| `vj.py` | Paths, settings, prompt templates, data / annotations, frame sampling, official resolution search and pair timeline, model loading (1 or 2 GPU) |
| `infer.py` | GPU stages `window` / `global` / `full` → `outputs/<ds>/raw_<stage>/` |
| `score.py` | CPU: c(t) for `gl_v10` / `gl_v12` / `gl_v12_qwin` plus the 20% candidates |
| `evaluate.py` | CPU: frame / candidate / video metrics (GT only here) |
| `run.sh` | Entry point (see `bash run.sh`) |

## 6. Experimental: InternVL-written queries (`gen_intern.py`, `infer.py window_gen`)

InternVL3.5-2B-HF (fp16, 1 card) watches 16 frames of each video:
- 8 frames uniform over the whole video;
- 4 frames inside each of the top-2 `gl_v10` candidates.

Greedy decoding, JSON output, no class list, no GT. Per video it writes:
- a grounding query `tg_query`;
- a question;
- options A (the unusual action), B (the ordinary activity of the scene) and C (before / after the incident).

Option D is fixed: "None of these is clearly visible." VideoJev answers this one set of texts in every 24 s window, so the scores of all windows of a video are comparable.

| Mode | c(t) | Status |
|---|---|---|
| `gl_gen` | P(A) × g_gen | Declared before the pilot. **Fails**: on normal videos InternVL describes the benign scene and P(A) ≈ 1 |
| `gl_gen_contrast` | P(A)/(P(A)+P(C)) × g_gen | Declared before the pilot. Fails for the same reason |
| `gl_gen_tg` | q_window (generic question) × g_gen | Declared 2026-10-30 after the pilot. Next: full UCF to decide, then XD once |

Pilot (10 pre-declared UCF videos: 6 abnormal, 4 normal):

| Mode | micro AUC / AP | macro AUC / AP | Normal frames ≥ 0.5 |
|---|---|---|---|
| gl_v10 | 87.20 / 11.88 | 84.69 / 61.63 | 1.9% |
| gl_gen | 27.54 / 1.60 | 83.50 / 67.48 | 91.4% |
| gl_gen_contrast | 27.77 / 1.60 | 85.74 / 67.04 | 91.9% |
| gl_gen_tg | 77.61 / 8.57 | 88.02 / 73.22 | 4.1% |

Run:

```bash
bash run.sh gen_all ucf G1 G2   # gen + window_gen + gl_gen_tg score/eval (~2.3 GPU-h of VideoJev for UCF)
bash run.sh gen_all xd G1 G2    # only after the UCF decision (~6 GPU-h)
```
