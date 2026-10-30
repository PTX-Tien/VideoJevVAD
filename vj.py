"""VideoJev Global-Local VAD (clean): shared pieces. Paths, prompts, data, frame sampling, model loading.

Self-contained (imports nothing from V10 / V11 / GlobalLocal / V12). The model code is the official VideoJev-4B code shipped
with the checkpoint (trust_remote_code), loaded from the local patched copy ../checkpoint_fp16 (BF16 -> FP16 autocast,
RTX 2080 Ti has no BF16). No event bank, no class names anywhere.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # VideoJevVAD
HF_SNAPSHOT = ROOT.parent / 'checkpoints/hf_cache/models--ZhuYuhan--VideoJev-4B/snapshots/1c1a253bf90dfa66176124d25a48ffb791d517d5'
LOCAL_CKPT = ROOT / 'checkpoint_fp16'                # symlinks to HF_SNAPSHOT + patched modeling_videojev.py
DATA = {
    'ucf': dict(root=Path('/mmlab_students/storageStudents/nguyenvd/anomaly_datasets/ucf_crime'),
                ann='annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'),
    'xd': dict(root=Path('/mmlab_students/storageStudents/nguyenvd/anomaly_datasets/xd_violence'),
               ann='annotations/temporal_anomaly_annotation_for_testing_videos.txt'),
}
OUT = HERE / 'outputs'

# ---- settings (all fixed in advance; none was chosen on test scores)
CONTEXT = 8192            # tokens per call for the 1-GPU stages
MAX_FRAMES = 96           # frame cap of the 1-GPU stages
WINDOW_SEC, STRIDE_SEC, WINDOW_FPS = 24., 12., 4.     # local windows (V10 setting)
GLOBAL_FPS = 0.5          # q_global look and full-video grounding
GRID = 0.5                # score grid (s)
BUDGET = 0.20             # candidate budget: 20 % of the duration
PIXEL_CAP = 640 * 640     # official per-frame pixel cap
FULL_MAX_TOKENS = 26880   # full-video grounding: visual tokens per call on 2 x 11 GB (probe: 672 frames, peak 9.0 / 9.9 GiB)
FULL_SPLIT_GIB = (3.6, 6.5)   # weights per card for the 2-GPU split (card 0 also holds vision tower / embeddings)
TEXT_PER_PAIR, TEXT_MARGIN = 24, 256   # timestamp text tokens per frame pair (measured ~22) + chat template / query

# ---- prompt templates (the only prompts used)
GENERIC_EVENT = 'A clearly unusual or dangerous incident occurs.'
ORDINARY_EVENT = 'People or vehicles perform ordinary activities without any visible accident or attack.'
WINDOW_Q = 'Does this video segment contain a clearly visible unusual, dangerous or harmful incident rather than ordinary activity?'
GLOBAL_Q = 'Does this entire video contain a clearly visible unusual, dangerous, or harmful incident?'

TG_GENERIC = {'id': 'tg_generic', 'task': 'grounding', 'query': GENERIC_EVENT}
TG_ORDINARY = {'id': 'tg_ordinary', 'task': 'grounding', 'query': ORDINARY_EVENT}
QA_WINDOW = {'id': 'qa_generic', 'task': 'choice', 'query': WINDOW_Q,
             'options': ['A. Yes, a clearly unusual or harmful event is visible.',
                         'B. No, only ordinary activity is visible.',
                         'C. Insufficient evidence to determine.']}
QA_GLOBAL = {'id': 'q_global', 'task': 'choice', 'query': GLOBAL_Q,
             'options': ['A. Yes', 'B. No', 'C. Insufficient evidence']}
# One causal sequence per window, in this order (V10 TG_first order without the event bank; tg_ordinary is kept only so
# q_window / g(t) are bit-for-bit the V10 cache values -- its output is not used).
WINDOW_QUERIES = [TG_GENERIC, TG_ORDINARY, QA_WINDOW]


# ---- data
def load_annotations(dataset):
    out = {}
    for line in (DATA[dataset]['root'] / DATA[dataset]['ann']).read_text(encoding='utf-8-sig').splitlines():
        x = line.strip().split()
        if not x or x[0].startswith('#'):
            continue
        a = list(map(int, x[2:]))
        out[Path(x[0]).stem] = [(s, e) for s, e in zip(a[::2], a[1::2]) if s >= 0 and e >= s]
    return out


def test_videos(dataset):
    """Names only (labels are not read)."""
    return sorted(load_annotations(dataset))


def index_videos(dataset):
    v = DATA[dataset]['root'] / 'videos'
    return {p.stem: p for p in sorted([*v.glob('*.mp4'), *v.glob('*/*.mp4')])}


def fixed_windows(duration):
    """24 s windows, 12 s stride, last window aligned to the video end."""
    dur = float(duration)
    if dur < .5:
        return []
    win = min(WINDOW_SEC, dur)
    if dur <= win + 1e-6:
        return [(0., dur)]
    starts = list(np.arange(0, max(0., dur - win) + 1e-8, STRIDE_SEC))
    if not starts or (dur - win) - starts[-1] > 1e-5:
        starts.append(dur - win)
    return [(float(x), float(min(dur, x + win))) for x in starts]


class Video:
    """decord reader; frames at the nearest source frame to each requested time."""

    def __init__(self, path):
        import decord
        self.vr = decord.VideoReader(str(path), num_threads=2)
        self.frames = len(self.vr)
        self.fps = float(self.vr.get_avg_fps())
        self.duration = self.frames / self.fps

    def window(self, start, end, fps, max_frames=MAX_FRAMES):
        """Official time grid on [start, end): arange(ceil(d*fps))/fps, frames < d; above max_frames, uniform."""
        d = float(end - start)
        times = np.arange(int(np.ceil(d * fps)), dtype=np.float64) / fps
        times = times[times < d]
        eff = float(fps)
        if max_frames and len(times) > max_frames:
            eff = max_frames / d
            times = np.arange(max_frames, dtype=np.float64) / eff
        idx = np.clip(np.rint((start + times) * self.fps).astype(int), 0, self.frames - 1)
        return self.vr.get_batch(idx.tolist()).asnumpy(), times, eff, d


def pair_timeline(frame_times, duration):
    """Official videojev_temporal.pair_timeline: one candidate per frame pair, centre = mean of the two timestamps,
    bounds halfway between pairs, first bound 0, last bound = duration."""
    t = np.asarray(frame_times, dtype=np.float64)
    first = t[::2]
    second = t[np.minimum(np.arange(0, len(t), 2) + 1, len(t) - 1)]
    edges = np.concatenate(([0.], (t[1:-1:2] + t[2::2]) / 2, [float(duration)]))
    return (first + second) / 2, np.stack((edges[:-1], edges[1:]), axis=1)


def prepare_window(processor, frames, times, eff_fps, duration, context=CONTEXT):
    """Official processing_videojev.prepare_video after decoding: largest per-frame pixel budget whose visual tokens fit
    80 % of the context, processor call, pair timeline."""
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
    from transformers.video_utils import VideoMetadata
    n, height, width, _ = frames.shape
    vp = processor.video_processor
    budget = context * 4 // 5

    def tokens(cap):
        h, w = smart_resize(n, height, width, temporal_factor=vp.temporal_patch_size,
                            factor=vp.patch_size * vp.merge_size, min_pixels=vp.size['shortest_edge'], max_pixels=n * cap)
        return (n + 1) // 2 * (h // 32) * (w // 32)
    low, high = max(1024, math.ceil(vp.size['shortest_edge'] / n)), PIXEL_CAP
    if tokens(low) > budget:
        raise ValueError('window cannot fit the context at minimum native resolution')
    while low < high:
        mid = (low + high + 1) // 2
        low, high = (mid, high) if tokens(mid) <= budget else (low, mid - 1)
    meta = VideoMetadata(total_num_frames=n, fps=eff_fps, frames_indices=list(range(n)))
    native = vp(videos=[frames], video_metadata=[meta], do_sample_frames=False,
                size=dict(shortest_edge=vp.size['shortest_edge'], longest_edge=n * low), return_tensors='pt')
    grid = native['video_grid_thw'][0].tolist()
    centers, bounds = pair_timeline(times, duration)
    record = dict(frames=n, fps=eff_fps, grid_thw=grid, tokens=int(np.prod(grid)) // 4,
                  score_times=centers.tolist(), score_bounds=bounds.tolist(), sampled_duration=duration)
    return native['pixel_values_videos'], record


def native_tokens_per_pair(processor, h, w):
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
    vp = processor.video_processor
    hh, ww = smart_resize(2, h, w, temporal_factor=vp.temporal_patch_size, factor=vp.patch_size * vp.merge_size,
                          min_pixels=vp.size['shortest_edge'], max_pixels=2 * PIXEL_CAP)
    return (hh // 32) * (ww // 32)


def full_context(n_frames, tokens_per_pair):
    pairs = (n_frames + 1) // 2
    vis = pairs * tokens_per_pair
    return int(math.ceil((max(vis / 0.8, vis + TEXT_PER_PAIR * pairs) + TEXT_MARGIN) / 64) * 64)


# ---- model
PATCH_OLD = "torch.autocast('cuda', dtype=torch.bfloat16)"
PATCH_NEW = "torch.autocast('cuda', dtype=torch.float16)"


def make_patched_checkpoint():
    """Symlinks to the shared HF snapshot; modeling_videojev.py gets BF16 -> FP16 autocast. HF cache never modified."""
    LOCAL_CKPT.mkdir(exist_ok=True)
    for src in HF_SNAPSHOT.iterdir():
        dst = LOCAL_CKPT / src.name
        if src.name != 'modeling_videojev.py' and not (dst.exists() or dst.is_symlink()):
            dst.symlink_to(src.resolve())
    raw = (HF_SNAPSHOT / 'modeling_videojev.py').read_text()
    if raw.count(PATCH_OLD) != 2:
        raise RuntimeError('official autocast code changed; inspect before patching')
    raw = raw.replace(PATCH_OLD, PATCH_NEW)
    target = LOCAL_CKPT / 'modeling_videojev.py'
    if not target.is_file() or target.read_text() != raw:
        target.write_text(raw)
    return hashlib.sha256(raw.encode()).hexdigest()


def load_model(n_gpus=1, split_gib=FULL_SPLIT_GIB):
    """fp16 weights, vision tower + decision head fp32 (official prepare_for_inference), SDPA with use_gqa_in_sdpa=False.
    n_gpus=2: accelerate device map over two cards (sequential, weights split_gib per card)."""
    import torch
    from transformers import AutoModel, AutoProcessor
    from transformers.integrations import sdpa_attention
    # torch 2.5 SDPA with enable_gqa=True only has the math kernel; repeating K/V (identical result) uses the
    # memory-efficient kernel on the 2080 Ti.
    sdpa_attention.use_gqa_in_sdpa = lambda *a, **k: False
    sha = make_patched_checkpoint()
    processor = AutoProcessor.from_pretrained(str(LOCAL_CKPT), trust_remote_code=True)
    kw = dict(device_map={'': 'cuda:0'}) if n_gpus == 1 else dict(
        device_map='sequential', max_memory={i: f'{g}GiB' for i, g in enumerate(split_gib)})
    model = AutoModel.from_pretrained(str(LOCAL_CKPT), trust_remote_code=True, dtype=torch.float16,
                                      attn_implementation='sdpa', **kw)
    model.prepare_for_inference()
    if n_gpus > 1:   # the official forward indexes the last hidden state with indices that live on the embedding device
        from accelerate.hooks import remove_hook_from_module
        dev0 = model.get_input_embeddings().weight.device
        remove_hook_from_module(model.head)
        model.head.to(dev0)

        def to_dev0(_m, _a, out):
            out.last_hidden_state = out.last_hidden_state.to(dev0)
            return out
        model.backbone.language_model.register_forward_hook(to_dev0)
    return model, processor, sha


def write_json(path, obj):
    """Atomic write, world-readable (shared project directory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.chmod(tmp, 0o666)
    os.replace(tmp, path)
