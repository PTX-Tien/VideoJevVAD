#!/usr/bin/env bash
# VideoJev Global-Local VAD (clean). Run as nguyenvd_drone. DS = ucf | xd. MODE = gl_v10 (main) | gl_v12 | gl_v12_qwin | gl_gen_tg | gl_gen | gl_gen_contrast.
#   bash run.sh reproduce                    # CPU: score + evaluate all modes on both datasets from the existing caches
#   bash run.sh smoke G [G2]                 # GPU: tiny run of each stage -> outputs/smoke (G2: also the 2-GPU full stage)
#   bash run.sh window DS G [G ...]          # GPU: local windows (24 s / 4 fps): q_window, g(t)   (1 shard per GPU)
#   bash run.sh global DS G [G ...]          # GPU: q_global (whole video, 0.5 fps, <= 96 frames)
#   bash run.sh full DS G1 G2                # 2 GPU: whole-video grounding g_full(t) (0.5 fps, native res)
#   bash run.sh gen DS G [G ...]             # GPU: InternVL3.5-2B writes the TG query / question / options per video
#   bash run.sh window_gen DS G [G ...]      # GPU: VideoJev answers the InternVL texts in every window (needs gen)
#   bash run.sh gen_all DS G [G ...]         # gen then window_gen, then score + eval gl_gen_tg (foreground)
#   bash run.sh score DS MODE [auto|clean|cache]   # CPU: timeline c(t) + 20 % candidates
#   bash run.sh eval DS MODE                 # CPU: metrics (GT read only here)
#   bash run.sh status
# GPU stages are resumable (done videos are skipped). Logs: outputs/logs/.
umask 000
cd "$(dirname "$(readlink -f "$0")")"
PY=/mmlab_students/storageStudents/nguyenvd/anaconda3/envs/vjepa2/bin/python
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PWD/../pydeps          # flash-linear-attention 0.2.2 (Triton 3.1 kernels)
mkdir -p outputs/logs
quiet() { grep --line-buffered -v -E "Loading weights|Triton version|cap_pixels_per_frame|causal_conv1d"; }
shards() {   # shards STAGE DS G...: one background process per GPU
  st=$1; ds=$2; shift 2; n=$#; i=0
  for g in "$@"; do
    if [ $st == gen ]; then cmd="gen_intern.py"; else cmd="infer.py $st"; fi
    CUDA_VISIBLE_DEVICES=$g nohup $PY -u $cmd --dataset $ds --num-shards $n --shard-index $i \
      > outputs/logs/${st}_${ds}_${i}of${n}.log 2>&1 < /dev/null &
    echo "$st $ds shard $i/$n on GPU $g -> outputs/logs/${st}_${ds}_${i}of${n}.log"; i=$((i + 1))
  done
}
case $1 in
  reproduce)
    for ds in ucf xd; do for m in gl_v10 gl_v12 gl_v12_qwin; do
      $PY score.py --dataset $ds --mode $m --source cache && $PY evaluate.py --dataset $ds --mode $m | sed -n 5p
    done; done ;;
  smoke)
    O=outputs/smoke
    CUDA_VISIBLE_DEVICES=$2 $PY -u infer.py window --dataset ucf --only-video Abuse028_x264 --max-windows 2 --redo --out $O/raw_window 2>&1 | quiet
    CUDA_VISIBLE_DEVICES=$2 $PY -u infer.py global --dataset ucf --only-video Abuse028_x264 --redo --out $O/raw_global 2>&1 | quiet
    [ -n "$3" ] && CUDA_VISIBLE_DEVICES=$2,$3 $PY -u infer.py full --dataset ucf --only-video RoadAccidents021_x264 \
      --only-video Shooting032_x264 --redo --out $O/raw_full 2>&1 | quiet ;;
  window|global|gen|window_gen) shards $1 $2 "${@:3}" ;;
  gen_all)
    shards gen $2 "${@:3}"; wait
    echo "gen errors: $(grep -l '"ok": false' outputs/$2/gen_intern/*.json 2>/dev/null | wc -l)"
    shards window_gen $2 "${@:3}"; wait
    $PY score.py --dataset $2 --mode gl_gen_tg && $PY evaluate.py --dataset $2 --mode gl_gen_tg ;;
  full)
    CUDA_VISIBLE_DEVICES=$3,$4 $PY -u infer.py full --dataset $2 2>&1 | quiet | tee -a outputs/logs/full_$2.log ;;
  score) $PY score.py --dataset $2 --mode $3 --source ${4:-auto} ;;
  eval) $PY evaluate.py --dataset $2 --mode $3 ;;
  status)
    for ds in ucf xd; do n=$(wc -l < <($PY -c "import vj; print('\n'.join(vj.test_videos('$ds')))"))
      for st in window global full window_gen; do echo "$ds $st: $(ls outputs/$ds/raw_$st 2>/dev/null | wc -l)/$n, errors: $(grep -l '"ok": false' outputs/$ds/raw_$st/*.json 2>/dev/null | wc -l)"; done
      echo "$ds gen_intern: $(ls outputs/$ds/gen_intern 2>/dev/null | wc -l)/$n, errors: $(grep -l '"ok": false' outputs/$ds/gen_intern/*.json 2>/dev/null | wc -l)"
    done ;;
  *) sed -n 2,14p "$0" ;;
esac
