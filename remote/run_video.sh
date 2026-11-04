#!/bin/bash
# Box A: v1 baseline -> video training (A) -> v2 eval -> Qwen3-VL yardstick. Training must end by 10:05 ET.
P=/workspace/proj; cd $P; export PROJ_ROOT=$P NANBEIGE_PATH=$P/models/nanbeige TZ=America/New_York PYTHONUNBUFFERED=1
PY=/workspace/venv/bin/python
$PY -B fetch_msrvtt.py > logs/fetch_msrvtt.log 2>&1
for w in vid1 vid2 vid3 examples; do $PY -B eval_video.py runs/vision/projector.safetensors runs/video_eval v1 $w >> logs/eval_v1.log 2>&1; done
echo "V1 BASELINE DONE $(date +%T)" >> logs/status.log
$PY -B video.py --out runs/smoke_video2 --init runs/vision/projector.safetensors --samples 100000 --micro 16 --accum 2 --lr 3e-4 --smoke_minutes 3 > logs/smoke_video2.log 2>&1
SPS=$(grep -o "SMOKE END s/sample [0-9.]*" logs/smoke_video2.log | awk '{print $4}'); SPS=${SPS:-0.1}
LEFT=$(( $(date -d "today 10:05" +%s) - $(date +%s) - 120 ))
N=$($PY -c "print(max(3200, int($LEFT/$SPS*0.97)//32*32))")
echo "VIDEO PLAN $N samples at $SPS s/sample $(date +%T)" >> logs/status.log
$PY -B video.py --out runs/video2 --init runs/vision/projector.safetensors --samples $N --micro 16 --accum 2 --lr 3e-4 --deadline 10:05 > logs/train_video.log 2>&1
echo "VIDEO TRAIN DONE $(date +%T)" >> logs/status.log
for w in vid1 vid2 vid3 examples; do $PY -B eval_video.py runs/video2/projector.safetensors runs/video_eval v2 $w >> logs/eval_v2.log 2>&1; done
echo "V2 EVAL DONE $(date +%T)" >> logs/status.log
$PY -B vlm_yardstick.py runs/video_eval > logs/yardstick.log 2>&1
echo "ALL DONE $(date +%T)" >> logs/status.log
