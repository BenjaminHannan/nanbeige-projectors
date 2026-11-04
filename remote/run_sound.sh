#!/bin/bash
# Box B: sound data -> smoke -> full audio training (B) -> eval. Training must end by 10:15 ET.
P=/workspace/proj; cd $P; export PROJ_ROOT=$P NANBEIGE_PATH=$P/models/nanbeige TZ=America/New_York PYTHONUNBUFFERED=1
PY=/workspace/venv/bin/python
Z=https://zenodo.org/api/records/4783391/files
mkdir -p data/clotho data/wavcaps data/audio
$PY -B wavcaps_range.py FreeSound ${NFSD:-40000} 30 ${WSTOP:-08:40} > logs/wr_fsd.log 2>&1 &
$PY -B wavcaps_range.py AudioSet_SL ${NAS:-30000} 11 ${WSTOP:-08:40} > logs/wr_as.log 2>&1 &
( cd data/clotho
  for f in clotho_captions_development.csv clotho_captions_evaluation.csv clotho_captions_validation.csv clotho_metadata_evaluation.csv LICENSE clotho_audio_evaluation.7z clotho_audio_development.7z clotho_audio_validation.7z; do
    curl -sL -C - --retry 10 -o $f "$Z/$f/content"; done )
curl -sL -o data/wavcaps/sb_final.json https://huggingface.co/datasets/cvssp/WavCaps/resolve/main/json_files/SoundBible/sb_final.json
curl -sL -o data/wavcaps/SoundBible.zip https://huggingface.co/datasets/cvssp/WavCaps/resolve/main/Zip_files/SoundBible/SoundBible.zip
echo "DOWNLOADED $(date +%T)" >> logs/status.log
for n in clotho_evaluation clotho_development clotho_validation soundbible; do [ -f data/audio/$n.json ] || $PY -B prep_audio.py $n >> logs/prep.log 2>&1 & done
wait
echo "PACKS DONE $(date +%T)" >> logs/status.log
ls -la data/audio >> logs/status.log
$PY -B train.py --modality audio --out runs/smoke_audio2 --samples 100000 --micro 16 --accum 2 --smoke_minutes 3 > logs/smoke_audio2.log 2>&1
SPS=$(grep -o "SMOKE END s/sample [0-9.]*" logs/smoke_audio2.log | awk '{print $4}'); SPS=${SPS:-0.12}
LEFT=$(( $(date -d "today 10:15" +%s) - $(date +%s) - 120 ))
N=$($PY -c "print(max(3200, int($LEFT/$SPS*0.97)//32*32))")
echo "AUDIO PLAN $N samples at $SPS s/sample $(date +%T)" >> logs/status.log
$PY -B train.py --modality audio --out runs/audio2 --samples $N --micro 16 --accum 2 --deadline 10:15 --ckpt_minutes 10 > logs/train_audio2.log 2>&1
echo "AUDIO TRAIN DONE $(date +%T)" >> logs/status.log
$PY -B eval_audio.py runs/audio2/projector.safetensors runs/audio2/eval 500 > logs/eval_audio2.log 2>&1
echo "ALL DONE $(date +%T)" >> logs/status.log
