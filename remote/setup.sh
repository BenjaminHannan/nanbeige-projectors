#!/bin/bash
# Fresh Linux CUDA box setup for nanbeige-projectors round 2.  usage: bash setup.sh video|sound
# Expects the scripts already copied to /workspace/proj (scp from the Mac) and v1 weights for video.
set -x
ROLE=$1; P=/workspace/proj; export ROLE PROJ_ROOT=$P HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p $P/models $P/data $P/logs $P/runs; cd $P
(apt-get update -qq && apt-get install -y -qq p7zip-full rsync > /dev/null) &
python -m venv --system-site-packages /workspace/venv
/workspace/venv/bin/pip install -q "transformers==4.57.3" "huggingface_hub[hf_transfer]==0.36.2" safetensors sentencepiece \
    protobuf "pyarrow>=21" soundfile librosa scipy av requests accelerate py7zr pillow
git clone -q https://github.com/BenjaminHannan/nanbeige-projectors.git /workspace/repo || true
PY=/workspace/venv/bin/python
$PY - <<'PYEOF'
import os
from huggingface_hub import snapshot_download, hf_hub_download
P = os.environ["PROJ_ROOT"]; role = os.environ.get("ROLE")
snapshot_download("Nanbeige/Nanbeige4.2-3B", revision="b82e54bd609793562a75cbf9337970a93369eab5", local_dir=f"{P}/models/nanbeige", max_workers=8)
print("nanbeige ok", flush=True)
if role == "video":
    snapshot_download("google/siglip2-base-patch16-256", local_dir=f"{P}/models/siglip2-base-patch16-256", max_workers=8)
    for i in [25] + list(range(0, 10)):
        hf_hub_download("lmms-lab/LLaVA-ReCap-558K", f"data/train-{i:05d}-of-00026.parquet", repo_type="dataset", local_dir=f"{P}/data/_recap")
        os.makedirs(f"{P}/data/recap", exist_ok=True)
        os.replace(f"{P}/data/_recap/data/train-{i:05d}-of-00026.parquet", f"{P}/data/recap/train-{i:05d}.parquet")
    snapshot_download("Qwen/Qwen3-VL-2B-Instruct", local_dir=f"{P}/models/qwen3-vl-2b", max_workers=8)
else:
    snapshot_download("openai/whisper-small", local_dir=f"{P}/models/whisper-small", allow_patterns=["*.json", "*.safetensors", "*.txt"], max_workers=8)
print("SETUP DOWNLOADS DONE", flush=True)
PYEOF
wait
echo SETUP DONE
