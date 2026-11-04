# Download LLaVA-ReCap-558K parquet shards + SigLIP2 encoder into the project folder.
import os, sys
os.environ.pop("REQUESTS_CA_BUNDLE", None); os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import hf_hub_download, snapshot_download
root = r"C:\Users\benja\premonition-projectors"
snapshot_download("google/siglip2-base-patch16-256", local_dir=os.path.join(root, "models", "siglip2-base-patch16-256"))
print("siglip done", flush=True)
shards = [int(x) for x in sys.argv[1:]] or [25, 0, 1, 2, 3]
for i in shards:
    p = hf_hub_download("lmms-lab/LLaVA-ReCap-558K", f"data/train-{i:05d}-of-00026.parquet", repo_type="dataset", local_dir=os.path.join(root, "data", "recap"))
    print("got", p, flush=True)
print("ALL DONE", flush=True)
