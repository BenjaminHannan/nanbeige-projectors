"""Data streams. Deterministic order (seeded) with an integer cursor so a run can resume exactly."""
import io, os, re, json, random, threading, queue
import numpy as np

ROOT = os.environ.get("PROJ_ROOT", r"C:\Users\benja\premonition-projectors")
RECAP_DIR = os.path.join(ROOT, "data", "recap")
HOLDOUT_SHARD = "train-00025.parquet"      # never trained on


def trim_caption(tok, text, max_tokens):
    """Keep whole sentences up to max_tokens (so the model learns to end cleanly), then add <|im_end|>."""
    from mm import IM_END
    text = " ".join(text.replace("\n", " ").split())
    sents = re.split(r"(?<=[.!?])\s+", text)
    out, n = [], 0
    for s in sents:
        ids = tok(("" if not out else " ") + s, add_special_tokens=False).input_ids
        if n + len(ids) > max_tokens - 1:
            break
        out.extend(ids); n += len(ids)
    if not out:                                  # one huge sentence: hard cut, no end token
        return tok(text, add_special_tokens=False).input_ids[:max_tokens]
    return out + [IM_END]


class RecapStream:
    """LLaVA-ReCap-558K parquet shards. Order = shuffled row groups, shuffled rows inside each, per epoch."""

    def __init__(self, files, seed=0):
        import pyarrow.parquet as pq
        self.files = files
        self.groups = []                          # (file, rg, nrows)
        for f in files:
            md = pq.ParquetFile(f).metadata
            for g in range(md.num_row_groups):
                self.groups.append((f, g, md.row_group(g).num_rows))
        self.n = sum(g[2] for g in self.groups)
        self.seed = seed

    def order(self, epoch):
        r = random.Random(self.seed * 1000 + epoch)
        gs = list(range(len(self.groups))); r.shuffle(gs)
        return gs

    def iterate(self, start):
        """Yield (pil_image, caption_text, id) from global cursor `start` onwards, forever."""
        import pyarrow.parquet as pq
        from PIL import Image
        epoch, pos = divmod(start, self.n)
        while True:
            c = 0
            for gi in self.order(epoch):
                f, g, nr = self.groups[gi]
                if c + nr <= pos:
                    c += nr; continue
                t = pq.ParquetFile(f).read_row_group(g, columns=["id", "image", "conversations"]).to_pylist()
                idx = list(range(nr)); random.Random(self.seed * 7919 + epoch * 131 + gi).shuffle(idx)
                for j in idx:
                    if c >= pos:
                        row = t[j]
                        try:
                            img = Image.open(io.BytesIO(row["image"]["bytes"]))
                            cap = [x["value"] for x in row["conversations"] if x["from"] == "gpt"][0]
                        except Exception:
                            img, cap = None, None
                        yield img, cap, row["id"]
                    c += 1
            epoch += 1; pos = 0


def load_holdout(n=500, seed=1234):
    """n images + captions from the held-out shard (fixed sample)."""
    import pyarrow.parquet as pq
    from PIL import Image
    pf = pq.ParquetFile(os.path.join(RECAP_DIR, HOLDOUT_SHARD))
    t, g = [], 0
    while len(t) < 4 * n and g < pf.num_row_groups:      # a pool of >= 4n rows, then a fixed random sample
        t += pf.read_row_group(g, columns=["id", "image", "conversations"]).to_pylist(); g += 1
    idx = list(range(len(t))); random.Random(seed).shuffle(idx)
    out = []
    for j in idx[:n]:
        r = t[j]
        out.append((Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"),
                    [c["value"] for c in r["conversations"] if c["from"] == "gpt"][0], r["id"]))
    return out


class Prefetch:
    """Run a generator in a background thread."""
    def __init__(self, gen, size=64):
        self.q = queue.Queue(size)
        def run():
            for x in gen:
                self.q.put(x)
        threading.Thread(target=run, daemon=True).start()

    def __next__(self):
        return self.q.get()

    def __iter__(self):
        return self
