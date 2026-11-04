"""Audio-caption stream over the packed 16 kHz int16 files made by prep_audio.py.
Train = Clotho development (+ validation if present) x 5 captions + WavCaps SoundBible. Held out = Clotho evaluation."""
import os, json, random
import numpy as np

ROOT = r"C:\Users\benja\premonition-projectors"
A = os.path.join(ROOT, "data", "audio")
TRAIN_PACKS = ["clotho_development", "clotho_validation", "soundbible"]
HOLDOUT_PACK = "clotho_evaluation"


def _open(name):
    meta = json.load(open(os.path.join(A, name + ".json")))
    pcm = np.memmap(os.path.join(A, name + ".pcm"), dtype=np.int16, mode="r")
    return meta, pcm


class AudioStream:
    def __init__(self, seed=0, packs=None):
        self.items = []                  # (pack_idx, clip_idx, caption)
        self.packs = []
        for name in packs or TRAIN_PACKS:
            if not os.path.exists(os.path.join(A, name + ".json")):
                print("audio pack missing (skipped):", name, flush=True); continue
            meta, pcm = _open(name)
            pi = len(self.packs); self.packs.append((name, meta, pcm))
            for ci, m in enumerate(meta):
                for c in m["captions"]:
                    if c and c.strip():
                        self.items.append((pi, ci, c.strip()))
        self.n = len(self.items)
        self.seed = seed
        print("audio packs", [(p[0], len(p[1])) for p in self.packs], "pairs", self.n, flush=True)

    def iterate(self, start):
        epoch, pos = divmod(start, self.n)
        while True:
            order = list(range(self.n)); random.Random(self.seed * 1000 + epoch).shuffle(order)
            for k in order[pos:]:
                pi, ci, cap = self.items[k]
                _, meta, pcm = self.packs[pi]
                m = meta[ci]
                wav = np.asarray(pcm[m["off"]: m["off"] + m["n"]], dtype=np.float32) / 32768.0
                yield wav, cap, m["name"]
            epoch += 1; pos = 0


def load_holdout(n=500, seed=1234):
    meta, pcm = _open(HOLDOUT_PACK)
    idx = list(range(len(meta))); random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        m = meta[i]
        out.append((np.asarray(pcm[m["off"]: m["off"] + m["n"]], dtype=np.float32) / 32768.0, m["captions"], m["name"]))
    return out
