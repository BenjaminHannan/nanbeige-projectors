"""Turn downloaded Clotho 7z / WavCaps SoundBible zip into compact 16 kHz mono int16 packs, then delete the
big archives. usage: prep_audio.py clotho_evaluation | clotho_development | clotho_validation | soundbible
Output: data/audio/<name>.pcm (int16, concatenated) + <name>.json [{name, off, n, captions}]"""
import os, sys, io, csv, json, glob, shutil, zipfile
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from math import gcd

ROOT = r"C:\Users\benja\premonition-projectors"
OUT = os.path.join(ROOT, "data", "audio"); os.makedirs(OUT, exist_ok=True)
MAX_S = 30


def to16k(x, sr):
    if x.ndim > 1:
        x = x.mean(1)
    x = x[: sr * MAX_S]
    if sr != 16000:
        g = gcd(16000, sr); x = resample_poly(x, 16000 // g, sr // g)
    return np.clip(x * 32767, -32768, 32767).astype(np.int16)


def pack(name, items):
    """items: iterator of (clip_name, raw_bytes, captions)"""
    meta, off = [], 0
    with open(os.path.join(OUT, name + ".pcm.tmp"), "wb") as f:
        for cn, raw, caps in items:
            try:
                x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            except Exception as e:
                print("skip", cn, e, flush=True); continue
            y = to16k(x, sr)
            if len(y) < 1600:
                continue
            f.write(y.tobytes()); meta.append({"name": cn, "off": off, "n": len(y), "captions": caps}); off += len(y)
            if len(meta) % 200 == 0:
                print(name, len(meta), flush=True)
    os.replace(os.path.join(OUT, name + ".pcm.tmp"), os.path.join(OUT, name + ".pcm"))
    json.dump(meta, open(os.path.join(OUT, name + ".json"), "w"))
    print("PACKED", name, len(meta), "clips", off / 16000 / 3600, "hours", flush=True)


def clotho(split):
    d = os.path.join(ROOT, "data", "clotho")
    caps = {}
    for r in csv.DictReader(open(os.path.join(d, f"clotho_captions_{split}.csv"), encoding="utf-8")):
        caps[r["file_name"]] = [r[f"caption_{k}"] for k in range(1, 6)]
    arch = os.path.join(d, f"clotho_audio_{split}.7z")
    tmp = os.path.join(d, "x_" + split)
    import py7zr
    with py7zr.SevenZipFile(arch, "r") as z:
        z.extractall(tmp)
    def items():
        for p in sorted(glob.glob(os.path.join(tmp, "**", "*.wav"), recursive=True)):
            b = os.path.basename(p)
            if b in caps:
                yield b, open(p, "rb").read(), caps[b]
    pack("clotho_" + split, items())
    shutil.rmtree(tmp); os.remove(arch)


def soundbible():
    d = os.path.join(ROOT, "data", "wavcaps")
    js = json.load(open(os.path.join(d, "sb_final.json"), encoding="utf-8"))
    caps = {str(x["id"]): x["caption"] for x in js["data"]}
    zp = os.path.join(d, "SoundBible.zip")
    def items():
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                stem = os.path.splitext(os.path.basename(n))[0]
                if stem in caps:
                    yield stem, z.read(n), [caps[stem]]
    pack("soundbible", items())
    os.remove(zp)


if __name__ == "__main__":
    w = sys.argv[1]
    if w.startswith("clotho_"):
        clotho(w.split("_", 1)[1])
    else:
        soundbible()
