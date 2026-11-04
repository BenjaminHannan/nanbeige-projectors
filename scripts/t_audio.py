# CPU test of the audio encoder path + stream (no GPU, no LLM).
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(4)
import mm, audio_data
enc = mm.AudioEncoder(device="cpu")
h = audio_data.load_holdout(3)
for w, caps, name in h:
    f, ns = enc.preprocess(w)
    t = time.time(); feats, nt = enc(f[None], torch.tensor([ns]))
    print(name, len(w) / 16000, "s ->", tuple(feats.shape), int(nt[0]), "std", float(feats.std()), "t", round(time.time() - t, 2), caps[0], flush=True)
s = audio_data.AudioStream(packs=["clotho_evaluation"])
it = s.iterate(5)
for _ in range(2):
    w, c, n = next(it); print("stream", n, len(w), c)
