"""Yardstick (reported only, not a pass mark): a real small VLM, Qwen3-VL-2B-Instruct, on the same retrieval tests
(V1 single images, VID1 sequences, VID3 MSR-VTT), scored by caption log-likelihood given the image(s).
Images are given at ~256x256 (= 64 visual tokens each after Qwen's 2x2 merge), the same budget as our projector.
usage: vlm_yardstick.py <out_dir>"""
import os, sys, json, time, random
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, evalcommon as E
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, AutoTokenizer

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
MP = os.path.join(os.environ.get("PROJ_ROOT", r"C:\Users\benja\premonition-projectors"), "models", "qwen3-vl-2b")
P = lambda *a: print(time.strftime("%H:%M:%S"), "qwen3vl", *a, flush=True)
proc = AutoProcessor.from_pretrained(MP, min_pixels=256 * 256, max_pixels=256 * 256)
model = Qwen3VLForConditionalGeneration.from_pretrained(MP, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ntok = AutoTokenizer.from_pretrained(os.environ.get("NANBEIGE_PATH", r"C:\Users\benja\premonition-wincheck\models\nanbeige-hf"), trust_remote_code=True)
res = {"model": "Qwen/Qwen3-VL-2B-Instruct", "image_pixels": "256x256 (64 visual tokens)"}
resf = os.path.join(out, "yardstick_qwen3vl2b.json")
def save(): json.dump(res, open(resf, "w"), indent=1)

def sq(im):
    return im.convert("RGB").resize((256, 256))

@torch.no_grad()
def score(content, images, prefix, cands):
    """content: list of {'type':'image'} / {'type':'text'} for the user turn; returns sum logprob of each cand."""
    msgs = [{"role": "user", "content": content}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    lp_base = proc(text=[prompt + prefix], images=images or None, return_tensors="pt")["input_ids"].shape[1]
    outs = []
    for c in cands:
        enc = proc(text=[prompt + prefix + c], images=images or None, return_tensors="pt").to("cuda")
        logits = model(**enc).logits[0, :-1].float()
        ids = enc["input_ids"][0, 1:]
        lp = -F.cross_entropy(logits, ids, reduction="none")
        outs.append(float(lp[lp_base - 1:].sum()))
    return outs

hold = data.load_holdout(500)
dec = lambda ids: ntok.decode([i for i in ids if i != 166101], skip_special_tokens=True).strip()
capfull = [dec(data.trim_caption(ntok, c, 160)) for _, c, _ in hold]
cap48 = [dec(data.trim_caption(ntok, c, 48)) for _, c, _ in hold]

# V1
cands = E.make_cands(500, 9, seed=0); hits = 0
for i in range(500):
    sc = score([{"type": "image"}, {"type": "text", "text": "Describe this image."}], [sq(hold[i][0])], "", [capfull[j] for j in cands[i]])
    hits += int(max(range(10), key=lambda k: sc[k]) == 0)
    if (i + 1) % 100 == 0: P("V1", i + 1, hits / (i + 1))
res["V1_top1"] = hits / 500; save()

# VID1 (same items as eval_video.py)
rng = random.Random(11); hits = 0; hs = 0; n = 0
for s in range(75):
    ids4 = list(range(4 * s, 4 * s + 4))
    content = []
    for k in range(4):
        content += [{"type": "text", "text": ("" if k == 0 else "\n") + f"Frame {k + 1} (t={2 * k}s):\n"}, {"type": "image"}]
    content.append({"type": "text", "text": "\nDescribe each frame."})
    imgs = [sq(hold[i][0]) for i in ids4]
    for k in range(4):
        d = [i for i in ids4 if i != ids4[k]] + rng.sample(range(300, 500), 6)
        cand = [cap48[ids4[k]]] + [cap48[i] for i in d]
        sc = score(content, imgs, f"Frame {k + 1}: ", cand)
        hits += int(max(range(10), key=lambda j: sc[j]) == 0)
        sc1 = score([{"type": "image"}, {"type": "text", "text": "Describe this image."}], [imgs[k]], "", cand)
        hs += int(max(range(10), key=lambda j: sc1[j]) == 0)
        n += 1
P("VID1", hits / n, "alone", hs / n)
res["VID1_top1"] = hits / n; res["VID1_same_images_alone_top1"] = hs / n; save()

# VID3
R = os.path.join(os.environ.get("PROJ_ROOT", r"C:\Users\benja\premonition-projectors"), "data", "msrvtt")
clips = json.load(open(os.path.join(R, "clips.json"))); N = len(clips)
caps = [c["caption"].strip().capitalize() + "." for c in clips]
cands = E.make_cands(N, 9, seed=3); hits = 0
for i, c in enumerate(clips):
    content = []
    for k in range(8):
        content += [{"type": "text", "text": ("" if k == 0 else "\n") + f"Frame {k + 1} (t={c['times'][k]}s):\n"}, {"type": "image"}]
    content.append({"type": "text", "text": "\nDescribe this video."})
    imgs = [sq(Image.open(os.path.join(R, c["video_id"], f"f{k}.jpg"))) for k in range(8)]
    sc = score(content, imgs, "", [caps[j] for j in cands[i]])
    hits += int(max(range(10), key=lambda k: sc[k]) == 0)
res["VID3_top1"] = hits / N; res["VID3_n"] = N; save()
P("YARDSTICK DONE", json.dumps(res))
