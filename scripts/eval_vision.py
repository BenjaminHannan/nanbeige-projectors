"""Vision pass marks V1-V3 + controls + video smoke. usage: eval_vision.py <projector.safetensors> <out_dir> [n=500]"""
import os, sys, json, time, random
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm, data, evalcommon as E
from safetensors.torch import load_file

proj_path, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 500
MAXCAP = 160
os.makedirs(out, exist_ok=True)
def P(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
tok, model = mm.load_llm(checkpointing=False)
enc = mm.VisionEncoder()
proj = mm.Projector(enc.dim).cuda()
proj.load_state_dict({k: v.float() for k, v in load_file(proj_path).items()}); proj.eval()
pre_ids, post_ids = mm.template_ids(tok, mm.PROMPTS["vision"])
P("loading holdout")
hold = data.load_holdout(N)
caps = [data.trim_caption(tok, c, MAXCAP) for _, c, _ in hold]
feats = []
for i in range(0, N, 32):
    f, _ = enc(torch.stack([enc.preprocess(im) for im, _, _ in hold[i:i + 32]]))
    feats.append(f)
feats = torch.cat(feats)                                   # (N,64,768) encoder features
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    me = proj(feats).float()                               # projected
    me_zero = proj(torch.zeros_like(feats[:1])).float()[0]
cands = E.make_cands(N, 9, seed=0)
res = {"n": N, "max_cap_tokens": MAXCAP, "projector": proj_path}

# V1 retrieval + controls
res["V1_top1"], res["V1_top1_lennorm"] = E.retrieval(model, pre_ids, post_ids, lambda i: me[i], caps, N, cands, label="true")
P("V1", res["V1_top1"])
perm = list(range(N)); random.Random(7).shuffle(perm)
perm = [p if p != i else (p + 1) % N for i, p in enumerate(perm)]   # never its own image
res["shuffled_top1"], res["shuffled_top1_lennorm"] = E.retrieval(model, pre_ids, post_ids, lambda i: me[perm[i]], caps, N, cands, label="shuffled")
P("shuffled", res["shuffled_top1"])
# zeroed features: same constant media for every caption -> score each caption once
ll_zero = E.caption_ll(model, pre_ids, post_ids, me_zero[None].expand(N, *me_zero.shape), torch.full((N,), 64), caps)
res["zeroed_top1"] = sum(int(max(range(10), key=lambda k: ll_zero[cands[i][k]]) == 0) for i in range(N)) / N
P("zeroed", res["zeroed_top1"])

# V2 held-out caption loss: with image vs no image at all (prompt text only)
ntok = sum(len(c) for c in caps)
ll_img = E.caption_ll(model, pre_ids, post_ids, me, torch.full((N,), 64), caps)
ll_none = E.caption_ll(model, pre_ids, post_ids, None, None, caps)
res["V2_loss_image"] = -sum(ll_img) / ntok
res["V2_loss_noimage"] = -sum(ll_none) / ntok
res["V2_loss_zeroed"] = -sum(ll_zero) / ntok
ll_shuf = E.caption_ll(model, pre_ids, post_ids, me[perm], torch.full((N,), 64), caps)
res["V2_loss_shuffled"] = -sum(ll_shuf) / ntok
res["V2_reduction"] = 1 - res["V2_loss_image"] / res["V2_loss_noimage"]
P("V2", res["V2_loss_image"], res["V2_loss_noimage"], res["V2_reduction"])

# V3 greedy captions on 50 held-out images
rows, gens = [], []
for i in range(50):
    g = E.greedy(model, tok, E.chat_parts(tok, [me[i], "\n" + mm.PROMPTS["vision"]]), 48)
    gens.append({"id": hold[i][2], "caption": g, "reference": hold[i][1][:400],
                 "empty": len(g.split()) == 0, "degenerate": E.ngram_degenerate(g)})
    rows.append((f"<img src='data:image/jpeg;base64,{E.thumb_b64(hold[i][0])}'>", g, hold[i][1]))
ok = sum(1 for x in gens if not x["empty"] and not x["degenerate"])
res["V3_ok_frac"] = ok / 50
E.html_sheet(os.path.join(out, "vision_captions.html"), "Nanbeige4.2-3B + SigLIP2 projector: 50 held-out images", rows)
json.dump(gens, open(os.path.join(out, "vision_captions.json"), "w"), indent=1)
P("V3", res["V3_ok_frac"])

# Video smoke: (a) a simulated 4-frame camera pan over one held-out image, (b) 4 different held-out images
def frames_parts(imgs, times):
    f, _ = enc(torch.stack([enc.preprocess(im) for im in imgs]))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pe = proj(f).float()
    parts = []
    for k, t in enumerate(times):
        parts += [f"frame {k + 1}, t={t}s\n", pe[k], "\n"]
    return parts
im = hold[50][0]; W, H = im.size; cw = int(W * 0.6)
pan = [im.crop((int((W - cw) * k / 3), 0, int((W - cw) * k / 3) + cw, H)) for k in range(4)]
video = {}
video["pan_reference"] = hold[50][1][:400]
video["pan_caption"] = E.greedy(model, tok, E.chat_parts(tok, frames_parts(pan, [0, 2, 4, 6]) +
                                ["Describe what is shown across these video frames."]), 96)
seq = [hold[j][0] for j in (51, 52, 53, 54)]
video["seq_references"] = [hold[j][1][:200] for j in (51, 52, 53, 54)]
video["seq_caption"] = E.greedy(model, tok, E.chat_parts(tok, frames_parts(seq, [0, 2, 4, 6]) +
                                ["Describe what is shown in each frame."]), 128)
res["video_smoke"] = video
P("video", json.dumps(video)[:600])
pan_rows = [(f"<img src='data:image/jpeg;base64,{E.thumb_b64(p)}'>", f"frame {k+1}", "") for k, p in enumerate(pan)]
E.html_sheet(os.path.join(out, "video_smoke_frames.html"), "Video smoke: pan frames. Output: " + video["pan_caption"], pan_rows)
json.dump(res, open(os.path.join(out, "vision_eval.json"), "w"), indent=1)
P("EVAL DONE", json.dumps({k: v for k, v in res.items() if k != "video_smoke"}))
