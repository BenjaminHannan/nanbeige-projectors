"""Round 2 (A) tests. usage: eval_video.py <projector.safetensors> <out_dir> <tag>
VID1: 75 held-out 4-frame sequences of unrelated held-out images; per frame k, rank caption k against 9 distractors
      (the other 3 frames' captions + 6 other held-out captions) as the continuation of "Frame k: ".
      Control: frame-shuffled sequences (frames reordered, labels/captions kept).
VID2: single-image V1 retrieval on the same 500 held-out images as round 1.
VID3: MSR-VTT 1k-A test clips (never trained on), 8 frames, prompt "Describe this video.", caption top-1 of 10."""
import os, sys, json, time, random
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm, data, evalcommon as E, video as V
from safetensors.torch import load_file
from PIL import Image

proj_path, out, tag = sys.argv[1], sys.argv[2], sys.argv[3]
only = sys.argv[4] if len(sys.argv) > 4 else "all"
os.makedirs(out, exist_ok=True)
P = lambda *a: print(time.strftime("%H:%M:%S"), tag, *a, flush=True)
tok, model = mm.load_llm(checkpointing=False)
enc = mm.VisionEncoder()
proj = mm.Projector(enc.dim).cuda()
proj.load_state_dict({k: v.float() for k, v in load_file(proj_path).items()}); proj.eval()
wrap = V.chat_wrap(tok)
res = {"tag": tag, "projector": proj_path}
resf = os.path.join(out, f"video_eval_{tag}.json")
if os.path.exists(resf):
    res.update(json.load(open(resf)))
def save(): json.dump(res, open(resf, "w"), indent=1)

@torch.no_grad()
def project(imgs):
    f, _ = enc(torch.stack([enc.preprocess(im) for im in imgs]))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return proj(f).float()

@torch.no_grad()
def score(segs, media, prefix_ids, cands):
    """Sum log-prob of each candidate's tokens after prefix_ids, same media for all."""
    samples = [(segs, media, prefix_ids + c) for c in cands]
    X, A, pos, ids = V.build(model, wrap, samples)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lp = mm.token_logprobs(model, X, A, pos, ids)
    outs, o = [], 0
    for c in cands:
        n = len(prefix_ids) + len(c)
        outs.append(float(lp[o + len(prefix_ids): o + n].sum())); o += n
    return outs

hold = data.load_holdout(500)
def cap48(c):
    ids = data.trim_caption(tok, c, V.FRAME_CAP)
    return ids[:-1] if ids and ids[-1] == mm.IM_END else ids

# ---------------- VID1 (+ frame-shuffled control, + the same images scored alone in single-image format)
# Pass mark (changed by Ben before any round-2 results): per-frame top-1 >= 60 %, gap to "same images alone" <= 5 points,
# and >= v1 baseline + 15 points.
NS = 75
rng = random.Random(11)
VID1_ITEMS = []                       # (frame image ids, k, candidate image ids [true first], perm)
for s in range(NS):
    ids4 = list(range(4 * s, 4 * s + 4))
    perm = [1, 2, 3, 0] if s % 3 == 0 else ([2, 3, 0, 1] if s % 3 == 1 else [3, 0, 1, 2])   # no frame stays put
    for k in range(4):
        d = [i for i in ids4 if i != ids4[k]] + rng.sample(range(300, 500), 6)
        VID1_ITEMS.append((ids4, k, [ids4[k]] + d, perm))
if only in ("all", "vid1") and "VID1_top1" not in res:
    caps = [cap48(hold[i][1]) for i in range(500)]
    hits = {"true": 0, "shuf": 0, "single": 0}; n = 0
    pre1, post1 = mm.template_ids(tok, mm.PROMPTS["vision"])
    cache = {}
    for ids4, k, cand_ids, perm in VID1_ITEMS:
        key = tuple(ids4)
        if key not in cache:
            cache = {key: project([hold[i][0] for i in ids4])}
        media = cache[key]
        segs = V.seq_user(tok, 4)
        cand = [caps[i] for i in cand_ids]
        pre = tok(f"Frame {k + 1}: ", add_special_tokens=False).input_ids
        for kind, med in (("true", media), ("shuf", media[perm])):
            sc = score(segs, med, pre, cand)
            hits[kind] += int(max(range(10), key=lambda j: sc[j]) == 0)
        # same image alone, single-image format (v1 prompt), same 10 candidate captions
        sc = E.caption_ll(model, pre1, post1, media[k][None].expand(10, *media[k].shape), torch.full((10,), 64), cand)
        hits["single"] += int(max(range(10), key=lambda j: sc[j]) == 0)
        n += 1
        if n % 100 == 0:
            P("VID1", n, {kk: round(v / n, 3) for kk, v in hits.items()})
    res["VID1_top1"] = hits["true"] / n; res["VID1_frame_shuffled_top1"] = hits["shuf"] / n
    res["VID1_same_images_alone_top1"] = hits["single"] / n; res["VID1_gap_points"] = 100 * (hits["single"] - hits["true"]) / n
    res["VID1_n"] = n
    save(); P("VID1 done", json.dumps({k: v for k, v in res.items() if k.startswith("VID1")}))

# ---------------- VID2 (single-image V1, same as round 1)
if only in ("all", "vid2") and "VID2_V1_top1" not in res:
    pre_ids, post_ids = mm.template_ids(tok, mm.PROMPTS["vision"])
    caps = [data.trim_caption(tok, c, 160) for _, c, _ in hold]
    me = torch.cat([project([im for im, _, _ in hold[i:i + 32]]) for i in range(0, 500, 32)])
    cands = E.make_cands(500, 9, seed=0)
    res["VID2_V1_top1"], _ = E.retrieval(model, pre_ids, post_ids, lambda i: me[i], caps, 500, cands, label=tag + " V1")
    save(); P("VID2 done", res["VID2_V1_top1"])

# ---------------- VID3 (real video, MSR-VTT test)
R = os.path.join(os.environ.get("PROJ_ROOT", r"C:\Users\benja\premonition-projectors"), "data", "msrvtt")
if only in ("all", "vid3") and "VID3_top1" not in res and os.path.exists(os.path.join(R, "clips.json")):
    clips = json.load(open(os.path.join(R, "clips.json")))
    N = len(clips)
    caps = [tok(c["caption"].strip().capitalize() + ".", add_special_tokens=False).input_ids + [mm.IM_END] for c in clips]
    cands = E.make_cands(N, 9, seed=3)
    hits = 0; hits_shuf = 0
    for i, c in enumerate(clips):
        media = project([Image.open(os.path.join(R, c["video_id"], f"f{k}.jpg")) for k in range(8)])
        segs = V.seq_user(tok, 8, c["times"], V.PROMPT_VID)
        sc = score(segs, media, [], [caps[j] for j in cands[i]])
        hits += int(max(range(10), key=lambda j: sc[j]) == 0)
    res["VID3_top1"] = hits / N; res["VID3_n"] = N
    save(); P("VID3 done", res["VID3_top1"], "n", N)

# ---------------- examples
if only in ("all", "examples") and "examples" not in res:
    ex = {}
    segs_imgs = [hold[i][0] for i in range(400, 404)]
    parts = []
    media = project(segs_imgs)
    for k in range(4):
        parts += [("" if k == 0 else "\n") + V.frame_label(k + 1, 2 * k), media[k]]
    parts.append("\n" + V.PROMPT_SEQ)
    ex["seq4"] = {"output": E.greedy(model, tok, E.chat_parts(tok, parts), 160),
                  "references": [hold[i][1][:160] for i in range(400, 404)]}
    if os.path.exists(os.path.join(R, "clips.json")):
        clips = json.load(open(os.path.join(R, "clips.json")))
        ex["videos"] = []
        for c in clips[:3]:
            media = project([Image.open(os.path.join(R, c["video_id"], f"f{k}.jpg")) for k in range(8)])
            parts = []
            for k in range(8):
                parts += [("" if k == 0 else "\n") + V.frame_label(k + 1, c["times"][k]), media[k]]
            parts.append("\n" + V.PROMPT_VID)
            ex["videos"].append({"video_id": c["video_id"], "reference": c["caption"],
                                 "output": E.greedy(model, tok, E.chat_parts(tok, parts), 80)})
    res["examples"] = ex; save(); P("examples", json.dumps(ex)[:800])
P("EVAL DONE", json.dumps({k: v for k, v in res.items() if k != "examples"}))
