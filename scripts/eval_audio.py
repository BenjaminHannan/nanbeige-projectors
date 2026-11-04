"""Sound pass marks A1-A3 + controls on held-out Clotho evaluation clips.
usage: eval_audio.py <projector.safetensors> <out_dir> [n=500]"""
import os, sys, json, time, random, html
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm, data, audio_data, evalcommon as E
from safetensors.torch import load_file

proj_path, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 500
MAXCAP = 160
os.makedirs(out, exist_ok=True)
def P(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
tok, model = mm.load_llm(checkpointing=False)
enc = mm.AudioEncoder()
proj = mm.Projector(enc.dim).cuda()
proj.load_state_dict({k: v.float() for k, v in load_file(proj_path).items()}); proj.eval()
pre_ids, post_ids = mm.template_ids(tok, mm.PROMPTS["audio"])
hold = audio_data.load_holdout(N)
caps = [data.trim_caption(tok, c[0], MAXCAP) for _, c, _ in hold]
mes, lens = [], []
for i in range(0, N, 16):
    chunk = [enc.preprocess(w) for w, _, _ in hold[i:i + 16]]
    f, nt = enc(torch.stack([c[0] for c in chunk]), torch.tensor([c[1] for c in chunk]))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pe = proj(f).float()
    for b in range(len(chunk)):
        mes.append(pe[b, : int(nt[b])]); lens.append(int(nt[b]))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    me_zero = proj(torch.zeros(1, 64, enc.dim, device="cuda")).float()[0]
cands = E.make_cands(N, 9, seed=0)
res = {"n": N, "max_cap_tokens": MAXCAP, "projector": proj_path, "mean_audio_tokens": sum(lens) / N}
res["A1_top1"], res["A1_top1_lennorm"] = E.retrieval(model, pre_ids, post_ids, lambda i: mes[i], caps, N, cands, label="true")
P("A1", res["A1_top1"])
perm = list(range(N)); random.Random(7).shuffle(perm)
perm = [p if p != i else (p + 1) % N for i, p in enumerate(perm)]
res["shuffled_top1"], res["shuffled_top1_lennorm"] = E.retrieval(model, pre_ids, post_ids, lambda i: mes[perm[i]], caps, N, cands, label="shuffled")
P("shuffled", res["shuffled_top1"])
ll_zero = E.caption_ll(model, pre_ids, post_ids, me_zero[None].expand(N, *me_zero.shape), torch.full((N,), 64), caps)
res["zeroed_top1"] = sum(int(max(range(10), key=lambda k: ll_zero[cands[i][k]]) == 0) for i in range(N)) / N
P("zeroed", res["zeroed_top1"])

def ll_each(media):
    return [E.caption_ll(model, pre_ids, post_ids, media[i][None], torch.tensor([media[i].shape[0]]), [caps[i]])[0] for i in range(N)]
ntok = sum(len(c) for c in caps)
ll_aud = ll_each(mes)
ll_none = E.caption_ll(model, pre_ids, post_ids, None, None, caps)
ll_shuf = ll_each([mes[perm[i]] for i in range(N)])
res["A2_loss_audio"] = -sum(ll_aud) / ntok
res["A2_loss_noaudio"] = -sum(ll_none) / ntok
res["A2_loss_zeroed"] = -sum(ll_zero) / ntok
res["A2_loss_shuffled"] = -sum(ll_shuf) / ntok
res["A2_reduction"] = 1 - res["A2_loss_audio"] / res["A2_loss_noaudio"]
P("A2", res["A2_loss_audio"], res["A2_loss_noaudio"], res["A2_reduction"])

gens = []
for i in range(30):
    g = E.greedy(model, tok, E.chat_parts(tok, [mes[i], "\n" + mm.PROMPTS["audio"]]), 48)
    gens.append({"clip": hold[i][2], "caption": g, "references": hold[i][1], "empty": len(g.split()) == 0,
                 "degenerate": E.ngram_degenerate(g)})
res["A3_ok_frac"] = sum(1 for x in gens if not x["empty"] and not x["degenerate"]) / 30
json.dump(gens, open(os.path.join(out, "audio_captions.json"), "w"), indent=1)
E.html_sheet(os.path.join(out, "audio_captions.html"), "Nanbeige4.2-3B + Whisper-small projector: 30 held-out Clotho clips",
             [(html.escape(x["clip"]), x["caption"], " | ".join(x["references"])) for x in gens])
json.dump(res, open(os.path.join(out, "audio_eval.json"), "w"), indent=1)
P("EVAL DONE", json.dumps(res))
