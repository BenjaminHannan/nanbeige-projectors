# Sanity: inputs_embeds path gives sane text, and one projector backward works. Tiny, ~1 min.
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm, evalcommon as E
tok, model = mm.load_llm(checkpointing=True)
model.eval()
print("text sanity:", E.greedy(model, tok, E.chat_parts(tok, ["What is the capital of France? Answer in one word."]), 12), flush=True)
model.train()
enc = mm.VisionEncoder()
from PIL import Image
img = Image.new("RGB", (300, 200), (200, 30, 30))
f, ml = enc(torch.stack([enc.preprocess(img)] * 8))
print("feat", f.shape, f.dtype, float(f.std()), flush=True)
proj = mm.Projector(768).cuda()
pre, post = mm.template_ids(tok, "Describe this image.")
caps = [tok("A plain red picture with nothing else in it. " * 20, add_special_tokens=False).input_ids[:160]] * 8
torch.cuda.reset_peak_memory_stats(); t = time.time()
for _ in range(3):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        me = proj(f.cuda())
    X, A, pos, ids = mm.build_batch(model, pre, post, me, ml, caps)
    lp = mm.token_logprobs(model, X, A, pos, ids)
    loss = -lp.mean(); loss.backward()
    torch.cuda.synchronize()
print("loss", float(loss), "grad", float(proj.fc1.weight.grad.norm()), "s/iter(8)", (time.time() - t) / 3,
      "peak GB", torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9, flush=True)
emb = model.get_input_embeddings().weight
print("embed row norm mean", float(emb.float().norm(dim=1).mean()), "proj out norm", float(me.float().norm(dim=-1).mean()))
