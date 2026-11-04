"""Train a projector (LLaVA-1.5 stage 1): frozen encoder + frozen Nanbeige4.2-3B (2 native loops), MLP trains.
usage: train.py --modality vision|audio --out runs\\vision --samples N [--micro 8 --accum 4 --max_cap 160]
       [--smoke_minutes 10]   (smoke: run that long, print s/sample + peak VRAM, no LR schedule commitment)
Resumable: rerun the same command; it continues from runs\\<out>\\ckpt.pt."""
import argparse, os, sys, time, json, glob, subprocess, random
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm, data

ap = argparse.ArgumentParser()
ap.add_argument("--modality", default="vision")
ap.add_argument("--out", required=True)
ap.add_argument("--samples", type=int, default=40000)
ap.add_argument("--micro", type=int, default=8)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--max_cap", type=int, default=160)
ap.add_argument("--smoke_minutes", type=float, default=0)
ap.add_argument("--ckpt_minutes", type=float, default=15)
ap.add_argument("--deadline", default="")          # "HH:MM" local: save + stop cleanly at this time
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
log = open(os.path.join(args.out, "train_log.jsonl"), "a")
def P(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
def smi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,power.draw,utilization.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return str(e)

torch.manual_seed(args.seed)
tok, model = mm.load_llm()
P("llm loaded", smi())
if args.modality == "vision":
    enc = mm.VisionEncoder()
    files = sorted(f for f in glob.glob(os.path.join(data.RECAP_DIR, "train-*.parquet")) if not f.endswith(data.HOLDOUT_SHARD))
    stream = data.RecapStream(files, seed=args.seed)
    P("train files", [os.path.basename(f) for f in files], "rows", stream.n)
    def gen(start):
        for img, cap, _id in stream.iterate(start):
            if img is None or not cap:
                yield None; continue
            yield enc.preprocess(img), data.trim_caption(tok, cap, args.max_cap), 64
else:
    import audio_data
    enc = mm.AudioEncoder()
    stream = audio_data.AudioStream(seed=args.seed)
    P("audio train clips", stream.n)
    def gen(start):
        for wav, cap, _id in stream.iterate(start):
            if wav is None or not cap:
                yield None; continue
            f, ns = enc.preprocess(wav)
            yield f, data.trim_caption(tok, cap, args.max_cap), ns

proj = mm.Projector(enc.dim).cuda()
opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
pre_ids, post_ids = mm.template_ids(tok, mm.PROMPTS[args.modality])
eff = args.micro * args.accum
total_steps = max(1, args.samples // eff)
ck = os.path.join(args.out, "ckpt.pt")
step, cursor = 0, 0
if os.path.exists(ck):
    s = torch.load(ck, map_location="cuda", weights_only=False)
    proj.load_state_dict(s["proj"]); opt.load_state_dict(s["opt"]); step, cursor = s["step"], s["cursor"]
    total_steps = s.get("total_steps", total_steps)
    P(f"RESUMED step {step} cursor {cursor} total_steps {total_steps}")
P(f"plan: samples {args.samples} eff_batch {eff} total_steps {total_steps}")

def save(tag="ckpt"):
    tmp = ck + ".tmp"
    torch.save({"proj": proj.state_dict(), "opt": opt.state_dict(), "step": step, "cursor": cursor,
                "total_steps": total_steps, "args": vars(args)}, tmp)
    os.replace(tmp, ck)
    from safetensors.torch import save_file
    save_file({k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in proj.state_dict().items()},
              os.path.join(args.out, "projector.safetensors"))
    P("saved", tag, "step", step, "cursor", cursor)

it = data.Prefetch(gen(cursor), 96)
t_start = time.time(); last_ck = time.time(); seen_run = 0
torch.cuda.reset_peak_memory_stats()
deadline = None
if args.deadline:
    hh, mi = map(int, args.deadline.split(":"))
    lt = time.localtime(); d = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mi, 0, 0, 0, -1))
    deadline = d if d > time.time() else d + 86400
ema = None
while step < total_steps:
    for g in opt.param_groups:
        g["lr"] = mm.cosine_lr(step, total_steps, args.lr)
    opt.zero_grad(set_to_none=True)
    tot_loss, tot_tok = 0.0, 0
    for a in range(args.accum):
        items = []
        while len(items) < args.micro:
            x = next(it); cursor += 1
            if x is not None:
                items.append(x)
        inp = torch.stack([x[0] for x in items])
        caps = [x[1] for x in items]
        if args.modality == "vision":
            feats, mlen = enc(inp)
        else:
            feats, mlen = enc(inp, torch.tensor([x[2] for x in items]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            me = proj(feats.cuda())
        X, A, pos, ids = mm.build_batch(model, pre_ids, post_ids, me, mlen, caps)
        lp = mm.token_logprobs(model, X, A, pos, ids)
        loss = -lp.sum() / (sum(len(c) for c in caps) * args.accum)   # mean token loss over the micro-batches
        loss.backward()
        tot_loss += float(-lp.sum()); tot_tok += len(ids)
    torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
    opt.step(); step += 1; seen_run += eff
    l = tot_loss / tot_tok; ema = l if ema is None else 0.98 * ema + 0.02 * l
    el = time.time() - t_start
    if step % 10 == 0 or step <= 3:
        rec = {"t": time.strftime("%H:%M:%S"), "step": step, "loss": round(l, 4), "ema": round(ema, 4),
               "lr": opt.param_groups[0]["lr"], "s_per_sample": round(el / seen_run, 4),
               "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}
        if step % 50 == 0 or step <= 3:
            rec["smi"] = smi()
        log.write(json.dumps(rec) + "\n"); log.flush(); P(json.dumps(rec))
    if time.time() - last_ck > args.ckpt_minutes * 60:
        save(); last_ck = time.time()
    if args.smoke_minutes and el > args.smoke_minutes * 60:
        P(f"SMOKE END s/sample {el/seen_run:.4f} peak_alloc {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
          f"peak_reserved {torch.cuda.max_memory_reserved()/1e9:.2f} GB smi {smi()}")
        save("smoke"); sys.exit(0)
    if deadline and time.time() > deadline:
        P("DEADLINE reached"); save("deadline"); sys.exit(0)
save("final")
P("TRAINING DONE")
