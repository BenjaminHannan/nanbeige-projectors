"""Round 2 (A): multi-frame / video training for the vision projector.
Mix per sample: ~50 % single image (v1 format), ~40 % sequence of 2-4 unrelated ReCap images,
~10 % "pan video" (4-8 overlapping crops of one image, prompt "Describe this video.").
Sequence format (user):  Frame 1 (t=0s):\n<64 tok>\nFrame 2 (t=2s):\n<64 tok>\n...Describe each frame.
Target:                  Frame 1: <caption 1>\nFrame 2: <caption 2>...
Only the projector trains (initialised from v1). No extra frame-position embedding (the text labels carry position)."""
import os, sys, time, json, random, argparse, glob
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm, data

FRAME_CAP = 48          # per-frame caption budget in sequences (tokens)
PROMPT_SEQ = "Describe each frame."
PROMPT_VID = "Describe this video."


def frame_label(k, t):
    return f"Frame {k} (t={t}s):\n"


def chat_wrap(tok):
    s = tok.apply_chat_template([{"role": "user", "content": mm.MEDIA}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    pre, post = s.split(mm.MEDIA)
    return tok(pre, add_special_tokens=False).input_ids, tok(post, add_special_tokens=False).input_ids


def seq_user(tok, n, times=None, prompt=PROMPT_SEQ):
    """Return list of segments: ints-list for text, int k for 'media slot k'."""
    times = times or [2 * i for i in range(n)]
    segs = []
    for k in range(n):
        segs.append(tok(("" if k == 0 else "\n") + frame_label(k + 1, times[k]), add_special_tokens=False).input_ids)
        segs.append(k)
    segs.append(tok("\n" + prompt, add_special_tokens=False).input_ids)
    return segs


def single_user(tok):
    return [0, tok("\n" + mm.PROMPTS["vision"], add_special_tokens=False).input_ids]


def build(model, wrap, samples, device="cuda"):
    """samples: list of (segments, media (n,64,d) projected, target_ids). Returns X, A, (pos_b,pos_t), ids, start positions."""
    emb = model.get_input_embeddings()
    pre, post = wrap
    E = lambda ids: emb(torch.tensor(ids, device=device))
    seqs, pb, pt, li, starts = [], [], [], [], []
    for b, (segs, media, tgt) in enumerate(samples):
        parts = [E(pre)]
        for s in segs:
            parts.append(media[s].to(parts[0].dtype) if isinstance(s, int) else E(s))
        parts.append(E(post))
        if tgt:
            parts.append(E(tgt))
        x = torch.cat(parts, 0)
        start = x.shape[0] - len(tgt)
        starts.append(start)
        for k, t in enumerate(tgt):
            pb.append(b); pt.append(start - 1 + k); li.append(t)
        seqs.append(x)
    L = max(s.shape[0] for s in seqs); d = seqs[0].shape[1]
    X = torch.zeros(len(seqs), L, d, dtype=seqs[0].dtype, device=device)
    A = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
    for b, s in enumerate(seqs):
        X[b, :s.shape[0]] = s; A[b, :s.shape[0]] = 1
    return X, A, (torch.tensor(pb, device=device), torch.tensor(pt, device=device)), torch.tensor(li, device=device)


def pan_crops(img, n):
    W, H = img.size
    cw, ch = int(W * 0.7), int(H * 0.85)
    out = []
    for k in range(n):
        f = k / max(1, n - 1)
        x0 = int((W - cw) * f); y0 = int((H - ch) * (0.5 + 0.5 * (f - 0.5)))
        out.append(img.crop((x0, y0, x0 + cw, y0 + ch)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--deadline", default="")
    ap.add_argument("--smoke_minutes", type=float, default=0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    P = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
    import subprocess
    smi = lambda: subprocess.run(["nvidia-smi", "--query-gpu=memory.used,power.draw,utilization.gpu", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip()
    torch.manual_seed(args.seed)
    tok, model = mm.load_llm()
    enc = mm.VisionEncoder()
    from safetensors.torch import load_file, save_file
    proj = mm.Projector(enc.dim).cuda()
    proj.load_state_dict({k: v.float() for k, v in load_file(args.init).items()})
    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=0.0)
    wrap = chat_wrap(tok)
    files = sorted(f for f in glob.glob(os.path.join(data.RECAP_DIR, "train-*.parquet")) if not f.endswith(data.HOLDOUT_SHARD))
    stream = data.RecapStream(files, seed=args.seed)        # different seed from v1 -> different order
    eff = args.micro * args.accum
    total = max(1, args.samples // eff)
    ck = os.path.join(args.out, "ckpt.pt")
    step, cursor = 0, 0
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cuda", weights_only=False)
        proj.load_state_dict(s["proj"]); opt.load_state_dict(s["opt"]); step, cursor, total = s["step"], s["cursor"], s["total"]
        P("RESUMED", step, cursor, total)
    P("plan", args.samples, "steps", total)

    def gen(start):
        rng = random.Random(args.seed * 99991 + start)
        it = stream.iterate(start)
        while True:
            r = rng.random()
            n = 1 if r < 0.5 else (rng.choice([2, 3, 4]) if r < 0.9 else -rng.choice([4, 6, 8]))
            k = 1 if n < 0 else n
            got = []
            while len(got) < k:
                img, cap, _ = next(it)
                if img is not None and cap:
                    got.append((img.convert("RGB"), cap))
            if n == 1:
                img, cap = got[0]
                yield single_user(tok), [enc.preprocess(img)], data.trim_caption(tok, cap, 160), k
            elif n > 1:
                tgt = []
                for j, (img, cap) in enumerate(got):
                    c = data.trim_caption(tok, cap, FRAME_CAP)
                    if c and c[-1] == mm.IM_END:
                        c = c[:-1]
                    tgt += tok(("" if j == 0 else "\n") + f"Frame {j + 1}: ", add_special_tokens=False).input_ids + c
                yield seq_user(tok, n), [enc.preprocess(im) for im, _ in got], tgt + [mm.IM_END], k
            else:
                img, cap = got[0]; m = -n
                fr = pan_crops(img, m)
                yield seq_user(tok, m, [i for i in range(m)], PROMPT_VID), [enc.preprocess(f) for f in fr], \
                    data.trim_caption(tok, cap, 96), k

    def save(tag):
        torch.save({"proj": proj.state_dict(), "opt": opt.state_dict(), "step": step, "cursor": cursor, "total": total}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        save_file({k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in proj.state_dict().items()},
                  os.path.join(args.out, "projector.safetensors"))
        P("saved", tag, step, cursor)

    # cursor counts ReCap rows consumed; gen() restarts from it (sample mix re-seeded from the cursor)
    pf = data.Prefetch(gen(cursor), 48)
    deadline = None
    if args.deadline:
        hh, mi = map(int, args.deadline.split(":")); lt = time.localtime()
        deadline = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mi, 0, 0, 0, -1))
    t0 = time.time(); last = time.time(); seen = 0; ema = None
    torch.cuda.reset_peak_memory_stats()
    while step < total:
        for g in opt.param_groups:
            g["lr"] = mm.cosine_lr(step, total, args.lr)
        opt.zero_grad(set_to_none=True)
        tl, tt = 0.0, 0
        for _ in range(args.accum):
            items = [next(pf) for _ in range(args.micro)]
            cursor += sum(x[3] for x in items)
            pix = torch.stack([p for x in items for p in x[1]])
            feats, _ = enc(pix)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pe = proj(feats.cuda())
            samples, o = [], 0
            for segs, ims, tgt, _ in items:
                samples.append((segs, pe[o:o + len(ims)], tgt)); o += len(ims)
            X, A, pos, ids = build(model, wrap, samples)
            lp = mm.token_logprobs(model, X, A, pos, ids)
            (-lp.sum() / (len(ids) * args.accum)).backward()
            tl += float(-lp.detach().sum()); tt += len(ids)
        torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        opt.step(); step += 1; seen += eff
        l = tl / tt; ema = l if ema is None else 0.98 * ema + 0.02 * l
        el = time.time() - t0
        if step % 10 == 0 or step <= 3:
            rec = {"t": time.strftime("%H:%M:%S"), "step": step, "loss": round(l, 4), "ema": round(ema, 4),
                   "lr": opt.param_groups[0]["lr"], "s_per_sample": round(el / seen, 4),
                   "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}
            if step % 50 == 0 or step <= 3:
                rec["smi"] = smi()
            log.write(json.dumps(rec) + "\n"); log.flush(); P(json.dumps(rec))
        if time.time() - last > 10 * 60:
            save("ckpt"); last = time.time()
        if args.smoke_minutes and el > args.smoke_minutes * 60:
            P(f"SMOKE END s/sample {el / seen:.4f} peak_reserved {torch.cuda.max_memory_reserved() / 1e9:.2f} smi {smi()}")
            save("smoke"); return
        if deadline and time.time() > deadline:
            P("DEADLINE"); save("deadline"); return
    save("final"); P("TRAINING DONE")


if __name__ == "__main__":
    main()
