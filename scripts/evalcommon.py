"""Shared evaluation: retrieval (V1/A1), caption loss vs no-media (V2/A2), greedy captions (V3/A3), controls."""
import random, json, time, html, io, base64, re
import torch
import mm


@torch.no_grad()
def caption_ll(model, pre_ids, post_ids, me, mlen, caps, bs=10):
    """Sum log-likelihood of each caption given its media tokens (me[b], mlen[b]); me=None -> no media."""
    out = []
    for i in range(0, len(caps), bs):
        c = caps[i:i + bs]
        X, A, pos, ids = mm.build_batch(model, pre_ids, post_ids, None if me is None else me[i:i + bs],
                                        None if me is None else mlen[i:i + bs], c)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = mm.token_logprobs(model, X, A, pos, ids)
        b = pos[0]
        for k in range(len(c)):
            out.append(float(lp[b == k].sum()))
    return out


def retrieval(model, pre_ids, post_ids, feats_of, caps, n, cands, seed=0, label=""):
    """feats_of(i) -> (me (m,d) projected tokens for item i). For each i rank its candidate set (true + 9)."""
    hits, hits_norm = 0, 0
    for i in range(n):
        me = feats_of(i)
        cs = cands[i]
        me_b = me[None].expand(len(cs), *me.shape)
        ml = torch.full((len(cs),), me.shape[0])
        ll = caption_ll(model, pre_ids, post_ids, me_b, ml, [caps[j] for j in cs])
        hits += int(max(range(len(cs)), key=lambda k: ll[k]) == 0)
        lln = [ll[k] / len(caps[cs[k]]) for k in range(len(cs))]
        hits_norm += int(max(range(len(cs)), key=lambda k: lln[k]) == 0)
        if (i + 1) % 100 == 0:
            print(time.strftime("%H:%M:%S"), label, i + 1, hits / (i + 1), flush=True)
    return hits / n, hits_norm / n


def make_cands(n, k=9, seed=0):
    r = random.Random(seed)
    return [[i] + r.sample([j for j in range(n) if j != i], k) for i in range(n)]


def ngram_degenerate(text, n=4, times=3):
    w = text.lower().split()
    cnt = {}
    for i in range(len(w) - n + 1):
        g = tuple(w[i:i + n]); cnt[g] = cnt.get(g, 0) + 1
        if cnt[g] >= times:
            return True
    return False


@torch.no_grad()
def greedy(model, tok, parts, max_new=48):
    """parts: list of str (rendered literally) or tensors (m,d) of projected media tokens, already in
    chat-template order. Greedy decoding without KV cache (simple and safe with the 2-loop model)."""
    emb = model.get_input_embeddings()
    xs = []
    for p in parts:
        if isinstance(p, str):
            ids = tok(p, add_special_tokens=False).input_ids
            if ids:
                xs.append(emb(torch.tensor(ids, device="cuda")))
        else:
            xs.append(p.to("cuda", emb.weight.dtype))
    X = torch.cat(xs, 0)[None]
    out = []
    for _ in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(inputs_embeds=X, use_cache=False)[0][:, -1]
            nxt = int(model.lm_head(h).float().argmax(-1))
        if nxt == mm.IM_END or nxt == tok.eos_token_id:
            break
        out.append(nxt)
        X = torch.cat([X, emb(torch.tensor([[nxt]], device="cuda"))], 1)
    return tok.decode(out, skip_special_tokens=True).strip()


def chat_parts(tok, user_parts):
    """Wrap interleaved user content (str/tensor list) in the chat template (default system prompt)."""
    s = tok.apply_chat_template([{"role": "user", "content": mm.MEDIA}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    pre, post = s.split(mm.MEDIA)
    return [pre] + user_parts + [post]


def thumb_b64(img, size=160):
    im = img.copy(); im.thumbnail((size, size))
    b = io.BytesIO(); im.convert("RGB").save(b, "JPEG", quality=80)
    return base64.b64encode(b.getvalue()).decode()


def html_sheet(path, title, rows):
    """rows: list of (thumb_html, generated, reference)."""
    h = [f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>body{{font-family:sans-serif}}"
         "td{vertical-align:top;padding:6px;border-bottom:1px solid #ddd}td.r{color:#777;font-size:12px}</style></head>"
         f"<body><h2>{html.escape(title)}</h2><table><tr><th></th><th>model caption (greedy, &le;48 tokens)</th>"
         "<th>reference caption (first 300 chars)</th></tr>"]
    for th, g, r in rows:
        h.append(f"<tr><td>{th}</td><td>{html.escape(g)}</td><td class='r'>{html.escape(r[:300])}</td></tr>")
    h.append("</table></body></html>")
    open(path, "w", encoding="utf-8").write("\n".join(h))
