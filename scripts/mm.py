"""Shared pieces: frozen Nanbeige4.2-3B, frozen encoders, the 2-layer MLP projector,
prompt building (chat template), and the loss on caption tokens only."""
import os, math, torch, torch.utils.checkpoint, torch.nn as nn, torch.nn.functional as F

os.environ.pop("REQUESTS_CA_BUNDLE", None)
ROOT = r"C:\Users\benja\premonition-projectors"
LLM_PATH = r"C:\Users\benja\premonition-wincheck\models\nanbeige-hf"   # bf16 Nanbeige/Nanbeige4.2-3B
SIGLIP_PATH = os.path.join(ROOT, "models", "siglip2-base-patch16-256")
WHISPER_PATH = os.path.join(ROOT, "models", "whisper-small")
MEDIA = "<<MEDIA>>"
PROMPTS = {"vision": "Describe this image.", "audio": "Describe this sound."}
IM_END = 166101


class Projector(nn.Module):
    """LLaVA-1.5 projector: Linear(enc_dim, d) -> GELU -> Linear(d, d)."""
    def __init__(self, enc_dim, d_model=3072):
        super().__init__()
        self.fc1 = nn.Linear(enc_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        with torch.no_grad():          # start outputs near Nanbeige's embedding scale (row norm ~1.5, not ~17)
            self.fc2.weight.mul_(0.1); self.fc2.bias.mul_(0.1)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def load_llm(device="cuda", checkpointing=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(LLM_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device)
    assert model.config.num_loops == 2, model.config.num_loops
    for p in model.parameters():
        p.requires_grad_(False)
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()   # dropout is 0.0 in this config; train() only switches checkpointing on
    else:
        model.eval()
    return tok, model


def template_ids(tok, prompt_text):
    """Token ids before and after the media slot for the user turn, ending with the assistant header."""
    s = tok.apply_chat_template([{"role": "user", "content": MEDIA + "\n" + prompt_text}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    pre, post = s.split(MEDIA)
    return tok(pre, add_special_tokens=False).input_ids, tok(post, add_special_tokens=False).input_ids


def caption_ids(tok, text, max_tokens):
    ids = tok(text.strip(), add_special_tokens=False).input_ids
    if len(ids) >= max_tokens:
        return ids[:max_tokens]       # truncated: no end token
    return ids + [IM_END]


# ---------------------------------------------------------------- encoders
class VisionEncoder:
    """SigLIP2-base-patch16-256, frozen. 16x16 patch grid -> 2x2 average pool -> 64 tokens of dim 768."""
    dim = 768

    def __init__(self, device="cuda"):
        from transformers import AutoModel
        m = AutoModel.from_pretrained(SIGLIP_PATH, torch_dtype=torch.bfloat16)
        self.vm = (m.vision_model if hasattr(m, "vision_model") else m).to(device).eval()
        for p in self.vm.parameters():
            p.requires_grad_(False)
        self.device = device

    @staticmethod
    def preprocess(img):
        """PIL image -> float tensor (3,256,256) in [-1,1] (SigLIP2 processor: resize 256 bicubic, mean=std=0.5)."""
        import numpy as np
        from PIL import Image
        img = img.convert("RGB").resize((256, 256), Image.BICUBIC)
        a = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        return (a - 0.5) / 0.5

    @torch.no_grad()
    def __call__(self, pix):            # pix (B,3,256,256)
        h = self.vm(pixel_values=pix.to(self.device, torch.bfloat16)).last_hidden_state   # (B,256,768)
        B = h.shape[0]
        h = h.view(B, 16, 16, 768).permute(0, 3, 1, 2)
        h = F.avg_pool2d(h.float(), 2)                                                   # (B,768,8,8)
        return h.flatten(2).transpose(1, 2).contiguous(), torch.full((B,), 64, dtype=torch.long)


class AudioEncoder:
    """Whisper-small encoder, frozen. 30 s window -> 1500 frames (20 ms); keep the real (unpadded) length,
    average-pool x4 over time -> up to 375 tokens of dim 768 (80 ms each)."""
    dim = 768

    def __init__(self, device="cuda"):
        from transformers import WhisperModel, WhisperFeatureExtractor
        self.fe = WhisperFeatureExtractor.from_pretrained(WHISPER_PATH)
        self.enc = WhisperModel.from_pretrained(WHISPER_PATH, torch_dtype=torch.bfloat16).encoder.to(device).eval()
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.device = device

    def preprocess(self, wav16k):
        """1-D float numpy at 16 kHz (<=30 s) -> (80,3000) log-mel and number of real samples."""
        wav16k = wav16k[: 16000 * 30]
        f = self.fe(wav16k, sampling_rate=16000, return_tensors="pt").input_features[0]
        return f, len(wav16k)

    @torch.no_grad()
    def __call__(self, feats, nsamples):   # feats (B,80,3000), nsamples (B,)
        h = self.enc(feats.to(self.device, torch.bfloat16)).last_hidden_state.float()   # (B,1500,768)
        frames = torch.clamp((torch.as_tensor(nsamples) + 319) // 320, 4, 1500)         # 20 ms per frame
        ntok = (frames + 3) // 4
        B = h.shape[0]
        T = int(ntok.max())
        h = h[:, : T * 4]
        mask = (torch.arange(T * 4)[None, :] < frames[:, None]).to(h.device).float()     # ignore padding frames
        hs = (h * mask[..., None]).view(B, T, 4, 768).sum(2)
        cnt = mask.view(B, T, 4).sum(2).clamp(min=1)
        return (hs / cnt[..., None]).contiguous(), ntok


# ---------------------------------------------------------------- batching + loss
def build_batch(model, pre_ids, post_ids, media_emb, media_len, caps, device="cuda"):
    """media_emb: (B, M, d) projected media tokens (padded), media_len: (B,) real counts.
    caps: list of caption token-id lists (may be empty for scoring prompts only).
    Returns inputs_embeds, attention_mask, label positions (b, t) and label ids for the caption tokens."""
    emb = model.get_input_embeddings()
    B = len(caps)
    pre = emb(torch.tensor(pre_ids, device=device))
    post = emb(torch.tensor(post_ids, device=device))
    seqs, lab_pos_b, lab_pos_t, lab_ids = [], [], [], []
    for b in range(B):
        m = int(media_len[b]) if media_emb is not None else 0
        parts = [pre]
        if m:
            parts.append(media_emb[b, :m].to(pre.dtype))
        parts.append(post)
        c = caps[b]
        if c:
            parts.append(emb(torch.tensor(c, device=device)))
        x = torch.cat(parts, 0)
        start = len(pre_ids) + m + len(post_ids)
        # logits at position t predict token t+1: caption token k is predicted at start-1+k
        for k, tid in enumerate(c):
            lab_pos_b.append(b); lab_pos_t.append(start - 1 + k); lab_ids.append(tid)
        seqs.append(x)
    L = max(s.shape[0] for s in seqs)
    d = seqs[0].shape[1]
    X = torch.zeros(B, L, d, dtype=seqs[0].dtype, device=device)
    A = torch.zeros(B, L, dtype=torch.long, device=device)
    for b, s in enumerate(seqs):
        X[b, : s.shape[0]] = s
        A[b, : s.shape[0]] = 1
    return X, A, (torch.tensor(lab_pos_b, device=device), torch.tensor(lab_pos_t, device=device)), \
        torch.tensor(lab_ids, device=device)


def token_logprobs(model, X, A, pos, ids, chunk=512):
    """Per-label-token log-probabilities (fp32), computing the 166k-way head only at label positions."""
    out = model.model(inputs_embeds=X, attention_mask=A, use_cache=False)
    h = out[0][pos[0], pos[1]]                             # (N, d)
    def piece(hh, yy):
        return -F.cross_entropy(model.lm_head(hh).float(), yy, reduction="none")
    lps = []
    for i in range(0, h.shape[0], chunk):
        if torch.is_grad_enabled():   # recompute the big head in backward instead of storing 166k-wide logits
            lps.append(torch.utils.checkpoint.checkpoint(piece, h[i:i + chunk], ids[i:i + chunk], use_reentrant=False))
        else:
            lps.append(piece(h[i:i + chunk], ids[i:i + chunk]))
    return torch.cat(lps) if lps else h.new_zeros(0)


def cosine_lr(step, total, peak, warm_frac=0.03):
    w = max(1, int(total * warm_frac))
    if step < w:
        return peak * (step + 1) / w
    p = (step - w) / max(1, total - w)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
