"""MLX loader for the Nanbeige4.2-3B vision / sound projectors.

The projector maps frozen-encoder features to Nanbeige input embeddings (d_model = 3072):
    fc2( gelu( fc1(x) ) )      fc1: enc_dim -> 3072, fc2: 3072 -> 3072   (LLaVA-1.5 style MLP)

Encoder features expected (computed outside this file, e.g. with transformers / mlx-vlm / mlx-whisper):
  vision: SigLIP2-base-patch16-256 vision tower `last_hidden_state` (256 x 768, image resized to 256x256,
          mean=std=0.5), reshaped to a 16x16 grid and 2x2 average-pooled -> 64 x 768 (row-major).
  sound : Whisper-small encoder `last_hidden_state` (1500 x 768 for a 30 s padded window); keep only the
          real frames ceil(n_samples/320), then average-pool x4 over time -> ceil(frames/4) x 768.

Prompt layout used in training (Nanbeige chat template, default system prompt, enable_thinking=False):
    <|im_start|>system ...<|im_end|>\n<|im_start|>user\n[MEDIA TOKENS]\nDescribe this image.<|im_end|>\n
    <|im_start|>assistant\n<think>\n\n</think>\n\n
Embed the text pieces with the model's embed_tokens and concatenate the projector output where
[MEDIA TOKENS] is (see `splice`). For video, repeat "frame k, t=Ns\n" + 64 tokens + "\n" per frame.

Note: GELU is the exact (erf) form, as in torch.nn.functional.gelu.
"""
import mlx.core as mx
import mlx.nn as nn


class Projector(nn.Module):
    def __init__(self, enc_dim: int = 768, d_model: int = 3072):
        super().__init__()
        self.fc1 = nn.Linear(enc_dim, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


def load_projector(path: str, dtype=mx.float16) -> Projector:
    """Load projector.safetensors (keys fc1.weight [3072,enc], fc1.bias, fc2.weight, fc2.bias)."""
    w = mx.load(path)
    enc_dim = w["fc1.weight"].shape[1]
    p = Projector(enc_dim, w["fc1.weight"].shape[0])
    p.load_weights([(k, v.astype(dtype)) for k, v in w.items()], strict=True)
    mx.eval(p.parameters())
    return p


def pool_siglip(last_hidden: mx.array) -> mx.array:
    """(B,256,768) -> (B,64,768): 16x16 grid, 2x2 average pool, row-major."""
    B, N, D = last_hidden.shape
    g = last_hidden.reshape(B, 8, 2, 8, 2, D)
    return g.mean(axis=(2, 4)).reshape(B, 64, D)


def pool_whisper(last_hidden: mx.array, n_samples: int) -> mx.array:
    """(1500,768) Whisper encoder output for one clip -> (ceil(frames/4),768), ignoring padding frames."""
    frames = max(4, min(1500, (n_samples + 319) // 320))
    T = (frames + 3) // 4
    h = last_hidden[: T * 4]
    mask = (mx.arange(T * 4) < frames).astype(h.dtype)[:, None]
    s = (h * mask).reshape(T, 4, -1).sum(axis=1)
    c = mask.reshape(T, 4).sum(axis=1, keepdims=True)
    return s / mx.maximum(c, 1)


def splice(embed_tokens, pre_ids, media_embeds, post_ids) -> mx.array:
    """Build inputs_embeds (1, L, d) = embed(pre) + media + embed(post)."""
    pre = embed_tokens(mx.array(pre_ids))
    post = embed_tokens(mx.array(post_ids))
    return mx.concatenate([pre, media_embeds.astype(pre.dtype), post], axis=0)[None]


if __name__ == "__main__":   # tiny shape test only (no big model loads)
    import tempfile, os
    import numpy as np
    enc, d = 8, 16
    rng = np.random.default_rng(0)
    W = {"fc1.weight": rng.standard_normal((d, enc)).astype(np.float32), "fc1.bias": rng.standard_normal(d).astype(np.float32),
         "fc2.weight": rng.standard_normal((d, d)).astype(np.float32), "fc2.bias": rng.standard_normal(d).astype(np.float32)}
    path = os.path.join(tempfile.mkdtemp(), "p.safetensors")
    mx.save_safetensors(path, {k: mx.array(v) for k, v in W.items()})
    p = load_projector(path, mx.float32)
    x = rng.standard_normal((2, 5, enc)).astype(np.float32)
    y = np.array(p(mx.array(x)))
    from math import erf
    h = x @ W["fc1.weight"].T + W["fc1.bias"]
    h = h * 0.5 * (1 + np.vectorize(erf)(h / np.sqrt(2)))
    ref = h @ W["fc2.weight"].T + W["fc2.bias"]
    assert y.shape == (2, 5, d) and np.allclose(y, ref, atol=1e-3), np.abs(y - ref).max()
    # pooling vs a reference loop
    lh = rng.standard_normal((1, 256, 4)).astype(np.float32)
    pooled = np.array(pool_siglip(mx.array(lh)))
    g = lh.reshape(16, 16, 4)
    refp = np.stack([g[2 * i:2 * i + 2, 2 * j:2 * j + 2].mean((0, 1)) for i in range(8) for j in range(8)])
    assert np.allclose(pooled[0], refp, atol=1e-5)
    wh = rng.standard_normal((1500, 4)).astype(np.float32)
    pw = np.array(pool_whisper(mx.array(wh), 16000 * 5 + 100))     # 5.006 s -> 251 frames -> 63 tokens
    assert pw.shape == (63, 4) and np.allclose(pw[-1], wh[248:251].mean(0), atol=1e-5) and np.allclose(pw[0], wh[:4].mean(0), atol=1e-5)
    emb = nn.Embedding(10, d)
    s = splice(emb, [1, 2], p(mx.array(x[0])), [3])
    assert s.shape == (1, 2 + 5 + 1, d)
    print("mlx_projector self-test OK")
