# nanbeige-projectors

Vision and sound "projectors" that let **Nanbeige4.2-3B** (a text-only, natively 2-loop LLM) take pictures,
video frames and sounds as input, LLaVA-style. As far as we could find, these are the first public vision/audio
adapters for Nanbeige (checked all Nanbeige repos on Hugging Face on 2026-11-03).

<!-- RESULTS_TABLE -->

## In plain English

A language model only understands words. A picture model (SigLIP2) turns a picture into a list of numbers
that describe what's in it, and a sound model (Whisper's listening half) does the same for sounds. Those
numbers are in the "wrong language" for Nanbeige. A **projector** is a tiny translator (two layers) that turns
them into the kind of numbers Nanbeige uses for words. We kept Nanbeige and the picture/sound models
completely frozen and trained only the translator, by showing it hundreds of thousands of
(picture, description) and (sound, description) pairs and asking Nanbeige to write the description.
Afterwards Nanbeige can "see" a picture or "hear" a sound: it describes it and can tell which description
matches. It is a small first step: it describes, it does not deeply reason about images.

## Technical details

**Architecture** (LLaVA-1.5 stage 1). Frozen encoder → 2-layer MLP `Linear(768, 3072) → GELU → Linear(3072, 3072)`
(≈ 11.8 M parameters, ~24 MB in bf16) → inserted as input embeddings into frozen Nanbeige4.2-3B
(`Nanbeige/Nanbeige4.2-3B`, bf16, its native `num_loops = 2`, no extra loops).

| | vision | sound |
|---|---|---|
| encoder (frozen) | `google/siglip2-base-patch16-256`, vision tower `last_hidden_state` | `openai/whisper-small` encoder `last_hidden_state` |
| tokens | 16×16 patches → 2×2 average pool → **64 tokens / image** | 20 ms frames, real (unpadded) length only → ×4 average pool → **1 token / 80 ms** (≤ 375 for 30 s) |
| prompt | `<media>\nDescribe this image.` | `<media>\nDescribe this sound.` |

Prompt = Nanbeige chat template with its default system prompt and `enable_thinking=False`
(assistant turn starts with `<think>\n\n</think>\n\n`). Loss on the caption tokens (+ `<|im_end|>`) only.
Captions are cut at a sentence boundary to ≤ 160 tokens, then end with `<|im_end|>` so the model learns to stop.

Video: each frame is 64 tokens, written as `frame 1, t=0s\n<64 tokens>\nframe 2, t=2s\n<64 tokens>\n...` followed
by the question. (Trained on single images only; multi-frame use is a zero-shot extension, see results.)

**Recipe.** AdamW, LR 1e-3, cosine, 3 % warmup, effective batch 32 (micro 8 × 4 accumulation), grad clip 1.0,
no weight decay, bf16 autocast, gradient checkpointing through the frozen LLM, 166k-vocab head evaluated only at caption
positions. fc2 initialised at 0.1× default scale so the first outputs match Nanbeige's embedding scale.
Trained on one RTX 5070 Ti (16 GB).

<!-- DATA_AND_RESULTS -->

## How to use (transformers)

```python
import torch, sys
sys.path.insert(0, "scripts")
import mm, evalcommon as E
from safetensors.torch import load_file
from PIL import Image
tok, model = mm.load_llm(checkpointing=False)          # edit mm.LLM_PATH to your Nanbeige4.2-3B folder
enc = mm.VisionEncoder()                               # edit mm.SIGLIP_PATH
proj = mm.Projector(768).cuda(); proj.load_state_dict({k: v.float() for k, v in load_file("weights/vision_projector.safetensors").items()})
feats, _ = enc(torch.stack([enc.preprocess(Image.open("photo.jpg"))]))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    media = proj(feats)[0]
print(E.greedy(model, tok, E.chat_parts(tok, [media, "\nDescribe this image."]), 96))
```

## How to use (MLX, Apple silicon)

`mlx_projector.py` loads the same safetensors into MLX (`load_projector`), has the exact pooling functions
(`pool_siglip`, `pool_whisper`) and `splice()` to put the projected tokens between the embedded prompt pieces.
Run `python mlx_projector.py` for its self-test (tiny random shapes). The encoders themselves must be run separately
(e.g. transformers on CPU/MPS, or an MLX port) to produce the features described in the file's docstring.

## Licenses and data terms

- Nanbeige4.2-3B: Apache-2.0. SigLIP2: Apache-2.0. Whisper: MIT. The projector weights and code here: Apache-2.0.
<!-- LICENSE_DATA -->
