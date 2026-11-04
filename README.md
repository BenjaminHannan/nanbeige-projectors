# nanbeige-projectors

Vision and sound "projectors" that let **Nanbeige4.2-3B** (a text-only, natively 2-loop LLM) take pictures,
video frames and sounds as input, LLaVA-style. As far as we could find, these are the first public vision/audio
adapters for Nanbeige (checked all Nanbeige repos on Hugging Face on 2026-11-03).

**Status:** vision projector ✅ passed all its pre-set tests (weights in `weights/vision_projector.safetensors`).
Sound projector ❌ **not released**: it missed its retrieval mark (A1: 30.2 % vs the required 40 %), although
it clearly hears something (shuffled-audio control 9 %). Its code, data recipe and results are here; the weights are not.

## In plain English

A language model only understands words. A picture model (SigLIP2) turns a picture into a list of numbers
that describe what's in it, and a sound model (Whisper's listening half) does the same for sounds. Those
numbers are in the "wrong language" for Nanbeige. A **projector** is a tiny translator (two layers) that turns
them into the kind of numbers Nanbeige uses for words. We kept Nanbeige and the picture/sound models
completely frozen and trained only the translator, by showing it tens of thousands of
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

## Data and training run

**Vision.** 4 parquet shards of `lmms-lab/LLaVA-ReCap-558K` (85,868 images with detailed LLaVA-NeXT-34B captions);
shard 25 held out and never trained on. 56,320 samples (1,760 steps × 32, ~0.66 epoch), 2 h 24 min on an RTX 5070 Ti
(0.153 s/sample, peak 10.5 GB allocated / 11.3 GB in nvidia-smi). Final train loss ≈ 1.01.

**Sound (not released).** Clotho v2.1 development + validation (5 captions per clip) + WavCaps SoundBible:
25,652 caption pairs, 33.6 h of audio, converted to 16 kHz mono. Held out: Clotho evaluation split. 31,008 samples
(969 of 1,077 planned steps: the PC rebooted for a Windows update at 04:47, training resumed from the 04:39
checkpoint and stopped at a fixed 05:32 deadline, at LR ≈ 3 % of peak). 0.24 s/sample (≈ 278 audio tokens per clip),
peak 11.7 GB reserved. Final train loss ≈ 2.4.

## Results (pass marks were fixed before training)

Held-out sets: 500 images from the never-trained ReCap shard 25; Clotho v2.1 *evaluation* split for sound.
Retrieval = rank the true caption against 9 random other held-out captions by caption log-likelihood given the media
(chance 10 %). Controls: **zeroed** = encoder features set to 0; **shuffled** = another item's features.
Loss = mean per-token caption loss (captions cut to ≤ 160 tokens at a sentence end).

| mark | target | result | controls | verdict |
|---|---|---|---|---|
| V1 retrieval top-1 of 10 (n=500) | ≥ 60 % | 67.6 % | zeroed 9.6 %, shuffled 10.8 % (chance 10 %) | **PASS** |
| V2 held-out caption loss vs no image | ≥ 25 % lower | 1.048 vs 2.239 (53.2 % lower) | zeroed 9.123, shuffled 1.614 | **PASS** |
| V3 50 greedy captions non-degenerate | ≥ 90 % | 100 % | | **PASS** |
| A1 retrieval top-1 of 10 (n=500) | ≥ 40 % | 30.2 % | zeroed 10.8 %, shuffled 9.0 % | **FAIL** |
| A2 held-out caption loss vs no audio | ≥ 15 % lower | 2.498 vs 7.051 (64.6 % lower) | zeroed 6.576, shuffled 3.141 | **PASS** |
| A3 30 greedy captions non-degenerate | ≥ 90 % | 100 % | | **PASS** |


Extra numbers: vision retrieval with length-normalised log-likelihood: 82.4 % (shuffled 10.0 %).
Important for reading V2: *any* image, even the wrong one, already lowers the loss (shuffled 1.61 vs no image 2.24),
because it tells the model "write a ReCap-style description". The fair content check is true vs shuffled image:
1.05 vs 1.61 (35 % lower), together with retrieval (67.6 % vs 10.8 %).

Sound, extra numbers: length-normalised retrieval 37.4 % (shuffled 8.6 %). The no-audio loss (7.05) is inflated
because without audio Nanbeige answers "Describe this sound." in a completely different way; the fair check is true vs
shuffled audio: 2.50 vs 3.14 (20 % lower). Greedy captions (`results/audio_captions.html`) are fluent and Clotho-style,
often partly right (crowds, wind, running water, machines) and sometimes wrong (bees heard as "a machine… car horn").

Sample captions: `results/vision_captions.html` (50 held-out images with thumbnails).

**Video smoke test (no pass mark).** 4 frames (`frame k, t=…s` + 64 tokens each), never trained on multi-image input.
A simulated camera pan over one held-out image gave: *"The video frames show a series of images that appear to be part of a media template or a news article template. The images are black and white and contain text and a blank space for content. The first frame shows a template with the title "The News" and a section for "The Headline" followed by a bla…"*.
Four unrelated images: the model described frame 1 but mixed in text from frame 2 (*"Health in Relationships:
Solutions for UK"* combines frame 1 "Hybrid Event Solutions UK" and frame 2 "Healing Toxic Injuries in Love
Relationships"). So multi-frame input works, but frames bleed into each other: a multi-image fine-tune is needed for real video.

**Limitations.** Stage-1 only (projector trained, LLM frozen): it describes, it does not follow complex visual
instructions. Small text in images is often misread or invented (e.g. book titles). 64 tokens per image at 256×256 loses
fine detail. Trained with Nanbeige's default system prompt; other system prompts are untested. Trained against the bf16
LLM; using it with a quantized (e.g. 8-bit MLX) Nanbeige is untested.

## Round 2 (2026-11-04): video v2 and sound v2, both failed, not released

Two single-change experiments on rented RTX 5090s. Pass marks were fixed before results; full details are in `REPORT.md`
(Round 2) and `results/round2/`. **Neither v2 projector passed, so the released weights are still the v1 vision projector above.**

**Video v2.** One change: continue v1 on multi-frame sequences (50 % single images, 40 % 2–4 unrelated images, 10 % crop
"pans"; format `Frame k (t=Ns):` + 64 tokens per frame). 49,888 samples, 0.172 s/sample, peak 17.1 GB reserved.

| test (held out) | target | v1 | v2 | Qwen3-VL-2B (yardstick, 64 tokens/image) |
|---|---|---|---|---|
| VID1: caption for frame k in a 4-image sequence, top-1 of 10 (n=300) | ≥ 60 %, gap ≤ 5, ≥ v1 + 15 | 28.7 % | **56.7 %** (FAIL) | 48.7 % |
| same images scored alone | | 72.0 % | 77.7 % | 75.0 % |
| VID1 with frames shuffled (control) | | 17.7 % | 10.0 % | |
| VID2: single-image V1 (n=500) | ≥ 62.6 % | 68.2 % | **71.2 %** (PASS) | 65.8 % |
| VID3: MSR-VTT test clips, 8 frames, top-1 of 10 (n=160) | ≥ v1 + 10 | 20.6 % | **18.8 %** (FAIL) | 28.1 % |

**Sound v2.** One change: 6.5× more audio (Clotho dev+val, SoundBible, 40,000 WavCaps-FreeSound and 30,000 WavCaps-AudioSet_SL
clips: 95,652 pairs, 220 h) and a full schedule. 69,088 samples, 0.107 s/sample, peak 15.8 GB reserved.
Leakage rule: FreeSound clips whose ID is a Clotho-evaluation sound ID or in WavCaps' test blacklist were removed:
1,833 clips, including all 917 Clotho-eval IDs present in WavCaps.

| test (500 held-out Clotho-eval clips) | target | v1 | v2 | Qwen2.5-Omni-3B thinker (yardstick, 4.7B params) |
|---|---|---|---|---|
| A1 retrieval top-1 of 10 | ≥ 40 % | 30.2 % | **30.6 %** (FAIL) (shuffled 10.2 %, zeroed 10.0 %) | 34.4 % |
| A2 loss vs no audio | ≥ 15 % lower | 64.6 % | 64.0 % (PASS) | |
| A3 non-degenerate captions | ≥ 90 % | 100 % | 100 % (PASS) | |

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
- **Training data terms (read before using the weights commercially):**
  - Vision: `lmms-lab/LLaVA-ReCap-558K` (captions written by LLaVA-NeXT-34B for the LCS-558K images, which come from
    LAION / CC3M / SBU). The dataset card gives no license; LLaVA's data is released for **research use**, and the
    caption-writer model has its own terms. Treat the vision projector as research-use.
  - Sound: Clotho v2.1 (Tampere University). Captions: **experimental and non-commercial use only**, commercial use
    prohibited (see Clotho's LICENSE; this notice is reproduced in `LICENSES/CLOTHO_LICENSE.txt`). Audio: Freesound
    clips under their own CC licenses. WavCaps SoundBible subset: WavCaps says **academic use only**.
  - So: code is Apache-2.0, but the trained weights inherit research / non-commercial restrictions from the data.
  - Round 2 (no weights released): WavCaps FreeSound and AudioSet_SL subsets are **academic use only** (WavCaps terms;
    FreeSound audio under each clip's CC license, AudioSet from YouTube). MSR-VTT test clips were used **for evaluation
    only** (frames are not redistributed; `results/round2/msrvtt_clips.json` lists video IDs and captions). Yardstick
    models Qwen3-VL-2B-Instruct and Qwen2.5-Omni-3B were only run for comparison; see their own licenses.
