"""Sound yardstick (report only, not a pass mark): Qwen2.5-Omni-3B *thinker* (audio encoder + 3B LLM, talker unused)
on the same A1 retrieval as eval_audio.py: 500 held-out Clotho-evaluation clips, true caption vs the same 9 distractors
(make_cands seed 0), scored by caption log-likelihood given the audio. usage: audio_yardstick.py <out_dir>"""
import os, sys, json, time
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_data, evalcommon as E
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
MP = os.path.join(os.environ.get("PROJ_ROOT", "."), "models", "qwen25-omni-3b")
P = lambda *a: print(time.strftime("%H:%M:%S"), "omni", *a, flush=True)
proc = Qwen2_5OmniProcessor.from_pretrained(MP)
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MP, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
nparam = sum(p.numel() for p in model.parameters())
N = 500
hold = audio_data.load_holdout(N)
caps = [c[0].strip() for _, c, _ in hold]
cands = E.make_cands(N, 9, seed=0)
msgs = [{"role": "user", "content": [{"type": "audio", "audio": "x"}, {"type": "text", "text": "Describe this sound."}]}]
prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def score(wav, cs):
    base = proc(text=[prompt], audio=[wav], sampling_rate=16000, return_tensors="pt")["input_ids"].shape[1]
    outs = []
    for c in cs:
        enc = proc(text=[prompt + c], audio=[wav], sampling_rate=16000, return_tensors="pt").to("cuda")
        enc = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in enc.items()}
        lg = model(**enc).logits[0, :-1].float(); ids = enc["input_ids"][0, 1:]
        outs.append(float(-F.cross_entropy(lg, ids, reduction="none")[base - 1:].sum()))
    return outs

hits = 0; t0 = time.time()
for i in range(N):
    sc = score(hold[i][0][: 16000 * 30], [caps[j] for j in cands[i]])
    hits += int(max(range(10), key=lambda k: sc[k]) == 0)
    if (i + 1) % 50 == 0: P(i + 1, hits / (i + 1), f"{(time.time() - t0) / (i + 1):.2f}s/clip")
res = {"model": "Qwen/Qwen2.5-Omni-3B (thinker only)", "thinker_params": nparam, "A1_top1": hits / N, "n": N,
       "note": "same 500 clips, same candidate sets as eval_audio.py; caption = first reference"}
json.dump(res, open(os.path.join(out, "yardstick_qwen25omni3b.json"), "w"), indent=1)
P("YARDSTICK DONE", json.dumps(res))
