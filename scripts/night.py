"""PC-side orchestrator for the rest of the night (so no GPU time is lost between steps):
wait for vision training -> vision eval -> audio smoke (4 min) -> audio training sized to end by AUDIO_END
-> audio eval. Each step is a child process; logs in logs\\night_*.log. Writes logs\\night_state.json."""
import os, sys, time, json, re, subprocess
R = r"C:\Users\benja\premonition-projectors"
PY = os.path.join(R, "venv", "Scripts", "python.exe")
AUDIO_END = "05:32"           # audio training must stop by then (eval ~15 min, Bedwars resume before 06:15)
state = {}
def P(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def save(): json.dump(state, open(os.path.join(R, "logs", "night_state.json"), "w"), indent=1)
def run(name, args):
    P("start", name, args); state[name] = {"start": time.strftime("%H:%M:%S")}; save()
    with open(os.path.join(R, "logs", f"night_{name}.log"), "a") as lf:
        r = subprocess.run([PY, "-B"] + args, cwd=R, stdout=lf, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"))
    state[name].update(end=time.strftime("%H:%M:%S"), rc=r.returncode); save(); P("end", name, r.returncode)
    return r.returncode
def at(hhmm):
    hh, mi = map(int, hhmm.split(":")); lt = time.localtime()
    d = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hh, mi, 0, 0, 0, -1))
    return d if d > time.time() - 12 * 3600 else d + 86400

# 1. wait for vision training to finish (its log says TRAINING DONE or DEADLINE)
vlog = os.path.join(R, "logs", "train_vision.log")
while True:
    t = open(vlog, encoding="utf-8", errors="ignore").read()
    if "TRAINING DONE" in t or "saved deadline" in t:
        break
    if "Traceback" in t[-5000:]:
        P("vision training crashed; evaluating last checkpoint anyway"); break
    time.sleep(30)
P("vision training finished")
# 2. vision eval
if not os.path.exists(os.path.join(R, "runs", "vision", "eval", "vision_eval.json")):
  run("eval_vision", [os.path.join(R, "eval_vision.py"), r"runs\vision\projector.safetensors", r"runs\vision\eval", "500"])
# 3. audio smoke
micro, accum = 8, 4
_sl = os.path.join(R, "logs", "night_smoke_audio.log")
if not (os.path.exists(_sl) and "SMOKE END" in open(_sl, encoding="utf-8", errors="ignore").read()):
  run("smoke_audio", [os.path.join(R, "train.py"), "--modality", "audio", "--out", r"runs\smoke_audio", "--samples", "40000",
                    "--micro", str(micro), "--accum", str(accum), "--smoke_minutes", "4"])
sl = open(os.path.join(R, "logs", "night_smoke_audio.log"), encoding="utf-8", errors="ignore").read()
m = re.search(r"SMOKE END s/sample ([\d.]+) peak_alloc ([\d.]+) GB peak_reserved ([\d.]+) GB", sl)
sps = float(m.group(1)) if m else 0.2
peak = float(m.group(3)) if m else 99
if peak > 13.5:
    micro, accum = 4, 8; sps *= 1.1
state["audio_smoke"] = {"s_per_sample": sps, "peak_reserved_gb": peak, "micro": micro, "accum": accum}
left = at(AUDIO_END) - time.time() - 120
samples = max(3200, int(left / sps) // 32 * 32)
state["audio_plan"] = {"samples": samples, "seconds_left": left}; save(); P("audio plan", samples, "samples")
# 4. audio training (deadline as safety)
run("train_audio", [os.path.join(R, "train.py"), "--modality", "audio", "--out", r"runs\audio", "--samples", str(samples),
                    "--micro", str(micro), "--accum", str(accum), "--deadline", AUDIO_END])
# 5. audio eval
run("eval_audio", [os.path.join(R, "eval_audio.py"), r"runs\audio\projector.safetensors", r"runs\audio\eval", "500"])
P("NIGHT DONE")
