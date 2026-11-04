# Wait for each audio archive from pdl_audio, then pack it (runs on CPU while vision trains).
import os, time, subprocess, sys
R = r"C:\Users\benja\premonition-projectors"
jobs = [("clotho_evaluation", r"data\clotho\clotho_audio_evaluation.7z"),
        ("clotho_development", r"data\clotho\clotho_audio_development.7z"),
        ("soundbible", r"data\wavcaps\SoundBible.zip"),
        ("clotho_validation", r"data\clotho\clotho_audio_validation.7z")]
for name, f in jobs:
    if os.path.exists(os.path.join(R, "data", "audio", name + ".json")):
        print("already", name, flush=True); continue
    while not os.path.exists(os.path.join(R, f)):
        time.sleep(30)
    print(time.strftime("%H:%M:%S"), "prep", name, flush=True)
    r = subprocess.run([sys.executable, "-B", os.path.join(R, "prep_audio.py"), name], cwd=R)
    print(time.strftime("%H:%M:%S"), "prep", name, "exit", r.returncode, flush=True)
print("CHAIN DONE", flush=True)
