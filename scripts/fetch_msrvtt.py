# MSR-VTT 1k-A test clips (VID3, never trained on): download N mp4s, keep 8 evenly spaced frames as JPEGs.
import os, json, random, io, requests
os.environ.pop("REQUESTS_CA_BUNDLE", None)
import av
R = os.path.join(os.environ.get("PROJ_ROOT", r"C:\Users\benja\premonition-projectors"), "data", "msrvtt"); os.makedirs(R, exist_ok=True)
B = "https://huggingface.co/datasets/VLM2Vec/MSR-VTT/resolve/main/"
test = requests.get(B + "msrvtt_test_1k.json", timeout=120).json()
random.Random(0).shuffle(test)
out, N = [], 160
for row in test:
    if len(out) >= N: break
    vid = row["video_id"]
    try:
        b = requests.get(B + "raw_videos/" + row["video"], timeout=120).content
        c = av.open(io.BytesIO(b)); s = c.streams.video[0]
        frames = [f.to_image() for f in c.decode(s)]
        if len(frames) < 8: continue
        fps = float(s.average_rate or 30)
        idx = [int((i + 0.5) * len(frames) / 8) for i in range(8)]
        d = os.path.join(R, vid); os.makedirs(d, exist_ok=True)
        for k, i in enumerate(idx):
            frames[i].convert("RGB").save(os.path.join(d, f"f{k}.jpg"), quality=90)
        out.append({"video_id": vid, "caption": row["caption"], "times": [round(i / fps) for i in idx], "nframes": len(frames)})
        print(len(out), vid, row["caption"], flush=True)
    except Exception as e:
        print("skip", vid, e, flush=True)
json.dump(out, open(os.path.join(R, "clips.json"), "w"), indent=1)
print("DONE", len(out))
