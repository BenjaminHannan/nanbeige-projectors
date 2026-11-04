"""Pull individual clips out of WavCaps' huge split zips (FreeSound = 121 x 5.4 GB parts) with HTTP range requests:
read the zip central directory, choose clips, fetch only their bytes, convert to 16 kHz mono int16 and pack.
usage: wavcaps_range.py <subset AudioSet_SL|FreeSound|BBC_Sound_Effects> <n_clips> <max_seconds> [stop_hhmm]
Leakage rule (FreeSound): drop every clip whose FreeSound ID is a Clotho evaluation sound_id or is in WavCaps'
own blacklist_exclude_test_ac.json. Counts go to data/audio/<pack>_leakage.json."""
import os, sys, io, csv, json, struct, random, threading, time, zlib
import requests, numpy as np, soundfile as sf
os.environ.pop("REQUESTS_CA_BUNDLE", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_audio import to16k

ROOT = os.environ.get("PROJ_ROOT", r"C:\Users\benja\premonition-projectors")
HF = "https://huggingface.co/datasets/cvssp/WavCaps/resolve/main/"
API = "https://huggingface.co/api/datasets/cvssp/WavCaps/tree/main/Zip_files/"
subset, NWANT, MAXS = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
STOP_AT = sys.argv[4] if len(sys.argv) > 4 else None
JSON = {"AudioSet_SL": "AudioSet_SL/as_final.json", "FreeSound": "FreeSound/fsd_final.json",
        "BBC_Sound_Effects": "BBC_Sound_Effects/bbc_final.json"}[subset]
OUT = os.path.join(ROOT, "data", "audio"); PACK = "wavcaps_" + subset.lower()
S = requests.Session()
def P(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

_cdn = {}; _cdn_lock = threading.Lock()
def cdn(url, refresh=False):
    """huggingface.co/resolve is rate-limited (429); resolve each file's signed storage URL once and reuse it
    for range requests (refreshed every 20 min or on 401/403)."""
    with _cdn_lock:
        u, t0 = _cdn.get(url, (None, 0))
        if u is None or refresh or time.time() - t0 > 1200:
            for t in range(20):
                r = S.head(url, allow_redirects=False, timeout=60)
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
                    u = requests.compat.urljoin(url, r.headers["Location"]); break
                if r.status_code == 200:
                    u = url; break
                time.sleep(3 + 3 * t)
            _cdn[url] = (u, time.time())
        return u

def get(url, a=None, b=None, tries=12):
    for t in range(tries):
        try:
            h = {"Range": f"bytes={a}-{b}"} if a is not None else {}
            u = cdn(url, refresh=t > 0)
            r = S.get(u, headers=h, timeout=120)
            if r.status_code in (200, 206):
                return r.content
        except Exception:
            pass
        time.sleep(2 + 2 * t)
    raise RuntimeError("fetch failed " + url)

# ---- parts (disk order: .z01, .z02, ..., .zip)
parts = [x for x in requests.get(API + subset, timeout=60).json() if x["type"] == "file"]
def disk_key(p):
    e = p["path"].rsplit(".", 1)[1]
    return 10 ** 6 if e == "zip" else int(e[1:])
parts.sort(key=disk_key)
urls = [HF + p["path"] for p in parts]; sizes = [p["size"] for p in parts]
base = np.cumsum([0] + sizes[:-1]).tolist()
P(subset, len(parts), "parts", sum(sizes) / 1e9, "GB")

def read_global(g, n):
    """Read n bytes at global offset g (may span parts)."""
    out = b""
    while n > 0:
        i = max(k for k in range(len(base)) if base[k] <= g)
        off = g - base[i]; take = min(n, sizes[i] - off)
        out += get(urls[i], off, off + take - 1); g += take; n -= take
    return out

# ---- central directory
last = len(parts) - 1
tail = get(urls[last], sizes[last] - 65536, sizes[last] - 1)
e = tail.rfind(b"PK\x05\x06")
disk_cd, cd_size, cd_off = struct.unpack("<H", tail[e + 6:e + 8])[0], struct.unpack("<I", tail[e + 12:e + 16])[0], struct.unpack("<I", tail[e + 16:e + 20])[0]
loc = tail.rfind(b"PK\x06\x07")
if loc >= 0:   # zip64
    z64_disk, z64_off = struct.unpack("<IQ", tail[loc + 4:loc + 16])
    z = read_global(base[z64_disk] + z64_off, 56)
    assert z[:4] == b"PK\x06\x06"
    disk_cd = struct.unpack("<I", z[20:24])[0]; cd_size, cd_off = struct.unpack("<QQ", z[40:56])
cd = read_global(base[disk_cd] + cd_off, cd_size)
P("central directory", len(cd) / 1e6, "MB")
entries, p = [], 0
while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
    method, = struct.unpack("<H", cd[p + 10:p + 12])
    csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
    nlen, elen, clen, disk = struct.unpack("<HHHH", cd[p + 28:p + 36])
    lofs, = struct.unpack("<I", cd[p + 42:p + 46])
    name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
    ex = cd[p + 46 + nlen:p + 46 + nlen + elen]; q = 0
    while q + 4 <= len(ex):
        hid, hl = struct.unpack("<HH", ex[q:q + 4]); body = ex[q + 4:q + 4 + hl]
        if hid == 1:
            r = 0
            if usize == 0xFFFFFFFF: usize, = struct.unpack("<Q", body[r:r + 8]); r += 8
            if csize == 0xFFFFFFFF: csize, = struct.unpack("<Q", body[r:r + 8]); r += 8
            if lofs == 0xFFFFFFFF: lofs, = struct.unpack("<Q", body[r:r + 8]); r += 8
            if disk == 0xFFFF: disk, = struct.unpack("<I", body[r:r + 4]); r += 4
        q += 4 + hl
    if not name.endswith("/"):
        entries.append((name, method, csize, disk, lofs))
    p += 46 + nlen + elen + clen
P("zip entries", len(entries), "e.g.", entries[:2])

# ---- captions + selection
meta = json.loads(get(HF + "json_files/" + JSON))["data"]
stem = lambda n: os.path.splitext(os.path.basename(n))[0]
info = {stem(str(m["id"])): m for m in meta}
leak = {"subset": subset}
banned = set()
if subset == "FreeSound":
    ev = csv.DictReader(io.StringIO(get("https://zenodo.org/api/records/4783391/files/clotho_metadata_evaluation.csv/content").decode("utf-8")))
    clotho_eval_ids = {r["sound_id"].strip() for r in ev if r.get("sound_id")}
    bl = json.loads(get(HF + "json_files/blacklist/blacklist_exclude_test_ac.json"))
    bl_ids = {str(x) for x in (bl.get("FreeSound", []) if isinstance(bl, dict) else bl)}
    banned = clotho_eval_ids | bl_ids
    in_wavcaps = {k for k in info}
    leak.update(clotho_eval_ids=len(clotho_eval_ids), wavcaps_blacklist_ids=len(bl_ids),
                clotho_eval_ids_present_in_wavcaps_freesound=len(clotho_eval_ids & in_wavcaps),
                blacklist_ids_present=len(bl_ids & in_wavcaps))
cands = []
for (name, method, csize, disk, lofs) in entries:
    s = stem(name); m = info.get(s)
    if m is None or not m.get("caption"):
        continue
    if s in banned:
        leak["removed"] = leak.get("removed", 0) + 1; continue
    if float(m.get("duration", 0)) > MAXS or float(m.get("duration", 0)) < 1.0:
        continue
    cands.append((name, method, csize, disk, lofs, m["caption"]))
leak["eligible_after_filter"] = len(cands)
random.Random(0).shuffle(cands)
cands = cands[:NWANT]
json.dump(leak, open(os.path.join(OUT, PACK + "_leakage.json"), "w"), indent=1)
P("leakage/selection", json.dumps(leak), "fetching", len(cands), "avg MB", sum(c[2] for c in cands) / max(1, len(cands)) / 1e6)

# ---- fetch + pack
lock = threading.Lock(); metaout = []; offs = [0]; done = [0]; byts = [0]; t0 = time.time()
fpcm = open(os.path.join(OUT, PACK + ".pcm.tmp"), "wb")
queue_ = list(cands)
def stop_now():
    if not STOP_AT: return False
    hh, mi = map(int, STOP_AT.split(":")); lt = time.localtime()
    return (lt.tm_hour, lt.tm_min) >= (hh, mi)
def worker():
    while True:
        with lock:
            if not queue_ or stop_now(): return
            name, method, csize, disk, lofs, cap = queue_.pop()
        try:
            g = base[disk] + lofs
            blob = read_global(g, 30 + 1024 + csize)
            nlen, elen = struct.unpack("<HH", blob[26:30])
            raw = blob[30 + nlen + elen: 30 + nlen + elen + csize]
            if method == 8:
                raw = zlib.decompress(raw, -15)
            x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            y = to16k(x, sr)[: int(16000 * 20)]
            if len(y) < 16000:
                continue
            with lock:
                fpcm.write(y.tobytes()); metaout.append({"name": stem(name), "off": offs[0], "n": len(y), "captions": [cap]})
                offs[0] += len(y); done[0] += 1; byts[0] += csize
                if done[0] % 500 == 0:
                    fpcm.flush()
                    json.dump(metaout, open(os.path.join(OUT, PACK + ".partial.json"), "w"))
                    P(done[0], f"{byts[0] / (time.time() - t0) / 1e6:.2f} MB/s", f"{offs[0] / 16000 / 3600:.1f} h")
        except Exception as ex:
            P("skip", name, str(ex)[:120])
ts = [threading.Thread(target=worker) for _ in range(12)]
[t.start() for t in ts]; [t.join() for t in ts]
fpcm.close()
os.replace(os.path.join(OUT, PACK + ".pcm.tmp"), os.path.join(OUT, PACK + ".pcm"))
json.dump(metaout, open(os.path.join(OUT, PACK + ".json"), "w"))
leak["packed_clips"] = len(metaout); leak["packed_hours"] = offs[0] / 16000 / 3600
json.dump(leak, open(os.path.join(OUT, PACK + "_leakage.json"), "w"), indent=1)
P("PACKED", PACK, len(metaout), "clips", offs[0] / 16000 / 3600, "h")
