# Parallel ranged downloader (the PC's Wi-Fi gives ~0.7 MB/s per connection).
# usage: pdl.py <queue file with lines "url<TAB>outpath">   ; skips files already complete.
import os, sys, threading, requests, time
os.environ.pop("REQUESTS_CA_BUNDLE", None)
NCONN = 12; CHUNK = 8 << 20
def size_of(url):
    r = requests.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=60, stream=True)
    cr = r.headers.get("Content-Range")
    final = r.url; r.close()
    return int(cr.split("/")[-1]), final
def fetch(url, out):
    if os.path.exists(out): print("have", out, flush=True); return
    n, final = size_of(url)
    tmp = out + ".part"; os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(tmp, "wb") as f: f.truncate(n)
    chunks = list(range(0, n, CHUNK)); lock = threading.Lock(); done = [0]; t0 = time.time()
    def worker():
        while True:
            with lock:
                if not chunks: return
                a = chunks.pop(0)
            b = min(a + CHUNK, n) - 1
            for attempt in range(20):
                try:
                    r = requests.get(final if attempt < 3 else url, headers={"Range": f"bytes={a}-{b}"}, timeout=120, allow_redirects=True)
                    if r.status_code in (200, 206) and len(r.content) == b - a + 1:
                        with lock:
                            with open(tmp, "r+b") as f: f.seek(a); f.write(r.content)
                            done[0] += len(r.content)
                        break
                except Exception as e: pass
                time.sleep(2 + attempt)
            else: raise RuntimeError(f"chunk {a} failed")
    ts = [threading.Thread(target=worker) for _ in range(NCONN)]
    [t.start() for t in ts]
    while any(t.is_alive() for t in ts):
        time.sleep(30); print(f"{os.path.basename(out)} {done[0]/n*100:.0f}% {done[0]/(time.time()-t0)/1e6:.2f} MB/s", flush=True)
    if done[0] != n: raise RuntimeError("incomplete " + out)
    os.replace(tmp, out); print("DONE", out, n, flush=True)
for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#"): continue
    url, out = line.split("\t")
    try: fetch(url, out)
    except Exception as e: print("FAIL", out, e, flush=True)
print("QUEUE DONE", flush=True)
