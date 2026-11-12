import os
import difflib, os, re, json, subprocess, sys, time, unicodedata
from concurrent.futures import ThreadPoolExecutor

def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", s)).strip()

KEY = os.environ["MISTRAL_API_KEY"]
BASE = r"C:\Users\Arnaud\dev\whisper-bench-real"
OUT = os.path.join(BASE, "full", "mistral")
os.makedirs(OUT, exist_ok=True)
DONE = os.path.join(OUT, "mistral.jsonl")

srcs = {}
for line in open(os.path.join(BASE, "full-indexed.txt")):
    p = line.strip().split(" ", 1)
    if len(p) == 2:
        srcs[p[0]] = "C:\\" + p[1][3:].replace("/", "\\")

# déjà faits (reprise)
done = set()
if os.path.exists(DONE):
    for line in open(DONE, encoding="utf-8"):
        try:
            done.add(json.loads(line)["idx"])
        except Exception:
            pass

# sélection : saines avec écart (sim < 1.0 mais >= 0.8) + dépairées (< 0.8)
todo = []
for idx, src in srcs.items():
    if idx in done:
        continue
    ref = norm(open(os.path.splitext(src)[0] + ".txt", encoding="utf-8", errors="replace").read())
    hyp = norm(open(os.path.join(BASE, "full", "wav", f"a{idx}.wav.txt"), encoding="utf-8", errors="replace").read())
    r = difflib.SequenceMatcher(None, ref, hyp).ratio() if ref and hyp else 0.0
    if r < 1.0:
        todo.append((idx, src))

print(f"a transcrire: {len(todo)} (deja faits: {len(done)})", flush=True)

lock = __import__("threading").Lock()
outf = open(DONE, "a", encoding="utf-8")

def transcribe(item):
    idx, src = item
    dst = os.path.join(OUT, f"a{idx}.txt")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        text = open(dst, encoding="utf-8").read()
    else:
        for attempt in range(4):
            try:
                r = subprocess.run(
                    ["curl", "-sL", "--max-time", "120",
                     "https://api.mistral.ai/v1/audio/transcriptions",
                     "-H", f"Authorization: Bearer {KEY}",
                     "-F", "model=voxtral-small-transcribe-3",
                     "-F", f"file=@{src}",
                     "-F", "language=fr"],
                    capture_output=True, text=True, timeout=130)
                d = json.loads(r.stdout)
                if "text" in d:
                    text = d["text"].strip()
                    open(dst, "w", encoding="utf-8").write(text)
                    break
                # 429 ou erreur -> backoff
                time.sleep(3 * (attempt + 1))
            except Exception:
                time.sleep(3 * (attempt + 1))
        else:
            with lock:
                outf.write(json.dumps({"idx": idx, "ok": False}) + "\n")
                outf.flush()
            return False
    with lock:
        outf.write(json.dumps({"idx": idx, "ok": True}) + "\n")
        outf.flush()
    return True

n_ok = 0
with ThreadPoolExecutor(max_workers=4) as ex:
    for i, ok in enumerate(ex.map(transcribe, todo), 1):
        n_ok += 1 if ok else 0
        if i % 25 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(todo)} — ok: {n_ok}", flush=True)

print(f"TERMINE: {n_ok}/{len(todo)} reussis", flush=True)
