import os, re, unicodedata, statistics, json

def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", s)).strip()

def wer(ref_words, hyp_words):
    # Levenshtein mot à mot standard
    d = list(range(len(hyp_words) + 1))
    for i in range(1, len(ref_words) + 1):
        prev = d[0]
        d[0] = i
        for j in range(1, len(hyp_words) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j-1] + 1, prev + (ref_words[i-1] != hyp_words[j-1]))
            prev = cur
    return d[-1]

BASE = r"C:\Users\Arnaud\dev\whisper-bench-real"
AUD  = r"C:\Users\Arnaud\Downloads\fine-tune-export\fine-tune\audio"
ARB  = os.path.join(BASE, "dataset-arbitre")

srcs = {}
for line in open(os.path.join(BASE, "full-indexed.txt")):
    p = line.strip().split(" ", 1)
    if len(p) == 2:
        srcs[p[0]] = "C:\\" + p[1][3:].replace("/", "\\")

# vérité arbitrée par idx
truth = {}
for idx, src in srcs.items():
    rel = os.path.relpath(src, AUD)
    tp = os.path.join(ARB, rel.replace(".opus", ".txt").replace(".ogg", ".txt"))
    if os.path.exists(tp):
        t = norm(open(tp, encoding="utf-8").read())
        if t:
            truth[idx] = t

# timing
times = {}
for line in open(os.path.join(BASE, "bench-models", "timing.log")):
    parts = line.split()
    if len(parts) == 3 and parts[0] in ("tiny", "base", "turbo"):
        times[parts[0]] = (float(parts[1]), int(parts[2]))

# durée audio de l'échantillon
import wave, glob
durs = {}
for w in glob.glob(os.path.join(BASE, "bench-models", "wavs_tmp", "a*.wav")):
    with wave.open(w) as f:
        durs[os.path.basename(w)[1:5]] = f.getnframes() / f.getframerate()
total_audio = sum(durs.values())

print(f"échantillon: {len(durs)} fichiers, {total_audio/60:.1f} min d'audio, vérité dispo pour {len([i for i in durs if i in truth])}")
print()
print(f"| Modèle | Vitesse NPU | Temps réel | Similarité médiane | WER médian | WER moyen | Fichiers WER<10% |")
print(f"|---|---|---|---|---|---|---|")
results = {}
for model in ["tiny", "base", "turbo"]:
    sims, wers = [], []
    n_ok = 0
    for f in glob.glob(os.path.join(BASE, "bench-models", model, "a*.wav.txt")):
        idx = os.path.basename(f)[1:5]
        if idx not in truth:
            continue
        t = truth[idx]
        h = norm(open(f, encoding="utf-8", errors="replace").read())
        if not h:
            wers.append(1.0); sims.append(0.0)
            continue
        tw, hw = t.split(), h.split()
        w = wer(tw, hw) / max(1, len(tw))
        wers.append(w)
        import difflib
        sims.append(difflib.SequenceMatcher(None, t, h).ratio())
        if w < 0.10:
            n_ok += 1
    n = len(wers)
    t_sec, files = times.get(model, (0, n))
    per_file = t_sec / max(1, files)
    rt = total_audio / t_sec if t_sec else 0
    results[model] = (per_file, rt, statistics.median(sims), statistics.median(wers), statistics.mean(wers), n_ok, n)
    print(f"| {model} | {per_file:.2f} s/fichier | ×{rt:.1f} | {statistics.median(sims):.1%} | {statistics.median(wers):.1%} | {statistics.mean(wers):.1%} | {n_ok}/{n} |")

print()
print("perte de qualité relative vs turbo (WER moyen):")
if "turbo" in results:
    for m in ["tiny", "base"]:
        if m in results and results[turbo][4] > 0:
            print(f"  {m}: {(results[m][4] - results['turbo'][4])/results['turbo'][4]:+.0%} de WER en plus")
