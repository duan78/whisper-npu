import difflib, os, re, unicodedata, statistics

def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", s)).strip()

srcs = {}
for line in open(r"C:\Users\Arnaud\dev\whisper-bench-real\full-indexed.txt"):
    p = line.strip().split(" ", 1)
    if len(p) == 2:
        srcs[p[0]] = "C:\\" + p[1][3:].replace("/", "\\")

items = []
for idx, src in srcs.items():
    ref = norm(open(os.path.splitext(src)[0] + ".txt", encoding="utf-8", errors="replace").read())
    hyp = norm(open(rf"C:\Users\Arnaud\dev\whisper-bench-real\full\wav\a{idx}.wav.txt", encoding="utf-8", errors="replace").read())
    if not ref or not hyp:
        continue
    r = difflib.SequenceMatcher(None, ref, hyp).ratio()
    if r < 0.80:
        continue
    rw, hw = ref.split(), hyp.split()
    sm = difflib.SequenceMatcher(None, rw, hw)
    ops = [(t, " ".join(rw[i1:i2]), " ".join(hw[j1:j2])) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal"]
    items.append({
        "f": os.path.basename(src), "d": os.path.basename(os.path.dirname(src)),
        "s": r, "rw": len(rw), "hw": len(hw), "ops": ops,
    })

items.sort(key=lambda x: x["s"])

def diff_md(ops, maxlen=220):
    out = []
    n = 0
    for t, ref_txt, hyp_txt in ops:
        if t == "delete":
            seg = f"~~{ref_txt}~~"
        elif t == "insert":
            seg = f"**{hyp_txt}**"
        else:
            seg = f"~~{ref_txt}~~/**{hyp_txt}**"
        if n + len(seg) > maxlen:
            out.append("…")
            break
        out.append(seg)
        n += len(seg) + 1
    return " ".join(out)

med = statistics.median(x["s"] for x in items)
identical = sum(1 for x in items if x["s"] == 1.0)
total_del = sum(len(o[1].split()) for x in items for o in x["ops"] if o[1])
total_ins = sum(len(o[2].split()) for x in items for o in x["ops"] if o[2])

L = []
L.append("# Dataset sain (611 paires) — NPU (large-v3-turbo) vs transcripts originaux")
L.append("")
L.append(f"- **Paires saines** (similarité ≥ 80 %) : **{len(items)}**")
L.append(f"- **Similarité médiane** : {med:.1%}")
L.append(f"- **Textes 100 % identiques** (après normalisation) : {identical} ({identical/len(items):.0%} %)")
L.append(f"- **Mots divergents** : −{total_del} (présents seulement dans l'original) / +{total_ins} (seulement dans la transcription NPU)")
L.append(f"- **Légende diff** : ~~barré~~ = seulement dans l'original · **gras** = seulement dans le NPU")
L.append("")
L.append("## Répartition par similarité")
L.append("")
L.append("| Similarité | Paires | Part |")
L.append("|---|---|---|")
for lo, hi, label in [(0.80, 0.90, "80–90 %"), (0.90, 0.95, "90–95 %"), (0.95, 0.99, "95–99 %"), (0.99, 0.9999, "99–100 %"), (0.9999, 1.01, "100 %")]:
    n = sum(1 for x in items if lo <= x["s"] < hi)
    L.append(f"| {label} | {n} | {n/len(items):.0%} |")
L.append("")
L.append("## Les 30 paires les plus divergentes")
L.append("")
L.append("| # | Jour | Fichier | Simil. | Mots | Différence principale |")
L.append("|---|---|---|---|---|---|")
for i, x in enumerate(items[:30], 1):
    d = diff_md(x["ops"])
    L.append(f"| {i} | {x['d']} | `{x['f']}` | {x['s']:.0%} | {x['rw']}→{x['hw']} | {d} |")
L.append("")
L.append("## Toutes les paires présentant un écart (< 100 %)")
L.append("")
L.append("| # | Jour | Fichier | Simil. | Mots | Différence |")
L.append("|---|---|---|---|---|---|")
for i, x in enumerate((y for y in items if y["s"] < 1.0), 1):
    d = diff_md(x["ops"], maxlen=150)
    L.append(f"| {i} | {x['d']} | `{x['f']}` | {x['s']:.0%} | {x['rw']}→{x['hw']} | {d} |")

OUT = r"C:\Users\Arnaud\dev\whisper-bench-real\COMPARAISON-NPU-vs-ORIGINAL.md"
open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("OK:", OUT)
print(f"paires avec écart: {sum(1 for x in items if x['s'] < 1.0)} / {len(items)}")
print(f"médiane: {med:.1%}, identiques: {identical}, -{total_del}/+{total_ins} mots")
