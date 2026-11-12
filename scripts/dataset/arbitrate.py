import difflib, os, re, json, unicodedata, statistics, shutil

def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", s)).strip()

BASE = r"C:\Users\Arnaud\dev\whisper-bench-real"
AUD  = r"C:\Users\Arnaud\Downloads\fine-tune-export\fine-tune\audio"
OUTDIR = os.path.join(BASE, "dataset-arbitre")
os.makedirs(OUTDIR, exist_ok=True)

srcs = {}
for line in open(os.path.join(BASE, "full-indexed.txt")):
    p = line.strip().split(" ", 1)
    if len(p) == 2:
        srcs[p[0]] = "C:\\" + p[1][3:].replace("/", "\\")

def sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0

stats = {"saines": 0, "orig_garde": 0, "npu_garde": 0, "equivalent": 0,
         "depairees": 0, "npu_valide": 0, "vox_remplace": 0, "revue": 0}
rows_vote, rows_revue = [], []
report = []

for idx, src in sorted(srcs.items()):
    ref = norm(open(os.path.splitext(src)[0] + ".txt", encoding="utf-8", errors="replace").read())
    npu = norm(open(os.path.join(BASE, "full", "wav", f"a{idx}.wav.txt"), encoding="utf-8", errors="replace").read())
    vox_p = os.path.join(BASE, "full", "mistral", f"a{idx}.txt")
    vox = norm(open(vox_p, encoding="utf-8").read()) if os.path.exists(vox_p) else ""
    day = os.path.basename(os.path.dirname(src))
    rel = os.path.relpath(src, AUD)

    r_refnpu = sim(ref, npu)

    if r_refnpu >= 0.995:
        # consensus 2 sources (identiques) : original conservé tel quel
        stats["saines"] += 1
        final, decision = ref, "consensus (original = NPU)"
    elif r_refnpu >= 0.80 and vox:
        # paire saine divergente -> arbitrage voxtral
        stats["saines"] += 1
        s_vox_ref, s_vox_npu = sim(vox, ref), sim(vox, npu)
        margin = s_vox_ref - s_vox_npu
        if max(s_vox_ref, s_vox_npu) < 0.80:
            if r_refnpu >= 0.85:
                # 2 sources sur 3 concordent déjà : voxtral est l'outlier -> original
                final, decision = ref, f"vox outlier (orig/npu {r_refnpu:.0%}) — original garde"
                stats["orig_garde"] += 1
            else:
                final, decision = None, "revue (trois sources divergent)"
                stats["revue"] += 1
                rows_revue.append((idx, day, os.path.basename(src), s_vox_ref, s_vox_npu, r_refnpu))
        elif margin >= 0.02:
            final, decision = ref, f"original (vox: {s_vox_ref:.0%} vs npu: {s_vox_npu:.0%})"
            stats["orig_garde"] += 1
        elif margin <= -0.02:
            final, decision = npu, f"NPU (vox: {s_vox_npu:.0%} vs orig: {s_vox_ref:.0%})"
            stats["npu_garde"] += 1
            rows_vote.append((idx, day, os.path.basename(src), s_vox_ref, s_vox_npu))
        else:
            final, decision = ref, f"equivalent (vox: {s_vox_ref:.0%}/{s_vox_npu:.0%}) — original garde"
            stats["equivalent"] += 1
    elif vox:
        # paire dépairée : NPU vs voxtral
        stats["depairees"] += 1
        s_vox_npu = sim(vox, npu)
        if s_vox_npu >= 0.90:
            final, decision = npu, f"depairee: NPU confirme par voxtral ({s_vox_npu:.0%})"
            stats["npu_valide"] += 1
        else:
            final, decision = vox, f"depairee: transcription voxtral adoptee (ecart NPU {s_vox_npu:.0%})"
            stats["vox_remplace"] += 1
    else:
        final, decision = (npu if r_refnpu < 0.8 else ref), "sans voxtral"
        stats["depairees" if r_refnpu < 0.8 else "saines"] += 1

    report.append({"idx": idx, "fichier": rel, "similarite_orig_npu": round(r_refnpu, 3), "decision": decision})

    if final is not None:
        # écriture du dataset arbitré : même arborescence, texte final
        dst = os.path.join(OUTDIR, rel.replace(".opus", ".txt").replace(".ogg", ".txt"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        raw = final  # texte normalisé minuscule (vérité terrain fine-tune)
        open(dst, "w", encoding="utf-8").write(raw + "\n")

# copie des audios ? non — on garde l'arborescence texte pointant les mêmes noms. Copie des audios pour dataset autonome:
# (optionnel, désactivé pour éviter la duplication — le rapport mappe les chemins)

with open(os.path.join(BASE, "ARBITRAGE.jsonl"), "w", encoding="utf-8") as f:
    for r in report:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("=== ARBITRAGE TERMINÉ ===")
for k, v in stats.items():
    print(f"  {k}: {v}")
print(f"\nDataset arbitré écrit dans: {OUTDIR}")
print(f"\nFichiers où le NPU gagne le vote ({len(rows_vote)}):")
for idx, day, f, sr, sn in rows_vote[:10]:
    print(f"  {day} {f}: vox/npu {sn:.0%} > vox/orig {sr:.0%}")
print(f"\nÀ revoir manuellement ({len(rows_revue)}):")
for idx, day, f, sr, sn, rr in rows_revue[:10]:
    print(f"  {day} {f}: vox/orig {sr:.0%}, vox/npu {sn:.0%}, orig/npu {rr:.0%}")
