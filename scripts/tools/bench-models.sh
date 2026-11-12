#!/bin/bash
# Benchmark qualité/vitesse des 3 modèles NPU sur un échantillon de 100 fichiers du dataset
cd /c/Users/Arnaud/dev/whisper-bench-real || exit 1
W=/c/Users/Arnaud/dev/whisper.cpp/build-pipe/bin/Release/whisper-cli.exe
M=/c/Users/Arnaud/dev/whisper-npu-bin/models
mkdir -p bench-models/{tiny,base,turbo}

# échantillon déterministe de 100 fichiers (seed 7)
(cd full && ls wav/a*.wav | sort | shuf -n 100 --random-source=<(yes 7)) > bench-models/sample.txt
mkdir -p bench-models/wavs_tmp
while read -r f; do cp "full/$f" bench-models/wavs_tmp/; done < bench-models/sample.txt
N_COPIED=$(ls bench-models/wavs_tmp/a*.wav 2>/dev/null | wc -l)
echo "copied $N_COPIED" >> bench-models/timing.log
[ "$N_COPIED" -eq 100 ] || { echo "ERREUR COPIE: $N_COPIED/100" >> bench-models/timing.log; exit 1; }

for model in tiny base turbo; do
  case $model in
    tiny)  MF=$M/ggml-tiny.bin;;
    base)  MF=$M/ggml-base.bin;;
    turbo) MF=$M/ggml-large-v3-turbo.bin;;
  esac
  echo "[$(date +%H:%M:%S)] MODELE $model ..." >> bench-models/timing.log
  s=$(date +%s.%N)
  (cd bench-models/wavs_tmp && $W -m $MF -t 16 -l fr -bs 1 -otxt $(ls a*.wav) > ../$model/run.log 2>&1)
  e=$(date +%s.%N)
  echo "$model $(python -c "print(f'{$e-$s:.1f}')") $(ls bench-models/wavs_tmp/*.txt 2>/dev/null | wc -l)" >> bench-models/timing.log
  # ranger les sorties dans le dossier du modèle
  mv bench-models/wavs_tmp/*.txt bench-models/$model/
done
echo "DONE-BENCH" >> bench-models/timing.log
