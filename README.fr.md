# whisper-npu

**Transcription vocale rapide et privée sur le NPU AMD XDNA 2 — une stack complète, fonctionnelle et mesurée.**

Fait tourner Whisper `large-v3-turbo` avec l'encodeur entièrement déporté sur le NPU Ryzen AI (50 TOPS), ajoute un patch de pipelining inter-fichiers (+35 % de débit), un serveur API persistant compatible OpenAI, et une boucle d'assistant vocal 100 % locale (micro → VAD → STT NPU → LLM → TTS) avec un endpoint WebSocket façon Realtime API.

Tout ce dépôt a été construit et mesuré sur une vraie machine (Ryzen AI Max+ 395 « Strix Halo », Windows 11) — y compris les modes de défaillance que personne ne documente.

[English documentation](README.md)

---

## Pourquoi ce dépôt existe

AMD publie un [fork](https://github.com/amd/whisper.cpp) de whisper.cpp qui déporte l'encodeur Whisper sur leurs NPU via des caches `.rai` précompilés. La documentation s'arrête là. Ce dépôt est tout ce qu'il a fallu comprendre pour en faire un outil de production :

- comment le compiler sous Windows (MSVC, pas MSYS) avec le runtime `flexmlrt` (et où le trouver quand le portail AMD bloque les scripts)
- les **règles dures du NPU XDNA 2** (les enfreindre corrompt silencieusement les sorties — voir plus bas)
- un patch de **pipelining inter-fichiers** pour occuper le NPU pendant les décodes CPU
- un **serveur HTTP persistant** (celui du fork segfaulte sur le VAD)
- une **boucle vocale** avec transcription spéculative et TTS streamé par phrase
- une tentative documentée de **recompilation du `.rai`** à fenêtre réduite, avec les raisons mesurées de l'échec et la piste restante

## Performances mesurées (matériel réel)

| Métrique | Valeur |
|---|---|
| Encodeur, NPU vs CPU 16 threads | **1160 ms vs 5540 ms par fenêtre 30 s (×4,8)** |
| Pipeline complet de transcription (greedy) | ×12,5 temps réel |
| WER sur notes vocales FR arbitrées à 3 sources | **2,9 % médian** (large-v3-turbo) |
| Qualité des modèles sur le français parlé | tiny : 44 % WER, base : 32 % — inutilisables ; seul turbo est viable |
| Latence serveur persistant (énoncé 6 s) | ~1,7 s |
| Boucle vocale, fin de parole → texte | **1,57 s** (STT spéculatif) |
| Boucle vocale, texte → premier mot parlé | **0,41 s** (TTS par demi-phrase) |

## Les règles du NPU (apprises à nos dépens)

1. **Un seul pipeline NPU à la fois.** La stack flexmlrt/XRT n'a pas de file multi-contexts fiable. Lancer 3 workers concurrents produit des erreurs `ERT_CMD_STATE_TIMEOUT` *et — pire — des transcriptions corrompues en silence* (texte fluide mais faux). Un process, tout sérialisé.
2. **Décodage greedy (`-bs 1`) pour le batch.** Le beam search occupe le CPU pendant que le NPU dort ; greedy divise la phase CPU par ~3 sans perte mesurée.
3. **La fenêtre du `.rai` est fixée à 30 s.** `-ac` est ignoré par le cache précompilé : un énoncé de 2 s coûte le même encode 1,16 s qu'un de 30 s. Les fenêtres courtes exigent une recompilation — et l'architecture de Whisper s'y oppose (voir [docs/08-recompilation.md](docs/08-recompilation.md)).
4. **Le VAD du fork segfaulte.** Ce dépôt le rend opt-in (`WHISPER_ALLOW_VAD=1`) ; faites le VAD dans votre orchestrateur.

## Démarrage rapide

Prérequis : Windows 11, VS 2022 Build Tools (C++), CMake, Git, Python 3.12, ffmpeg.

```bat
:: 1. cloner le fork AMD et appliquer notre patch
git clone https://github.com/amd/whisper.cpp
cd whisper.cpp
git apply ..\whisper-npu\patches\0001-whisper-npu-full-patch.patch

:: 2. runtime flexmlrt (miroir si le portail AMD bloque :
::    github.com/lemonade-sdk/whisper.cpp-builds/releases/tag/deps)

:: 3. compiler (scripts adaptés dans scripts/build/)
build-npu.bat

:: 4. modèles : ggml-large-v3-turbo.bin (ggerganov/whisper.cpp)
::              ggml-large-v3-turbo-encoder-vitisai.rai (amd/whisper-large-turbo-onnx-npu)

:: 5. transcrire
whisper-cli -m models\ggml-large-v3-turbo.bin -f audio.wav -l fr -bs 1 -t 16
```

Guide complet avec tous les pièges : [docs/01-installation.md](docs/01-installation.md) (EN)

## Contenu du dépôt

```
whisper-npu/
├── patches/                 modifications du fork (pipeline, fix VAD, pont ONNX)
├── scripts/
│   ├── build/               scripts MSVC (base / pipeline / pont ORT / serveur)
│   ├── server/              serveur ASR OpenAI-compatible, boucle vocale, endpoint WebSocket
│   ├── tools/               benchmarks, comparaison de modèles, génération de rapports
│   └── dataset/             arbitrage 3 sources, recompilation .rai
├── docs/                    la connaissance (index dans le README EN)
├── bench/                   résultats mesurés
└── LICENSE                  MIT
```

## Licence

MIT — voir [LICENSE](LICENSE). whisper.cpp est MIT ; le fork AMD et les modèles `.rai` suivent leurs licences respectives.

## Crédits

Construit par essais, erreurs et mesures — une longue soirée sur du vrai matériel, par Arnaud avec Claude Code. Le fork [`amd/whisper.cpp`](https://github.com/amd/whisper.cpp) et le miroir [lemonade-sdk](https://github.com/lemonade-sdk) ont rendu cela possible.
