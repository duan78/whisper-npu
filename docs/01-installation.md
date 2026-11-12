# Installation — the complete path, traps included

Tested: Windows 11, VS 2022 Build Tools (MSVC 14.44), CMake 3.29, Python 3.12.
Hardware: Ryzen AI Max+ 395 (works the same on Ryzen AI 300 series — same XDNA 2).

## 0. Prerequisites

- **Visual Studio 2022 Build Tools** with the C++ workload (`cl.exe` under
  `VC\Tools\MSVC\*\bin\Hostx64\x64`). **MSYS/MinGW will not work** — the flexmlrt
  import library is MSVC-only.
- CMake ≥ 3.20, Git, ffmpeg in PATH, Python 3.12 (for servers/tools).
- **NPU driver ≥ .280** (check Device Manager → "NPU Compute Accelerator Device" →
  driver version; anything from 2025 onward is fine).

## 1. The whisper.cpp fork

```bash
git clone https://github.com/amd/whisper.cpp
cd whisper.cpp
git apply /path/to/whisper-npu/patches/0001-whisper-npu-full-patch.patch
```

The patch adds: inter-file pipeline, VAD opt-out, the ONNX-bridge mode, and the
OpenAI-server VAD fix. It applies cleanly on fork commit `b40e6c8`.

## 2. The flexmlrt runtime (the part AMD makes hard)

The fork needs `flexmlrt 1.7.0` (headers + `flexmlrt.dll/.lib` + CMake config).
Official source: AMD's portal
(`account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=flexmlrt1.7.0-win.zip`)
— **which blocks scripted downloads** (curl hangs / EULA wall). Community mirror that
worked for us:

```
https://github.com/lemonade-sdk/whisper.cpp-builds/releases/download/deps/flexmlrt1.7.0-win.zip
```

Extract anywhere; note the path to `flexmlrt\share\cmake\FlexmlRT` (that's the
`FlexmlRT_DIR` CMake needs).

⚠️ **Version lock**: use 1.7.0 with this fork. flexmlrt 1.9 (from the Ryzen AI SDK)
cannot load the HuggingFace `.rai` files (`KeyError: dry_run` / C++ crash).

## 3. Models

From HuggingFace, into one `models/` folder:

| File | Source | Purpose |
|---|---|---|
| `ggml-large-v3-turbo.bin` | `ggerganov/whisper.cpp` | full model (CPU decoder side) |
| `ggml-large-v3-turbo-encoder-vitisai.rai` | `amd/whisper-large-turbo-onnx-npu` | precompiled NPU encoder cache |

**Naming rule (the fork enforces it):** the `.rai` must sit next to the `.bin` and be
named `<bin-name>-encoder-vitisai.rai`. Other model sizes exist in the same AMD HF
collection (tiny/base/small/medium/large-v3) — but see
[benchmarks](03-benchmarks.md): only turbo is worth using.

Override for experiments: `set OVERRIDE_VITISAI_MODEL_PATH=C:\...\something.rai`
(or `.onnx` with the bridge build).

## 4. Build

Use `scripts/build/build-npu.bat` (base) / `build-pipe.bat` (inter-file pipeline) /
`build-ort.bat` (adds the ONNX bridge — requires the Ryzen AI SDK, see
[08-recompilation](08-recompilation.md)). Essence:

```bat
call vcvars64.bat
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
  -DWHISPER_VITISAI=1 -DWHISPER_PIPELINE=1 ^
  -DFlexmlRT_DIR=C:\path\to\flexmlrt\share\cmake\FlexmlRT
cmake --build build --config Release -j
copy C:\path\to\flexmlrt\lib\flexmlrt.dll build\bin\Release\
```

## 5. Smoke test

```bat
build\bin\Release\whisper-cli.exe -m models\ggml-large-v3-turbo.bin ^
  -f samples\jfk.wav -t 16 -l fr -bs 1
```

Expected in the log: `XRT build version: …`, `whisper_init_state: Vitis AI model
loaded`, `system_info: … VITISAI = 1`, then the transcription. Audio must be
16 kHz mono WAV (`ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`). **Opus is not decoded
by whisper.cpp** despite the help text — convert first.

## 6. Servers (optional)

```bash
pip install sounddevice onnxruntime piper-tts websockets
```

- OpenAI-compatible REST: `python scripts/server/npu-asr-server.py` (port 9200)
- Voice loop + WebSocket: see [06-voice-loop](06-voice-loop.md)

Next: [02-usage](02-usage.md).
