# Troubleshooting — every error we hit, and its fix

## Build / load

| Symptom | Cause | Fix |
|---|---|---|
| CMake can't find `FlexmlRT` | `FlexmlRT_DIR` must point at the **`share\cmake\FlexmlRT`** subfolder | Pass `-DFlexmlRT_DIR=...\flexmlrt\share\cmake\FlexmlRT` |
| Link errors / missing `FlexMLClient.h` | Built with MinGW instead of MSVC | Use the VS generator + vcvars64 |
| `whisper_init_state: failed to load Vitis AI model` + runner exception | `.rai` missing or misnamed | `<bin>-encoder-vitisai.rai` beside the `.bin` (or `OVERRIDE_VITISAI_MODEL_PATH`) |
| flexmlrt 1.9 DLL swapped in → crash (rc 127 / segfault) | 1.9 can't run 1.5-era `.rai` | Stay on 1.7.0 with the fork |
| Segfault on silence/noise-only audio | Known hallucination issue, not a crash — see below | Add VAD (orchestrator-side) or filter tiny outputs |

## Runtime

| Symptom | Cause | Fix |
|---|---|---|
| **Fluent but wrong transcripts** in parallel runs | Silent NPU corruption — stale encoder buffer on ERT timeout ([07-npu-rules](07-npu-rules.md)) | Serialize NPU access; re-run serially |
| `ERT_CMD_STATE_TIMEOUT` in logs | Multiple NPU contexts racing | One process at a time |
| NPU at 0 % in Task Manager, job "stuck" | Same contention, escalated | Kill stragglers, restart serial |
| `whisper_vad` segfault (server/VAD runs) | Fork bug | Our patch disables it; `WHISPER_ALLOW_VAD=1` re-enables at your own risk |
| Transcription is empty (`encode time = 0`) | Input was Opus — whisper.cpp's decoder silently produces nothing | Convert to 16 kHz mono WAV first |
| Server returns in ~4.5 s instead of ~1.7 s | Non-persistent path reloading the model per request | Use the patched persistent `whisper-server` (port 9199 in our stack) |
| `M_PI` undeclared after adding includes | MSVC `<cmath>` ordering | Keep includes after the existing list or define `_USE_MATH_DEFINES` first |

## ONNX bridge / recompilation (advanced)

| Symptom | Cause | Fix |
|---|---|---|
| Segfault ~3 s after launch with `WHISPER_ORT_ONNX` | Missing `onnxruntime_vitisai_ep.dll` (517 MB) next to the exe | Copy **all** of `onnxruntime/bin/*.dll` + `dyn_dispatch_core.dll` + the EP dll |
| `ShapeInferenceError: Incompatible dimensions` after surgery | Stale `value_info` annotations | `del m.graph.value_info[:]` before saving |
| Protobuf `EncodeError` on save | Inline model > 2 GB | `save_as_external_data=True` |
| Compiled model transcribes garbage (even original 3000) | Provider default quantization ([08](08-recompilation.md)) | Open issue — disable quantizing passes in vaip config |
| 21-minute hang at session creation | Cold AIE compilation | Normal once; cached in `C:\temp\<user>\vaip\CACHE` |

## Still stuck?

Open an issue with: NPU driver version (Device Manager), flexmlrt version, the exact
command, and the first 20 lines of log output — `Vitis AI model loaded` (or where it
stopped) is the key line.
