# Recompiling the NPU encoder — full experiment log

Goal: shrink the encoder's 30 s window (fixed 1.16 s NPU forward) to ~10 s
(~0.4 s) for conversational latency. Status: **toolchain 100 % working, two
measured blockers, one path remains open.**

## The working toolchain (reusable for any ONNX on this NPU)

```
Ryzen AI Software 1.9 MSI          (6.6 GB, public CDN — no AMD account needed)
  download.amd.com/opendownload/RyzenAI/1.9.0/ryzen-ai-1.9.0-msi.zip
  → extract without installing:  msiexec /a ryzen-ai-1.9.0.msi /qn TARGETDIR=...
  → install the local wheels into a Python 3.12 venv:
      flexml, flexmlrt, voe, onnxruntime_vitisai, onnxruntime_providers_ryzenai,
      ryzenai_dynamic_dispatch(+_bins)
ONNX surgery                        scripts/dataset/onnx-surgery.py
AIE compilation                     happens on first inference with the
                                    VitisAIExecutionProvider (~21 min cold)
Persistent cache                    C:\temp\<user>\vaip\CACHE*  (auto, survives reboots)
C++ bridge into whisper.cpp         patches/0001: WHISPER_ORT_ONNX mode
```

### Key traps (each cost us an hour)

| Trap | Fix |
|---|---|
| `pypi.amd.com` returns 403 (Akamai) for pip and curl | Use the MSI from the public CDN |
| PyPI package named `flexml` is an unrelated AutoML tool | Only use AMD's wheels |
| ONNX surgery: `ShapeInferenceError: Incompatible dimensions` | Delete `graph.value_info[:]` — stale intermediate shapes block re-inference |
| Protobuf `EncodeError` on save | Model > 2 GB: save with `save_as_external_data` |
| C++ `Ort::Session` ctor: no overload matches | Windows wants a wide path (`wchar_t`) |
| Patch compiled but feature absent | CMake `WHISPER_EXTRA_FLAGS` applies to target `whisper`, but the encoder lives in target `whisper.vitisai` — apply the define to both |
| Segfault at provider init (3 s) with all obvious DLLs present | You also need `onnxruntime_vitisai_ep.dll` (**517 MB**, embeds the AIE compiler) next to the exe |

## What we achieved

| Milestone | Result |
|---|---|
| Window-1000 encoder built, compiled, executed in whisper.cpp | **255 ms/window (4.5× vs 1160 ms)** |
| Cold compile time | 21 min (once; cached afterwards — reload is instant) |
| Original window-3000 model through our bridge | runs, 1151 ms (parity with AMD's `.rai`) |

## Blocker 1 — short windows are out-of-distribution (measured)

cos-sim between the window-1000 encoder and the same model fed 30 s-padded input,
same signal: **0.48**. Whisper was trained with the full 30 s window **including
zero padding**; its global attention treats the padding as part of the input. Remove
it and the embeddings drift so far the decoder hallucinates ("not to be." on French
audio).

**Implication: no compiler config can fix this.** A short-window encoder requires
fine-tuning Whisper with short windows — out of scope here. This also explains why
upstream whisper.cpp's `-ac` flag is known to degrade quality.

## Blocker 2 — the SDK provider quantizes by default (the open path)

Feeding the **original, unmodified** window-3000 ONNX through our bridge produced
garbage transcriptions too. Isolation test (same input, same model):

| Execution | cos-sim vs CPU reference |
|---|---|
| CPU | 1.0 |
| VitisAI EP, default provider options | **0.25** |
| VitisAI EP, `config_file=vaip_config.json` (SDK's own config) | **0.25** (no change; the cache isn't keyed on the config) |

The provider's default passes quantize the graph (W8A8-style fusions — see
`fuse_wb_flow_waic` etc. in `vaip_config.json`) with no calibration data, which
destroys a float32 encoder. AMD's official `.rai` files don't suffer this — they were
built by AMD's internal pipeline with proper precision handling.

**Remaining path:** disable the quantizing passes in a custom `vaip_config.json`
(set the `fuse_DD`/`waic` entries `disabled: true`, keep the rest), or provide
calibration data. Then recompile the window-3000 model, validate cos-sim ≥ 0.99 vs
CPU, and only then attempt production use. Scripts to reproduce every measurement:
`scripts/dataset/`, `scripts/dataset/compile-npu.py`.

## Why the bridge is still a win

Even with the `.rai` path unchanged for production, the `WHISPER_ORT_ONNX` mode in
our patch lets whisper.cpp consume **any ONNX encoder** on the NPU — the full
compile-cache-execute loop is proven. That's the foundation for custom encoders
(distilled, fine-tuned, other architectures) the day the quantization config is
solved.
