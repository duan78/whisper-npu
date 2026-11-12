# The XDNA 2 NPU rules — behavior we measured, not documented

These findings come from an evening of systematic testing on a Ryzen AI Max+ 395
(Strix Halo). Everything below was reproduced and measured. AMD documents none of it.

## Rule 1: One NPU pipeline at a time — or silent corruption

**The most important rule in this repo.**

### What we did

To speed up batch transcription we ran **3 concurrent `whisper-cli` processes** on the
same NPU. Symptoms appeared within minutes:

- Task Manager showed the NPU **dropping to 0 %** (stuck)
- The logs filled with `FlexMLHWRunner.cpp: Subgraph vaiml_par_0 execution failed:
  ERT_CMD_STATE_TIMEOUT` / `ERT_CMD_STATE_ERROR`
- All 806 files eventually produced output files — **the job "succeeded"**

### The trap

Most outputs were **wrong**. The transcriptions were fluent French — same speaker,
same vocabulary — but *not what the audio said*. Cause: when a window's forward pass
fails with an exception, the fork's `whisper_vitisai_encode()` returns 0 but the
return value is **ignored** (see `src/whisper.cpp`, the `#elif defined(WHISPER_USE_VITISAI)`
branch). The decoder then reads a **stale `embd_enc` buffer** from a previous window
and confidently decodes the wrong content.

We only caught it because we had a 3-source-arbitrated ground truth: a re-test showed
"corrupted" files jumping from 18 % similarity to 97.8 % when re-transcribed serially.

### The rule

- **Serialize all NPU access.** One whisper-cli process at a time, or a mutex/queue in
  your orchestrator (see `scripts/server/voice-stream-server.py`).
- **Never trust fluent output from a parallel run.** The corruption is silent and
  statistically plausible.
- If you must parallelize (we don't recommend it): re-verify outputs against a second
  pass on anything suspicious.

## Rule 2: The `.rai` input window is fixed — audio length doesn't matter

The precompiled `.rai` cache expects exactly **1500 mel frames (30 s)**. Consequences:

| Utterance | Encode time (NPU) |
|---|---|
| 2 s | 1.41 s |
| 6 s | 1.44 s |
| 14 s | 1.77 s |

A 2-second utterance costs **the same 1.16 s forward** as 30 seconds. Short-window
recompilation is the theoretical fix — see [08-recompilation](08-recompilation.md) for
why it's harder than it looks.

`-ac` (audio context) is wired through the fork (`exp_n_audio_ctx`) and correctly
builds a shorter mel window — **but the `.rai` still runs its fixed shape**. We
measured `-ac 500` producing an identical 1091 ms encode.

## Rule 3: Greedy decoding for throughput, beams for nothing

Default beam search (5 beams) keeps the CPU busy ~2.5 s/window while the NPU idles.
With `-bs 1` (greedy):

- CPU decode phase drops ~3×
- Quality on clean speech: unchanged (our arbitrated dataset showed no measurable
  WER difference on conversational French)
- Net effect: the NPU becomes the bottleneck, which is what you want

## Rule 4: The fork's VAD segfaults — do VAD yourself

`whisper_full(params.vad=true)` crashes in `whisper_vad()` even when a valid Silero
model is loaded. The crash path exists in `whisper-server` (where VAD gets enabled
despite CLI defaults). Our patch:

- disables VAD inside `whisper.cpp` unless `WHISPER_ALLOW_VAD=1` is set
- pins `wparams.vad = false` in `server.cpp`

Run [Silero VAD](https://github.com/snakers4/silero-vad) (ONNX, 1 ms/frame) in your
orchestrator instead — it's better anyway: you control thresholds, pre-roll, and
speculative triggers (see `scripts/server/voice-loop.py`).

## Rule 5: flexmlrt versions are not interchangeable

- The `.rai` files on AMD's HuggingFace were built with an older format. flexmlrt
  **1.9 refuses them** (`KeyError: dry_run` in Python; hard crash in C++ after load).
- Conversely, flexmlrt 1.7 (the runtime AMD's whisper fork documents) cannot load
  plain ONNX.
- Match versions: fork ↔ flexmlrt 1.7 ↔ HF `.rai` files. The Ryzen AI SDK 1.9 stack
  (voe/VitisAI EP) is a separate, parallel world — see [08-recompilation](08-recompilation.md).

## Rule 6: The NPU is a server, the CPU is the client

Mental model that matches every measurement: the XDNA 2 runs **one serialized queue**
of compiled-graph executions. Your job is to always have work pending while never
having two clients racing:

- Whisper's structure (encode NPU → decode CPU → encode NPU...) leaves the NPU ~50 %
  idle by design
- Our [inter-file pipeline](04-pipeline.md) overlaps *file N+1's* encode with *file
  N's* decode — the NPU utilization rises without ever having concurrent forwards
