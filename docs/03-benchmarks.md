# Benchmarks — all measurements, real hardware

Machine: AMD Ryzen AI Max+ 395 (16 C Zen 5, Radeon 8060S, XDNA 2 NPU 50 TOPS),
128 GB LPDDR5X-8000, Windows 11 Pro. Driver 32.0.20102.3930.

## Model comparison on conversational French (NPU encoder)

100 files, 24.3 min of audio, quality measured against a 3-source arbitrated
ground truth (original transcript + NPU + Mistral voxtral-small-transcribe-3,
majority vote — see `scripts/dataset/arbitrate.py`).

| Model | Speed (NPU, greedy) | Real-time factor | Similarity median | WER median | WER mean | Files WER < 10 % |
|---|---|---|---|---|---|---|
| tiny | 0.22 s/file | ×65.6 | 65.3 % | 43.9 % | 52.2 % | 3/100 |
| base | 0.33 s/file | ×43.8 | 77.8 % | 31.6 % | 45.3 % | 6/100 |
| **large-v3-turbo** | 1.16 s/file | ×12.5 | **99.6 %** | **2.9 %** | 19.3 %* | **72/100** |

\* mean dragged by a few whispered/very short files; median is the honest figure.

**Conclusion: on conversational French with technical jargon, only large-v3-turbo is
viable.** tiny/base transcribe phonetically without comprehension ("cloud code" →
"claude cotte"). And turbo at ×12.5 real-time is already fast — the speed/quality
trade-off doesn't exist here.

## Encoder: NPU vs CPU

Same machine, whisper-cli, greedy, 16 threads CPU:

| | 30 s window encode |
|---|---|
| CPU (16t AVX-512) | 5542 ms |
| **NPU (XDNA 2, `.rai`)** | **1160 ms** |
| Speedup | **4.8×** |

## The window is everything (latency per utterance)

Persistent server, turbo:

| Utterance | Latency |
|---|---|
| 2 s | 1.41 s |
| 6 s | 1.44 s |
| 14 s | 1.77 s |

The fixed 30 s window dominates — see [07-npu-rules](07-npu-rules.md).

## Serving stack (2026-11-12, facade :9200 → persistent engine :9199)

| Request | Latency | Notes |
|---|---|---|
| 1.8 s utterance, engine up | **1.11–1.16 s** (×3 runs) | fast path: resident model + `.rai` |
| 40 s dense file, engine up | **3.3–3.5 s** | 2 NPU windows (0.9 s each) + ~1.3 s decode |
| 1.8 s utterance, engine down (fallback) | 3.5–5.3 s | whisper-cli spawn — still transcribes |

Why the fallback is slower — per-spawn phase breakdown (6 s file, whisper-cli):

| Phase | Time | Paid |
|---|---|---|
| load (ggml 1.6 GB + `.rai` context init) | 1335 ms | every spawn |
| encode (NPU) | 887 ms | every request |
| decode (CPU, 41 tokens × 7 ms) | 288 ms | every request |

The persistent engine pays the 1335 ms **once at boot**; each request is then just
encode + decode. This is why the engine-first architecture exists — and why the
fallback exists: if the engine crashes, transcription survives at +2 s.

## Inter-file pipeline

Batch of 30 mixed files (turbo, greedy), single model load:

| Configuration | Wall clock |
|---|---|
| Sequential | ~60 s |
| Pipeline v1 (naive prefetch before `whisper_full`) | 83 s — **inversion bug**, worse than sequential |
| **Pipeline v2 (callback-at-decode prefetch)** | **39.3 s (+35 %)** |

Quality: byte-identical outputs vs sequential on the 30-file sample. Design details
in [04-pipeline](04-pipeline.md).

## Voice loop (end-to-end)

Simulated real-time microphone (audio injected at capture speed), turbo + Silero VAD
+ speculative STT:

| Stage | Measured |
|---|---|
| End of speech → text (speculative trigger at 150 ms silence) | **1.57 s** (STT 1.42 s, original direct-engine setup) · **1.46 s** (STT 1.31 s, re-measured 2026-11-12 through facade + persistent engine) |
| Text → first spoken word (sentence-streamed TTS, stub LLM) | **0.41 s** (0.50 s re-measured) |

## Dataset verification (the workload that motivated everything)

806 real voice notes (198 min) verified with the NPU, arbitrated with Mistral Voxtral:

- 611 pairs healthy (76 %) — median similarity 98.8 % vs original transcripts
- 195 desynchronized (24 %) — export bug, permutations confirmed visually
- Full 806-file transcription: ~30 min wall clock (mix of sequential + pipeline)
