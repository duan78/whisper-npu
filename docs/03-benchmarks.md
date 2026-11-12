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
| End of speech → text (speculative trigger at 150 ms silence) | **1.57 s** (STT itself 1.42 s) |
| Text → first spoken word (sentence-streamed TTS, stub LLM) | **0.41 s** |

## Dataset verification (the workload that motivated everything)

806 real voice notes (198 min) verified with the NPU, arbitrated with Mistral Voxtral:

- 611 pairs healthy (76 %) — median similarity 98.8 % vs original transcripts
- 195 desynchronized (24 %) — export bug, permutations confirmed visually
- Full 806-file transcription: ~30 min wall clock (mix of sequential + pipeline)
