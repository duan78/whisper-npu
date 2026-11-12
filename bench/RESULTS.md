# Raw benchmark data

Condensed from the measurement session of 2026-11-11/12 (see docs/03-benchmarks.md
for context). Machine: Ryzen AI Max+ 395, 128 GB, Win 11, driver 32.0.20102.3930.

## Encoder micro-benchmarks (jfk.wav, 11 s)

| Model | NPU encode | CPU encode (16t) | Speedup |
|---|---|---|---|
| tiny | 49 ms | 164 ms | 3.3× |
| base | 109 ms | ~450 ms | ~4× |
| large-v3-turbo | 1157-1190 ms | 5542 ms | 4.65× |

## Batch throughput (mixed real files, turbo, greedy)

| Run | Files | Audio | Wall | Rate |
|---|---|---|---|---|
| Sequential 30-file sample | 30 | 356 s | 60 s | ×5.9 real-time |
| Pipeline v2, same sample | 30 | 356 s | 39.3 s | ×9.1 real-time |
| Full dataset (seq + pipeline mix) | 806 | 11 898 s | ~1 900 s | ×6.3 real-time |

## Model quality (100 files vs 3-source arbitrated truth)

tiny: WER median 43.9 %, mean 52.2 %, 3/100 usable — rejected
base: WER median 31.6 %, mean 45.3 %, 6/100 usable — rejected
turbo: WER median 2.9 %, mean 19.3 %*, 72/100 < 10 % — adopted

*tail-heavy: a handful of whispered/very-short files dominate the mean.

## Server latency (persistent engine, turbo)

2 s utterance: 1.41 s · 6 s: 1.44 s · 14 s: 1.77 s (fixed 30 s window — see docs/07)

## Voice loop (simulated real-time mic)

end of speech → text: 1.57 s (STT 1.42 s, speculative trigger at 150 ms silence)
text → first spoken word: 410 ms (sentence+comma-streamed TTS, stub LLM)

## Recompilation experiment

window-1000 encoder on NPU: 255 ms/window in whisper.cpp (×4.5) — but cos-sim 0.48
vs 30 s-padded model (out of distribution) and provider default quantization cos-sim
0.25 vs CPU on the unmodified model. See docs/08.
