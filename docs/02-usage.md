# Usage — CLI, batch, and model choice

## Single file

```bat
whisper-cli.exe -m models\ggml-large-v3-turbo.bin -f audio.wav -t 16 -l fr -bs 1
```

- `-bs 1` greedy decoding: fastest, quality unchanged on clean speech
  ([why](07-npu-rules.md#rule-3))
- `-l fr` / `-l auto` — pin the language when you know it
- Audio: **16 kHz mono WAV** (`ffmpeg -i in.opus -ar 16000 -ac 1 out.wav`).
  Opus is listed as supported but silently yields empty transcripts.
- Outputs: add `-otxt` / `-osrt` / `-ovtt`; multiple files per invocation share one
  model load (cheaper than per-file runs)

## Batch (hundreds of files)

One invocation, many files — the inter-file pipeline (if built with
`-DWHISPER_PIPELINE=1`) overlaps encodes and decodes automatically:

```bat
whisper-cli.exe -m models\ggml-large-v3-turbo.bin -t 16 -l fr -bs 1 -otxt a1.wav a2.wav ...
```

Command-line length limit (~32 k): chunk at ~100 files, e.g.
`scripts/tools/bench-models.sh` shows the pattern. Sequential invocations are fine
(one NPU pipeline at a time — never parallel processes, see
[07-npu-rules](07-npu-rules.md)).

## Model choice

See [benchmarks](03-benchmarks.md): **large-v3-turbo only**, on conversational
speech. tiny/base are unusable for accented/fast speech or jargon (30-44 % WER).
`small` is broken on NPU at the time of writing (its `.rai` segfaults with flexmlrt
1.7 — sha verified, it's an AMD-side bug).

## Useful flags

| Flag | Effect |
|---|---|
| `-bs 1` | greedy decode (recommended default) |
| `-otxt` | write `<file>.txt` next to each input |
| `-l fr` | language hint |
| `-tr` | translate → English |
| `OVERRIDE_VITISAI_MODEL_PATH=...` env | point at another `.rai`/`.onnx` (bridge build) |
| `WHISPER_NO_PIPELINE=1` env | disable inter-file pipeline for A/B |

## Verifying the NPU is actually used

1. Log line: `whisper_init_state: Vitis AI model loaded` and `VITISAI = 1`
2. Per-window `encode time ≈ 1.1-1.2 s` (CPU would be ~5.5 s at 16 threads)
3. Task Manager → Performance → NPU spikes during encoding
