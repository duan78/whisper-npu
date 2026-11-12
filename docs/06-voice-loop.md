# The local voice loop — a private "GPT Live"

```
Mic ─→ Silero VAD ─→ STT (NPU, speculative) ─→ LLM (Lemonade/local or cloud)
                     │                              │ (sentence-streamed)
                     └── barge-in (interrupt) ──→ TTS (Piper) ─→ Speakers
```

`scripts/server/voice-loop.py` — one file, threads only, ~350 lines.

## Architecture

| Component | Role | Measured |
|---|---|---|
| **Silero VAD** (ONNX, CPU) | speech/silence per 512-sample frame (~1 ms) | 83 % speech detected on real notes |
| **Speculative STT** | fires at **150 ms** of trailing silence — the trailing silence adds nothing to transcription; a generation counter discards the speculation if speech resumes | end of speech → text: **1.57 s** |
| **LLM worker** | streams tokens; flushes TTS at every sentence end **and at commas** once ≥ 30 chars | first spoken word: **+0.41 s** after transcript |
| **TTS** (Piper, fr_FR-siwis-medium) | synthesizes per segment, queue-played | ~0.3 s/segment |
| **Barge-in** | speaking while the agent talks cuts playback and generation | — |

Threads: PortAudio callback → VAD state machine → STT worker → LLM worker → speaker
worker. Listening **never blocks** — transcription, LLM and TTS all run while the mic
stays live. NPU access is serialized through the HTTP engine (single forward at a
time, per the rules).

## Usage

```bash
# engine (once)
whisper-server.exe -m models\ggml-large-v3-turbo.bin -l fr --port 9199 -bs 1 -t 16

# loop
python voice-loop.py                       # default mic, gpt-oss:20b via Lemonade
python voice-loop.py --mic shokz           # pick mic by name fragment
python voice-loop.py --model qwen3:4b      # faster agent
python voice-loop.py --simulate note.wav   # no-mic integration test (real-time injection)
```

Per-turn console output includes the two latency figures above — watch them to tune.

## Tunables (top of file)

| Constant | Default | Effect |
|---|---|---|
| `SPEC_SILENCE_MS` | 150 | lower = snappier, more false triggers |
| `END_SILENCE_MS` | 350 | turn confirmation |
| `VAD_THRESHOLD` | 0.5 | raise in noisy rooms |
| Blacklist | … | known Whisper hallucinations on silence ("Sous-titrage Société Radio-Canada", …) |

## Known behavior

- On quick speech resumption, a speculative segment can occasionally repeat — the
  generation counter discards most cases.
- The LLM endpoint targets [Lemonade Server](https://github.com/amd/lemonade)
  (`/api/v1/chat-completions`, SSE) on the iGPU; swap `llm_stream()` for any
  OpenAI-compatible URL for other backends.
- Whisper hallucinates on pure silence/noise — the blacklist catches the classics;
  add yours.
