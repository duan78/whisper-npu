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
| **Speculative STT** | fires at **150 ms** of trailing silence — the trailing silence adds nothing to transcription; a generation counter discards the speculation if speech resumes | end of speech → text: **1.46 s** (STT 1.31 s, measured through the facade + persistent engine; 1.57 s on the original direct-engine setup) |
| **LLM worker** | streams tokens; flushes TTS at every sentence end **and at commas** once ≥ 30 chars | first spoken word: **+0.41 s** after transcript |
| **TTS** (Piper, fr_FR-siwis-medium) | synthesizes per segment, queue-played | ~0.3 s/segment |
| **Barge-in** | speaking while the agent talks cuts playback and generation | — |

Threads: PortAudio callback → VAD state machine → STT worker → LLM worker → speaker
worker. Listening **never blocks** — transcription, LLM and TTS all run while the mic
stays live. NPU access is serialized through the HTTP engine (single forward at a
time, per the rules).

## Usage

```bat
:: start the whole stack (persistent NPU engine :9199, OpenAI facade :9200, WS :9201)
scripts\server\start-servers.bat
```

```bash
# mic loop (default mic, gpt-oss:20b via Lemonade)
python voice-loop.py
python voice-loop.py --mic shokz           # pick mic by name fragment
python voice-loop.py --model qwen3:4b      # faster agent
python voice-loop.py --simulate note.wav   # no-mic integration test (real-time injection)

# or drive the WebSocket endpoint (any language)
python test-ws-client.py note.wav 1 1      # agent=1, stub LLM=1
```

STT goes through the facade at `:9200` (`STT_URL` at the top of `voice-loop.py`,
overridable via the `STT_URL`/`LLM_URL` environment variables — e.g. to target a
different engine or LLM backend).

## Local assets (gitignored — fetch once into `scripts/server/`)

| File | What it is | Where to get it |
|---|---|---|
| `silero_vad.onnx` | Silero VAD v5, 512-sample frames | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) (v5 ONNX export) |
| `fr_FR-siwis-medium.onnx` + `.onnx.json` | Piper French voice | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) (`fr/fr_FR/siwis/medium/`) |

Both are loaded from the script's own directory (`BASE`) — `start-servers.bat`
refuses to start without the VAD model. Python deps: `numpy`, `onnxruntime`,
`sounddevice`, `piper-tts`, `websockets`.

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
