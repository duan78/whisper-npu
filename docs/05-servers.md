# Servers — OpenAI-compatible REST + WebSocket realtime endpoint

Two complementary ways to expose the NPU over HTTP. Both serialize NPU access
internally ([the rule](07-npu-rules.md)).

## Option A: persistent engine + REST proxy (simple)

- **Engine**: the fork's `whisper-server.exe` (rebuilt with our patch — the stock one
  segfaults on VAD). Loads the model **once** (ggml + `.rai` NPU context stay
  resident), serves `POST /inference` at ~1.1-1.4 s per utterance.
- **Proxy**: `scripts/server/npu-asr-server.py` — pure-stdlib Python. Accepts any
  audio format (ffmpeg converts), speaks OpenAI. Forwards to the persistent engine
  (fast path); if the engine is down, falls back to spawning whisper-cli
  (~+2 s model reload — transcription still works, just slower).

```bat
scripts\server\start-servers.bat    :: engine :9199 + proxy :9200 + WS :9201 (idempotent)
scripts\server\stop-servers.bat     :: stops all three
```

Endpoints: `POST /v1/audio/transcriptions` (multipart: `file`, `language`,
`response_format` = json|text|verbose_json), `GET /v1/models`, `GET /health`
(reports `"backend": "up (persistant)"` or `"down (repli whisper-cli)"`).
Tested with the official OpenAI Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9200/v1", api_key="local-npu")
print(client.audio.transcriptions.create(
    model="whisper-large-v3-turbo", file=open("note.ogg", "rb"), language="fr").text)
```

Latency: **~1.1-1.5 s per request** with the engine up (measured: 1.11-1.16 s for a
1.8 s utterance); ~3.5-5 s in fallback mode. The engine is started with `-l fr`;
requests asking for another language only get it honored in fallback mode — restart
the engine with the language you need otherwise.

## Option B: the realtime voice endpoint (WebSocket)

`scripts/server/voice-stream-server.py` — a mini local Realtime API:

```
ws://127.0.0.1:9201
```

Client → server:
- binary frames: continuous PCM16 mono 16 kHz microphone audio
- `{"type":"configure","agent":true|false,"model":"...","stub":false}`
- `{"type":"text","text":"..."}` (keyboard input → agent response)

Server → client:
- `{"type":"turn.start"}` — speech detected
- `{"type":"transcript.final","text":"...","latency":{"stt":1.42,"voice_to_text":1.57}}`
- `{"type":"response.delta","text":"..."}` — LLM tokens (agent mode)
- `{"type":"response.audio","sr":22050,"audio":"<base64 PCM16>"}` — streamed TTS
- `{"type":"turn.end"}`

Test client: `scripts/server/test-ws-client.py file.wav [agent] [stub]` streams a WAV
at real-time capture speed and prints the event sequence. `stub=1` fakes the LLM —
useful to develop clients without a backend.

The engine behind both: the patched persistent `whisper-server` on port 9199.
