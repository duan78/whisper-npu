"""Client de test du voice-stream-server : streame un wav au rythme temps réel
et affiche les événements reçus. Usage: test-ws-client.py [wav] [agent] [stub]"""
import base64
import json
import sys
import time
import wave

from websockets.sync.client import connect

WAV = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Arnaud\dev\whisper-bench-real\full\wav\a0002.wav"
AGENT = len(sys.argv) > 2 and sys.argv[2] == "1"
STUB = len(sys.argv) > 3 and sys.argv[3] == "1"
SR = 16000

with wave.open(WAV) as w:
    data = w.readframes(w.getnframes())

ws = connect("ws://127.0.0.1:9201")
ws.send(json.dumps({"type": "configure", "agent": AGENT, "stub": STUB}))

events = []
done = threading_done = False
t_stream_end = None

import threading

def reader():
    global done
    while True:
        try:
            msg = ws.recv(timeout=30)
        except Exception:
            break
        d = json.loads(msg)
        events.append(d)
        t = d.get("type")
        if t == "ready":
            print("[evt] ready")
        elif t == "config":
            print(f"[evt] config: {d}")
        elif t == "turn.start":
            print(f"[evt] turn.start  ({time.time():.1f})")
        elif t == "transcript.final":
            print(f"[evt] TRANSCRIPT: {d['text']!r}  latence={d['latency']}")
        elif t == "response.delta":
            print(f"[evt] delta: {d['text']!r}", end="", flush=True)
            if not d["text"].strip():
                print()
        elif t == "response.audio":
            n = len(base64.b64decode(d["audio"])) // 2
            print(f"\n[evt] audio: {n} samples @{d['sr']}Hz")
        elif t == "latency.first_audio":
            print(f"[evt] PREMIER AUDIO PARLÉ: {d['ms']} ms après le transcript")
        elif t == "turn.end":
            print(f"[evt] turn.end  total={d.get('total_ms')} ms")
            break
        elif t == "error":
            print(f"[evt] ERREUR: {d['message']}")

rt = threading.Thread(target=reader, daemon=True)
rt.start()

# stream temps réel : chunks de 32 ms
CH = 512
t0 = time.time()
for i in range(0, len(data), CH * 2):
    ws.send(data[i:i + CH * 2])
    target = t0 + (i + CH * 2) / (SR * 2)
    d = target - time.time()
    if d > 0:
        time.sleep(d)
print(f"[cli] {len(data)/2/SR:.1f}s d'audio streamés, attente des événements...")
time.sleep(0.4)
ws.send(b"\x00\x00" * SR)  # 1s de silence pour déclencher la fin de parole
rt.join(timeout=40)
ws.close()

types = [e["type"] for e in events]
print("\n=== BILAN ===")
print("séquence:", " -> ".join(types))
ok = ("turn.start" in types and "transcript.final" in types
      and ("turn.end" in types)
      and (not AGENT or any(t == "response.audio" for t in types)))
print("RÉSULTAT:", "OK ✓" if ok else "ÉCHEC ✗")
