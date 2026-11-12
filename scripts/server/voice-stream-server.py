#!/usr/bin/env python3
"""
Voice Stream Server — endpoint WebSocket style « GPT Live » (Realtime API), 100 % local.

 ws://127.0.0.1:9201

 Client → Serveur :
   - frames binaires : audio PCM16 mono 16 kHz (flux micro continu)
   - frames texte JSON :
       {"type":"configure","agent":true|false,"model":"...","stub":false}
       {"type":"text","text":"..."}        (question clavier -> réponse agent)

 Serveur → Client (JSON) :
   {"type":"ready"}
   {"type":"turn.start"}                   début de prise de parole détecté
   {"type":"transcript.final","text":"...","latency":{"stt":1.42,"voice_to_text":1.57}}
   {"type":"response.delta","text":"..."}  tokens de la réponse (mode agent)
   {"type":"response.audio","sr":22050,"audio":"<base64 PCM16>"}  TTS streamé
   {"type":"turn.end","total_ms":...}
   {"type":"error","message":"..."}

 Mode agent=false : transcription seule. Mode stub=true : réponse factice
 (tests sans Lemonade). Moteur : VAD Silero + STT NPU (npu-asr-server :9200) + Lemonade + Piper.
"""
import argparse
import base64
import importlib.util
import json
import os
import queue
import sys
import threading
import time

import numpy as np
from websockets.sync.server import serve

BASE = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location("vl", os.path.join(BASE, "voice-loop.py"))
vl = importlib.util.module_from_spec(spec)
sys.argv = ["voice-loop.py"]
spec.loader.exec_module(vl)

VAD_THRESHOLD = vl.VAD_THRESHOLD
VAD_MIN_SPEECH_MS = vl.VAD_MIN_SPEECH_MS
SPEC_SILENCE_MS = vl.SPEC_SILENCE_MS
END_SILENCE_MS = vl.END_SILENCE_MS
SR = vl.SR

STUB_RESPONSE = ("Bien sûr, je peux te répondre en streaming. Voici la première partie, "
                 "qui arrive dès la première virgule. Et voici la suite de la phrase, "
                 "avec une pause naturelle. Enfin la conclusion arrive maintenant.")


class VoiceSession:
    """Une session = un client WebSocket : sa machine à états VAD/STT/agent."""

    def __init__(self, ws):
        self.ws = ws
        self.send_lock = threading.Lock()
        self.cfg = {"agent": False, "model": "gpt-oss:20b", "stub": False}
        self.vad = vl.VAD()
        self.voice = None  # Piper chargé à la demande (mode agent)
        self.gen = 0
        self.gen_lock = threading.Lock()
        self.audio_q = queue.Queue()
        self.stt_q = queue.Queue()
        self.text_q = queue.Queue()
        self.barge = threading.Event()
        self.stats = {"t_spec": None}
        self.threads = [
            threading.Thread(target=self._stt_worker, daemon=True),
            threading.Thread(target=self._agent_worker, daemon=True),
        ]
        for t in self.threads:
            t.start()

    # ---------- utilitaires ----------
    def send(self, obj):
        data = json.dumps(obj, ensure_ascii=False)
        with self.send_lock:
            try:
                self.ws.send(data)
            except Exception:
                pass

    # ---------- worker STT ----------
    def _stt_worker(self):
        while True:
            g, samples = self.stt_q.get()
            if g is None:
                return
            while not self.stt_q.empty():
                g2, s2 = self.stt_q.get_nowait()
                if g2 > g:
                    g, samples = g2, s2
            t0 = time.time()
            try:
                text = vl.transcribe_wav(samples)
                self.text_q.put((g, text, t0, time.time()))
            except Exception as e:
                self.send({"type": "error", "message": f"stt: {e}"})

    # ---------- worker agent (transcript -> réponse streamée + TTS) ----------
    def stub_llm_stream(self):
        for w in STUB_RESPONSE.split(" "):
            yield w + " "
            time.sleep(0.03)

    def _agent_worker(self):
        while True:
            item = self.text_q.get()
            if item is None:
                return
            g, text, t0, t1 = item
            with self.gen_lock:
                if g != self.gen:
                    continue
            if not text or text.lower() in vl.BLACKLIST:
                continue
            t_spec = self.stats.get("t_spec")
            lat = {"stt": round(t1 - t0, 2)}
            if t_spec:
                lat["voice_to_text"] = round(t1 - (t_spec - SPEC_SILENCE_MS / 1000), 2)
            self.send({"type": "transcript.final", "text": text, "latency": lat})

            if not self.cfg["agent"]:
                self.send({"type": "turn.end"})
                continue

            if self.voice is None:
                self.voice = vl.make_tts()
            self.barge.clear()
            t_start_agent = time.time()

            def say(seg, first=False):
                try:
                    s, sr = vl.tts_samples(self.voice, seg)
                    pcm16 = (np.clip(s, -1, 1) * 32767).astype(np.int16).tobytes()
                    self.send({"type": "response.audio", "sr": sr,
                               "audio": base64.b64encode(pcm16).decode()})
                    if first:
                        self.send({"type": "latency.first_audio",
                                   "ms": int((time.time() - t_start_agent) * 1000)})
                except Exception as e:
                    self.send({"type": "error", "message": f"tts: {e}"})

            sentence, full, first = "", "", True
            stream = (self.stub_llm_stream() if self.cfg.get("stub")
                      else vl.llm_stream([{"role": "system", "content": vl.SYSTEM_PROMPT},
                                          {"role": "user", "content": text}],
                                         self.cfg["model"]))
            try:
                for tok in stream:
                    if self.barge.is_set():
                        break
                    sentence += tok
                    full += tok
                    self.send({"type": "response.delta", "text": tok})
                    st = sentence.strip()
                    if (st.endswith((".", "!", "?", "…")) and len(st) > 2) or \
                       (st.endswith((",", ";", ":")) and len(st) >= 30):
                        say(st, first)
                        first = False
                        sentence = ""
                rest = sentence.strip()
                if rest and not self.barge.is_set():
                    say(rest, first)
            except Exception as e:
                self.send({"type": "error", "message": f"llm: {e.__class__.__name__} — Lemonade lancé ?"})
            self.send({"type": "turn.end", "total_ms": int((time.time() - t_start_agent) * 1000)})

    # ---------- machine à états VAD (appelée par le thread récepteur) ----------
    def on_chunk(self, chunk):
        p = self.vad.prob(chunk)
        if p > VAD_THRESHOLD:
            if not self._in_speech:
                self._in_speech = True
                self.send({"type": "turn.start"})
                if self._spec_sent:
                    with self.gen_lock:
                        self.gen += 1
                    self._spec_sent = False
            self._speech_ms += 512 / SR * 1000
            self._silence_ms = 0
            self._buf.append(chunk)
            if self._speech_ms > VAD_MIN_SPEECH_MS and self.cfg["agent"] and self._tts_playing():
                if not self.barge.is_set():
                    self.barge.set()
        else:
            if self._in_speech:
                self._silence_ms += 512 / SR * 1000
                self._buf.append(chunk)
                if self._silence_ms >= SPEC_SILENCE_MS and not self._spec_sent and self._speech_ms >= VAD_MIN_SPEECH_MS:
                    with self.gen_lock:
                        self.stt_q.put((self.gen, np.concatenate(self._buf)))
                    self._spec_sent = True
                    self.stats["t_spec"] = time.time()
                if self._silence_ms >= END_SILENCE_MS:
                    self._in_speech = False
                    self._speech_ms = self._silence_ms = 0.0
                    self._spec_sent = False
                    self._buf = []
                    self.vad.reset()

    def _tts_playing(self):
        return False  # v1: barge-in côté client (il coupe la lecture localement)

    _in_speech = False
    _speech_ms = 0.0
    _silence_ms = 0.0
    _spec_sent = False
    _buf = []


def handler(ws):
    sess = VoiceSession(ws)
    sess.send({"type": "ready"})
    try:
        while True:
            msg = ws.recv()
            if isinstance(msg, bytes):
                pcm = np.frombuffer(msg, dtype=np.int16).astype(np.float32) / 32768.0
                for i in range(0, len(pcm), 512):
                    sess.on_chunk(pcm[i:i + 512])
            else:
                try:
                    d = json.loads(msg)
                except Exception:
                    continue
                t = d.get("type")
                if t == "configure":
                    for k in ("agent", "model", "stub"):
                        if k in d:
                            sess.cfg[k] = d[k]
                    sess.send({"type": "config", **sess.cfg})
                elif t == "text":
                    with sess.gen_lock:
                        sess.gen += 1
                    sess.text_q.put((sess.gen, d.get("text", ""), time.time(), time.time()))
    except Exception:
        pass
    finally:
        with sess.gen_lock:
            sess.stt_q.put((None, None))
            sess.text_q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9201)
    args = ap.parse_args()
    print(f"[voice-stream] ws://127.0.0.1:{args.port} — prêt (GPT-Live local)", file=sys.stderr)
    with serve(handler, "127.0.0.1", args.port) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()
