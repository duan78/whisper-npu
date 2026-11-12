#!/usr/bin/env python3
"""
Voice Loop v2 — discussion naturelle locale, latence minimisée.

Chaîne : Micro → VAD Silero → STT NPU (npu-asr-server :9200) → LLM Lemonade (:9000) → TTS Piper.

Optimisations v2 :
- STT en worker thread : l'écoute ne bloque jamais, même pendant transcription/LLM/TTS
- Déclenchement SPÉCULATIF du STT dès 150 ms de silence (le silence de fin n'ajoute
  rien à la transcription) ; si la parole reprend, jeton de génération -> résultat jeté
- Confirmation de fin de parole à 350 ms (au lieu de 450)
- Latences perçues mesurées et affichées par tour

Usage : python voice-loop.py [--mic NOM] [--model M] [--simulate FICHIER.wav]
"""
import argparse
import io
import json
import os
import queue
import sys
import threading
import time
import urllib.request
import wave

import numpy as np
import onnxruntime as ort
import sounddevice as sd

BASE = os.path.dirname(os.path.abspath(__file__))
SR = 16000

# STT : npu-asr-server.py (ce dépôt, port 9200) — même format multipart que l'API
# OpenAI. Surchargés par les variables d'environnement STT_URL / LLM_URL
# (ex. pour viser un autre moteur).
STT_URL = os.environ.get("STT_URL", "http://127.0.0.1:9200/v1/audio/transcriptions")
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:9000/api/v1/chat-completions")
SYSTEM_PROMPT = (
    "Tu es un assistant vocal local. Tu réponds en français, de façon naturelle et "
    "concise : 1 à 3 phrases, style parlé, pas de listes ni de markdown. "
    "Si on te demande du code ou des détails longs, propose de les écrire dans un fichier."
)

VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 300    # parole minimale pour une prise de parole
SPEC_SILENCE_MS = 150      # silence déclenchant le STT spéculatif
END_SILENCE_MS = 350       # silence confirmant la fin de parole
PRE_ROLL_MS = 300

BLACKLIST = {"...", "…", ".", "sous-titrage société radio-canada", "merci d'avoir regardé", "you"}


# ---------------- VAD Silero v5 ----------------
class VAD:
    def __init__(self):
        self.sess = ort.InferenceSession(os.path.join(BASE, "silero_vad.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.ctx = np.zeros(64, dtype=np.float32)

    def prob(self, chunk):
        if len(chunk) < 512:  # silero exige des fenêtres de taille exacte
            chunk = np.concatenate([chunk, np.zeros(512 - len(chunk), dtype=np.float32)])
        x = np.concatenate([self.ctx, chunk]).astype(np.float32)
        outs = self.sess.run(None, {
            "input": x[None, :], "state": self.state, "sr": np.array(SR, dtype=np.int64)})
        prob = float(np.asarray(outs[0]).flatten()[0])
        self.state = np.asarray(outs[1], dtype=np.float32)
        if len(outs) > 2:
            self.ctx = np.asarray(outs[2], dtype=np.float32)
        return prob


# ---------------- STT (NPU persistant) ----------------
def samples_to_wav_bytes(samples_f32):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(samples_f32, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()

def transcribe_wav(samples_f32):
    wav_bytes = samples_to_wav_bytes(samples_f32)
    boundary = "----npuvoice" + str(int(time.time() * 1000))
    body = (("--" + boundary + "\r\n"
             'Content-Disposition: form-data; name="file"; filename="utt.wav"\r\n'
             "Content-Type: audio/wav\r\n\r\n").encode()
            + wav_bytes
            + ("\r\n--" + boundary + "\r\n"
               'Content-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'
               "--" + boundary + "--\r\n").encode())
    req = urllib.request.Request(STT_URL, data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8")).get("text", "").strip()


# ---------------- LLM (Lemonade, SSE) ----------------
def llm_stream(messages, model):
    body = json.dumps({"model": model, "messages": messages,
                       "stream": True, "max_tokens": 400, "temperature": 0.6}).encode()
    req = urllib.request.Request(LLM_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
                delta = d["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]
            except Exception:
                continue


# ---------------- TTS (Piper 1.3) ----------------
def make_tts():
    from piper import PiperVoice
    return PiperVoice.load(os.path.join(BASE, "fr_FR-siwis-medium.onnx"))

def tts_samples(voice, text):
    parts, sr = [], 22050
    for chunk in voice.synthesize(text):
        parts.append(chunk.audio_float_array)
        sr = chunk.sample_rate
    if not parts:
        return np.zeros(0, dtype=np.float32), sr
    return np.concatenate(parts).astype(np.float32), sr


# ---------------- Boucle principale ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", default=None, help="nom partiel du micro")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--simulate", default=None, help="simuler le micro avec un fichier wav")
    args = ap.parse_args()

    vad = VAD()
    print("[voice-loop] chargement TTS Piper...", file=sys.stderr)
    voice = make_tts()

    audio_q = queue.Queue()          # chunks micro
    stt_q = queue.Queue()            # (gen, samples) à transcrire
    text_q = queue.Queue()           # (gen, text, t_spec) résultats STT
    speak_q = queue.Queue()          # (samples, sr) à jouer
    barge = threading.Event()
    gen = 0                          # jeton anti-spéculation-périmée
    gen_lock = threading.Lock()

    stats = {"t_end_speech": None, "t_spec": None, "t_last_voice": None}

    # --- worker STT : transcrit, résultats horodatés du jeton courant
    def stt_worker():
        while True:
            g, samples = stt_q.get()
            if g is None:
                return
            # ne garder que la dernière demande (les spéculations dépassées sont sautées)
            while not stt_q.empty():
                g2, s2 = stt_q.get_nowait()
                if g2 > g:
                    g, samples = g2, s2
            t0 = time.time()
            try:
                text = transcribe_wav(samples)
                text_q.put((g, text, t0, time.time()))
            except Exception as e:
                print(f"[stt err] {e}", file=sys.stderr)

    # --- worker LLM+TTS : traite le texte validé, phrase par phrase
    def llm_worker():
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        while True:
            item = text_q.get()
            if item is None:
                return
            g, text, t_stt0, t_stt1 = item
            with gen_lock:
                if g != gen:
                    continue  # spéculation périmée : la parole a repris
            if not text or text.lower() in BLACKLIST:
                continue
            delay = ""
            t_spec = stats.get("t_spec")
            if t_spec:
                # le STT spéculatif part à ~150 ms de silence après la fin réelle de la voix
                fin_voix = t_spec - SPEC_SILENCE_MS / 1000
                delay = f"  (voix finie -> texte: {t_stt1 - fin_voix:.2f}s, dont STT {t_stt1 - t_stt0:.2f}s)"
            print(f"\n[VOUS] {text}{delay}", flush=True)

            barge.clear()
            history.append({"role": "user", "content": text})
            sentence, full, t_first_audio = "", "", None

            def say(txt):
                nonlocal t_first_audio
                print(f"[AGENT] {txt}", flush=True)
                try:
                    s, sr = tts_samples(voice, txt)
                    if t_first_audio is None:
                        t_first_audio = time.time()
                        t_spec = stats.get("t_spec")
                        if t_spec:
                            fin_voix = t_spec - SPEC_SILENCE_MS / 1000
                            print(f"       (voix finie -> 1er mot parlé: {t_first_audio - fin_voix:.2f}s)", file=sys.stderr)
                    speak_q.put((s, sr))
                except Exception as e:
                    print(f"[tts err] {e}", file=sys.stderr)

            try:
                for tok in llm_stream(history, args.model):
                    if barge.is_set():
                        break
                    sentence += tok
                    full += tok
                    st = sentence.strip()
                    # flush sur fin de phrase, ou demi-phrase (virgule) si déjà ~30 chars :
                    # le premier mot parlé arrive plus tôt sans attendre la phrase complète
                    if (st.endswith((".", "!", "?", "…")) and len(st) > 2) or \
                       (st.endswith((",", ";", ":")) and len(st) >= 30):
                        say(st)
                        sentence = ""
                rest = sentence.strip()
                if rest and not barge.is_set():
                    say(rest)
            except Exception as e:
                say(f"Je n'arrive pas à joindre Lemonade. Vérifie qu'il est lancé.")
                print(f"[llm err] {e}", file=sys.stderr)
            if full:
                history.append({"role": "assistant", "content": full})
            if len(history) > 22:
                history = [history[0]] + history[-20:]

    # --- haut-parleur : joue la queue, barge-in coupe
    def speaker():
        while True:
            item = speak_q.get()
            if item is None:
                return
            samples, sr = item
            if len(samples) == 0:
                continue
            stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
            stream.start()
            CH = 4096
            for i in range(0, len(samples), CH):
                if barge.is_set():
                    break
                stream.write(samples[i:i+CH, None])
            stream.stop(); stream.close()

    threading.Thread(target=stt_worker, daemon=True).start()
    threading.Thread(target=llm_worker, daemon=True).start()
    threading.Thread(target=speaker, daemon=True).start()

    def audio_cb(indata, frames, t, status):
        audio_q.put(indata[:, 0].copy())

    def feed_simulated(path):
        """Simule un micro temps réel depuis un wav (+1s de silence final)."""
        with wave.open(path) as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        data = np.concatenate([data, np.zeros(SR, dtype=np.float32)])
        n = len(data) / SR
        t0 = time.time()
        for i in range(0, len(data), 512):
            audio_q.put(data[i:i+512])
            target = t0 + (i + 512) / SR
            d = target - time.time()
            if d > 0:
                time.sleep(d)
        print(f"[simu] {n:.1f}s d'audio injectés en temps réel", file=sys.stderr)

    pre_roll = np.zeros(int(SR * PRE_ROLL_MS / 1000), dtype=np.float32)
    buf, in_speech, speech_ms, silence_ms = [], False, 0.0, 0.0
    spec_sent = False

    print("[voice-loop] PRÊT — parlez. Ctrl+C pour arrêter.", file=sys.stderr)

    def on_chunk(chunk):
        nonlocal buf, in_speech, speech_ms, silence_ms, spec_sent, pre_roll, gen
        p = vad.prob(chunk)
        if p > VAD_THRESHOLD:
            stats["t_last_voice"] = time.time()
            if not in_speech:
                in_speech = True
                if spec_sent:
                    # la parole reprend : la spéculaiton en cours est périmée
                    with gen_lock:
                        gen += 1
                    spec_sent = False
                    print("[reprise de parole — spéculation annulée]", file=sys.stderr)
            speech_ms += 512 / SR * 1000
            silence_ms = 0
            buf.append(chunk)
            if speech_ms > VAD_MIN_SPEECH_MS and (speak_q.qsize() > 0 or barge.is_set()):
                if not barge.is_set() and speak_q.qsize() > 0:
                    barge.set()
                    print("\n[interruption]", file=sys.stderr)
        else:
            if in_speech:
                silence_ms += 512 / SR * 1000
                buf.append(chunk)
                # STT SPÉCULATIF : le silence de fin n'apporte rien à la transcription
                if silence_ms >= SPEC_SILENCE_MS and not spec_sent and speech_ms >= VAD_MIN_SPEECH_MS:
                    utterance = np.concatenate([pre_roll] + buf)
                    with gen_lock:
                        stt_q.put((gen, utterance))
                    spec_sent = True
                    stats["t_spec"] = time.time()
                if silence_ms >= END_SILENCE_MS:
                    in_speech, speech_ms, silence_ms, spec_sent = False, 0.0, 0.0, False
                    stats["t_end_speech"] = time.time() - (END_SILENCE_MS / 1000)
                    print(f"[VAD] fin de parole confirmée ({stats['t_end_speech']:.2f})", file=sys.stderr)
                    buf = []
                    vad.reset()
                    pre_roll = np.zeros(int(SR * PRE_ROLL_MS / 1000), dtype=np.float32)
            else:
                pre_roll = np.concatenate([pre_roll, chunk])[-len(pre_roll):]

    try:
        if args.simulate:
            threading.Thread(target=feed_simulated, args=(args.simulate,), daemon=True).start()
            deadline = time.time() + 25
            while time.time() < deadline:
                try:
                    on_chunk(audio_q.get(timeout=1))
                except queue.Empty:
                    pass
        else:
            device = None
            if args.mic:
                for i, d in enumerate(sd.query_devices()):
                    if d["max_input_channels"] > 0 and args.mic.lower() in d["name"].lower():
                        device = i
                        print(f"[voice-loop] micro: {d['name']}", file=sys.stderr)
                        break
            with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                blocksize=512, callback=audio_cb, device=device):
                while True:
                    on_chunk(audio_q.get())
    except KeyboardInterrupt:
        print("\n[voice-loop] arrêt.", file=sys.stderr)
    finally:
        stt_q.put((None, None)); text_q.put(None); speak_q.put(None)


if __name__ == "__main__":
    main()
