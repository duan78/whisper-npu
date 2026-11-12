#!/usr/bin/env python3
"""
Serveur ASR local NPU — API compatible OpenAI.

Endpoints:
  POST /v1/audio/transcriptions  (multipart: file, model, language, response_format)
  GET  /v1/models
  GET  /health

Moteur: whisper-server persistant (fork AMD, modèle ggml + cache .rai résidents sur
le NPU — zéro rechargement par appel, lancé par start-servers.bat sur :9199, greedy).
Repli automatique sur whisper-cli (spawn par requête) si le moteur est down.
Un seul appel NPU à la fois (verrou) — règle d'or du XDNA 2.
"""
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOST = "127.0.0.1"
PORT = 9200
BASE = os.path.dirname(os.path.abspath(__file__))
WHISPER = r"C:\Users\Arnaud\dev\whisper.cpp\build-pipe\bin\Release\whisper-cli.exe"
MODEL = r"C:\Users\Arnaud\dev\whisper-npu-bin\models\ggml-large-v3-turbo.bin"
MODEL_ID = "whisper-large-v3-turbo"
TMP = os.path.join(BASE, "tmp")
os.makedirs(TMP, exist_ok=True)

# moteur persistant (voir start-servers.bat) — surchargeable via BACKEND_URL
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:9199/inference")
_u = urlsplit(BACKEND_URL)
BACKEND_ADDR = (_u.hostname or "127.0.0.1", _u.port or 80)

npu_lock = threading.Lock()  # un seul forward NPU à la fois


def parse_multipart(body, content_type):
    """Parseur multipart/form-data minimal (stdlib, sans cgi)."""
    m = re.search(r'boundary="?([^";]+)"?', content_type)
    if not m:
        return {}, {}
    boundary = m.group(1).encode()
    fields, files = {}, {}
    for part in body.split(b"--" + boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue
        header_blob, data = part.split(b"\r\n\r\n", 1)
        headers = header_blob.decode("utf-8", errors="replace")
        name_m = re.search(r'name="([^"]+)"', headers)
        file_m = re.search(r'filename="([^"]+)"', headers)
        if not name_m:
            continue
        name = name_m.group(1)
        if file_m:
            files[name] = (file_m.group(1), data)
        else:
            fields[name] = data.decode("utf-8", errors="replace")
    return fields, files


def transcribe_backend(wav):
    """Chemin rapide : moteur persistant :9199 (modèle + .rai résidents).
    Retourne le texte, ou None si le moteur est injoignable (repli whisper-cli)."""
    with open(wav, "rb") as f:
        wav_bytes = f.read()
    boundary = "----npuasr" + str(int(time.time() * 1000))
    body = (("--" + boundary + "\r\n"
             'Content-Disposition: form-data; name="file"; filename="utt.wav"\r\n'
             "Content-Type: audio/wav\r\n\r\n").encode()
            + wav_bytes
            + ("\r\n--" + boundary + "\r\n"
               'Content-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'
               "--" + boundary + "--\r\n").encode())
    req = urllib.request.Request(BACKEND_URL, data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=600) as r:
        return " ".join(json.loads(r.read().decode("utf-8")).get("text", "").split())


def transcribe_spawn(wav, language):
    """Chemin de repli : spawn whisper-cli (paye ~1,4 s de chargement modèle+.rai)."""
    txt = wav + ".txt"
    r = subprocess.run(
        [WHISPER, "-m", MODEL, "-t", "16", "-bs", "1", "-otxt",
         "-l", (language or "fr"), "-np", "-f", wav],
        capture_output=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError("whisper rc=" + str(r.returncode))
    if not os.path.exists(txt):
        raise RuntimeError("whisper: aucune sortie")
    with open(txt, encoding="utf-8", errors="replace") as f:
        return " ".join(f.read().split())


def transcribe(audio_bytes, filename, language):
    """Sauvegarde, convertit en wav 16k mono, transcrit sur NPU. Retourne (texte, durée_audio)."""
    suffix = os.path.splitext(filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(dir=TMP, suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        src = f.name
    wav = src + ".wav"
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav],
            capture_output=True, timeout=120)
        if r.returncode != 0 or not os.path.exists(wav):
            raise RuntimeError("ffmpeg: " + r.stderr.decode(errors="replace")[:200])

        import wave
        with wave.open(wav) as w:
            duration = w.getnframes() / w.getframerate()

        with npu_lock:  # règle NPU : jamais deux forwards simultanés
            t0 = time.time()
            text = None
            try:
                text = transcribe_backend(wav)
                engine = "persistant"
            except (OSError, urllib.error.URLError, ValueError):
                text = transcribe_spawn(wav, language)
                engine = "repli whisper-cli"
            elapsed = time.time() - t0
        sys.stderr.write(f"[npu-asr] {filename} {duration:.1f}s audio -> {elapsed:.1f}s NPU ({engine})\n")
        return text, duration
    finally:
        for p in (src, wav, wav + ".txt"):
            try:
                os.unlink(p)
            except OSError:
                pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            try:
                with socket.create_connection(BACKEND_ADDR, timeout=1):
                    backend = "up (persistant)"
            except OSError:
                backend = "down (repli whisper-cli)"
            self._json(200, {"status": "ok", "engine": "npu-vitisai",
                             "backend": backend, "model": MODEL_ID})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "local-npu"}]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != "/v1/audio/transcriptions":
            self._json(404, {"error": {"message": "not found, POST /v1/audio/transcriptions"}})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            fields, files = parse_multipart(body, self.headers.get("Content-Type", ""))

            if "file" not in files:
                self._json(400, {"error": {"message": "champ 'file' manquant"}})
                return
            filename, data = files["file"]
            if len(data) < 100:
                self._json(400, {"error": {"message": "fichier audio vide ou trop court"}})
                return
            language = fields.get("language", "fr")
            response_format = fields.get("response_format", "json")

            text, duration = transcribe(data, filename, language)

            if response_format == "text":
                out = text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            elif response_format == "verbose_json":
                self._json(200, {"task": "transcribe", "language": language,
                                 "duration": round(duration, 2), "text": text})
            else:
                self._json(200, {"text": text})
        except Exception as e:
            self._json(500, {"error": {"message": str(e)[:300]}})

    def log_message(self, fmt, *args):
        pass  # logs propres via stderr dans transcribe()


if __name__ == "__main__":
    sys.stderr.write(f"[npu-asr] moteur: {WHISPER}\n[npu-asr] modèle: {MODEL_ID} (NPU XDNA 2)\n")
    sys.stderr.write(f"[npu-asr] écoute sur http://{HOST}:{PORT}/v1/audio/transcriptions\n")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
