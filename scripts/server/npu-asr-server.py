#!/usr/bin/env python3
"""
Serveur ASR local NPU — API compatible OpenAI.

Endpoints:
  POST /v1/audio/transcriptions  (multipart: file, model, language, response_format)
  GET  /v1/models
  GET  /health

Moteur: whisper-cli (fork AMD, encodeur large-v3-turbo sur NPU XDNA 2), greedy,
VitisAI. Un seul appel NPU à la fois (verrou) — règle d'or du XDNA 2.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
PORT = 9200
WHISPER = r"C:\Users\Arnaud\dev\whisper.cpp\build-pipe\bin\Release\whisper-cli.exe"
MODEL = r"C:\Users\Arnaud\dev\whisper-npu-bin\models\ggml-large-v3-turbo.bin"
MODEL_ID = "whisper-large-v3-turbo"
TMP = r"C:\Users\Arnaud\dev\npu-asr\tmp"
os.makedirs(TMP, exist_ok=True)

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

        txt = wav + ".txt"
        with npu_lock:  # règle NPU : jamais deux forwards simultanés
            t0 = time.time()
            r = subprocess.run(
                [WHISPER, "-m", MODEL, "-t", "16", "-bs", "1", "-otxt",
                 "-l", (language or "fr"), "-np", "-f", wav],
                capture_output=True, timeout=600)
            elapsed = time.time() - t0
        if r.returncode != 0:
            raise RuntimeError("whisper rc=" + str(r.returncode))
        if not os.path.exists(txt):
            raise RuntimeError("whisper: aucune sortie")
        with open(txt, encoding="utf-8", errors="replace") as f:
            text = f.read().strip()
        sys.stderr.write(f"[npu-asr] {filename} {duration:.1f}s audio -> {elapsed:.1f}s NPU\n")
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
            self._json(200, {"status": "ok", "engine": "npu-vitisai", "model": MODEL_ID})
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
