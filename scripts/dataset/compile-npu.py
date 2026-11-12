"""Compilation NPU de l'encoder whisper fenêtre 1000 via VitisAI EP.
Le premier run déclenche la compilation AIE (long) et écrit un cache."""
import numpy as np
import onnxruntime as ort
import time, os, glob

os.chdir(r"C:\Users\Arnaud\dev\rai-build")
MODEL = "encoder_model_w1000.onnx"

print("providers disponibles:", ort.get_available_providers())
so = ort.SessionOptions()
so.log_severity_level = 1

t0 = time.time()
sess = ort.InferenceSession(MODEL, so, providers=["VitisAIExecutionProvider"])
print(f"session créée en {time.time()-t0:.0f}s (compilation AIE incluse)")

x = np.zeros((1, 128, 1000), dtype=np.float32)
t0 = time.time()
out = sess.run(None, {"x": x})
print(f"run 1: {time.time()-t0:.2f}s, sortie {out[0].shape}")
t0 = time.time()
out = sess.run(None, {"x": x})
print(f"run 2: {time.time()-t0:.2f}s")

# localiser le cache généré
print("\n=== fichiers récents dans rai-build ===")
for f in sorted(glob.glob("**/*", recursive=True), key=os.path.getmtime, reverse=True):
    if os.path.isfile(f) and time.time() - os.path.getmtime(f) < 300 and not f.startswith("msi"):
        print(f"  {time.time()-os.path.getmtime(f):6.0f}s ago  {os.path.getsize(f):>12,}  {f}")
        if f.endswith((".onnx", ".py", ".zip", ".data")):
            continue
