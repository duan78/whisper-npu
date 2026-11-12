"""Chirurgie ONNX : fenêtre encoder whisper 3000 -> 1000 frames (~10 s).
1. Recense les valeurs 3000/1500 dans les données des initializers (Reshape shapes, etc.)
2. Ajuste input/output + positional_embedding + tous les shapes concernés
3. Valide numériquement sur CPU vs le modèle 3000 d'origine (tronqué)
"""
import numpy as np
import onnx
from onnx import numpy_helper

SRC = r"C:\Users\Arnaud\dev\rai-build\onnx-src\encoder_model.onnx"
DST = r"C:\Users\Arnaud\dev\rai-build\encoder_model_w1000.onnx"
SRC_WINDOW, DST_WINDOW = 3000, 1000  # frames mel
SRC_HALF, DST_HALF = 1500, 500       # frames encodeur (stride 2)

m = onnx.load(SRC)  # charge les données externes en mémoire

# ---------- 1. recensement ----------
def map_val(v):
    return {3000: DST_WINDOW, 1500: DST_HALF}.get(v, v)

changed = []
for init in m.graph.initializer:
    arr = numpy_helper.to_array(init)
    if np.isin(arr, [3000, 1500]).any():
        new = np.vectorize(map_val)(arr)
        if not np.array_equal(arr, new):
            changed.append((init.name, list(arr.shape), f"{(arr != new).sum()} valeurs"))
            init.CopyFrom(numpy_helper.from_array(new.astype(arr.dtype), init.name))

print(f"initializers modifiés: {len(changed)}")
for c in changed[:20]:
    print("   ", c)

# positional_embedding [1500,1280] -> [500,1280] (slice, pas resize)
for init in m.graph.initializer:
    if init.name == "positional_embedding":
        arr = numpy_helper.to_array(init)
        assert arr.shape == (SRC_HALF, arr.shape[1]), arr.shape
        init.CopyFrom(numpy_helper.from_array(arr[:DST_HALF].copy(), init.name))
        print(f"positional_embedding: {arr.shape} -> {(DST_HALF, arr.shape[1])}")

# Constant nodes (données 3000/1500)
for node in m.graph.node:
    if node.op_type != "Constant":
        continue
    for attr in node.attribute:
        t = onnx.helper.get_attribute_value(attr)
        if t.int64_data and np.isin(list(t.int64_data), [3000, 1500]).any():
            new_vals = [map_val(v) for v in t.int64_data]
            t.int64_data[:] = new_vals
            print("Constant ajusté:", node.name, new_vals[:10])

# input / output
inp = m.graph.input[0]
dims = inp.type.tensor_type.shape.dim
assert dims[2].dim_value == SRC_WINDOW
dims[2].dim_value = DST_WINDOW
out = m.graph.output[0]
od = out.type.tensor_type.shape.dim
od[1].dim_value = DST_HALF

# supprimer les annotations de shape intermédiaires : elles contiennent les
# anciennes dimensions 3000/1500 et bloquent la shape inference (recalculées par ORT)
del m.graph.value_info[:]

# inliner tous les tensors (sinon conflit external_data lors de la sérialisation)
for init in m.graph.initializer:
    if init.data_location == onnx.TensorProto.EXTERNAL:
        arr = numpy_helper.to_array(init)
        init.CopyFrom(numpy_helper.from_array(arr.copy(), init.name))

onnx.save_model(m, DST, save_as_external_data=True, all_tensors_to_one_file=True,
                location="encoder_model_w1000.onnx.data", size_threshold=1024)
print("sauvé:", DST)

# ---------- 3. validation numérique CPU ----------
import onnxruntime as ort
import time

t0 = time.time()
s_full = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
s_short = ort.InferenceSession(DST, providers=["CPUExecutionProvider"])
print(f"sessions CPU prêtes ({time.time()-t0:.0f}s)")

x = np.random.randn(1, 128, DST_WINDOW).astype(np.float32) * 0.5
y_full = s_full.run(None, {"x": np.pad(x, ((0,0),(0,0),(0,SRC_WINDOW-DST_WINDOW)))})[0]
y_short = s_short.run(None, {"x": x})[0]
print("sorties:", y_full.shape, y_short.shape)
ref = y_full[:, :DST_HALF, :]
err = np.abs(ref - y_short).max()
rel = err / (np.abs(ref).max() + 1e-9)
print(f"erreur max: {err:.4f} (relative {rel:.2%})")
print("VALIDATION:", "OK ✓" if rel < 1e-3 else "ÉCHEC ✗")
