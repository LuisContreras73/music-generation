"""Agregados de referencia del corpus y CALIBRACION del gen_score.

Salidas en data/processed/:
  ref_stats.npz / .json   histogramas de referencia (val, train, val_excerpt)
                          + calibracion: techo y suelo de OA por componente

Por que hace falta la calibracion
---------------------------------
El area de solapamiento (OA) entre dos muestras finitas de la MISMA
distribucion no vale 1, y hay componentes que un modelo trivial ya acierta.
Se miden por tanto dos anclas empiricas, con el MISMO numero de fragmentos y
la MISMA longitud que se usaran al evaluar generaciones:

  ceiling_k  OA medio entre una muestra real (del split de train, para no
             contaminar la referencia que sale de val) y la referencia.
             Es el maximo alcanzable: mas alla solo hay ruido de muestreo.
  floor_k    OA medio entre ruido i.i.d. con la densidad marginal del corpus
             y la referencia. Es lo que consigue un modelo que no aprende nada.

El gen_score reescala cada componente a [0,1] entre esas dos anclas y pondera
por el poder discriminativo medido (ceiling - floor).
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.datasets import get_roll, load_meta            # noqa: E402
from metrics import (aggregate_features, overlap, HIST_KEYS,   # noqa: E402
                     SCORE_KEYS, gen_score)

GEN_STEPS = 800          # longitud de las generaciones que se evaluaran
N_SAMPLES = 16           # cfg.n_gen_samples: la calibracion depende de este N
N_BOOT = 40              # repeticiones del bootstrap
DENSITY = 0.005142671821564932

m, splits = load_meta()
out = {}

# ---------------------------------------------------------------- referencias
for split in ("val", "train"):
    idx = splits[split]
    rng = np.random.default_rng(7)
    sel = idx if len(idx) <= 300 else rng.choice(idx, 300, replace=False).tolist()
    agg = aggregate_features([get_roll(int(i)) for i in sel])
    out[split] = agg
    print("%-6s n=%3d density %.5f poly %.3f empty %.3f notes/s %.2f repeat8 %.4f"
          % (split, agg["n_pieces"], agg["density"], agg["mean_poly"],
             agg["empty_ratio"], agg["notes_per_sec"], agg["repeat8"]))


def excerpts(pool, n, rng, steps=GEN_STEPS):
    """n fragmentos de `steps` pasos tomados de piezas distintas de `pool`."""
    got = []
    for i in rng.permutation(pool):
        r = get_roll(int(i))
        if len(r) > steps + 400:
            s = int(rng.integers(200, len(r) - steps))
            got.append(r[s:s + steps])
        if len(got) >= n:
            break
    return got


rng = np.random.default_rng(11)
out["val_excerpt"] = aggregate_features(excerpts(splits["val"], 250, rng))
ref = out["val_excerpt"]
print("val_excerpt (%d pasos) n=%d density %.5f poly %.3f repeat8 %.4f"
      % (GEN_STEPS, ref["n_pieces"], ref["density"], ref["mean_poly"], ref["repeat8"]))

# ---------------------------------------------------------------- calibracion
print("\ncalibrando con N=%d fragmentos de %d pasos, %d repeticiones..."
      % (N_SAMPLES, GEN_STEPS, N_BOOT))
ceil_acc = {k: [] for k in SCORE_KEYS}
floor_acc = {k: [] for k in SCORE_KEYS}
for b in range(N_BOOT):
    r1 = np.random.default_rng(2000 + b)
    real = aggregate_features(excerpts(splits["train"], N_SAMPLES, r1))
    noise = aggregate_features([(np.random.default_rng(5000 + b * 97 + j)
                                 .random((GEN_STEPS, 88)) < DENSITY).astype(np.uint8)
                                for j in range(N_SAMPLES)])
    for k in SCORE_KEYS:
        ceil_acc[k].append(overlap(real[k], ref[k]))
        floor_acc[k].append(overlap(noise[k], ref[k]))

calib = dict(
    ceiling={k: float(np.mean(v)) for k, v in ceil_acc.items()},
    floor={k: float(np.mean(v)) for k, v in floor_acc.items()},
    ceiling_std={k: float(np.std(v)) for k, v in ceil_acc.items()},
    n_samples=N_SAMPLES, gen_steps=GEN_STEPS, n_boot=N_BOOT,
)
gains = {k: calib["ceiling"][k] - calib["floor"][k] for k in SCORE_KEYS}
gsum = sum(max(g, 0.0) for g in gains.values())
print("\n%-16s %8s %8s %8s %8s" % ("componente", "suelo", "techo", "margen", "peso"))
for k in SCORE_KEYS:
    print("%-16s %8.3f %8.3f %8.3f %7.1f%%" %
          (k, calib["floor"][k], calib["ceiling"][k], gains[k],
           100 * max(gains[k], 0.0) / gsum))
out["calib"] = calib

# ---------------------------------------------------------------- verificacion
np.savez(ROOT / "data" / "processed" / "ref_stats.npz",
         **{s + "__" + k: np.asarray(v)
            for s, d in out.items() if s != "calib" for k, v in d.items()})
json.dump({s: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
           for s, d in out.items()},
          open(ROOT / "data" / "processed" / "ref_stats.json", "w"), indent=1)
json.dump(calib, open(ROOT / "data" / "processed" / "calibration.json", "w"), indent=1)

print("\nverificacion del score calibrado (deberia dar ~100 real, ~0 ruido):")
rv = np.random.default_rng(31)
real_test = aggregate_features(excerpts(splits["test"], N_SAMPLES, rv))
noise_test = aggregate_features([(np.random.default_rng(777 + j).random((GEN_STEPS, 88))
                                  < DENSITY).astype(np.uint8) for j in range(N_SAMPLES)])
silence = aggregate_features([np.zeros((GEN_STEPS, 88), np.uint8) for _ in range(N_SAMPLES)])
loop = aggregate_features([np.tile(get_roll(int(splits["test"][0]))[:8], (GEN_STEPS // 8, 1))
                           for _ in range(N_SAMPLES)])
for tag, agg in (("fragmentos reales (test)", real_test), ("ruido i.i.d.", noise_test),
                 ("silencio total", silence), ("bucle de 8 frames", loop)):
    s = gen_score(agg, ref, calib)
    print("   %-26s gen_score %6.2f   (sin calibrar %6.2f)"
          % (tag, s["gen_score"], gen_score(agg, ref)["gen_score"]))
print("\nOK -> ref_stats.npz / ref_stats.json / calibration.json")
