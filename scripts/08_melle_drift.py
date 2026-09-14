"""Diagnostico: por que MELLE tiene buena verosimilitud y genera basura.

    python scripts/08_melle_drift.py [--exp melle]

melle logro test_bpt 3.2152 (frente a 4.0921 de la referencia trivial), es
decir aprendio la distribucion de frames, y sin embargo sus generaciones dan
gen_score 0.00 con densidad ~4.3 onsets/paso frente a 0.45 real: casi diez
veces mas notas.

Las dos explicaciones posibles son distinguibles midiendo:

  (a) MALA CALIBRACION. Si el modelo predijera de por si demasiada masa, la
      densidad esperada en TEACHER FORCING -- la media de sigmoid(logits) sobre
      datos reales -- ya seria alta. Se puede medir directamente.
  (b) DERIVA AUTOREGRESIVA (exposure bias). Si en teacher forcing la densidad
      esperada es correcta (~0.0051) pero al muestrear crece paso a paso,
      entonces el problema es la realimentacion: un frame con demasiadas notas
      queda fuera de distribucion y empuja al siguiente a tener aun mas.

Este script mide ambas cosas y ademas la densidad generada EN FUNCION DEL PASO,
que es la firma inequivoca de la deriva: si crece monotonamente con el paso,
es (b).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import Config                      # noqa: E402
from data.datasets import FrameWindows         # noqa: E402
from models import build_model                 # noqa: E402
import registry as reg                         # noqa: E402
import prompts as pr                           # noqa: E402
import generate as gen                         # noqa: E402

DENSITY_REAL = 0.005142671821564932


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="melle")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    ck_path = ROOT / "experiments" / a.exp / "checkpoints" / a.ckpt
    if not ck_path.exists():
        print("no existe %s" % ck_path)
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = reg.load_checkpoint(ck_path, map_location=dev)
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
    model = build_model(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print("%s paso %s | %s" % (a.exp, ck.get("step"), model.param_report()))
    print("densidad real del corpus: %.6f onsets/(paso*nota)\n" % DENSITY_REAL)

    # ---------- (a) densidad esperada en TEACHER FORCING ----------
    ds = FrameWindows("val", cfg.seq_len, 40)
    dl = DataLoader(ds, batch_size=8, num_workers=0)
    tot_p = 0.0; tot_n = 0; tot_y = 0.0
    with torch.no_grad():
        for x, y, mask in dl:
            x, y, mask = x.to(dev), y.to(dev), mask.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = model(x)
            p = (out["logits"] if isinstance(out, dict) else out).float().sigmoid()
            m3 = mask.unsqueeze(-1)
            tot_p += float((p * m3).sum()); tot_y += float((y * m3).sum())
            tot_n += int(m3.sum()) * 1
    dens_tf = tot_p / max(tot_n, 1) * 1.0
    dens_true = tot_y / max(tot_n, 1) * 1.0
    print("(a) TEACHER FORCING sobre datos reales de validacion")
    print("    densidad esperada por el modelo   %.6f" % dens_tf)
    print("    densidad real de esas ventanas    %.6f" % dens_true)
    print("    cociente modelo/real              %.2fx" % (dens_tf / max(dens_true, 1e-9)))

    # ---------- (b) densidad generada en funcion del paso ----------
    px, srcs = pr.make_frame_prompts(a.n, cfg.gen_prime_steps, seed=4242)
    res = gen.sample_frames(model, px, a.steps, seq_len=cfg.seq_len, device=dev)
    new = np.stack([r for r in res["new"]])            # [n, steps, 88]
    per_step = new.reshape(len(new), a.steps, -1).sum(-1).mean(0)   # media sobre muestras
    print("\n(b) MUESTREO AUTOREGRESIVO (%d muestras de %d pasos)" % (len(new), a.steps))
    print("    densidad global generada          %.6f  (%.1fx la real)"
          % (new.mean(), new.mean() / DENSITY_REAL))
    blocks = 8
    bs = a.steps // blocks
    print("    onsets por paso segun el tramo generado:")
    for b in range(blocks):
        seg = per_step[b * bs:(b + 1) * bs]
        print("      pasos %4d-%4d  %6.2f notas/paso" % (b * bs, (b + 1) * bs - 1, seg.mean()))
    real_per_step = DENSITY_REAL * 88
    print("    referencia real:  %.2f notas/paso" % real_per_step)

    # --- descomposicion del exceso total en sus dos factores multiplicativos ---
    gen_per_step = float(per_step.mean())
    f_calib = dens_tf / max(dens_true, 1e-9)          # lo que el modelo ya sobreestima
    f_ampl = gen_per_step / max(dens_tf, 1e-9)        # lo que anade la realimentacion
    growth = per_step[-bs:].mean() / max(per_step[:bs].mean(), 1e-9)

    print("\nDESCOMPOSICION DEL EXCESO DE DENSIDAD")
    print("  el exceso NO tiene una sola causa: es el producto de dos factores.")
    print("    calibracion   %5.2fx   (%.2f notas/frame esperadas vs %.2f reales, teacher forcing)"
          % (f_calib, dens_tf, dens_true))
    print("    realimentacion%5.2fx   (%.2f notas/paso muestreadas vs %.2f que el modelo espera)"
          % (f_ampl, gen_per_step, dens_tf))
    print("    ------------------------")
    print("    total         %5.2fx   (%.2f vs %.2f notas/paso reales)"
          % (f_calib * f_ampl, gen_per_step, real_per_step))
    print("\n  crecimiento del primer al ultimo tramo: %.2fx" % growth)
    if growth < 1.2:
        print("  -> NO es deriva acumulativa: la densidad se estabiliza alta desde el principio.")
        print("     La realimentacion actua de golpe al salir del prefijo real, no poco a poco.")
    else:
        print("  -> hay deriva acumulativa: la densidad sigue creciendo a lo largo de la muestra.")
    print("\n  Por que la calibracion es mala aun teniendo buen bits/paso: la perdida que se")
    print("  optimiza NO es la BCE pura, sino BCE + KL + regresion + flux. La flux loss premia")
    print("  la variacion entre frames, que en un roll binario disperso significa encender")
    print("  notas. El experimento ablacion_melle_sin_flux (flux_weight=0) aisla esa contribucion.")

    import json
    out = dict(exp=a.exp, step=int(ck.get("step", 0)),
               flux_weight=float(cfg.flux_weight), kl_weight=float(cfg.kl_weight),
               tf_expected_per_frame=float(dens_tf), tf_real_per_frame=float(dens_true),
               gen_per_step=gen_per_step, real_per_step=float(real_per_step),
               factor_calibracion=float(f_calib), factor_realimentacion=float(f_ampl),
               factor_total=float(f_calib * f_ampl), crecimiento_tramos=float(growth),
               per_step_por_tramo=[float(per_step[b * bs:(b + 1) * bs].mean())
                                   for b in range(blocks)])
    dst = ROOT / "experiments" / a.exp / "logs" / "drift.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=2)
    print("\nguardado %s" % dst.relative_to(ROOT))


if __name__ == "__main__":
    main()
