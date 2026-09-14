"""Figuras de interpretabilidad del TFT entrenado.

    python scripts/06_tft_interpret.py [--exp tft]

El Temporal Fusion Transformer expone dos cosas que ninguna de las otras
arquitecturas del laboratorio da:

  * los pesos de la Variable Selection Network: cuanto usa el modelo cada una de
    las features derivadas del token (identidad, tipo, pitch, clase de pitch,
    duracion del shift, posicion). Dice QUE informacion le resulta util.
  * los pesos de la atencion interpretable: como reparte el contexto en el
    tiempo. Dice HASTA DONDE mira hacia atras.

Salida: experiments/<exp>/figures/tft_vsn.png y tft_attention.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import Config          # noqa: E402
from models import build_model     # noqa: E402
import registry as reg             # noqa: E402
import prompts as pr               # noqa: E402
import viz                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="tft")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--len", type=int, default=512, help="longitud de contexto a analizar")
    a = ap.parse_args()

    exp = ROOT / "experiments" / a.exp
    ck_path = exp / "checkpoints" / a.ckpt
    if not ck_path.exists():
        print("no existe %s (entrena antes tft)" % ck_path)
        return
    ck = reg.load_checkpoint(ck_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
    if cfg.model != "tft":
        print("%s no es un experimento TFT (modelo=%s)" % (a.exp, cfg.model))
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print("cargado %s paso %s: %s" % (a.exp, ck.get("step"), model.param_report()))

    if not hasattr(model, "interpret"):
        print("el modelo no expone interpret()")
        return

    # contexto real del corpus, no aleatorio
    px, pst, srcs = pr.make_token_prompts(4, prime_steps=int(a.len / 0.697), seed=99)
    x = px[:, -a.len:].to(dev)
    with torch.no_grad():
        out = model.interpret(x)

    figs = exp / "figures"
    written = []

    vsn = out.get("vsn_weights") if isinstance(out, dict) else None
    if vsn is not None:
        w = np.asarray(vsn.detach().float().cpu())
        w = w.reshape(-1, w.shape[-1]).mean(0)          # promedio sobre lote (y tiempo)
        names = list(out.get("variable_names") or [])[: len(w)] or \
            ["var%d" % i for i in range(len(w))]
        print("pesos de la VSN:")
        for n, v in sorted(zip(names, w), key=lambda t: -t[1]):
            print("   %-16s %6.2f%%" % (n, 100 * v))
        written.append(viz.plot_vsn_weights(
            dict(zip(names, w.tolist())), figs / "tft_vsn.png",
            title="%s: seleccion de variables (paso %s)" % (a.exp, ck.get("step"))))

    att = out.get("attention") if isinstance(out, dict) else None
    if att is not None:
        raw = np.asarray(att.detach().float().cpu())     # [B, bloques, L, L]
        A = raw
        while A.ndim > 2:                                # solo para las estadisticas
            A = A.mean(0)
        near = float(np.mean([A[i, max(0, i - 32):i + 1].sum() for i in range(1, len(A))]))
        print("atencion %s: masa en los ultimos 32 pasos = %.3f (1.0 = solo mira de cerca)"
              % (raw.shape, near))
        # se pasa sin promediar: viz dibuja tambien el perfil por distancia relativa
        written.append(viz.plot_attention(
            raw, figs / "tft_attention.png",
            title="%s: atencion interpretable (paso %s)" % (a.exp, ck.get("step"))))

    for p in written:
        p = Path(p)
        print("%-40s %7.1f KB" % (p.name, p.stat().st_size / 1024))
    if not written:
        print("interpret() no devolvio ni 'vsn' ni 'attention'; claves: %s"
              % (sorted(out.keys()) if isinstance(out, dict) else type(out)))


if __name__ == "__main__":
    main()
