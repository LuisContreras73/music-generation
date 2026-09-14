"""Barrido de temperatura y nucleus sobre el mejor checkpoint de cada experimento.

    python scripts/09_sampling_sweep.py                 # todos los experimentos token
    python scripts/09_sampling_sweep.py --exp music_transformer
    python scripts/09_sampling_sweep.py --quick         # rejilla reducida

Por que hace falta
------------------
En tft se observo que al mejorar la verosimilitud EMPEORA el gen_score:
val_bpt 2.07 -> 1.87 mientras gen_score 35.3 -> 23.6. La descomposicion del
score senala una sola causa: la penalizacion por bucles, porque la fraccion de
8-gramas de frames repetidos sube de 0.372 a 0.527 (el corpus real esta en
0.217). Todo lo demas mejora -- la densidad se acerca a la real y el histograma
de intervalos armonicos se mantiene alto.

Eso es la degeneracion por muestreo descrita por Holtzman et al. (2020): un
modelo mas afilado, muestreado con temperatura 1.0, repite mas. No es un defecto
del modelo sino de la configuracion de muestreo, y comparar todos los modelos a
temperatura fija los juzga a todos con la configuracion optima de ninguno.

Este script mide gen_score en una rejilla (temperatura x top_p) y guarda el
resultado en experiments/<exp>/logs/sampling_sweep.json, de modo que la
comparacion final pueda hacerse con cada modelo en su mejor punto de muestreo
ademas de en el punto comun.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import Config                      # noqa: E402
from models import build_model                 # noqa: E402
import registry as reg                         # noqa: E402
import prompts as pr                           # noqa: E402
import generate as gen                         # noqa: E402
import evaluate as ev                          # noqa: E402

GRID_FULL = [(0.85, 0.95), (0.95, 0.95), (1.0, 0.95), (1.05, 0.95), (1.15, 0.95),
             (1.0, 0.99), (1.0, 1.0), (1.1, 0.99), (1.25, 0.99)]
GRID_QUICK = [(0.95, 0.95), (1.0, 0.95), (1.1, 0.99), (1.25, 0.99)]


def sweep(name: str, ckpt: str = "best.pt", quick: bool = False,
          n: int = 16, steps: int = 800, device: str = "cuda") -> dict:
    exp = ROOT / "experiments" / name
    ck_path = exp / "checkpoints" / ckpt
    if not ck_path.exists():
        print("  %s: no existe %s" % (name, ckpt))
        return {}
    ck = reg.load_checkpoint(ck_path, map_location=device)
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
    if cfg.family != "token":
        print("  %s: familia %s, el barrido de temperatura no aplica igual" % (name, cfg.family))
        return {}
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print("\n%s (paso %s, %s)" % (name, ck.get("step"), model.param_report()))
    print("  %5s %5s | %8s %8s %8s %8s %8s" %
          ("temp", "top_p", "gen_sc", "pen_loop", "repeat8", "densidad", "p_impos"))

    px, pst, srcs = pr.make_token_prompts(n, cfg.gen_prime_steps, seed=777)
    rows = []
    for temp, top_p in (GRID_QUICK if quick else GRID_FULL):
        t0 = time.time()
        res = gen.sample_tokens(model, px, steps, seq_len=cfg.seq_len, temperature=temp,
                                top_k=0, top_p=top_p, device=device)
        rolls = []
        for b in range(len(res["new"])):
            r = gen.tokens_to_roll(res["new"][b])
            if len(r) < steps:
                r = np.pad(r, ((0, steps - len(r)), (0, 0)))
            rolls.append(r[:steps])
        m, agg, ref = ev.evaluate_generations(rolls)
        row = dict(temperature=temp, top_p=top_p, gen_score=float(m["gen_score"]),
                   pen_loop=float(m.get("gen_pen_loop", float("nan"))),
                   pen_density=float(m.get("gen_pen_density", float("nan"))),
                   oa_weighted=float(m.get("gen_oa_weighted", float("nan"))),
                   repeat8=float(m.get("gen_repeat8", float("nan"))),
                   density=float(m.get("gen_density", float("nan"))),
                   grammar_prob_mass=float(res["grammar_prob_mass"]),
                   seconds=time.time() - t0)
        rows.append(row)
        print("  %5.2f %5.2f | %8.2f %8.3f %8.3f %8.4f %8.1e" %
              (temp, top_p, row["gen_score"], row["pen_loop"], row["repeat8"],
               row["density"], row["grammar_prob_mass"]))

    best = max(rows, key=lambda r: r["gen_score"])
    base = [r for r in rows if r["temperature"] == 1.0 and r["top_p"] == 0.95]
    out = dict(exp=name, step=int(ck.get("step", 0)), n_samples=n, gen_steps=steps,
               grid=rows, best=best,
               baseline=base[0] if base else None,
               ref_repeat8=float(ref["repeat8"]) if ref else None)
    print("  mejor: temp=%.2f top_p=%.2f -> gen_score %.2f%s" %
          (best["temperature"], best["top_p"], best["gen_score"],
           ("  (frente a %.2f en el punto comun temp=1.0/top_p=0.95)" % base[0]["gen_score"])
           if base else ""))
    json.dump(out, open(exp / "logs" / "sampling_sweep.json", "w"), indent=2)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = [a.exp] if a.exp else sorted(
        p.name for p in (ROOT / "experiments").iterdir()
        if p.is_dir() and not p.name.startswith("smoke_")
        and (p / "checkpoints" / a.ckpt).exists())
    outs = [sweep(n, a.ckpt, a.quick, a.n, device=device) for n in names]
    outs = [o for o in outs if o]
    if len(outs) > 1:
        print("\n%-14s %-22s %-22s" % ("experimento", "punto comun (1.0/0.95)", "su mejor punto"))
        for o in outs:
            b, s = o["best"], o["baseline"]
            print("%-14s %-22s temp=%.2f top_p=%.2f -> %.2f" %
                  (o["exp"], ("%.2f" % s["gen_score"]) if s else "n/d",
                   b["temperature"], b["top_p"], b["gen_score"]))


if __name__ == "__main__":
    main()
