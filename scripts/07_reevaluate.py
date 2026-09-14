"""Re-evalua TODOS los experimentos con el codigo de evaluacion actual.

    python scripts/07_reevaluate.py            # todos
    python scripts/07_reevaluate.py --exp music_transformer

Por que existe: durante el laboratorio la evaluacion se refino (score calibrado,
barrido de umbral en la familia frame). Un experimento entrenado antes de un
refinamiento lleva numeros de la version anterior. Este script recarga cada
best.pt, recalcula val/test y genera de nuevo, y reescribe logs/summary.json.
Asi TODAS las cifras del informe salen del mismo codigo y son comparables.

No re-entrena nada: solo re-mide.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import Config                                  # noqa: E402
from data.datasets import TokenWindows, FrameWindows       # noqa: E402
from models import build_model                             # noqa: E402
import evaluate as ev                                      # noqa: E402
import registry as reg                                     # noqa: E402
import train as T                                          # noqa: E402


def reevaluate(name: str, ckpt: str = "best.pt", device: str = "cuda",
               repeats: int = 3) -> dict:
    exp = ROOT / "experiments" / name
    ck_path = exp / "checkpoints" / ckpt
    if not ck_path.exists():
        print("  %s: no existe %s" % (name, ckpt))
        return {}
    ck = reg.load_checkpoint(ck_path, map_location=device)
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    DS = TokenWindows if cfg.family == "token" else FrameWindows
    ef = ev.eval_token_model if cfg.family == "token" else ev.eval_frame_model
    out = {"n_params": model.n_params(), "reeval_step": int(ck.get("step", 0))}
    for split in ("val", "test"):
        ds = DS(split, cfg.seq_len, cfg.val_windows)
        # sin workers: con spawn en Windows mueren al cerrar y tumban la evaluacion
        dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0, pin_memory=True)
        m = ef(model, dl, device=device, amp=cfg.amp)
        pref = "" if split == "val" else "test_"
        for k, v in m.items():
            out[pref + k.replace("val_", "" if split == "test" else "val_")] = v

    # --- generacion repetida con semillas distintas -------------------------
    # El gen_score de una sola tanda de 16 muestras tiene una varianza enorme:
    # en music_transformer se observaron 58.98, 28.94 y 69.70 en evaluaciones sucesivas
    # del mismo entrenamiento. Decidir el ganador con una sola medicion no es
    # fiable, asi que se repite con semillas distintas y se promedia. Cada
    # repeticion mantiene N=16 y 800 pasos, que es como esta calibrado el score.
    dirs = cfg.subdirs()
    scores, gms = [], []
    for rep in range(repeats):
        gm = T.run_generation(model, cfg, device, int(ck.get("step", 0)) + rep, dirs,
                              save_figs=(rep == 0))
        gms.append(gm); scores.append(float(gm.get("gen_score", float("nan"))))
    gm = gms[0]
    out.update({"final_" + k: v for k, v in gm.items()})
    if repeats > 1:
        import statistics
        out["final_gen_score"] = float(statistics.mean(scores))
        out["gen_score_std"] = float(statistics.pstdev(scores))
        out["gen_score_reps"] = scores
        print("    gen_score en %d repeticiones: %s -> media %.2f (sd %.2f)"
              % (repeats, " ".join("%.1f" % s for s in scores),
                 out["final_gen_score"], out["gen_score_std"]))

    # El tiempo y los tokens salen del metrics.csv, no del summary previo: si un
    # experimento se reanudo tras un fallo, los contadores de la ultima ejecucion
    # valen ~0 mientras que el CSV conserva el historial completo.
    import csv as _csv
    th, tk = 0.0, 0
    csv_path = exp / "logs" / "metrics.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = list(_csv.DictReader(f))
        th = max((float(r["time_s"]) for r in rows if r.get("time_s")), default=0.0) / 3600
        tk = int(max((float(r["tokens_seen"]) for r in rows if r.get("tokens_seen")), default=0))
    out["train_hours"] = th
    out["tokens_seen"] = tk
    reg.write_summary(cfg, out)
    print("  %-12s bpt val %.4f  test %.4f  gen_score %.2f%s"
          % (name, out.get("val_bpt", float("nan")), out.get("test_bpt", float("nan")),
             gm.get("gen_score", float("nan")),
             ("  F1@%.2f %.3f" % (out.get("val_frame_best_threshold", 0),
                                 out.get("val_frame_f1_best", 0)))
             if cfg.family == "frame" else ""))
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--repeats", type=int, default=3,
                    help="tandas de generacion con semillas distintas a promediar")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = [a.exp] if a.exp else sorted(
        p.name for p in (ROOT / "experiments").iterdir()
        if p.is_dir() and not p.name.startswith("smoke_") and (p / "checkpoints" / a.ckpt).exists())
    if not names:
        print("no hay experimentos con %s" % a.ckpt)
        return
    print("re-evaluando %d experimentos con el codigo actual:" % len(names))
    t0 = time.time()
    for n in names:
        reevaluate(n, a.ckpt, device, a.repeats)
    reg.refresh_all()
    print("\nlisto en %.1f min" % ((time.time() - t0) / 60))
    import pandas as pd
    lb = ROOT / "reports" / "leaderboard.csv"
    if lb.exists():
        print(pd.read_csv(lb).to_string(index=False))


if __name__ == "__main__":
    main()
