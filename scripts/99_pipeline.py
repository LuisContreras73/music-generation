"""Reproduce el laboratorio completo, de train.npz al informe final.

    python scripts/99_pipeline.py --smoke      # valida todo en pocos minutos
    python scripts/99_pipeline.py              # el laboratorio entero (~3 h de GPU)
    python scripts/99_pipeline.py --from 4     # retoma desde la etapa 4

Etapas:
    0  analisis exploratorio          -> data/processed/data_stats.json
    1  tokenizacion y particiones     -> tokens.bin, rolls_packed.bin, splits.json
    2  referencias y calibracion      -> ref_stats.npz, calibration.json
    3  figuras del dataset            -> reports/figures/
    4  entrenamiento de la matriz     -> experiments/*, reports/best/
    5  re-evaluacion homogenea        -> summary.json coherentes entre experimentos
    6  interpretabilidad del TFT      -> experiments/tft/figures/
    7  exportacion a MIDI             -> reports/best/midi/
    8  informe                        -> docs/03_resultados.md

Cada etapa es idempotente: volver a ejecutarla no rompe nada.
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STAGES = [
    ("analisis exploratorio", ["scripts/00_explore.py"], False),
    ("tokenizacion y particiones", ["scripts/01_prepare.py"], False),
    ("referencias y calibracion", ["scripts/02_ref_stats.py"], False),
    ("figuras del dataset", ["scripts/03_data_figures.py"], False),
    ("entrenamiento de la matriz", ["scripts/run_experiments.py"], True),
    ("re-evaluacion homogenea", ["scripts/07_reevaluate.py"], False),
    ("interpretabilidad del TFT", ["scripts/06_tft_interpret.py"], False),
    ("exportacion a MIDI", ["scripts/04_export_best.py"], False),
    ("informe", ["scripts/05_report.py"], False),
]


def run(cmd, tag):
    print("\n" + "=" * 78)
    print("[%s] %s" % (tag, " ".join(cmd)))
    print("=" * 78, flush=True)
    t0 = time.time()
    r = subprocess.run([PY] + cmd, cwd=ROOT)
    dt = (time.time() - t0) / 60
    print("[%s] %s en %.1f min" % (tag, "OK" if r.returncode == 0 else
                                   "FALLO (codigo %d)" % r.returncode, dt), flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="entrenamientos cortos (150 pasos) para validar el flujo")
    ap.add_argument("--from", dest="start", type=int, default=0, help="primera etapa")
    ap.add_argument("--to", dest="end", type=int, default=len(STAGES) - 1)
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    results = []
    for i, (name, cmd, is_train) in enumerate(STAGES):
        if i < a.start or i > a.end:
            continue
        if is_train and a.skip_train:
            print("etapa %d (%s) omitida" % (i, name)); continue
        c = list(cmd)
        if is_train and a.smoke:
            c.append("--smoke")
        ok = run(c, "etapa %d/%d %s" % (i, len(STAGES) - 1, name))
        results.append((i, name, ok))
        if not ok and i <= 2:                    # sin datos preparados no sigue nada
            print("\nla etapa %d es imprescindible; se detiene el pipeline" % i)
            break

    print("\n" + "=" * 78)
    print("PIPELINE: %.1f min" % ((time.time() - t0) / 60))
    for i, name, ok in results:
        print("  %d  %-30s %s" % (i, name, "OK" if ok else "FALLO"))
    lb = ROOT / "reports" / "leaderboard.csv"
    if lb.exists():
        print("\nleaderboard:")
        print(lb.read_text().strip()[:2000])


if __name__ == "__main__":
    main()
