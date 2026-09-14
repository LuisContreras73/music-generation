"""Exporta a MIDI las mejores generaciones y las deja en reports/best/midi/.

    python scripts/04_export_best.py                # el mejor experimento del leaderboard
    python scripts/04_export_best.py --exp music_transformer  # uno concreto
    python scripts/04_export_best.py --all          # todos los experimentos

Se exportan tambien fragmentos REALES del corpus como referencia auditiva, para
poder comparar oyendo, no solo mirando metricas.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from midi_export import roll_to_midi                      # noqa: E402
from data.datasets import get_roll                        # noqa: E402


def latest_gen(exp_dir: Path):
    """El .npz de generacion mas reciente de un experimento."""
    gens = sorted((exp_dir / "generations").glob("gen_step*.npz"))
    return gens[-1] if gens else None


def best_gen_npz(exp_dir: Path):
    """El .npz del paso con mejor gen_score, segun logs/best.json o el ultimo."""
    bj = exp_dir / "logs" / "best.json"
    if bj.exists():
        try:
            d = json.load(open(bj))
            step = d.get("best_gen_score_step") or d.get("best_gen_step")
            if step:
                p = exp_dir / "generations" / ("gen_step%07d.npz" % int(step))
                if p.exists():
                    return p
        except Exception:
            pass
    return latest_gen(exp_dir)


def export_exp(exp_dir: Path, out_dir: Path, limit: int = 6) -> list:
    npz = best_gen_npz(exp_dir)
    if npz is None:
        print("  %s: sin generaciones" % exp_dir.name)
        return []
    z = np.load(npz)
    rolls = z["rolls"]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, r in enumerate(rolls[:limit]):
        n_on = int(np.asarray(r).sum())
        p = roll_to_midi(r, out_dir / ("%s_%s_m%02d.mid" % (exp_dir.name, npz.stem[-7:], i + 1)))
        written.append(p)
        print("  %-52s %6d bytes  %4d onsets  %5.1f s"
              % (p.name, p.stat().st_size, n_on, len(r) * 0.05))
    return written


def export_reference(out_dir: Path, n: int = 3, steps: int = 800) -> list:
    """Fragmentos reales del corpus, como referencia para comparar de oido."""
    splits = json.load(open(ROOT / "data" / "processed" / "splits.json"))
    rng = np.random.default_rng(5)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for k, i in enumerate(rng.choice(splits["test"], n * 3, replace=False)):
        r = get_roll(int(i))
        if len(r) < steps + 400:
            continue
        s = int(rng.integers(200, len(r) - steps))
        p = roll_to_midi(r[s:s + steps], out_dir / ("CORPUS_real_%02d.mid" % (len(written) + 1)))
        written.append(p)
        print("  %-52s %6d bytes  (referencia real)" % (p.name, p.stat().st_size))
        if len(written) >= n:
            break
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None, help="nombre de un experimento concreto")
    ap.add_argument("--all", action="store_true", help="exportar todos los experimentos")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--no-reference", action="store_true")
    a = ap.parse_args()

    exp_root = ROOT / "experiments"
    best_dir = ROOT / "reports" / "best" / "midi"

    # Solo el ganador elegido por el leaderboard se copia a reports/best/, para
    # que ese directorio nunca contenga artefactos de un experimento cualquiera.
    is_winner = False
    if a.all:
        targets = sorted(p for p in exp_root.iterdir()
                         if p.is_dir() and not p.name.startswith("smoke_"))
    elif a.exp:
        targets = [exp_root / a.exp]
    else:
        is_winner = True
        lb = ROOT / "reports" / "leaderboard.csv"
        if not lb.exists():
            print("no hay reports/leaderboard.csv; usa --exp o --all")
            return
        import pandas as pd
        df = pd.read_csv(lb)
        df = df[~df["name"].astype(str).str.startswith("smoke_")]
        if df.empty:
            print("el leaderboard no tiene experimentos reales todavia")
            return
        targets = [exp_root / str(df.iloc[0]["name"])]
        print("mejor experimento del leaderboard: %s" % targets[0].name)

    total = []
    for t in targets:
        if not t.is_dir():
            print("no existe %s" % t)
            continue
        print("%s:" % t.name)
        out = t / "generations" / "midi"
        total += export_exp(t, out, a.limit)
        if is_winner:                             # solo el ganador va a reports/best
            total += export_exp(t, best_dir, a.limit)

    if not a.no_reference:
        print("referencia del corpus:")
        total += export_reference(best_dir if is_winner else ROOT / "reports" / "figures")
    print("\n%d archivos MIDI escritos" % len(total))


if __name__ == "__main__":
    main()
