"""Galeria de audio: convierte a WAV y MIDI las generaciones ya guardadas.

    python scripts/11_audio_gallery.py              # todos los experimentos
    python scripts/11_audio_gallery.py --n 4        # 4 muestras por modelo

No usa GPU ni vuelve a generar nada: lee los .npz que cada experimento dejo en
generations/ y los sintetiza con audio_play.py (timbre de caja de musica).

Salida en reports/audio/, con las carpetas numeradas de mejor a peor gen_score
para que el orden de escucha sea evidente, y una carpeta 00_CORPUS_real con
fragmentos reales del dataset como referencia.
"""
from __future__ import annotations
import argparse, json, sys, shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import audio_play                              # noqa: E402
from midi_export import roll_to_midi           # noqa: E402
import metrics as mx                           # noqa: E402
from data.datasets import get_roll             # noqa: E402

STEP_SEC = 0.05
NOTE_MIN = 21


def synth(roll, out_wav: Path, note_dur=None):
    audio, sr = audio_play.synthesize_musicbox_roll(
        np.asarray(roll), step_sec=STEP_SEC, note_min=NOTE_MIN,
        representation="onset", note_duration_sec=note_dur)
    audio_play.save_wav(out_wav, audio, sr)
    return out_wav


def best_gen_npz(exp_dir: Path):
    """El .npz del paso con mejor gen_score segun logs/best.json, o el ultimo."""
    bj = exp_dir / "logs" / "best.json"
    if bj.exists():
        try:
            d = json.load(open(bj))
            for key in ("best_gen_score_step", "best_gen_step"):
                if d.get(key):
                    p = exp_dir / "generations" / ("gen_step%07d.npz" % int(d[key]))
                    if p.exists():
                        return p
        except Exception:
            pass
    gens = sorted((exp_dir / "generations").glob("gen_step*.npz"))
    return gens[-1] if gens else None


def exp_score(name: str, lb) -> float:
    if lb is None:
        return -1.0
    row = lb[lb["name"] == name]
    if row.empty:
        return -1.0
    try:
        return float(row.iloc[0]["best_gen_score"])
    except Exception:
        return -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="muestras por modelo")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_root = Path(a.out) if a.out else ROOT / "reports" / "audio" / "1_continuaciones_corpus"
    if out_root.exists():
        shutil.rmtree(out_root)                 # idempotente: nunca mezcla tandas
    out_root.mkdir(parents=True)

    lb = None
    lb_path = ROOT / "reports" / "leaderboard.csv"
    if lb_path.exists():
        import pandas as pd
        lb = pd.read_csv(lb_path)

    exps = sorted((p for p in (ROOT / "experiments").iterdir()
                   if p.is_dir() and not p.name.startswith("smoke_")),
                  key=lambda p: -exp_score(p.name, lb))

    lines = ["# Galeria de audio\n",
             "Generado por `scripts/11_audio_gallery.py` desde las continuaciones que cada",
             "experimento guardo durante el entrenamiento. Timbre de caja de musica",
             "(`audio_play.py`). Cada `.wav` va acompanado del `.mid` equivalente.\n",
             "Las carpetas estan numeradas **de mejor a peor** `gen_score`.\n",
             "| carpeta | modelo | gen_score | que se oye |", "|---|---|---|---|"]

    # --- referencia real del corpus ---
    ref_dir = out_root.parent / "0_referencia_corpus"
    ref_dir.mkdir(exist_ok=True)   # vive fuera de out_root, que si se borra antes
    rng = np.random.default_rng(5)
    splits = json.load(open(ROOT / "data" / "processed" / "splits.json"))
    n_ref = 0
    for i in rng.choice(splits["test"], 12, replace=False):
        r = get_roll(int(i))
        if len(r) < 1200:
            continue
        s = int(rng.integers(200, len(r) - 800))
        seg = r[s:s + 800]
        synth(seg, ref_dir / ("corpus_real_%02d.wav" % (n_ref + 1)))
        roll_to_midi(seg, ref_dir / ("corpus_real_%02d.mid" % (n_ref + 1)))
        n_ref += 1
        if n_ref >= 3:
            break
    print("00_CORPUS_real: %d fragmentos reales de referencia" % n_ref)
    lines.append("| `00_CORPUS_real` | *(dataset)* | 87.6 (techo) | musica real, la referencia |")

    # --- una carpeta por experimento ---
    for k, exp in enumerate(exps, start=1):
        npz = best_gen_npz(exp)
        sc = exp_score(exp.name, lb)
        if npz is None:
            print("%s: sin generaciones, se omite" % exp.name)
            continue
        z = np.load(npz)
        rolls = z["rolls"]
        tag = exp.name
        d = out_root / tag
        d.mkdir()
        dens = []
        for i, r in enumerate(rolls[: a.n]):
            f = mx.roll_features(np.asarray(r))
            dens.append(f["density"])
            base = "%s_m%02d" % (exp.name, i + 1)
            synth(r, d / (base + ".wav"))
            roll_to_midi(r, d / (base + ".mid"))
        cfg = json.load(open(exp / "config.json")) if (exp / "config.json").exists() else {}
        note = {"melle": "familia frames: genera ~9x mas notas de las reales, se oye saturado",
                "lstm": "linea base recurrente",
                "tft": "TFT adaptado",
                "music_transformer": ("sin atencion relativa (ablacion)"
                                      if not cfg.get("rel_attn", True)
                                      else "Music Transformer con atencion relativa")}.get(
                    cfg.get("model", ""), "")
        print("%-28s %d muestras | gen_score %5.1f | densidad media %.3f  (%s)"
              % (tag, min(a.n, len(rolls)), sc, float(np.mean(dens)) if dens else 0, npz.name))
        lines.append("| `%s` | %s | %.1f | %s |" % (tag, cfg.get("model", exp.name), sc, note))

    lines += ["", "## Como escuchar", "",
              "Los `.wav` se abren con cualquier reproductor. Los `.mid` suenan mejor con un",
              "sintetizador de piano real (por ejemplo VLC con un SoundFont, o importandolos",
              "en un DAW).", "",
              "### Que escuchar", "",
              "1. Empieza por `00_CORPUS_real`: es el objetivo.",
              "2. Compara con la carpeta `01_...`, que es el modelo con mejor `gen_score`.",
              "3. Baja por la lista: las ultimas carpetas suenan saturadas o repetitivas, y",
              "   eso es exactamente lo que miden las metricas.", "",
              "Nota sobre la duracion: el dataset guarda **ataques sin duracion**, asi que la",
              "duracion de cada nota es una reconstruccion (hasta el siguiente ataque de la",
              "misma nota, tope 1.5 s en MIDI; timbre percusivo con decaimiento en WAV).",
              ]
    (out_root / "LEEME.md").write_text("\n".join(lines), encoding="utf-8")
    n_wav = len(list(out_root.rglob("*.wav")))
    print("\n%d archivos WAV y %d MIDI en %s" %
          (n_wav, len(list(out_root.rglob("*.mid"))), out_root))
    print("indice: %s" % (out_root / "LEEME.md"))


if __name__ == "__main__":
    main()
