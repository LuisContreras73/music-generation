"""Inferencia: genera musica con un modelo entrenado.

EJEMPLOS
--------
  # continuar un prefijo de tu .npz (lo mas habitual)
  python scripts/infer.py --npz external_eval_prefix_5s.npz --seg 30

  # continuar una pieza del corpus, 60 segundos
  python scripts/infer.py --corpus 7 --seg 60

  # generar desde cero, sin prefijo
  python scripts/infer.py --scratch --seg 30

  # elegir modelo y carpeta de salida
  python scripts/infer.py --exp estilo_llama_10ep --npz mis_prefijos.npz --out mi_carpeta

  # comparar decodificaciones (con y sin anti-bucle)
  python scripts/infer.py --corpus 7 --sampling baseline
  python scripts/infer.py --corpus 7 --sampling best

QUE PRODUCE, en reports/audio/<escenario>/<modelo>/ por defecto:
  <nombre>.wav        audio (caja de musica)
  <nombre>.mid        MIDI, suena mejor con un piano real
  <nombre>.png        piano-roll, con el prefijo sombreado si lo hay
  <nombre>.npz        piano-roll [T,88] en el formato del dataset

LO QUE HAY QUE SABER
--------------------
Un modelo entrenado NO basta: la DECODIFICACION importa tanto como los pesos.
Con el muestreo por defecto (temperatura 1.0, nucleus 0.95) estos modelos
puntuan 0.01 sobre melodias monofonicas porque entran en bucle; con la
configuracion calibrada en data/processed/best_sampling.json, 17.40 sobre un
techo de 21.56. Es el mismo modelo. Por eso --sampling best es el defecto.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import Config                              # noqa: E402
from models import build_model                         # noqa: E402
import registry as reg                                 # noqa: E402
import generate as gen                                 # noqa: E402
import metrics as mx                                   # noqa: E402
from data.tokenizer import encode_roll_fast, BOS       # noqa: E402
from midi_export import roll_to_midi                   # noqa: E402


def elegir_modelo(nombre=None) -> str:
    """Sin --exp, coge el de menor bits/paso del leaderboard."""
    if nombre:
        return nombre
    lb = ROOT / "reports" / "leaderboard.csv"
    if lb.exists():
        import pandas as pd
        df = pd.read_csv(lb)
        df = df[df["best_val_bpt"].notna()]
        if len(df):
            return str(df.sort_values("best_val_bpt").iloc[0]["name"])
    return "estilo_llama_24ep"


def cargar(exp: str, ckpt: str, device: str):
    p = ROOT / "experiments" / exp / "checkpoints" / ckpt
    if not p.exists():
        disponibles = sorted(d.name for d in (ROOT / "experiments").iterdir()
                             if (d / "checkpoints" / ckpt).exists())
        raise SystemExit("no existe %s\nmodelos disponibles: %s" % (p, ", ".join(disponibles)))
    ck = reg.load_checkpoint(p, map_location=device)
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
    m = build_model(cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, cfg, int(ck.get("step", 0))


def prefijos(a) -> list:
    """Devuelve [(nombre, roll_prefijo o None)]."""
    if a.scratch:
        return [("desde_cero", None)]
    if a.npz:
        z = np.load(ROOT / a.npz if not Path(a.npz).is_absolute() else a.npz, allow_pickle=True)
        off = z["offsets"].astype(np.int64)
        ids = [str(s) for s in z["ids"]]
        R = z["rolls_flat"]
        return [(ids[i], np.asarray(R[off[i]:off[i + 1]], np.uint8)) for i in range(len(ids))]
    from data.datasets import get_roll
    r = get_roll(int(a.corpus))
    n = int(a.prime_steps)
    return [("corpus_%d" % a.corpus, r[:n])]


def main():
    ap = argparse.ArgumentParser(description="genera musica con un modelo entrenado")
    ap.add_argument("--exp", default=None, help="modelo; por defecto el mejor del leaderboard")
    ap.add_argument("--ckpt", default="best.pt", choices=["best.pt", "best_gen.pt", "last.pt"])
    src = ap.add_argument_group("de donde sale el prefijo (elige uno)")
    src.add_argument("--npz", default=None, help="fichero de prefijos, formato del dataset")
    src.add_argument("--corpus", type=int, default=None, help="indice de pieza del corpus")
    src.add_argument("--scratch", action="store_true", help="sin prefijo, desde cero")
    ap.add_argument("--prime-steps", type=int, default=100, help="pasos de prefijo si --corpus")
    ap.add_argument("--seg", type=float, default=30.0, help="segundos a generar")
    ap.add_argument("--sampling", default="best", choices=["best", "baseline", "greedy"],
                    help="best = calibrada anti-bucle, ajustada sobre melodias monofonicas "
                         "fuera de distribucion; baseline = T 1.0 / top-p 0.95, la del "
                         "entrenamiento. MEDIDO: sobre prefijos del CORPUS la calibrada "
                         "sobrecorrige (densidad 0.34 frente a 0.45 real) y baseline gana en "
                         "5 de 6 modelos; sobre las melodias de prueba gana la calibrada. "
                         "Ver docs/03_resultados.md, seccion 7")
    ap.add_argument("--temp", type=float, default=None, help="sobrescribe la temperatura")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true", help="lista los modelos entrenados y sale")
    a = ap.parse_args()

    if a.list:
        lb = ROOT / "reports" / "leaderboard.csv"
        anchura = 18
        if lb.exists():
            import pandas as pd
            df = pd.read_csv(lb)
            anchura = max(18, int(df["name"].astype(str).str.len().max()))
        print(("%-" + str(anchura) + "s %10s %10s") % ("modelo", "bits/paso", "gen_score"))
        fila = "%-" + str(anchura) + "s %10s %10s"
        if lb.exists():
            for _, r in df.sort_values("best_val_bpt").iterrows():
                print(fila % (r["name"],
                              ("%.4f" % r["best_val_bpt"]) if r["best_val_bpt"] == r["best_val_bpt"] else "-",
                              ("%.2f" % r["best_gen_score"]) if r["best_gen_score"] == r["best_gen_score"] else "-"))
        return

    if not (a.npz or a.corpus is not None or a.scratch):
        a.corpus = 0
        print("(sin prefijo indicado: se usa --corpus 0; mira --help para las opciones)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp = elegir_modelo(a.exp)
    model, cfg, paso = cargar(exp, a.ckpt, device)
    n_steps = int(round(a.seg / 0.05))

    # --- decodificacion ---
    if a.sampling == "best":
        f = ROOT / "data" / "processed" / "best_sampling.json"
        strat = json.load(open(f)) if f.exists() else dict(temperature=1.0, top_p=0.95)
    elif a.sampling == "greedy":
        strat = dict(greedy=True)
    else:
        strat = dict(temperature=1.0, top_p=0.95)
    if a.temp is not None:
        strat["temperature"] = a.temp

    # Sin --out, la salida va al arbol de reports/audio/ ordenado POR ESCENARIO
    # (de donde sale el prefijo), que es lo que hace comparables dos resultados.
    if a.out:
        out_dir = Path(a.out)
    elif a.scratch:
        out_dir = ROOT / "reports" / "audio" / "3_generacion_libre" / exp
    elif a.npz:
        conj = ("melodias_5s" if "5s" in Path(a.npz).stem else
                "muestras_densas" if "02" in Path(a.npz).stem else Path(a.npz).stem)
        out_dir = ROOT / "reports" / "audio" / "2_prefijos_de_prueba" / conj / exp
    else:
        out_dir = ROOT / "reports" / "audio" / "1_continuaciones_corpus" / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        destino = out_dir.relative_to(ROOT)
    except ValueError:                      # --out fuera del repo
        destino = out_dir
    print("modelo %s (paso %d, %s) | %s | %.0f s por muestra -> %s"
          % (exp, paso, device, a.sampling, a.seg, destino))
    print("  decodificacion: %s\n" % {k: v for k, v in strat.items() if k != "generator"})

    import sampling as S
    import audio_play
    for nombre, pref in prefijos(a):
        t0 = time.time()
        tk = (np.array([BOS], np.int64) if pref is None
              else encode_roll_fast(pref, add_bos=True, add_eos=False).astype(np.int64))
        r = S.sample_with_strategy(model, tk, n_steps, strategy=strat,
                                   seq_len=cfg.seq_len, device=device, seed=a.seed)
        new = r["new"] if isinstance(r, dict) else r
        arr = np.asarray(new[0] if isinstance(new, (list, tuple)) else new).ravel()
        cont = gen.tokens_to_roll(arr)
        if len(cont) < n_steps:
            cont = np.pad(cont, ((0, n_steps - len(cont)), (0, 0)))
        cont = cont[:n_steps]
        full = cont if pref is None else np.concatenate([pref, cont])

        audio, sr = audio_play.synthesize_musicbox_roll(full, step_sec=0.05, note_min=21,
                                                        representation="onset")
        audio_play.save_wav(out_dir / ("%s.wav" % nombre), audio, sr)
        roll_to_midi(full, out_dir / ("%s.mid" % nombre))
        np.savez_compressed(out_dir / ("%s.npz" % nombre),
                            rolls_flat=full.astype(np.uint8),
                            offsets=np.array([0, len(full)], np.int64),
                            ids=np.array([nombre]), step_sec=np.array([0.05], np.float32),
                            note_min=np.array([21], np.int16), note_max=np.array([108], np.int16),
                            num_positions=np.array([88], np.int16),
                            representation=np.array(["onset"]))
        try:
            import viz
            viz.plot_pianoroll(full, out_dir / ("%s.png" % nombre), title="%s | %s" % (nombre, exp),
                               prime_steps=(len(pref) if pref is not None else None))
        except Exception:
            pass
        f = mx.roll_features(cont)
        print("  %-26s %4d notas | densidad %.3f | polifonia %.2f | %4.1f s" %
              (nombre, f["n_onsets"], f["density"], f["mean_poly"], time.time() - t0))
    print("\nlisto: %s" % out_dir)


if __name__ == "__main__":
    main()
