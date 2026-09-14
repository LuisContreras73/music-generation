"""Generacion libre comparable: varios modelos componen desde cero, mismas semillas.

    python scripts/20_generacion_libre.py                       # los 6 modelos comparados
    python scripts/20_generacion_libre.py --exps music_transformer lstm
    python scripts/20_generacion_libre.py --muestreo calibrado

Escenario: **sin prefijo**. La entrada al modelo es un unico token BOS, asi que no
hay nada que lo ancle; es el caso mas duro y el que mas rapido delata a un modelo
que solo sabe continuar lo que ya sonaba.

Por que un script y no `infer.py --scratch` repetido: infer.py escribe siempre el
mismo nombre de fichero (se sobreescribe entre semillas), no guarda el roll para
poder medirlo despues, y no fuerza que todos los modelos compartan semilla y
decodificacion, que es lo unico que hace comparables dos generaciones.

Decodificacion por defecto: la del entrenamiento (T 1.0, top-p 0.95). Medido en
la comparacion pareada, la calibrada de best_sampling.json SOBRECORRIGE fuera del
escenario para el que se ajusto (densidad 0.34 frente a 0.45 real). Con
--muestreo calibrado se obtiene la otra.

Salida en reports/audio/3_generacion_libre/<modelo>/ y una tabla en
reports/generacion_libre/resumen.csv con las metricas de cada generacion.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import generate as gen                               # noqa: E402
import evaluate as ev                                # noqa: E402
import metrics as mx                                 # noqa: E402
import registry as reg                               # noqa: E402
import audio_play                                    # noqa: E402
from config import Config                            # noqa: E402
from models import build_model                       # noqa: E402
from data.tokenizer import BOS                       # noqa: E402
from midi_export import roll_to_midi                 # noqa: E402

AUDIO = ROOT / "reports" / "audio" / "3_generacion_libre"
TABLA = ROOT / "reports" / "generacion_libre"
POR_DEFECTO = ["music_transformer", "lstm", "estilo_llama_ctx2048_24ep",
               "estilo_llama_24ep", "perceiver_ar", "deep_lstm"]


def cargar(exp: str, ckpt: str, device: str):
    p = ROOT / "experiments" / exp / "checkpoints" / ckpt
    if not p.exists():
        return None, None
    ck = reg.load_checkpoint(p, map_location=device)
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
    if cfg.family != "token":
        print("  %s: familia '%s', no aplica la generacion por tokens" % (exp, cfg.family))
        return None, None
    m = build_model(cfg).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*", default=None)
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--seg", type=float, default=40.0)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--audio", type=int, default=3,
                    help="cuantas semillas se sintetizan a wav/mid (las demas solo cuentan "
                         "para la metrica). El gen_score esta calibrado con N=16, asi que con "
                         "menos muestras queda sesgado; pero 16 wav por modelo no se escuchan.")
    ap.add_argument("--muestreo", default="entrenamiento", choices=["entrenamiento", "calibrado"])
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exps = a.exps or POR_DEFECTO
    n_steps = int(round(a.seg / 0.05))
    TABLA.mkdir(parents=True, exist_ok=True)

    estrategia = None
    if a.muestreo == "calibrado":
        estrategia = json.load(open(ROOT / "data" / "processed" / "best_sampling.json",
                                    encoding="utf-8"))
    print("generacion libre (sin prefijo, solo BOS) | %d modelos x %d semillas | %.0f s | %s"
          % (len(exps), a.semillas, a.seg, a.muestreo))

    ref = ev.load_ref_agg("val_excerpt")
    filas = []
    for exp in exps:
        model, cfg = cargar(exp, a.ckpt, device)
        if model is None:
            continue
        d = AUDIO / exp
        d.mkdir(parents=True, exist_ok=True)
        for f in list(d.glob("desde_cero*")):        # idempotente: no mezcla tandas
            f.unlink()
        rolls = []
        for semilla in range(1, a.semillas + 1):
            t0 = time.time()
            tk = np.array([BOS], np.int64)
            g = torch.Generator(device=device); g.manual_seed(semilla)
            if estrategia is None:
                x = torch.from_numpy(tk)[None, :].to(device)
                res = gen.sample_tokens(model, x, n_steps, seq_len=cfg.seq_len,
                                        temperature=1.0, top_k=0, top_p=0.95,
                                        device=device, generator=g)
                nuevo = res["new"][0]
            else:
                import sampling as S
                res = S.sample_with_strategy(model, tk, n_steps, strategy=estrategia,
                                             seq_len=cfg.seq_len, device=device, generator=g)
                nuevo = np.asarray(res["new"][0] if isinstance(res["new"], (list, tuple))
                                   else res["new"]).ravel()
            r = gen.tokens_to_roll(np.asarray(nuevo).ravel())
            if len(r) < n_steps:
                r = np.pad(r, ((0, n_steps - len(r)), (0, 0)))
            r = np.asarray(r[:n_steps], np.uint8)
            rolls.append(r)

            base = "desde_cero_s%d" % semilla
            if semilla <= a.audio:
                audio, sr = audio_play.synthesize_musicbox_roll(
                    r, step_sec=0.05, note_min=21, representation="onset")
                audio_play.save_wav(d / (base + ".wav"), audio, sr)
                roll_to_midi(r, d / (base + ".mid"))
                np.savez_compressed(d / (base + ".npz"), roll=r, exp=exp, semilla=semilla,
                                    muestreo=a.muestreo)
            f = mx.roll_features(r)
            filas.append(dict(modelo=exp, semilla=semilla, muestreo=a.muestreo,
                              densidad=round(float(f["density"]), 4),
                              repeat8=round(float(f["repeat8"]), 4),
                              polifonia=round(float(f["mean_poly"]), 3),
                              alturas_distintas=int(f["n_distinct_pitch"]),
                              notas=int(r.sum()), seg=round(time.time() - t0, 1)))
            print("  %-28s semilla %d: %3d notas | densidad %.3f | repeat8 %.3f | %.0f s"
                  % (exp, semilla, int(r.sum()), f["density"], f["repeat8"], time.time() - t0))
        # gen_score del conjunto de las semillas de ese modelo
        m, _, _ = ev.evaluate_generations(rolls)
        for fila in filas[-len(rolls):]:
            fila["gen_score_del_lote"] = round(float(m["gen_score"]), 2)
        print("  %-28s gen_score del lote: %.1f\n" % (exp, m["gen_score"]))
        del model
        torch.cuda.empty_cache()

    if not filas:
        print("nada generado"); return
    sufijo = "" if a.muestreo == "entrenamiento" else "_calibrado"
    out = TABLA / ("resumen%s.csv" % sufijo)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)

    print("referencia del corpus: densidad %.3f | repeat8 %.3f | polifonia %.2f"
          % (ref["density"], ref["repeat8"], ref["mean_poly"]))
    print("\n%-28s %9s %9s %9s" % ("modelo", "gen_score", "densidad", "repeat8"))
    for e in exps:
        f = [x for x in filas if x["modelo"] == e]
        if f:
            print("%-28s %9.1f %9.3f %9.3f"
                  % (e, f[0]["gen_score_del_lote"],
                     np.mean([x["densidad"] for x in f]), np.mean([x["repeat8"] for x in f])))
    print("\ntabla: %s" % out.relative_to(ROOT))
    print("audio: reports/audio/3_generacion_libre/<modelo>/desde_cero_sN.wav")


if __name__ == "__main__":
    main()
