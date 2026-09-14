"""Comparacion PAREADA: los mismos prefijos para todos los modelos, varias semillas.

    python scripts/18_comparacion_pareada.py                     # los 6 mejores de eventos
    python scripts/18_comparacion_pareada.py --exps lstm estilo_llama_ctx2048_24ep
    python scripts/18_comparacion_pareada.py --n 16 --semillas 3

Por que hace falta: el `gen_score` del leaderboard es el MAXIMO sobre las
evaluaciones del entrenamiento, y cada evaluacion sorteaba prefijos distintos
(la semilla era 1000+paso, que difiere de un modelo a otro). Medido sobre
`estilo_llama_ctx2048_24ep`: 79.3 en el paso 22500 y 62.7 en el 23800, con el
val_bpt pasando de 1.5185 a 1.5172, o sea el mismo modelo. Con ese ruido, una
diferencia de 3 puntos entre dos modelos no dice nada.

Que cambia aqui:

  1. UN solo juego de prefijos, identico para todos los modelos, del split de
     TEST, que no se uso para elegir ningun checkpoint.
  2. Varias semillas de muestreo por modelo, con generador explicito, asi que
     cada numero es reproducible y se obtiene una desviacion, no un dato suelto.
  3. Checkpoint `best.pt`, elegido por val_bpt (determinista), en vez de
     `best_gen.pt`, elegido por el gen_score ruidoso: ese es justo el sesgo que
     se quiere quitar.
  4. Mismo muestreo para todos (el del entrenamiento: T=1.0, top-p=0.95), para
     que la unica variable sea el modelo.

Solo cubre la familia de eventos; los de frames (melle) generan por otra ruta y
se omiten con aviso.

Salida en reports/comparacion_pareada/:
    resultados.csv   una fila por (modelo, semilla)
    resumen.csv      media, desviacion y rango por modelo, mas el dato del leaderboard
    comparacion.png  cada semilla como punto, la media como barra
    audio/prefijo_NN/<modelo>.wav   el MISMO prefijo continuado por cada modelo,
                                    junto a REAL.wav, que es lo que hacia la pieza
"""
from __future__ import annotations
import argparse, csv, importlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate as gen                               # noqa: E402
import evaluate as ev                                # noqa: E402
import prompts as pr                                 # noqa: E402
import registry as reg                               # noqa: E402
from config import Config                            # noqa: E402
from models import build_model                       # noqa: E402
from data.datasets import get_tokens, get_roll       # noqa: E402
from data.tokenizer import decode_tokens             # noqa: E402
from midi_export import roll_to_midi                 # noqa: E402

par = importlib.import_module("17_par_real_vs_modelo")   # localizar() y synth_par()

SALIDA = ROOT / "reports" / "comparacion_pareada"
SEMILLA_PREFIJOS = 2024          # fija el juego de prefijos, igual para todos
GEN_STEPS = 800                  # 40 s, como en el entrenamiento
POR_DEFECTO = ["lstm", "estilo_llama_ctx2048_24ep", "estilo_llama_24ep",
               "music_transformer", "perceiver_ar", "deep_lstm"]


def cargar(exp: str, ckpt: str, device: str):
    p = ROOT / "experiments" / exp / "checkpoints" / ckpt
    if not p.exists():
        return None, None
    ck = reg.load_checkpoint(p, map_location=device)
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
    if cfg.family != "token":
        print("  %s: familia '%s', se omite (genera por otra ruta)" % (exp, cfg.family))
        return None, None
    m = build_model(cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, cfg


def continuacion_real(src: int, pref_tokens: np.ndarray, n_pref: int):
    """Lo que la pieza hacia de verdad tras el prefijo, o None si no se localiza."""
    tk = np.asarray(get_tokens(int(src)))
    s = par.localizar(pref_tokens, tk)
    if s is None:
        return None, None
    roll = get_roll(int(src))
    ini = len(decode_tokens(tk[:s]))
    return (np.asarray(roll[ini:ini + n_pref], np.uint8),
            np.asarray(roll[ini + n_pref: ini + n_pref + GEN_STEPS], np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=16, help="prefijos (los mismos para todos)")
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--pasos", type=int, default=GEN_STEPS)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--audio", type=int, default=2, help="cuantos prefijos sintetizar")
    ap.add_argument("--muestreo", default="entrenamiento",
                    choices=["entrenamiento", "calibrado"],
                    help="entrenamiento = T 1.0 / top-p 0.95 (el que produjo el leaderboard); "
                         "calibrado = data/processed/best_sampling.json (anti-bucle)")
    ap.add_argument("--salida", default=None, help="subcarpeta de reports/ (por defecto comparacion_pareada)")
    a = ap.parse_args()

    global SALIDA
    if a.salida:
        SALIDA = ROOT / "reports" / a.salida
    elif a.muestreo == "calibrado":
        SALIDA = ROOT / "reports" / "comparacion_pareada_calibrada"

    estrategia = None
    if a.muestreo == "calibrado":
        f = ROOT / "data" / "processed" / "best_sampling.json"
        if not f.exists():
            print("no existe %s" % f); return
        estrategia = json.load(open(f, encoding="utf-8"))
        print("muestreo calibrado: %s" % estrategia)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exps = a.exps or POR_DEFECTO
    SALIDA.mkdir(parents=True, exist_ok=True)

    # --- el juego de prefijos, UNA vez -------------------------------------
    px, pst, srcs = pr.make_token_prompts(a.n, 200, split="test", seed=SEMILLA_PREFIJOS)
    print("%d prefijos fijos del split de TEST (semilla %d), %d-%d pasos (%.1f-%.1f s)"
          % (a.n, SEMILLA_PREFIJOS, pst.min(), pst.max(), pst.min() * .05, pst.max() * .05))
    print("modelos: %s | %d semillas de muestreo | %d pasos generados (%.0f s) | %s\n"
          % (", ".join(exps), a.semillas, a.pasos, a.pasos * .05, device))

    filas, audio_guardado = [], {}
    for exp in exps:
        model, cfg = cargar(exp, a.ckpt, device)
        if model is None:
            continue
        for semilla in range(1, a.semillas + 1):
            t0 = time.time()
            g = torch.Generator(device=device); g.manual_seed(semilla)
            if estrategia is None:
                res = gen.sample_tokens(model, px.to(device), a.pasos, seq_len=cfg.seq_len,
                                        temperature=1.0, top_k=0, top_p=0.95,
                                        device=device, generator=g)
                nuevos = res["new"]
            else:
                # sample_with_strategy lleva por fila el historial de repeticion y
                # los n-gramas, que es lo que la estrategia anti-bucle necesita
                import sampling as S
                res = S.sample_with_strategy(model, px.numpy(), a.pasos, strategy=estrategia,
                                             seq_len=cfg.seq_len, device=device, generator=g)
                nuevos = res["new"]
            rolls = []
            for b in range(len(nuevos)):
                r = gen.tokens_to_roll(np.asarray(nuevos[b]).ravel())
                if len(r) < a.pasos:
                    r = np.pad(r, ((0, a.pasos - len(r)), (0, 0)))
                rolls.append(r[: a.pasos])
            m, _, _ = ev.evaluate_generations(rolls)
            filas.append(dict(modelo=exp, semilla=semilla,
                              gen_score=float(m["gen_score"]),
                              densidad=float(m["gen_density"]),
                              repeat8=float(m["gen_repeat8"]),
                              seg=round(time.time() - t0, 1)))
            print("  %-28s semilla %d: gen_score %5.1f  densidad %.3f  repeat8 %.3f  (%.0f s)"
                  % (exp, semilla, m["gen_score"], m["gen_density"], m["gen_repeat8"],
                     time.time() - t0))
            if semilla == 1 and a.audio:
                audio_guardado[exp] = [np.asarray(r, np.uint8) for r in rolls[: a.audio]]
        del model
        torch.cuda.empty_cache()

    if not filas:
        print("ningun modelo evaluado"); return

    with open(SALIDA / "resultados.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)

    # --- resumen ------------------------------------------------------------
    lb = {}
    p_lb = ROOT / "reports" / "leaderboard.csv"
    if p_lb.exists():
        import pandas as pd
        for _, r in pd.read_csv(p_lb).iterrows():
            lb[str(r["name"])] = (r.get("best_gen_score"), r.get("best_val_bpt"))

    resumen = []
    for exp in exps:
        g = [f["gen_score"] for f in filas if f["modelo"] == exp]
        if not g:
            continue
        antes, bpt = lb.get(exp, (float("nan"), float("nan")))
        resumen.append(dict(modelo=exp, n=len(g), media=round(float(np.mean(g)), 2),
                            desv=round(float(np.std(g, ddof=1)) if len(g) > 1 else 0.0, 2),
                            minimo=round(min(g), 2), maximo=round(max(g), 2),
                            leaderboard=round(float(antes), 2) if antes == antes else "",
                            val_bpt=round(float(bpt), 4) if bpt == bpt else ""))
    resumen.sort(key=lambda d: -d["media"])
    with open(SALIDA / "resumen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumen[0].keys())); w.writeheader(); w.writerows(resumen)

    print("\n%-28s %7s %6s %13s %10s %8s" % ("modelo", "media", "desv", "rango", "leaderb.", "bits/paso"))
    for d in resumen:
        print("%-28s %7.1f %6.1f  %5.1f - %-5.1f %9s %9s"
              % (d["modelo"], d["media"], d["desv"], d["minimo"], d["maximo"],
                 d["leaderboard"], d["val_bpt"]))

    # --- figura -------------------------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 0.62 * len(resumen) + 2.2))
        y = np.arange(len(resumen))
        for k, d in enumerate(resumen):
            g = [f["gen_score"] for f in filas if f["modelo"] == d["modelo"]]
            ax.barh(k, d["media"], color="#4a7fb5", alpha=.55, height=.55)
            ax.plot(g, [k] * len(g), "o", color="#1b3b5a", ms=6, zorder=3)
            if d["leaderboard"] != "":
                ax.plot(d["leaderboard"], k, "x", color="#c1452e", ms=11, mew=2.2, zorder=4)
        ax.set_yticks(y); ax.set_yticklabels([d["modelo"] for d in resumen], fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel("gen_score"); ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=.3)
        ax.set_title("Mismos %d prefijos de test para todos, %d semillas de muestreo\n"
                     "puntos = cada semilla | barra = media | x roja = el maximo que reportaba el leaderboard"
                     % (a.n, a.semillas), fontsize=10)
        fig.tight_layout(); fig.savefig(SALIDA / "comparacion.png", dpi=140); plt.close(fig)
        print("\nfigura: %s" % (SALIDA / "comparacion.png"))
    except Exception as e:
        print("no se pudo dibujar la figura: %s" % e)

    # --- audio A/B: el mismo prefijo por todos los modelos ------------------
    if a.audio and audio_guardado:
        import audio_play                              # noqa: F401  (lo usa synth_par)
        for j in range(a.audio):
            pref_tokens = px[j].numpy()[1:]
            prefijo, real = continuacion_real(int(srcs[j]), pref_tokens, int(pst[j]))
            if prefijo is None:
                print("prefijo %d: no localizado en la pieza, sin audio" % (j + 1)); continue
            d = SALIDA / "audio" / ("prefijo_%02d" % (j + 1))
            d.mkdir(parents=True, exist_ok=True)
            pistas = {}
            if len(real):
                pistas["REAL"] = np.concatenate([prefijo, real])
            for exp, rolls in audio_guardado.items():
                pistas[exp] = np.concatenate([prefijo, rolls[j]])
            # una ganancia comun para todas: si cada una se normaliza por su
            # pico, el prefijo -- identico -- suena a volumenes distintos y se
            # confunde un cambio de densidad con uno de calidad
            par.synth_par(pistas, lambda et, d=d: d / ("%s.wav" % et))
            for et, full in pistas.items():
                roll_to_midi(full, d / ("%s.mid" % et))
            json.dump({"pieza": int(srcs[j]), "split": "test",
                       "prefijo_pasos": int(pst[j]),
                       "prefijo_segundos": round(int(pst[j]) * .05, 1),
                       "semilla_prefijos": SEMILLA_PREFIJOS, "semilla_muestreo": 1,
                       "nota": "todas las pistas comparten prefijo y ganancia; REAL es la pieza original"},
                      open(d / "info.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            print("audio A/B: %s (%d pistas, pieza %d)"
                  % (d.relative_to(ROOT), len(pistas), int(srcs[j])))

    print("\ntablas en %s" % SALIDA.relative_to(ROOT))


if __name__ == "__main__":
    main()
