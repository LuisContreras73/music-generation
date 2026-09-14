"""Continuaciones para los conjuntos de evaluacion externos, con audio.

    python scripts/10_external_eval.py                       # mejor modelo, los dos sets
    python scripts/10_external_eval.py --exp music_transformer
    python scripts/10_external_eval.py --npz external_eval_prefix_5s.npz --steps 600
    python scripts/10_external_eval.py --all-models          # compara todos los experimentos

Entrada: un .npz con prefijos en el mismo formato que train.npz
    rolls_flat [T,88] uint8, offsets, ids, step_sec, note_min, note_max,
    num_positions, representation, is_prefix, prefix_steps

Salida en reports/external_eval/<set>/<modelo>/:
    continuations.npz   prefijo + continuacion, MISMO formato que la entrada,
                        listo para audio_play.py
    *.wav               audio sintetizado (caja de musica) de cada pieza
    *.mid               MIDI de cada pieza
    *_roll.png          piano-roll con la costura prefijo/generado marcada
    summary.json        metricas por pieza

Aviso de dominio: los prefijos de melodias conocidas (Ode to Joy, Twinkle
Twinkle...) son MONOFONICOS y muy simples, mientras que el corpus de
entrenamiento es piano interpretado, polifonico y denso (1.86 notas por paso
activo). Es un test de generalizacion FUERA DE DISTRIBUCION, y conviene leer
los resultados con eso en mente.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import Config                      # noqa: E402
from models import build_model                 # noqa: E402
import registry as reg                         # noqa: E402
import generate as gen                         # noqa: E402
import metrics as mx                           # noqa: E402
from data.tokenizer import encode_roll_fast, BOS, EOS   # noqa: E402
from midi_export import roll_to_midi           # noqa: E402


# ---------------------------------------------------------------- entrada/salida
def load_prefixes(npz_path) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    rolls_flat = z["rolls_flat"]
    off = z["offsets"].astype(np.int64)
    ids = [str(s) for s in z["ids"]]
    rolls = [np.asarray(rolls_flat[off[i]:off[i + 1]], np.uint8) for i in range(len(ids))]
    meta = {k: z[k] for k in z.files if k not in ("rolls_flat", "offsets", "ids")}
    return dict(rolls=rolls, ids=ids, meta=meta,
                step_sec=float(z["step_sec"][0]) if "step_sec" in z.files else 0.05,
                note_min=int(z["note_min"][0]) if "note_min" in z.files else 21)


def save_like_input(rolls, ids, out_path, src_meta) -> Path:
    """Guarda [T,88] por pieza en el mismo formato que el npz de entrada."""
    lens = [len(r) for r in rolls]
    payload = dict(rolls_flat=np.concatenate(rolls).astype(np.uint8),
                   offsets=np.r_[0, np.cumsum(lens)].astype(np.int64),
                   ids=np.asarray(ids))
    for k, v in src_meta.items():
        if k not in ("is_prefix", "prefix_steps"):
            payload[k] = v
    payload["is_prefix"] = np.asarray([False])
    payload["prefix_steps"] = np.asarray([int(src_meta.get("prefix_steps", [100])[0])
                                          if hasattr(src_meta.get("prefix_steps", None), "__len__")
                                          else 100], np.int32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return out_path


# ---------------------------------------------------------------- generacion
def _load_best_sampling():
    """Estrategia de muestreo optima, medida sobre los prefijos monofonicos.

    No es un detalle menor: con la decodificacion por defecto (temperatura 1.0,
    nucleus 0.95) las melodias monofonicas puntuan 0.01 porque el modelo entra en
    bucle. Con penalizacion de repeticion y prohibicion de n-gramas, 16.12 sobre
    un techo medido de 21.56. Mismo modelo, mismos pesos.
    """
    f = ROOT / "data" / "processed" / "best_sampling.json"
    if f.exists():
        try:
            return json.load(open(f))
        except Exception:
            pass
    return None


def continue_prefixes(model, cfg, prefixes: list, n_steps: int, device: str,
                      temperature: float, top_p: float, seed: int = 0,
                      strategy=None) -> dict:
    """Continua cada prefijo por separado (tienen longitudes distintas en tokens)."""
    outs, gram = [], []
    for i, pref_roll in enumerate(prefixes):
        if cfg.family == "token":
            tk = encode_roll_fast(pref_roll, add_bos=True, add_eos=False).astype(np.int64)
            if strategy is not None:
                import sampling as S
                r = S.sample_with_strategy(model, tk, n_steps, strategy=strategy,
                                           seq_len=cfg.seq_len, device=device,
                                           seed=seed * 1000 + i)
                new = r["new"] if isinstance(r, dict) else r
                arr = np.asarray(new[0] if isinstance(new, (list, tuple)) else new).ravel()
                cont = gen.tokens_to_roll(arr)
                gram.append(float(r.get("grammar_prob_mass", 0.0)) if isinstance(r, dict) else 0.0)
            else:
                x = torch.from_numpy(tk)[None, :].to(device)
                g = torch.Generator(device=device); g.manual_seed(seed * 1000 + i)
                res = gen.sample_tokens(model, x, n_steps, seq_len=cfg.seq_len,
                                        temperature=temperature, top_k=0, top_p=top_p,
                                        device=device, generator=g)
                cont = gen.tokens_to_roll(res["new"][0])
                gram.append(res["grammar_prob_mass"])
        else:
            x = torch.from_numpy(pref_roll.astype(np.float32))[None].to(device)
            res = gen.sample_frames(model, x, n_steps, seq_len=cfg.seq_len,
                                    temperature=temperature, device=device)
            cont = np.asarray(res["new"][0], np.uint8)
            gram.append(0.0)
        if len(cont) < n_steps:
            cont = np.pad(cont, ((0, n_steps - len(cont)), (0, 0)))
        outs.append(cont[:n_steps].astype(np.uint8))
    return dict(continuations=outs, grammar_prob_mass=float(np.mean(gram)) if gram else 0.0)


# ---------------------------------------------------------------- principal
def run_set(npz_path: Path, exp: str, n_steps: int, device: str,
            temperature: float, top_p: float, ckpt: str = "best.pt",
            no_strategy: bool = False) -> dict:
    data = load_prefixes(npz_path)
    ck_path = ROOT / "experiments" / exp / "checkpoints" / ckpt
    if not ck_path.exists():
        print("  %s: no existe %s" % (exp, ck_path)); return {}
    ck = reg.load_checkpoint(ck_path, map_location=device)
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
    model = build_model(cfg).to(device); model.load_state_dict(ck["model"]); model.eval()

    set_name = Path(npz_path).stem
    out_dir = ROOT / "reports" / "external_eval" / set_name / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n%s | %s (paso %s) -> %s"
          % (set_name, exp, ck.get("step"), out_dir.relative_to(ROOT)))

    strat = None if no_strategy else _load_best_sampling()
    if strat:
        print("   muestreo optimizado: T%.2f penalty %.1f ventana %d ngram %d"
              % (strat.get("temperature", 1.0), strat.get("penalty", 1.0),
                 strat.get("penalty_window", 0), strat.get("ngram", 0)))
    res = continue_prefixes(model, cfg, data["rolls"], n_steps, device, temperature,
                            top_p, strategy=strat)
    full = [np.concatenate([p, c]) for p, c in zip(data["rolls"], res["continuations"])]

    save_like_input(full, data["ids"], out_dir / "continuations.npz", data["meta"])
    save_like_input(res["continuations"], data["ids"],
                    out_dir / "continuations_only.npz", data["meta"])

    # --- audio y MIDI ---
    import audio_play
    per_piece = []
    for i, (pid, roll_full, cont) in enumerate(zip(data["ids"], full, res["continuations"])):
        wav = out_dir / ("%s.wav" % pid)
        audio, sr = audio_play.synthesize_musicbox_roll(
            roll_full, step_sec=data["step_sec"], note_min=data["note_min"],
            representation="onset")
        audio_play.save_wav(wav, audio, sr)
        roll_to_midi(roll_full, out_dir / ("%s.mid" % pid), step_sec=data["step_sec"],
                     note_min=data["note_min"])
        f = mx.roll_features(cont)
        per_piece.append(dict(id=pid, prefix_steps=int(len(data["rolls"][i])),
                              gen_steps=int(len(cont)), gen_onsets=int(f["n_onsets"]),
                              gen_density=f["density"], gen_mean_poly=f["mean_poly"],
                              gen_distinct_pitch=int(f["n_distinct_pitch"]),
                              gen_repeat8=f["repeat8"]))
        print("   %-24s prefijo %3d + generado %3d pasos | %4d onsets | densidad %.3f | "
              "polifonia %.2f" % (pid, len(data["rolls"][i]), len(cont), f["n_onsets"],
                                  f["density"], f["mean_poly"]))
        try:
            import viz
            viz.plot_pianoroll(roll_full, out_dir / ("%s_roll.png" % pid),
                               title="%s | %s (prefijo %d pasos + %d generados)"
                                     % (pid, exp, len(data["rolls"][i]), len(cont)),
                               prime_steps=len(data["rolls"][i]),
                               step_sec=data["step_sec"])
        except Exception as e:
            print("     [viz] %s: %s" % (type(e).__name__, e))

    summary = dict(set=set_name, exp=exp, step=int(ck.get("step", 0)),
                   model=cfg.model, family=cfg.family, temperature=temperature,
                   top_p=top_p, gen_steps=n_steps,
                   grammar_prob_mass=res["grammar_prob_mass"], pieces=per_piece)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    del model
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="un .npz concreto; por defecto los dos sets")
    ap.add_argument("--exp", default=None, help="experimento; por defecto el mejor del leaderboard")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--steps", type=int, default=600, help="pasos a generar (600 = 30 s)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--no-strategy", action="store_true",
                    help="usar el muestreo por defecto en vez del optimizado")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sets = [Path(a.npz)] if a.npz else [ROOT / "external_eval_prefix_5s.npz",
                                        ROOT / "eval_set_02_prefix.npz"]
    sets = [s for s in sets if s.exists()]
    if not sets:
        print("no se encontro ningun npz de prefijos"); return

    if a.all_models:
        exps = sorted(p.name for p in (ROOT / "experiments").iterdir()
                      if p.is_dir() and not p.name.startswith("smoke_")
                      and (p / "checkpoints" / a.ckpt).exists())
    elif a.exp:
        exps = [a.exp]
    else:
        lb = ROOT / "reports" / "leaderboard.csv"
        if not lb.exists():
            print("no hay leaderboard; usa --exp"); return
        import pandas as pd
        df = pd.read_csv(lb)
        df = df[~df["name"].astype(str).str.startswith("smoke_")]
        exps = [str(df.iloc[0]["name"])]
        print("mejor experimento del leaderboard: %s" % exps[0])

    print("prefijos: %s | modelos: %s | %d pasos generados (%.0f s)"
          % ([s.stem for s in sets], exps, a.steps, a.steps * 0.05))
    for s in sets:
        for e in exps:
            run_set(s, e, a.steps, device, a.temperature, a.top_p, a.ckpt,
                    no_strategy=a.no_strategy)
    print("\nsalida en reports/external_eval/")


if __name__ == "__main__":
    main()
