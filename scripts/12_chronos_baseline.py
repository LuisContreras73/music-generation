"""Chronos como baseline zero-shot de continuacion musical.

    python scripts/12_chronos_baseline.py --size tiny
    python scripts/12_chronos_baseline.py --size small --npz eval_set_02_prefix.npz

Que es Chronos y por que probarlo
---------------------------------
Chronos (Ansari et al., Amazon) es un modelo FUNDACIONAL de prevision de series
temporales: tokeniza valores continuos escalados en bins y predice la
distribucion de los valores futuros, zero-shot, sin entrenar en el dominio.

Hipotesis previa (que este script existe para FALSAR, no para confirmar):
  no deberia funcionar bien aqui, por tres razones estructurales:
  1. Es un FORECASTER: optimiza error puntual o cuantiles, lo que empuja hacia la
     prediccion media. En un roll con 75.6% de pasos vacios, la media es silencio.
  2. Trata cada serie por separado. Las 88 notas de un instante se predicen sin
     modelar que suenan JUNTAS, que es justo la limitacion que hunde a la familia
     frame (MELLE pierde 1.4 bits/paso por eso).
  3. Sus tokens son bins de un valor escalar continuo; los nuestros son eventos
     simbolicos discretos.
  Evidencia interna que lo anticipa: el TFT, tambien un forecaster de series
  adaptado, obtuvo gen_score 26 frente a 62-83 de los modelos nativos.

Aun asi se mide, porque un argumento no es un resultado y un baseline negativo
bien medido es informacion util.

Adaptacion (documentada, porque condiciona la lectura del resultado)
  Se prueban dos codificaciones del roll a series continuas:
    "pitch"   la nota mas aguda activa en cada paso (0 si silencio) -> 1 serie.
              Captura la melodia, que es lo que tienen los prefijos de prueba.
    "multi"   una serie por nota (88 series binarias) -> multivariado.
  Se prevee el horizonte pedido y se reconstruye un roll binario. Cualquiera de
  las dos pierde informacion respecto a la representacion de eventos; se dice.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from metrics import aggregate_features, gen_score, roll_features   # noqa: E402
from evaluate import load_ref_agg                                  # noqa: E402

N_PITCH = 88


def roll_to_pitch_series(roll: np.ndarray) -> np.ndarray:
    """Nota mas aguda activa por paso; 0 = silencio. Serie de enteros [T]."""
    out = np.zeros(len(roll), np.float32)
    for t in range(len(roll)):
        nz = np.flatnonzero(roll[t])
        if len(nz):
            out[t] = nz[-1] + 1          # +1 para reservar el 0 al silencio
    return out


def pitch_series_to_roll(series: np.ndarray, n_pitch: int = N_PITCH) -> np.ndarray:
    """Inversa aproximada: cada valor > 0.5 se redondea a una nota."""
    T = len(series)
    roll = np.zeros((T, n_pitch), np.uint8)
    for t, v in enumerate(series):
        p = int(round(float(v))) - 1
        if 0 <= p < n_pitch and float(v) >= 0.5:
            roll[t, p] = 1
    return roll


def load_prefixes(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    off = z["offsets"].astype(np.int64)
    ids = [str(s) for s in z["ids"]]
    R = z["rolls_flat"]
    return [np.asarray(R[off[i]:off[i + 1]], np.uint8) for i in range(len(ids))], ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny", choices=["tiny", "mini", "small", "base"])
    ap.add_argument("--npz", default="external_eval_prefix_5s.npz")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--mode", default="pitch", choices=["pitch", "multi"])
    ap.add_argument("--samples", type=int, default=20)
    a = ap.parse_args()

    try:
        from chronos import BaseChronosPipeline
    except Exception as e:
        print("chronos no disponible: %s" % e)
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "amazon/chronos-t5-%s" % a.size
    print("cargando %s en %s ..." % (model_id, dev), flush=True)
    t0 = time.time()
    try:
        pipe = BaseChronosPipeline.from_pretrained(
            model_id, device_map=dev, torch_dtype=torch.float32)
    except Exception as e:
        print("no se pudo cargar (sin red o sin cache local?): %s: %s" % (type(e).__name__, str(e)[:200]))
        return
    print("cargado en %.1f s" % (time.time() - t0))

    prefixes, ids = load_prefixes(ROOT / a.npz)
    print("\n%s | modo=%s | horizonte %d pasos\n" % (Path(a.npz).stem, a.mode, a.steps))
    rolls = []
    for pid, pref in zip(ids, prefixes):
        t1 = time.time()
        if a.mode == "pitch":
            ctx = torch.tensor(roll_to_pitch_series(pref))[None]
            # Chronos limita el horizonte por llamada; se encadena si hace falta
            got = []
            remaining = a.steps
            cur = ctx
            while remaining > 0:
                h = min(remaining, 64)
                q, mean = pipe.predict_quantiles(context=cur, prediction_length=h,
                                                 quantile_levels=[0.1, 0.5, 0.9])
                med = q[0, :, 1].cpu().numpy()          # mediana
                got.append(med)
                cur = torch.cat([cur, torch.tensor(med, dtype=torch.float32)[None]], dim=1)
                remaining -= h
            series = np.concatenate(got)[: a.steps]
            roll = pitch_series_to_roll(series)
        else:
            roll = np.zeros((a.steps, N_PITCH), np.uint8)
            for p in range(N_PITCH):
                s = pref[:, p].astype(np.float32)
                if s.sum() == 0:
                    continue
                q, _ = pipe.predict_quantiles(context=torch.tensor(s)[None],
                                              prediction_length=min(a.steps, 64),
                                              quantile_levels=[0.5])
                med = q[0, :, 0].cpu().numpy()
                roll[: len(med), p] = (med > 0.5).astype(np.uint8)
        f = roll_features(roll)
        rolls.append(roll)
        print("  %-24s %4d onsets | densidad %.4f | polifonia %.2f | %5.1f s"
              % (pid, f["n_onsets"], f["density"], f["mean_poly"], time.time() - t1), flush=True)

    agg = aggregate_features(rolls)
    ref = load_ref_agg("val_excerpt")
    calib = json.load(open(ROOT / "data" / "processed" / "calibration.json"))
    sc = gen_score(agg, ref, calib)
    print("\nRESULTADO CHRONOS-%s (%s)" % (a.size, a.mode))
    print("  gen_score %.2f | repeat8 %.3f | densidad %.4f | polifonia %.2f"
          % (sc["gen_score"], agg["repeat8"], agg["density"], agg["mean_poly"]))
    print("\n  escala: ruido 1.6 | corpus real 87.6")
    print("  para comparar, sobre estas mismas melodias:")
    print("    music_transformer 2.43 | lstm 2.04 | music_transformer_con_augmentacion 0.00 (clava la textura pero entra en bucle)")

    out = ROOT / "reports" / "external_eval" / Path(a.npz).stem / ("chronos_%s_%s" % (a.size, a.mode))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "continuations_only.npz",
                        rolls_flat=np.concatenate(rolls).astype(np.uint8),
                        offsets=np.r_[0, np.cumsum([len(r) for r in rolls])].astype(np.int64),
                        ids=np.asarray(ids))
    json.dump(dict(model=model_id, mode=a.mode, gen_score=float(sc["gen_score"]),
                   repeat8=float(agg["repeat8"]), density=float(agg["density"]),
                   mean_poly=float(agg["mean_poly"])), open(out / "summary.json", "w"), indent=2)
    print("\nguardado en %s" % out.relative_to(ROOT))


if __name__ == "__main__":
    main()
