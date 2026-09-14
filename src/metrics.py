"""Metricas musicales para piano-roll de onsets [T,88] y comparacion vs corpus real.

Idea central
------------
* bpt (bits per timestep) es la metrica de verosimilitud COMPARABLE entre
  familias de modelos: la tokenizacion evento<->roll es biyectiva, luego
  log p(tokens) == log p(roll).  bpt = NLL_bits / n_timesteps.
* gen_score (0-100) mide cuanto se parece la DISTRIBUCION de la musica
  generada a la del corpus real, via Overlapping Area (OA = sum min(p,q))
  de varios histogramas + penalizaciones por degeneracion (silencio, bucles).
"""
from __future__ import annotations
import numpy as np

N_PITCH = 88
NOTE_MIN_MIDI = 21
STEP_SEC = 0.05
MAX_IOI = 64          # bins de inter-onset interval (pasos de 50 ms)
MAX_POLY = 12
MAX_INTERVAL = 24     # +/- semitonos para intervalos melodicos
MAX_HARM = 24         # semitonos para intervalos armonicos (notas simultaneas)


def _norm(h):
    h = np.asarray(h, np.float64)
    s = h.sum()
    return h / s if s > 0 else np.full_like(h, 1.0 / max(len(h), 1))


def overlap(p, q) -> float:
    """Overlapping Area entre dos histogramas -> [0,1]. 1 = identicos."""
    return float(np.minimum(_norm(p), _norm(q)).sum())


def roll_features(roll: np.ndarray) -> dict:
    """Histogramas y escalares descriptivos de un piano-roll de onsets."""
    roll = np.asarray(roll)
    if roll.ndim != 2 or roll.shape[1] != N_PITCH:
        raise ValueError("roll debe ser [T,88]")
    T = roll.shape[0]
    per_step = roll.sum(1).astype(np.int64)
    active = per_step > 0
    n_on = int(per_step.sum())

    pitch = roll.sum(0).astype(np.float64)
    pc = np.zeros(12)
    for p in range(N_PITCH):
        pc[(p + NOTE_MIN_MIDI) % 12] += pitch[p]

    if active.any():
        poly = np.bincount(per_step[active], minlength=MAX_POLY + 1)[: MAX_POLY + 1]
    else:
        poly = np.zeros(MAX_POLY + 1)

    at = np.flatnonzero(active)
    if len(at) > 1:
        ioi = np.bincount(np.clip(np.diff(at), 0, MAX_IOI), minlength=MAX_IOI + 1)[: MAX_IOI + 1]
        top = np.array([np.flatnonzero(roll[t])[-1] for t in at])
        d = np.clip(np.diff(top), -MAX_INTERVAL, MAX_INTERVAL) + MAX_INTERVAL
        interval = np.bincount(d, minlength=2 * MAX_INTERVAL + 1)[: 2 * MAX_INTERVAL + 1]
    else:
        ioi = np.zeros(MAX_IOI + 1)
        interval = np.zeros(2 * MAX_INTERVAL + 1)

    # --- intervalos ARMONICOS: distancias entre notas de un MISMO paso ---------
    # Discrimina musica de ruido con las mismas marginales: un modelo que
    # muestrea notas al azar produce intervalos armonicos uniformes, mientras
    # que el corpus concentra terceras, quintas y octavas.
    harm = np.zeros(MAX_HARM + 1)
    poly_at = at[per_step[at] > 1] if len(at) else at
    if len(poly_at):
        cap = 4000                                  # techo de coste para piezas largas
        sel = poly_at if len(poly_at) <= cap else poly_at[:: len(poly_at) // cap + 1]
        for t in sel:
            pp = np.flatnonzero(roll[t])
            dif = np.clip(pp[1:] - pp[:-1], 0, MAX_HARM)      # intervalos adyacentes
            harm += np.bincount(dif, minlength=MAX_HARM + 1)[: MAX_HARM + 1]

    rep8 = 0.0                      # fraccion de 8-gramas de frames repetidos (bucles)
    if T >= 16:
        step = max(1, T // 4000)
        w = roll[::step]
        if len(w) >= 16:
            v = np.packbits(w, axis=1)
            g = np.lib.stride_tricks.sliding_window_view(v, (8, v.shape[1]))[:, 0]
            g = np.ascontiguousarray(g.reshape(len(g), -1), dtype=np.uint8)
            # np.unique(axis=0) falla en numpy 2.x con ciertos arrays ("Cannot
            # compare structured arrays..."). Contar filas distintas por sus
            # bytes es equivalente, mas rapido y no depende de esa via.
            n_uniq = len({row.tobytes() for row in g})
            rep8 = float(1.0 - n_uniq / len(g))

    return dict(
        n_steps=T, n_onsets=n_on,
        pitch_hist=pitch, pitch_class=pc, poly_hist=poly.astype(np.float64),
        ioi_hist=ioi.astype(np.float64), interval_hist=interval.astype(np.float64),
        harm_hist=harm.astype(np.float64),
        density=float(n_on / max(T, 1)),
        notes_per_sec=float(n_on / max(T * STEP_SEC, 1e-9)),
        empty_ratio=float(1.0 - active.mean()) if T else 1.0,
        mean_poly=float(per_step[active].mean()) if active.any() else 0.0,
        pitch_range=int(np.ptp(np.flatnonzero(pitch))) if (pitch > 0).any() else 0,
        n_distinct_pitch=int((pitch > 0).sum()),
        repeat8=rep8,
    )


HIST_KEYS = ("pitch_hist", "pitch_class", "poly_hist", "ioi_hist", "interval_hist",
             "harm_hist")
SCALAR_KEYS = ("density", "notes_per_sec", "empty_ratio", "mean_poly",
               "pitch_range", "n_distinct_pitch", "repeat8")


def aggregate_features(rolls) -> dict:
    """Suma los histogramas de varios rolls y promedia los escalares."""
    fs = [roll_features(r) for r in rolls if np.asarray(r).size]
    if not fs:
        return {}
    out = {}
    for k in HIST_KEYS:
        out[k] = np.sum([f[k] for f in fs], axis=0)
    for k in SCALAR_KEYS:
        out[k] = float(np.mean([f[k] for f in fs]))
    out["n_steps"] = int(np.sum([f["n_steps"] for f in fs]))
    out["n_onsets"] = int(np.sum([f["n_onsets"] for f in fs]))
    out["n_pieces"] = len(fs)
    return out


SCORE_KEYS = HIST_KEYS          # componentes candidatos del score
MIN_GAIN = 0.02                 # margen techo-suelo minimo para que un componente cuente

# Pesos de reserva, solo si no hay calibracion disponible. Con calibracion los
# pesos se DERIVAN del poder discriminativo medido de cada componente.
FALLBACK_WEIGHTS = dict(pitch_hist=0.75, pitch_class=1.0, poly_hist=1.0,
                        ioi_hist=1.0, interval_hist=1.25, harm_hist=2.0)


def gen_score(gen_agg: dict, ref_agg: dict, calib: dict | None = None) -> dict:
    """Compara agregados generado vs referencia -> componentes + score 0-100.

    Por que hace falta calibrar
    ---------------------------
    El area de solapamiento (OA) entre dos muestras FINITAS de la misma
    distribucion no vale 1: el error de muestreo la baja. Con 16 fragmentos de
    800 pasos el techo real ronda 0.88, no 1.0. Y hay componentes que un modelo
    trivial ya acierta: el histograma de inter-onset interval de ruido i.i.d.
    con la densidad correcta solapa ~0.80 con el corpus, porque la densidad
    marginal ya determina la distribucion geometrica de huecos.

    Sin corregir eso el score no discrimina: medido empiricamente, fragmentos
    reales del corpus daban 68.1, un modelo sin entrenar 69.6 y ruido puro 61.7.

    Con calibracion (dict con "floor" y "ceiling" por componente, producido por
    scripts/02_ref_stats.py) cada componente se reescala a [0,1]:

        z_k = (OA_k - floor_k) / (ceiling_k - floor_k)

    donde floor_k = OA de ruido i.i.d. con la densidad del corpus y
    ceiling_k = OA de una muestra real del mismo tamano. Asi z=0 significa
    "no mejor que ruido" y z=1 "indistinguible del corpus".

    Los pesos se derivan del PODER DISCRIMINATIVO medido, w_k ~ ceiling_k -
    floor_k, en vez de fijarse a mano: un componente que el ruido ya acierta
    recibe peso casi nulo automaticamente.

        score = 100 * sum_k w_k * z_k * penal_densidad * penal_bucle
    """
    if not gen_agg or not ref_agg:
        return dict(gen_score=0.0)
    comp = {}
    for k in SCORE_KEYS:
        comp["oa_" + k] = overlap(gen_agg[k], ref_agg[k])

    if calib and "floor" in calib and "ceiling" in calib:
        floor, ceil = calib["floor"], calib["ceiling"]
        # Se DESCARTAN los componentes sin poder discriminativo real (margen por
        # debajo de MIN_GAIN): incluirlos con peso cero daria un z divergente,
        # porque el denominador seria casi nulo.
        gains = {k: float(ceil[k]) - float(floor[k])
                 for k in SCORE_KEYS if k in floor and k in ceil
                 and float(ceil[k]) - float(floor[k]) >= MIN_GAIN}
        gsum = sum(gains.values())
        if gsum > 1e-9:
            w = {k: g / gsum for k, g in gains.items()}
            z = {}
            for k in w:
                span = float(ceil[k]) - float(floor[k])
                z[k] = float(np.clip((comp["oa_" + k] - float(floor[k])) / span, -0.5, 1.3))
            oa = float(sum(w[k] * z[k] for k in w))
            oa = float(np.clip(oa, 0.0, 1.15))
            extra = {"z_" + k: float(z[k]) for k in z}
            extra.update({"w_" + k: float(w[k]) for k in w})
            calibrated = True
        else:
            calibrated = False
    else:
        calibrated = False

    if not calibrated:                          # sin calibracion: OA ponderado a mano
        wsum = sum(FALLBACK_WEIGHTS.values())
        oa = sum(FALLBACK_WEIGHTS[k] * comp["oa_" + k] for k in FALLBACK_WEIGHTS) / wsum
        extra = {}

    d_gen = max(gen_agg["density"], 1e-6)
    d_ref = max(ref_agg["density"], 1e-6)
    pen_density = float(np.exp(-abs(np.log(d_gen / d_ref))))
    excess_rep = max(0.0, gen_agg["repeat8"] - ref_agg["repeat8"])
    pen_loop = float(np.clip(1.0 - 2.0 * excess_rep, 0.0, 1.0))

    res = dict(gen_score=float(100.0 * oa * pen_density * pen_loop),
               oa_weighted=float(oa), pen_density=pen_density, pen_loop=pen_loop,
               density_ratio=float(d_gen / d_ref), calibrated=bool(calibrated))
    res.update(comp)
    res.update(extra)
    return res


def frame_prf(pred: np.ndarray, true: np.ndarray) -> dict:
    """Precision/recall/F1 a nivel de frame (protocolo Boulanger-Lewandowski)."""
    pred = np.asarray(pred).astype(bool)
    true = np.asarray(true).astype(bool)
    n = min(len(pred), len(true))
    pred, true = pred[:n], true[:n]
    tp = float((pred & true).sum()); fp = float((pred & ~true).sum()); fn = float((~pred & true).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(frame_precision=prec, frame_recall=rec, frame_f1=f1,
                frame_acc=tp / (tp + fp + fn) if tp + fp + fn else 0.0)
