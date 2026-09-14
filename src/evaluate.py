"""Evaluacion: verosimilitud comparable entre familias + metricas musicales.

bits per timestep (bpt)
-----------------------
Metrica central del laboratorio, la unica comparable entre un modelo de eventos
y uno de frames:

  bpt = -log2 p(piano_roll) / n_timesteps

* familia token: la tokenizacion roll<->eventos es BIYECTIVA, luego
  log p(tokens) = log p(roll). Se acumula la NLL de todos los tokens y se divide
  por el numero de pasos de 50 ms que esos tokens representan (suma de las
  duraciones de los tokens SHIFT).
* familia frame: NLL Bernoulli de las 88 notas de cada frame, sumada sobre las
  88 dimensiones y dividida por el numero de frames.

Un modelo que asigna la misma probabilidad al mismo roll obtiene el mismo bpt
independientemente de su parametrizacion. Referencia trivial: un modelo que
predice notas i.i.d. con la densidad marginal del corpus (p=0.005143) da
  88 * H2(0.005143) = 88 * 0.04585 = 4.03 bits/paso.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F

from data.tokenizer import PAD, SHIFT_OFF_ID, NOTE_OFF_ID, N_PITCH
from metrics import aggregate_features, gen_score, frame_prf, roll_features

LN2 = math.log(2.0)
MARGINAL_DENSITY = 0.005142671821564932


def marginal_bpt(p: float = MARGINAL_DENSITY) -> float:
    """bits/paso de la referencia i.i.d. Bernoulli con densidad marginal."""
    h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return N_PITCH * h


def _amp_ctx(amp: str, device: str = "cuda"):
    if amp == "bf16" and device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if amp == "fp16" and device == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cuda", enabled=False) if device == "cuda" else torch.enable_grad()


@torch.no_grad()
def eval_token_model(model, loader, device: str = "cuda", amp: str = "bf16",
                     max_batches: int = 0) -> dict:
    """NLL/ppl/bpt y accuracy desglosada por tipo de token."""
    model.eval()
    nll_sum = 0.0; n_tok = 0; n_steps = 0
    corr = 0; corr_note = 0; n_note = 0; corr_shift = 0; n_shift = 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with _amp_ctx(amp, device):
            logits = model(x)
        logits = logits.float()
        m = y != PAD
        if not bool(m.any()):
            continue
        ls = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                             reduction="none").view_as(y)
        nll_sum += float((ls * m).sum())
        n_tok += int(m.sum())
        is_sh = (y >= SHIFT_OFF_ID) & m
        n_steps += int(((y - SHIFT_OFF_ID + 1) * is_sh).sum())
        pred = logits.argmax(-1)
        ok = (pred == y) & m
        corr += int(ok.sum())
        is_nt = (y >= NOTE_OFF_ID) & (y < SHIFT_OFF_ID) & m
        corr_note += int((ok & is_nt).sum()); n_note += int(is_nt.sum())
        corr_shift += int((ok & is_sh).sum()); n_shift += int(is_sh.sum())
    if n_tok == 0:
        return dict(val_bpt=float("nan"), val_nll=float("nan"))
    nll_tok = nll_sum / n_tok
    return dict(
        val_nll=nll_tok,
        val_ppl=float(math.exp(min(nll_tok, 30))),
        val_bpt=float(nll_sum / LN2 / max(n_steps, 1)),
        val_bits_per_token=float(nll_tok / LN2),
        val_token_acc=corr / n_tok,
        val_note_acc=corr_note / max(n_note, 1),
        val_shift_acc=corr_shift / max(n_shift, 1),
        val_n_tokens=n_tok, val_n_steps=n_steps,
    )


THRESHOLDS = (0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5)


@torch.no_grad()
def eval_frame_model(model, loader, device: str = "cuda", amp: str = "bf16",
                     max_batches: int = 0, threshold: float = 0.5) -> dict:
    """NLL Bernoulli -> bpt, mas precision/recall/F1 de frame (teacher forcing).

    Se barre el umbral en vez de fijarlo en 0.5. Con la BCE pura (sin pos_weight)
    el modelo queda CALIBRADO a la densidad real del corpus, que es 0.0051, asi
    que casi ninguna nota supera 0.5 y el F1 a ese umbral sale practicamente
    cero: mediria la calibracion, no la capacidad de discriminar. Se reporta
    val_frame_f1 (umbral 0.5, comparable con la literatura que lo usa) y
    val_frame_f1_best con su umbral optimo, que es el numero informativo aqui.
    """
    model.eval()
    nll_sum = 0.0; n_frames = 0
    tp = {t: 0.0 for t in THRESHOLDS}
    fp = {t: 0.0 for t in THRESHOLDS}
    fn = {t: 0.0 for t in THRESHOLDS}
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y, mask = [t.to(device, non_blocking=True) for t in batch]
        with _amp_ctx(amp, device):
            out = model(x)
        logits = (out["logits"] if isinstance(out, dict) else out).float()
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(-1)  # [B,T]
        nll_sum += float((bce * mask).sum())
        n_frames += int(mask.sum())
        m3 = mask.unsqueeze(-1)
        p = logits.sigmoid() * m3
        yt = y * m3
        for th in THRESHOLDS:
            pred = (p > th).float() * m3
            tp[th] += float((pred * yt).sum())
            fp[th] += float((pred * (1 - yt) * m3).sum())
            fn[th] += float(((1 - pred) * yt).sum())
    if n_frames == 0:
        return dict(val_bpt=float("nan"))

    def prf(th):
        pr = tp[th] / (tp[th] + fp[th]) if tp[th] + fp[th] else 0.0
        rc = tp[th] / (tp[th] + fn[th]) if tp[th] + fn[th] else 0.0
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

    best_th = max(THRESHOLDS, key=lambda t: prf(t)[2])
    p5, r5, f5 = prf(threshold if threshold in tp else 0.5)
    pb, rb, fb = prf(best_th)
    return dict(
        val_nll=nll_sum / n_frames,
        val_bpt=float(nll_sum / LN2 / n_frames),
        val_frame_precision=p5, val_frame_recall=r5, val_frame_f1=f5,
        val_frame_f1_best=fb, val_frame_best_threshold=best_th,
        val_frame_precision_best=pb, val_frame_recall_best=rb,
        val_n_frames=n_frames,
    )


def load_ref_agg(key: str = "val_excerpt") -> dict:
    """Agregados de referencia del corpus (data/processed/ref_stats.npz)."""
    from config import PROC
    z = np.load(PROC / "ref_stats.npz")
    pref = key + "__"
    out = {}
    for k in z.files:
        if k.startswith(pref):
            v = z[k]
            out[k[len(pref):]] = v if v.ndim else v.item()
    return out


def load_calibration() -> dict | None:
    """Techo y suelo de OA por componente (scripts/02_ref_stats.py).

    Sin este fichero el gen_score cae a pesos fijos y NO discrimina bien:
    medido empiricamente, fragmentos reales daban 79.7 y ruido puro 63.7.
    Con calibracion: 87.6 frente a 1.6.
    """
    from config import PROC
    p = PROC / "calibration.json"
    if not p.exists():
        return None
    import json
    return json.load(open(p))


_CALIB_WARNED = False


def _check_calibration(calib, n_pieces, n_steps_each):
    """La calibracion depende del numero y largo de los fragmentos evaluados.

    El techo de OA se mueve con el tamano de muestra, asi que evaluar con un N
    distinto al calibrado daria un score sesgado en silencio. Se avisa una vez.
    """
    global _CALIB_WARNED
    if not calib or _CALIB_WARNED:
        return
    want_n, want_s = calib.get("n_samples"), calib.get("gen_steps")
    if (want_n and n_pieces and abs(n_pieces - want_n) > 0) or \
       (want_s and n_steps_each and abs(n_steps_each - want_s) > 0.05 * want_s):
        _CALIB_WARNED = True
        print("  [aviso] la calibracion se hizo con N=%s fragmentos de %s pasos, pero se "
              "evalua con N=%s de %s. Vuelve a ejecutar scripts/02_ref_stats.py o el "
              "gen_score quedara sesgado." % (want_n, want_s, n_pieces, n_steps_each))


def evaluate_generations(rolls, ref_key: str = "val_excerpt"):
    """Agregados de las generaciones + gen_score calibrado frente al corpus."""
    agg = aggregate_features(rolls)
    if not agg:
        return dict(gen_score=0.0, gen_n_pieces=0), {}, {}
    ref = load_ref_agg(ref_key)
    calib = load_calibration()
    _check_calibration(calib, agg.get("n_pieces"),
                       agg["n_steps"] // max(agg.get("n_pieces", 1), 1))
    sc = gen_score(agg, ref, calib)
    out = {"gen_" + k: v for k, v in sc.items()}
    out["gen_score"] = sc["gen_score"]
    for k in ("density", "mean_poly", "empty_ratio", "notes_per_sec",
              "repeat8", "n_distinct_pitch"):
        out["gen_" + k] = agg[k]
        out["ref_" + k] = ref[k]
    out["gen_n_pieces"] = agg["n_pieces"]
    return out, agg, ref
