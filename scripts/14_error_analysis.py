"""Donde se equivoca el modelo: desglose del error por categoria.

    python scripts/14_error_analysis.py --exp estilo_llama_24ep

Responde a "que le cuesta" en vez de dar un solo numero agregado:
  * por TIPO de token (nota / salto de tiempo / fin)
  * por ALTURA (que notas predice peor)
  * por POSICION en la ventana (necesita contexto?)
  * por DENSIDAD de la pieza (falla en lo raro o en lo comun?)
  * por PASO de tiempo (los saltos largos son mas dificiles?)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import Config                    # noqa: E402
from models import build_model               # noqa: E402
import registry as reg                       # noqa: E402
from gpu_data import GPUTokenStream          # noqa: E402

PAD, BOS, EOS, NOTE_OFF, SHIFT_OFF = 0, 1, 2, 3, 91
LN2 = float(np.log(2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="estilo_llama_24ep")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--cpu", action="store_true", help="forzar CPU si la GPU esta ocupada")
    a = ap.parse_args()
    dev = "cpu" if a.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    ck = reg.load_checkpoint(ROOT / "experiments" / a.exp / "checkpoints" / "best.pt",
                             map_location=dev)
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in Config.__dataclass_fields__})
    m = build_model(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
    bs = 4 if dev == "cpu" else cfg.batch_size     # en CPU, lotes pequenos
    va = GPUTokenStream("val", cfg.seq_len, bs, device=dev,
                        max_windows=a.batches * bs)
    print("%s (paso %s) | analisis de error sobre validacion\n" % (a.exp, ck.get("step")))

    nll_tok = torch.zeros(155, device=dev)      # suma de perdida por token objetivo
    cnt_tok = torch.zeros(155, device=dev)
    nll_pos = torch.zeros(cfg.seq_len, device=dev)
    cnt_pos = torch.zeros(cfg.seq_len, device=dev)
    dens_bins = [[] for _ in range(5)]

    with torch.no_grad():
        for i, (x, y) in enumerate(va.eval_batches()):
            if i >= a.batches:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = m(x)
            ls = F.cross_entropy(logits.float().reshape(-1, 155), y.reshape(-1),
                                 reduction="none").view_as(y)
            msk = (y != PAD).float()
            nll_tok.index_add_(0, y.reshape(-1), (ls * msk).reshape(-1))
            cnt_tok.index_add_(0, y.reshape(-1), msk.reshape(-1))
            nll_pos += (ls * msk).sum(0); cnt_pos += msk.sum(0)
            # densidad de la ventana = fraccion de tokens de nota
            dens = ((y >= NOTE_OFF) & (y < SHIFT_OFF)).float().mean(1)
            per = (ls * msk).sum(1) / msk.sum(1).clamp(min=1)
            for b in range(len(dens)):
                k = min(4, int(dens[b].item() * 5 / 0.75))
                dens_bins[k].append(float(per[b]))

    t = (nll_tok / cnt_tok.clamp(min=1)) / LN2      # bits por token
    c = cnt_tok

    print("=== 1. por TIPO de token ===")
    grupos = {"NOTE_ON (88)": slice(NOTE_OFF, SHIFT_OFF),
              "SHIFT (64)": slice(SHIFT_OFF, 155),
              "EOS": slice(EOS, EOS + 1)}
    for g, sl in grupos.items():
        w = c[sl]; v = t[sl]
        if float(w.sum()) == 0: continue
        print("  %-14s %7.3f bits  (%5.1f%% de los tokens)"
              % (g, float((v * w).sum() / w.sum()), 100 * float(w.sum() / c.sum())))

    print("\n=== 2. las 6 ALTURAS mas dificiles y las 6 mas faciles ===")
    nv = t[NOTE_OFF:SHIFT_OFF].cpu().numpy(); nc = c[NOTE_OFF:SHIFT_OFF].cpu().numpy()
    ok = nc > 50
    idx = np.argsort(np.where(ok, nv, -1))
    dur = [i for i in idx[::-1] if ok[i]][:6]
    fac = [i for i in idx if ok[i]][:6]
    print("  dificiles: " + ", ".join("MIDI %d (%.2f bits, n=%d)" % (i+21, nv[i], nc[i]) for i in dur))
    print("  faciles  : " + ", ".join("MIDI %d (%.2f bits, n=%d)" % (i+21, nv[i], nc[i]) for i in fac))

    print("\n=== 3. por DURACION del salto de tiempo ===")
    sv = t[SHIFT_OFF:].cpu().numpy(); sc = c[SHIFT_OFF:].cpu().numpy()
    for lo, hi, tag in ((0, 2, "1-2 pasos (0.05-0.1 s)"), (2, 8, "3-8 pasos"),
                        (8, 24, "9-24 pasos"), (24, 64, "25-64 pasos (>1.2 s)")):
        w = sc[lo:hi]
        if w.sum() > 0:
            print("  %-24s %7.3f bits  (%5.1f%% de los shifts)"
                  % (tag, float((sv[lo:hi]*w).sum()/w.sum()), 100*w.sum()/sc.sum()))

    print("\n=== 4. por POSICION en la ventana (necesita contexto?) ===")
    pos = (nll_pos / cnt_pos.clamp(min=1)).cpu().numpy() / LN2
    for lo, hi in ((0, 32), (32, 128), (128, 512), (512, cfg.seq_len)):
        if hi <= len(pos):
            print("  posiciones %4d-%4d: %7.3f bits" % (lo, hi, float(np.mean(pos[lo:hi]))))

    print("\n=== 5. por DENSIDAD de la ventana ===")
    for k, lab in enumerate(["muy rala", "rala", "media", "densa", "muy densa"]):
        if dens_bins[k]:
            print("  %-11s %7.3f nats/token  (%d ventanas)"
                  % (lab, float(np.mean(dens_bins[k])), len(dens_bins[k])))
