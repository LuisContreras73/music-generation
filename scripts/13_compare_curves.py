"""Curva de aprendizaje de TODOS los experimentos en una sola figura.

Eje x en EPOCAS EQUIVALENTES (tokens procesados / tamano del corpus), que es lo
unico comparable entre experimentos con distinto batch: 4900 pasos con batch 24
equivalen a 7300 con batch 16.
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TOK = 32_245_043
TRIVIAL = 4.0921

COLS = {"estilo_llama_24ep": "#1b6ca8", "estilo_llama_10ep": "#2e9e8f", "perceiver_ar": "#c1452e",
        "music_transformer": "#7a4fb5", "music_transformer_con_augmentacion": "#b58b2e", "lstm": "#4a8f2e",
        "ablacion_sin_atencion_relativa": "#999999", "tft": "#c96fa8", "music_transformer_ctx2048": "#5a7fa8",
        "melle": "#bbbbbb", "ablacion_melle_sin_flux": "#cccccc", "deep_lstm": "#8fbf5a"}

def series(name):
    import pandas as pd
    f = ROOT / "experiments" / name / "logs" / "metrics.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    tr = d[(d.split == "train") & d.tokens_seen.notna()]
    ev = d[(d.split == "eval") & d.val_bpt.notna()]
    if not len(tr) or not len(ev):
        return None
    tps = float(tr.tokens_seen.max()) / float(tr.step.max())
    return ev.step.values * tps / TOK, ev.val_bpt.values

names = sorted([p.name for p in (ROOT / "experiments").iterdir() if p.is_dir()])
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
ax, ax2 = axes

for n in names:
    s = series(n)
    if s is None:
        continue
    x, y = s
    c = COLS.get(n, "#aaaaaa")
    lw = 2.4 if n in ("estilo_llama_24ep", "perceiver_ar", "music_transformer") else 1.3
    for a in (ax, ax2):
        a.plot(x, y, marker="o", ms=3, lw=lw, color=c,
               label="%s (min %.3f)" % (n, y.min()), alpha=0.95 if lw > 2 else 0.75)

for a, tit, ylim in ((ax, "(a) todos los experimentos", (1.4, 4.3)),
                     (ax2, "(b) detalle de la zona baja", (1.5, 2.2))):
    a.axhline(TRIVIAL, color="#d33", ls="--", lw=1.1, alpha=0.7)
    a.set_xlabel("epocas equivalentes  (tokens procesados / corpus de 32.2 M)")
    a.set_ylabel("bits por paso de 50 ms (validacion)")
    a.set_title(tit, fontweight="bold", fontsize=11)
    a.grid(alpha=0.25); a.set_ylim(*ylim)
ax.annotate("modelo trivial i.i.d. = 4.0921", (0.35, TRIVIAL), textcoords="offset points",
            xytext=(4, 6), color="#d33", fontsize=8.5)
ax.legend(fontsize=7.2, loc="upper right", ncol=1, framealpha=0.9)
ax2.legend(fontsize=7.2, loc="upper right", framealpha=0.9)

fig.suptitle("Curvas de aprendizaje comparadas  |  menor es mejor",
             fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "reports" / "figures" / "curvas_comparadas.png"
fig.savefig(out, dpi=140); plt.close(fig)
print("escrito %s (%.0f KB)" % (out, out.stat().st_size / 1024))
