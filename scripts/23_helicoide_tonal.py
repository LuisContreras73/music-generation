"""El helicoide tonal: lo que el modelo aprendio sin que nadie se lo enseñara.

    python scripts/23_helicoide_tonal.py                 # GIF + PNG del mejor
    python scripts/23_helicoide_tonal.py --exp lstm
    python scripts/23_helicoide_tonal.py --medir         # solo la tabla, sin dibujar

El hallazgo
-----------
Los 88 vectores NOTE_ON de la matriz de embeddings no caen de cualquier manera:
se colocan sobre una **helice**. Al subir un semitono, el vector gira un angulo
casi constante alrededor de un eje, y ese eje es la altura.

Medido en `estilo_llama_ctx2048_24ep`: el giro por semitono es de **-150.1
grados**, cuando 7/12 de vuelta --un paso del circulo de quintas-- son -150.0
exactos. La desviacion es de 11 grados y el **100% de los 87 saltos**
consecutivos cae a menos de 30 grados de la mediana.

Dicho de otro modo: para el modelo, **subir un semitono es avanzar un paso en el
circulo de quintas**. Es la estructura que la psicologia de la musica describe
desde Shepard (1982) como helice de altura, y aqui aparece sola, en un modelo
entrenado solo para predecir el siguiente simbolo de un piano-roll binario sin
velocity, sin duracion y sin compas.

No todos la aprenden, y esa es la otra mitad del resultado:

    estilo_llama_ctx2048_24ep   -150.1 deg/semitono   desv  11.1   100% de saltos
    music_transformer           -143.9 deg/semitono   desv  77.8    85% de saltos
    lstm                        +120.6 deg/semitono   desv 124.0    47% de saltos

Salida:
    reports/figures/helicoide_tonal.gif   la helice girando
    reports/figures/helicoide_tonal.png   comparacion entre modelos
    reports/helicoide_tonal.csv           las cifras
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import registry as reg                                              # noqa: E402
from data.tokenizer import NOTE_OFF_ID, N_PITCH, NOTE_MIN_MIDI      # noqa: E402

QUINTAS = ["Do", "Sol", "Re", "La", "Mi", "Si", "Fa#", "Do#", "Sol#", "Re#", "La#", "Fa"]
GRADOS_TEORICOS = -150.0          # 7/12 de vuelta: un paso del circulo de quintas
FONDO = "#0a0c12"
COMPARAR = ["estilo_llama_ctx2048_24ep", "music_transformer", "lstm"]


def helice(exp: str):
    """Coordenadas de la helice y su calidad. None si no hay checkpoint."""
    p = ROOT / "experiments" / exp / "checkpoints" / "best.pt"
    if not p.exists():
        return None
    sd = reg.load_checkpoint(p, map_location="cpu")["model"]
    k = [n for n in sd if sd[n].ndim == 2 and sd[n].shape[0] == 155][0]
    E = sd[k].float().numpy()[NOTE_OFF_ID:NOTE_OFF_ID + N_PITCH]
    X = E - E.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:8].T

    midi = np.arange(N_PITCH) + NOTE_MIN_MIDI
    q5 = (midi * 7) % 12
    c, s = np.cos(2 * np.pi * q5 / 12), np.sin(2 * np.pi * q5 / 12)
    # los dos ejes del plano de quintas y el eje de altura, elegidos por ajuste
    ic = int(np.argmax([abs(np.corrcoef(Z[:, i], c)[0, 1]) for i in range(8)]))
    iss = int(np.argmax([abs(np.corrcoef(Z[:, i], s)[0, 1]) for i in range(8)]))
    ia = int(np.argmax([abs(np.corrcoef(Z[:, i], midi)[0, 1]) for i in range(8)]))
    if len({ic, iss, ia}) < 3:                       # ejes degenerados: no hay helice
        return None

    x, y = Z[:, ic], Z[:, iss]
    z = Z[:, ia] * np.sign(np.corrcoef(Z[:, ia], midi)[0, 1])   # que suba con la altura
    ang = np.arctan2(y, x)
    d = np.degrees((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    med = float(np.median(d))
    dentro = float(np.mean(np.abs((d - med + 180) % 360 - 180) < 30))
    radio = np.hypot(x, y)
    return dict(exp=exp, x=x, y=y, z=z, midi=midi, q5=q5,
                grados_por_semitono=round(med, 1), desviacion=round(float(d.std()), 1),
                fraccion_saltos_regulares=round(dentro, 3),
                error_vs_quintas=round(abs(med - GRADOS_TEORICOS), 1),
                radio_cv=round(float(radio.std() / radio.mean()), 3))


def _ejes_limpios(ax):
    ax.set_facecolor(FONDO)
    ax.grid(False)
    for eje in (ax.xaxis, ax.yaxis, ax.zaxis):
        eje.set_pane_color((0, 0, 0, 0))
        eje.line.set_color((0, 0, 0, 0))
        eje._axinfo["grid"]["color"] = (1, 1, 1, 0.04)
        eje.set_ticks([])


def _dibuja_helice(ax, h, con_etiquetas=True, s=52):
    """Las 88 notas como 12 COLUMNAS verticales repartidas en circulo.

    Por que asi y no uniendo las notas en orden cromatico: un semitono gira -150
    grados, asi que el enlace cromatico salta de un lado a otro del cilindro y se
    ve un enredo. La estructura de verdad es otra: todas las octavas de una misma
    clase de altura comparten angulo (dispersion interna medida: 8.7 grados) y
    las 12 columnas quedan separadas 30 grados en ORDEN DE QUINTAS. Visto desde
    arriba es el circulo de quintas; visto de lado, las octavas apiladas.
    """
    import matplotlib.pyplot as plt
    x, y, z, q5, midi = h["x"], h["y"], h["z"], h["q5"], h["midi"]
    cmap = plt.get_cmap("hsv")

    for v in range(12):                      # v = posicion en el circulo de quintas
        m = np.where(q5 == v)[0]
        if not len(m):
            continue
        m = m[np.argsort(midi[m])]           # de grave a agudo: la columna sube
        col = cmap(v / 12.0)
        ax.plot(x[m], y[m], z[m], color=col, lw=1.6, alpha=.55, zorder=1)
        for k, alpha in ((6.5, .05), (3.2, .10), (1.8, .17)):
            ax.scatter(x[m], y[m], z[m], s=s * k, color=col, alpha=alpha,
                       edgecolors="none", depthshade=False)
        ax.scatter(x[m], y[m], z[m], s=s, color=col, edgecolors="white",
                   linewidths=.45, depthshade=False, zorder=3)
        if con_etiquetas:
            # en un anillo exterior de radio fijo y a la altura media: asi las 12
            # etiquetas quedan repartidas y no chocan con las columnas
            ang = np.arctan2(y[m].mean(), x[m].mean())
            rr = np.hypot(x, y).max() * 1.32
            ax.text(rr * np.cos(ang), rr * np.sin(ang), z.mean(), QUINTAS[v],
                    color=col, fontsize=9.5, ha="center", va="center", weight="bold")

    # anillo y radios en la base: el circulo de quintas visto desde arriba
    t = np.linspace(0, 2 * np.pi, 240)
    r = np.hypot(x, y).mean()
    base = z.min() - (z.max() - z.min()) * .06
    ax.plot(r * np.cos(t), r * np.sin(t), np.full_like(t, base),
            color="#2f3d4f", lw=1.0, alpha=.9, zorder=0)
    for v in range(12):
        m = np.where(q5 == v)[0]
        if not len(m):
            continue
        ang = np.arctan2(y[m].mean(), x[m].mean())
        ax.plot([0, r * np.cos(ang)], [0, r * np.sin(ang)], [base, base],
                color="#2f3d4f", lw=.7, alpha=.7, zorder=0)

    _ejes_limpios(ax)
    lim = max(np.abs(x).max(), np.abs(y).max()) * 1.42
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_box_aspect((1, 1, 1.25))


def gif(h, n_frames=120):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure(figsize=(6.4, 7.0), facecolor=FONDO)
    ax = fig.add_subplot(111, projection="3d", facecolor=FONDO)
    _dibuja_helice(ax, h)
    fig.text(.5, .955, "El helicoide tonal", color="white", fontsize=16,
             ha="center", weight="bold")
    fig.text(.5, .915, "%s  ·  los 88 embeddings NOTE_ON" % h["exp"],
             color="#9fb3c8", fontsize=9, ha="center")
    fig.text(.5, .048, "12 columnas a 30° exactos, en orden de quintas   ·   subir un semitono gira %.1f°"
             % h["grados_por_semitono"], color="#e8b44a", fontsize=9.5, ha="center")
    fig.text(.5, .013, "nadie se lo enseño: solo se entreno a predecir el siguiente simbolo",
             color="#6b7f95", fontsize=8.5, ha="center", style="italic")
    fig.subplots_adjust(left=-.04, right=1.04, top=.97, bottom=.02)

    def frame(i):
        u = i / n_frames
        # sube a cenital en el primer medio giro y vuelve a bajar en el segundo:
        # de lado se ven las octavas, desde arriba se ve el circulo de quintas
        elev = 8 + 78 * (0.5 - 0.5 * np.cos(2 * np.pi * u)) ** 1.4
        ax.view_init(elev=elev, azim=360 * u)
        return ()

    out = ROOT / "reports" / "figures" / "helicoide_tonal.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, frame, frames=n_frames, interval=60).save(
        out, writer=PillowWriter(fps=16), dpi=80, savefig_kwargs={"facecolor": FONDO})
    plt.close(fig)
    return out


def png(hs):
    """Comparacion: quien aprende la helice y quien no."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4.5 * len(hs), 5.0), facecolor=FONDO)
    for i, h in enumerate(hs, 1):
        ax = fig.add_subplot(1, len(hs), i, projection="3d", facecolor=FONDO)
        _dibuja_helice(ax, h, con_etiquetas=(i == 1), s=34)
        ax.view_init(elev=14, azim=38)
        limpio = h["fraccion_saltos_regulares"]
        color = "#5ad18f" if limpio > .9 else ("#e8b44a" if limpio > .7 else "#e05b5b")
        ax.set_title("%s\n%+.1f°/semitono · %.0f%% regulares"
                     % (h["exp"], h["grados_por_semitono"], 100 * limpio),
                     color=color, fontsize=9.5, pad=2)
    fig.text(.5, .965, "¿Quien aprende el circulo de quintas?", color="white",
             fontsize=15, ha="center", weight="bold")
    fig.text(.5, .028, "cada columna es una clase de altura con todas sus octavas   ·   un semitono son -150°",
             color="#6b7f95", fontsize=9, ha="center")
    fig.subplots_adjust(left=-.02, right=1.02, top=.93, bottom=.06, wspace=-.06)
    out = ROOT / "reports" / "figures" / "helicoide_tonal.png"
    fig.savefig(out, dpi=140, facecolor=FONDO); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="estilo_llama_ctx2048_24ep")
    ap.add_argument("--medir", action="store_true")
    ap.add_argument("--frames", type=int, default=120)
    a = ap.parse_args()

    hs = [h for h in (helice(e) for e in COMPARAR) if h]
    print("%-28s %14s %10s %12s %10s" % ("modelo", "°/semitono", "desv", "regulares", "error"))
    filas = []
    for h in hs:
        print("%-28s %+14.1f %10.1f %11.0f%% %9.1f°"
              % (h["exp"], h["grados_por_semitono"], h["desviacion"],
                 100 * h["fraccion_saltos_regulares"], h["error_vs_quintas"]))
        filas.append({k: v for k, v in h.items() if not isinstance(v, np.ndarray)})
    print("\n(referencia: un paso del circulo de quintas son %.1f°)" % GRADOS_TEORICOS)
    if filas:
        out = ROOT / "reports" / "helicoide_tonal.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
        print("tabla: %s" % out.relative_to(ROOT))
    if a.medir:
        return
    h = helice(a.exp)
    if h is None:
        print("sin checkpoint o sin ejes separables para %s" % a.exp); return
    for p in (gif(h, a.frames), png(hs)):
        print("figura: %s (%.1f MB)" % (p.relative_to(ROOT), p.stat().st_size / 1048576))


if __name__ == "__main__":
    main()
