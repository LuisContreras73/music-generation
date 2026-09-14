"""El espacio de embeddings de los 155 tokens, en 3D y en movimiento.

    python scripts/22_espacio_embeddings.py                    # el modelo por defecto
    python scripts/22_espacio_embeddings.py --exp lstm
    python scripts/22_espacio_embeddings.py --todos            # tabla comparativa

Que NO es esto: un espacio latente tipo VAE. Los modelos de este laboratorio son
autorregresivos y no tienen variable latente (la excepcion es MELLE, que si lleva
un muestreo variacional). Lo que se dibuja aqui es la **matriz de embeddings**:
los 155 vectores que el modelo aprende, uno por simbolo del vocabulario. Es su
representacion interna del alfabeto musical, y nadie le dijo como organizarla.

Que se encuentra, y esta medido en `reports/espacio_embeddings.csv`:

  1. Un eje separa los NOTE_ON de los SHIFT: altura y tiempo viven en
     direcciones distintas.
  2. Una componente correlaciona con la DURACION del shift: |r| = 0.94 a 0.96 en
     los cinco modelos medidos.
  3. Otra correlaciona con la ALTURA de la nota: |r| = 0.82 a 0.92.
  4. Y una componente menor ordena las doce clases de altura por el **circulo de
     quintas**, |r| = 0.73 a 0.86, frente a 0.09-0.28 del orden cromatico. El
     modelo agrupa las notas como lo hace la armonia tonal, no como estan
     colocadas en el teclado.

El punto 4 se contrasta contra azar con 200 permutaciones de las filas de
embeddings, porque el estadistico busca el mejor de 8 componentes y eso lo infla
por construccion. El azar llega a 0.30 (percentil 95); los cinco modelos dan
p = 0.005, que es el minimo alcanzable con 200 permutaciones.

Salida:
    reports/figures/espacio_embeddings.gif   giro de 360 grados del espacio 3D
    reports/figures/espacio_embeddings.png   version estatica + circulo de quintas
    reports/espacio_embeddings.csv           las correlaciones, por modelo
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import registry as reg                                      # noqa: E402
from data.tokenizer import (NOTE_OFF_ID, SHIFT_OFF_ID, VOCAB_SIZE,  # noqa: E402
                            N_PITCH, NOTE_MIN_MIDI, MAX_SHIFT)

N_PERM = 200            # permutaciones del contraste contra azar

NOMBRES_PC = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
POR_DEFECTO = ["music_transformer", "estilo_llama_ctx2048_24ep", "lstm",
               "estilo_llama_24ep", "perceiver_ar"]


def embeddings(exp: str) -> np.ndarray | None:
    """Matriz [155, d] de embeddings de token del checkpoint de menor val_bpt."""
    p = ROOT / "experiments" / exp / "checkpoints" / "best.pt"
    if not p.exists():
        return None
    sd = reg.load_checkpoint(p, map_location="cpu")["model"]
    # se localiza por forma, no por nombre: cada arquitectura la llama distinto
    cand = [k for k in sd if sd[k].ndim == 2 and sd[k].shape[0] == VOCAB_SIZE]
    return sd[cand[0]].float().numpy() if cand else None


def pca(X: np.ndarray, k: int = 8):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T, (S ** 2 / (S ** 2).sum())[:k]


def analiza(E: np.ndarray) -> dict:
    """Correlaciones de cada componente con las magnitudes musicales."""
    notas = np.arange(NOTE_OFF_ID, NOTE_OFF_ID + N_PITCH)
    shifts = np.arange(SHIFT_OFF_ID, VOCAB_SIZE)
    Z, var = pca(E)
    altura = np.arange(N_PITCH)
    duracion = np.arange(1, MAX_SHIFT + 1)

    def mejor(ix, objetivo):
        rs = [abs(np.corrcoef(Z[ix, c], objetivo)[0, 1]) for c in range(Z.shape[1])]
        c = int(np.argmax(rs)); return c + 1, rs[c]

    c_dur, r_dur = mejor(shifts, duracion)
    c_alt, r_alt = mejor(notas, altura)

    # circularidad: se hace PCA SOLO sobre los NOTE_ON y se quita la tendencia
    # lineal de altura, porque si no la domina y tapa la estructura de 12
    Zn, _ = pca(E[notas], k=8)
    midi = np.arange(N_PITCH) + NOTE_MIN_MIDI
    pc, q5 = midi % 12, (midi * 7) % 12
    def circ(v):
        c, s = np.cos(2 * np.pi * v / 12), np.sin(2 * np.pi * v / 12)
        rs = [max(abs(np.corrcoef(Zn[:, i], c)[0, 1]),
                  abs(np.corrcoef(Zn[:, i], s)[0, 1])) for i in range(Zn.shape[1])]
        i = int(np.argmax(rs)); return i, rs[i]
    i_pc, r_pc = circ(pc)
    i_q5, r_q5 = circ(q5)

    # Contraste contra azar. Se busca el MEJOR de 8 componentes, asi que el
    # maximo esta inflado por construccion; sin un nulo, el 0.86 no significa
    # nada. Se permutan las 88 filas de embeddings (rompiendo la relacion entre
    # vector y altura, pero conservando la geometria de la nube) y se recalcula
    # el mismo estadistico. El p-valor es la fraccion de permutaciones que lo
    # igualan o superan.
    rng = np.random.default_rng(0)
    nulo = np.empty(N_PERM)
    for b in range(N_PERM):
        Zp, _ = pca(E[notas][rng.permutation(N_PITCH)], k=8)
        c, sn = np.cos(2 * np.pi * q5 / 12), np.sin(2 * np.pi * q5 / 12)
        nulo[b] = max(max(abs(np.corrcoef(Zp[:, i], c)[0, 1]),
                          abs(np.corrcoef(Zp[:, i], sn)[0, 1])) for i in range(Zp.shape[1]))
    p_val = float((np.sum(nulo >= r_q5) + 1) / (N_PERM + 1))
    sep = float(np.linalg.norm(Z[notas].mean(0) - Z[shifts].mean(0)))
    return dict(Z=Z, var=var, Zn=Zn, notas=notas, shifts=shifts,
                comp_duracion=c_dur, r_duracion=round(r_dur, 3),
                comp_altura=c_alt, r_altura=round(r_alt, 3),
                comp_cromatica=i_pc + 1, r_cromatica=round(r_pc, 3),
                comp_quintas=i_q5 + 1, r_quintas=round(r_q5, 3),
                nulo_quintas_p95=round(float(np.quantile(nulo, .95)), 3),
                p_valor_quintas=round(p_val, 4),
                separacion_note_shift=round(sep, 3))


def ejes_interpretables(a: dict):
    """Tres ejes con significado, en vez de PC1/PC2/PC3 a secas."""
    Z = a["Z"]
    cd, ca = a["comp_duracion"] - 1, a["comp_altura"] - 1
    resto = [c for c in range(Z.shape[1]) if c not in (cd, ca)]
    return Z[:, cd], Z[:, ca], Z[:, resto[0]]


def figuras(exp: str, a: dict, n_frames: int = 90):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    x, y, z = ejes_interpretables(a)
    notas, shifts = a["notas"], a["shifts"]
    midi = np.arange(N_PITCH) + NOTE_MIN_MIDI
    out_dir = ROOT / "reports" / "figures"; out_dir.mkdir(parents=True, exist_ok=True)

    def dibuja(ax):
        ax.scatter(x[notas], y[notas], z[notas], c=midi, cmap="viridis", s=38,
                   depthshade=False, edgecolors="none", label="NOTE_ON (88, color = altura)")
        ax.scatter(x[shifts], y[shifts], z[shifts], c=np.arange(1, MAX_SHIFT + 1),
                   cmap="autumn", s=38, marker="^", depthshade=False, edgecolors="none",
                   label="SHIFT (64, color = duracion)")
        ax.set_xlabel("eje de DURACION\n|r| = %.2f" % a["r_duracion"], fontsize=8, labelpad=-4)
        ax.set_ylabel("eje de ALTURA\n|r| = %.2f" % a["r_altura"], fontsize=8, labelpad=-4)
        ax.set_zlabel("3a componente", fontsize=8, labelpad=-4)
        for f in (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels):
            f([])

    # --- GIF: una vuelta completa ---
    fig = plt.figure(figsize=(6.4, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    dibuja(ax)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=.9)
    ax.set_title("Espacio de embeddings de %s\nel modelo separa altura y tiempo sin que nadie se lo diga"
                 % exp, fontsize=10)
    fig.tight_layout()

    def frame(i):
        ax.view_init(elev=18 + 8 * np.sin(2 * np.pi * i / n_frames), azim=360 * i / n_frames)
        return ()

    gif = out_dir / "espacio_embeddings.gif"
    FuncAnimation(fig, frame, frames=n_frames, interval=70, blit=False).save(
        gif, writer=PillowWriter(fps=14), dpi=78)
    plt.close(fig)

    # --- PNG: la vista 3D y el circulo de quintas ---
    fig = plt.figure(figsize=(12.5, 5.6))
    ax1 = fig.add_subplot(121, projection="3d")
    dibuja(ax1); ax1.view_init(elev=20, azim=42)
    ax1.legend(loc="upper left", fontsize=7.5)
    ax1.set_title("Los 155 tokens: altura y tiempo en ejes distintos", fontsize=10.5)

    # las dos componentes con mas estructura circular, sobre los NOTE_ON
    Zn = a["Zn"]; q5 = (midi * 7) % 12
    c, s = np.cos(2 * np.pi * q5 / 12), np.sin(2 * np.pi * q5 / 12)
    rc = [abs(np.corrcoef(Zn[:, i], c)[0, 1]) for i in range(Zn.shape[1])]
    rs = [abs(np.corrcoef(Zn[:, i], s)[0, 1]) for i in range(Zn.shape[1])]
    i, j = int(np.argmax(rc)), int(np.argmax(rs))
    if i == j:
        j = int(np.argsort(rs)[-2])
    ax2 = fig.add_subplot(122)
    sc = ax2.scatter(Zn[:, i], Zn[:, j], c=q5, cmap="hsv", s=58, edgecolors="k", linewidths=.4)
    for pcid in range(12):
        m = q5 == pcid
        if m.any():
            ax2.annotate(NOMBRES_PC[(pcid * 7) % 12],
                         (Zn[m, i].mean(), Zn[m, j].mean()), fontsize=9, weight="bold",
                         ha="center", va="center",
                         bbox=dict(fc="white", ec="none", alpha=.75, pad=1.2))
    ax2.set_title("Las 12 clases de altura se ordenan por el CIRCULO DE QUINTAS\n"
                  "|r| = %.2f frente a %.2f del orden cromatico"
                  % (a["r_quintas"], a["r_cromatica"]), fontsize=10.5)
    ax2.set_xlabel("componente %d de los NOTE_ON" % (i + 1))
    ax2.set_ylabel("componente %d" % (j + 1))
    ax2.grid(alpha=.3)
    fig.colorbar(sc, ax=ax2, label="posicion en el circulo de quintas", fraction=.046)
    fig.tight_layout()
    png = out_dir / "espacio_embeddings.png"
    fig.savefig(png, dpi=135); plt.close(fig)
    return gif, png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="music_transformer")
    ap.add_argument("--todos", action="store_true", help="tabla comparativa de todos")
    ap.add_argument("--frames", type=int, default=90)
    a = ap.parse_args()

    exps = POR_DEFECTO if a.todos else [a.exp]
    filas = []
    print("%-28s %9s %9s %9s %9s %9s %8s"
          % ("modelo", "duracion", "altura", "quintas", "cromatica", "azar p95", "p-valor"))
    for e in exps:
        E = embeddings(e)
        if E is None:
            print("%-28s (sin checkpoint)" % e); continue
        d = analiza(E)
        print("%-28s %9.3f %9.3f %9.3f %9.3f %9.3f %8.4f"
              % (e, d["r_duracion"], d["r_altura"], d["r_quintas"], d["r_cromatica"],
                 d["nulo_quintas_p95"], d["p_valor_quintas"]))
        filas.append({k: v for k, v in d.items() if not isinstance(v, np.ndarray)} | {"modelo": e})

    if filas:
        out = ROOT / "reports" / "espacio_embeddings.csv"
        campos = ["modelo"] + [k for k in filas[0] if k != "modelo"]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=campos); w.writeheader(); w.writerows(filas)
        print("\ntabla: %s" % out.relative_to(ROOT))

    E = embeddings(a.exp)
    if E is not None:
        gif, png = figuras(a.exp, analiza(E), a.frames)
        for p in (gif, png):
            print("figura: %s (%.1f MB)" % (p.relative_to(ROOT), p.stat().st_size / 1048576))


if __name__ == "__main__":
    main()
