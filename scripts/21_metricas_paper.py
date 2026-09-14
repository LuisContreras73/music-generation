"""Las metricas del paper de Music Transformer, aplicadas a nuestros modelos.

    python scripts/21_metricas_paper.py

Que metricas usa Huang et al. (2018, arXiv:1809.04281):

  1. **NLL de validacion/test**, que es su unica cifra cuantitativa. Se reporta
     por TOKEN. Nuestra lectura, que `docs/05_estado_del_arte.md` marca como NO
     confirmada en la fuente primaria, es que estan en nats (Tensor2Tensor
     reporta en nats). Nosotros registramos `val_nll` y `test_nll`, que son
     exactamente eso: media de la entropia cruzada por token, en nats.
  2. **Test de escucha por pares**: su evidencia de calidad NO es una metrica
     automatica, son humanos comparando dos fragmentos. Este script prepara la
     hoja para hacerlo (ver scripts/22_test_escucha.py si existe); aqui solo se
     deja constancia de que la parte cuantitativa del paper NO cubre calidad.

Por que importa distinguirlo: su NLL mide prediccion, igual que nuestro
bits/paso, y NINGUNA de las dos mide si la musica suena bien. Eso en el paper lo
resuelven con humanos. Nosotros lo sustituimos por gen_score, que es
distribucional, y ahi es donde aparece la disociacion que documenta este
informe.

AVISO sobre comparar cifras con el paper, que este script imprime siempre:
nuestro NLL y el suyo NO son la misma escala. Su vocabulario son 388 simbolos
con velocity y NOTE_OFF sobre MAESTRO; el nuestro son 155 sin velocity, sin
duracion y sin pedal, sobre otro corpus y otra rejilla temporal. Que a alguno de
nuestros modelos le salga un numero parecido al suyo es una coincidencia
aritmetica, no un empate.

Salida: reports/metricas_paper.csv y reports/figures/prediccion_vs_generacion.png
"""
from __future__ import annotations
import csv, math, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NLL_PAPER_MT = 1.84        # Music Transformer en MAESTRO, segun la Seccion A.2 (V=388)
NLL_PAPER_PAR = 1.82       # Perceiver AR en el mismo sitio, misma tokenizacion


def filas_metricas() -> list:
    out = []
    for d in sorted((ROOT / "experiments").iterdir()):
        p = d / "logs" / "metrics.csv"
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        ev = [r for r in rows if r.get("val_nll") not in (None, "", "nan")]
        te = [r for r in rows if r.get("split") == "test" and r.get("test_nll") not in (None, "", "nan")]
        if not ev:
            continue
        mejor = min(ev, key=lambda r: float(r["val_bpt"]))
        fila = dict(modelo=d.name,
                    val_nll_nats_token=round(float(mejor["val_nll"]), 4),
                    val_bits_token=round(float(mejor["val_nll"]) / math.log(2), 4),
                    val_bits_paso=round(float(mejor["val_bpt"]), 4),
                    val_ppl=round(float(mejor["val_ppl"]), 3) if mejor.get("val_ppl") else "")
        # tokens por paso: no todos los metrics.csv guardan los conteos, pero la
        # relacion bpt = nll * tok_por_paso / ln2 permite despejarlo
        try:
            fila["tokens_por_paso"] = round(
                float(mejor["val_n_tokens"]) / float(mejor["val_n_steps"]), 4)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            fila["tokens_por_paso"] = round(
                float(mejor["val_bpt"]) * math.log(2) / float(mejor["val_nll"]), 4)
        if te:
            # los metrics.csv de las primeras tandas no tienen todas las columnas
            t = te[-1]
            def _f(clave, alt=None):
                v = t.get(clave)
                if v not in (None, "", "nan"):
                    return round(float(v), 4)
                return alt
            fila.update(test_nll_nats_token=_f("test_nll"),
                        test_bits_token=_f("test_bits_per_token",
                                           round(float(t["test_nll"]) / math.log(2), 4)),
                        test_bits_paso=_f("test_bpt"),
                        test_token_acc=_f("test_token_acc"))
            fila = {k: v for k, v in fila.items() if v is not None}
        out.append(fila)
    return sorted(out, key=lambda r: r["val_nll_nats_token"])


def lee_resumen(carpeta: str) -> dict:
    p = ROOT / "reports" / carpeta / "resumen.csv"
    if not p.exists():
        return {}
    return {r["modelo"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def figura(filas, pareada, libre):
    """Prediccion frente a generacion, que es la tesis del informe."""
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("sin matplotlib: %s" % e); return
    modelos = [f["modelo"] for f in filas if f["modelo"] in pareada]
    if not modelos:
        print("no hay modelos con comparacion pareada; no se dibuja"); return
    nll = [float(next(f for f in filas if f["modelo"] == m)["val_nll_nats_token"]) for m in modelos]
    gp = [float(pareada[m]["media"]) for m in modelos]
    gl = [float(libre[m]["gen_score_del_lote"]) if m in libre else float("nan") for m in modelos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    ax1.scatter(nll, gp, s=90, c="#1b6ca8", zorder=3, label="prefijo del corpus (pareado)")
    ax1.scatter(nll, gl, s=90, c="#c1452e", marker="^", zorder=3, label="generacion libre")
    for x, y, y2, m in zip(nll, gp, gl, modelos):
        ax1.annotate(m, (x, y), fontsize=7.5, xytext=(4, 5), textcoords="offset points")
        if y2 == y2:
            ax1.plot([x, x], [y, y2], color="#999999", lw=.8, zorder=2)
    ax1.set_xlabel("NLL de validacion (nats/token) -- la metrica del paper\n<- mejor prediccion")
    ax1.set_ylabel("gen_score")
    ax1.set_title("Predecir mejor no es generar mejor", fontsize=11)
    ax1.grid(alpha=.3); ax1.legend(fontsize=8)

    y = np.arange(len(modelos))
    orden = np.argsort(gp)[::-1]
    ax2.barh(y - .2, [gp[i] for i in orden], height=.38, color="#1b6ca8", label="prefijo del corpus")
    ax2.barh(y + .2, [0 if gl[i] != gl[i] else gl[i] for i in orden], height=.38,
             color="#c1452e", label="generacion libre")
    ax2.set_yticks(y); ax2.set_yticklabels([modelos[i] for i in orden], fontsize=8.5)
    ax2.invert_yaxis(); ax2.set_xlabel("gen_score"); ax2.set_xlim(0, 100)
    ax2.set_title("El mismo modelo, dos escenarios", fontsize=11)
    ax2.grid(axis="x", alpha=.3); ax2.legend(fontsize=8)
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "prediccion_vs_generacion.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print("figura: %s" % out.relative_to(ROOT))


def main():
    filas = filas_metricas()
    if not filas:
        print("no hay metrics.csv utilizables"); return
    out = ROOT / "reports" / "metricas_paper.csv"
    campos = sorted({k for f in filas for k in f}, key=lambda k: (k != "modelo", k))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos); w.writeheader(); w.writerows(filas)

    print("METRICA DEL PAPER (Huang et al. 2018): NLL por token, en nats\n")
    print("%-36s %10s %10s %10s %10s" % ("modelo", "val NLL", "test NLL", "bits/token", "bits/paso"))
    for f in filas:
        print("%-36s %10.4f %10s %10.4f %10.4f"
              % (f["modelo"], f["val_nll_nats_token"],
                 ("%.4f" % f["test_nll_nats_token"]) if "test_nll_nats_token" in f else "-",
                 f["val_bits_token"], f["val_bits_paso"]))

    print("\nAVISO: el paper reporta %.2f (Music Transformer) y %.2f (Perceiver AR) en MAESTRO."
          % (NLL_PAPER_MT, NLL_PAPER_PAR))
    cerca = [f for f in filas if "test_nll_nats_token" in f
             and abs(f["test_nll_nats_token"] - NLL_PAPER_MT) < 0.05]
    for f in cerca:
        print("      `%s` da %.4f, que se PARECE a %.2f. Es una coincidencia aritmetica:"
              % (f["modelo"], f["test_nll_nats_token"], NLL_PAPER_MT))
    print("      su vocabulario son 388 simbolos CON velocity y NOTE_OFF sobre MAESTRO; el")
    print("      nuestro son 155 SIN velocity, duracion ni pedal, sobre otro corpus y otra")
    print("      rejilla. Una NLL por token depende del vocabulario y del dato: no se comparan.")
    print("      Ver docs/05_estado_del_arte.md, punto (3).")
    print("\ntabla: %s" % out.relative_to(ROOT))

    figura(filas, lee_resumen("comparacion_pareada"), lee_resumen("generacion_libre"))


if __name__ == "__main__":
    main()
