"""Figuras del laboratorio 1 (generacion de musica simbolica).

Contrato comun de todas las funciones publicas
----------------------------------------------
* reciben la ruta de salida de forma EXPLICITA (out_png),
* crean el directorio padre si no existe,
* cierran la figura con plt.close,
* devuelven la ruta del PNG como str,
* toleran datos parciales, vacios o con NaN sin lanzar excepcion: un
  experimento recien arrancado tiene 2 filas de CSV y ninguna generacion,
  y aun asi la figura se produce (con paneles que dicen "sin datos").

Detalles de estilo
------------------
Backend "Agg" (sin ventana ni display), DPI 140, fuente DejaVu Sans (viene con
matplotlib, no requiere internet), grid suave, paleta legible en escala de
grises tambien. Titulos y etiquetas en espanol SIN TILDES (consola cp1252).

Dependencias: numpy + matplotlib. Se apoya en src/metrics.py para el overlap
(OA) y el gen_score, con implementacion de reserva si ese modulo no se puede
importar (asi viz.py sigue siendo utilizable de forma aislada).
"""
from __future__ import annotations

import csv
import functools
import inspect
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")           # obligatorio ANTES de importar pyplot
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import Rectangle                         # noqa: E402

__all__ = [
    "plot_training_curves", "plot_pianoroll", "plot_pianoroll_grid",
    "plot_distribution_comparison", "plot_leaderboard",
    "plot_attention", "plot_vsn_weights", "plot_data_overview",
    "FAILURES",
]

# --------------------------------------------------------------------------
# constantes del dominio (duplicadas a proposito para no depender de imports)
# --------------------------------------------------------------------------
N_PITCH = 88
NOTE_MIN_MIDI = 21
STEP_SEC = 0.05
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

DPI = 140

# paleta: azul/naranja/verde/violeta, distinguibles tambien en gris
C_TRAIN = "#1f6fb2"
C_VAL = "#d1642a"
C_BEST = "#20794b"
C_GEN = "#6a3d9a"
C_REF = "#78838f"
C_LR = "#0f766e"
C_SPD = "#8a6d1d"
C_BAD = "#a4262c"
C_INK = "#2d3440"
C_GRID = "#b9c0c8"

# Registro de fallos del decorador _robust: la verificacion comprueba que este
# vacio. Si una figura falla, aqui queda (funcion, mensaje) y en stderr el
# traceback completo; la funcion devuelve igualmente un PNG con el aviso.
FAILURES: list = []

# Estilo del laboratorio. NO se aplica de forma global (importar viz no debe
# cambiar el aspecto de las figuras de otros modulos): el decorador _robust
# lo activa con plt.rc_context solo mientras se construye cada figura.
_RC = {
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.0,
    "axes.facecolor": "#fcfcfd",
    "axes.edgecolor": "#5b6472",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": C_GRID,
    "grid.alpha": 0.45,
    "grid.linewidth": 0.6,
    "legend.frameon": True,
    "legend.framealpha": 0.85,
    "legend.fontsize": 8.0,
    "legend.borderpad": 0.35,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "xtick.color": C_INK,
    "ytick.color": C_INK,
    "text.color": C_INK,
    "axes.labelcolor": C_INK,
    "lines.linewidth": 1.4,
}

# --------------------------------------------------------------------------
# metricas (import tolerante)
# --------------------------------------------------------------------------
#: pesos de reserva de los componentes del gen_score. Replica exacta de
#: metrics.FALLBACK_WEIGHTS (incluido harm_hist, el de mayor peso): si el
#: import de metrics funciona se sustituye por el objeto real.
_HIST_W = dict(pitch_hist=0.75, pitch_class=1.0, poly_hist=1.0,
               ioi_hist=1.0, interval_hist=1.25, harm_hist=2.0)


def _fallback_overlap(p, q) -> float:
    """OA de reserva, identico a metrics.overlap."""
    def _n(h):
        h = np.asarray(h, np.float64)
        s = h.sum()
        return h / s if s > 0 else np.full_like(h, 1.0 / max(len(h), 1))
    p, q = _n(p), _n(q)
    n = min(len(p), len(q))
    return float(np.minimum(p[:n], q[:n]).sum())


# CADA import va en su PROPIO try: si uno falla no debe arrastrar a los otros.
# (Con un unico try, un nombre inexistente dejaba _gen_score=None en silencio y
# el panel (f) no mostraba nunca el gen_score.)
_overlap, _gen_score = _fallback_overlap, None
for _base in ("", str(Path(__file__).resolve().parent)):
    if _base and _base not in sys.path:
        sys.path.insert(0, _base)               # viz.py ejecutado suelto
    try:
        from metrics import overlap as _overlap                  # noqa: F811
        from metrics import gen_score as _gen_score              # noqa: F811
    except Exception:                           # pragma: no cover
        continue
    break
del _base
try:                                            # nombre real en metrics.py
    from metrics import FALLBACK_WEIGHTS as _HIST_W              # noqa: F811
except Exception:                               # pragma: no cover
    pass
try:                                            # componentes que puntuan
    from metrics import HIST_KEYS as _HIST_KEYS
except Exception:                               # pragma: no cover
    _HIST_KEYS = tuple(_HIST_W)


def _calib():
    """Calibracion (floor/ceiling por componente) usada por gen_score.

    Sin ella metrics.gen_score cae a pesos fijos y NO discrimina: el score de
    la figura no coincidiria con el del CSV ni con el del leaderboard, que se
    calculan con calibracion (evaluate.evaluate_generations). Se lee de forma
    perezosa y solo se memoiza el EXITO, para que un calibration.json creado
    despues de importar viz (o despues de la primera figura) se recoja igual.
    """
    if _calib.cache is not _MISS and _calib.cache is not None:
        return _calib.cache
    d = None
    try:
        from config import PROC as _proc
        p = Path(_proc) / "calibration.json"
    except Exception:                           # pragma: no cover
        p = Path(__file__).resolve().parents[1] / "data" / "processed" / "calibration.json"
    try:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
            if isinstance(j, dict) and "floor" in j and "ceiling" in j:
                d = j
    except Exception:                           # pragma: no cover
        d = None
    _calib.cache = d
    return d


_MISS = object()
_calib.cache = _MISS

# --------------------------------------------------------------------------
# utilidades de figura
# --------------------------------------------------------------------------


def _prep(out_png) -> Path:
    p = Path(out_png)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _finish(fig, out_png) -> str:
    """Guarda, cierra y devuelve la ruta (str). Punto unico de salida."""
    p = _prep(out_png)
    try:
        fig.savefig(p, dpi=DPI)
    finally:
        plt.close(fig)
    return str(p)


def _msg(ax, text: str) -> None:
    """Convierte un eje en un cartel de 'sin datos'."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            fontsize=8.5, color="#6b7280", linespacing=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_alpha(0.25)


def _blank(ax) -> None:
    """Deja el eje limpio para escribir texto libre encima."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_alpha(0.25)


def _box(ax, text: str, loc: str = "tl", fontsize: float = 7.4) -> None:
    """Caja de anotaciones en una esquina del eje."""
    xy = dict(tl=(0.008, 0.985), tr=(0.992, 0.985),
              bl=(0.008, 0.015), br=(0.992, 0.015))[loc]
    ha = "left" if loc[1] == "l" else "right"
    va = "top" if loc[0] == "t" else "bottom"
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fontsize, linespacing=1.35, zorder=6,
            bbox=dict(boxstyle="round,pad=0.32", fc="white",
                      ec="#9aa3ad", lw=0.6, alpha=0.9))


def _robust(fn):
    """Ninguna figura debe tumbar el entrenamiento: captura y deja constancia.

    Si el cuerpo falla, se registra en FAILURES, se imprime el traceback en
    stderr y se guarda un PNG con el aviso, devolviendo su ruta.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        opened = set(plt.get_fignums())
        try:
            with plt.rc_context(_RC):        # estilo local, sin efectos globales
                return fn(*args, **kwargs)
        except Exception as exc:                                # noqa: BLE001
            traceback.print_exc()
            FAILURES.append((fn.__name__, repr(exc)))
            for num in set(plt.get_fignums()) - opened:
                plt.close(num)               # no dejar figuras a medias abiertas
            out = kwargs.get("out_png")
            if out is None:
                try:
                    b = inspect.signature(fn).bind_partial(*args, **kwargs)
                    out = b.arguments.get("out_png")
                except Exception:
                    out = None
            if out is None:
                return ""
            # el PNG de aviso tambien puede fallar (ruta invalida, disco
            # lleno): ni asi debe propagar la excepcion ni dejar la figura
            # abierta, porque el llamador es el bucle de entrenamiento.
            fig = None
            try:
                with plt.rc_context(_RC):
                    fig, ax = plt.subplots(figsize=(9.0, 5.0))
                    _msg(ax, "No se pudo generar la figura\n" + fn.__name__ +
                         "\n" + str(exc)[:300])
                    ax.set_title("Error al graficar")
                    return _finish(fig, out)
            except Exception as exc2:                           # noqa: BLE001
                traceback.print_exc()
                FAILURES.append((fn.__name__ + ":aviso", repr(exc2)))
                return ""
            finally:
                if fig is not None:
                    plt.close(fig)          # idempotente: _finish ya la cerro
    return wrapper


def _ann_offset(x, xs, up=True):
    """Coloca el texto de una anotacion al lado del punto, hacia dentro.

    Devuelve (dx, dy, ha) en puntos tipograficos segun donde caiga x dentro
    del rango xs: si el punto esta en el tercio derecho el texto va a la
    izquierda y viceversa, para que la flecha no cruce el panel entero.
    """
    xs = np.asarray(xs, float)
    if xs.size < 2 or not np.isfinite(xs).any():
        return 8.0, (14.0 if up else -14.0), "left"
    lo, hi = float(np.nanmin(xs)), float(np.nanmax(xs))
    frac = (float(x) - lo) / max(hi - lo, 1e-12)
    if frac > 0.62:
        return -10.0, (16.0 if up else -16.0), "right"
    return 10.0, (16.0 if up else -16.0), "left"


def _movavg(y, k: int):
    """Media movil con bordes replicados (misma longitud que y)."""
    y = np.asarray(y, float)
    if y.size == 0 or k <= 1:
        return y
    k = int(min(k, y.size))
    if k <= 1:
        return y
    pad = k // 2
    yp = np.r_[np.full(pad, y[0]), y, np.full(k - 1 - pad, y[-1])]
    return np.convolve(yp, np.ones(k) / k, mode="valid")


def _plot_trend(ax, x, y, color, label=None, k=9, min_n=14, lw=1.5):
    """Serie ruidosa: cruda tenue + media movil encima.

    Con pocos puntos (menos de min_n) la media movil enganaria, asi que se
    dibuja solo la serie cruda.
    """
    if x.size == 0:
        return
    if x.size >= min_n:
        ax.plot(x, y, color=color, alpha=0.30, lw=1.0)
        ax.plot(x, _movavg(y, k), color=color, lw=lw, label=label)
    else:
        ax.plot(x, y, color=color, lw=lw, marker="o", ms=2.6, label=label)


def _c_ticks(lo: int, hi: int):
    """Ticks en los Do (MIDI multiplo de 12) dentro de [lo,hi] -> (pos, labels)."""
    pos = [m for m in range(int(lo), int(hi) + 1) if m % 12 == 0]
    if len(pos) > 10:
        pos = pos[::2]
    return pos, [f"C{m // 12 - 1}" for m in pos]


def _norm(h):
    """Normaliza a suma 1 para dibujar; si esta vacio devuelve ceros."""
    h = np.asarray(h, np.float64).ravel()
    s = h.sum()
    return h / s if s > 0 else np.zeros_like(h)


def _f(v, default: float = float("nan")) -> float:
    """float() a prueba de None / texto / basura (json real trae nulls)."""
    if v is None:
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x


def _load_json(src) -> dict:
    if isinstance(src, dict):
        return src
    p = Path(src)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _dict_hist(d, kmax: int | None = None):
    """dict {clave->conteo} o lista -> (claves int ordenadas, valores float)."""
    if d is None:
        return np.array([]), np.array([])
    if isinstance(d, dict):
        pairs = []
        for k, v in d.items():
            try:
                pairs.append((int(float(k)), float(v)))
            except (TypeError, ValueError):
                continue
        pairs.sort()
    else:
        arr = np.asarray(d, float).ravel()
        pairs = [(i, float(v)) for i, v in enumerate(arr)]
    if kmax is not None:
        pairs = [p for p in pairs if p[0] <= kmax]
    if not pairs:
        return np.array([]), np.array([])
    ks = np.array([p[0] for p in pairs], float)
    vs = np.array([p[1] for p in pairs], float)
    return ks, vs


def _log_bars(ax, x, y, color, label=None, width=0.9):
    """Barras con eje y logaritmico seguro (ignora ceros para el limite)."""
    y = np.asarray(y, float)
    ax.bar(x, np.maximum(y, 0.0), width=width, color=color, label=label,
           linewidth=0)
    pos = y[y > 0]
    if pos.size:
        ax.set_yscale("log")
        ax.set_ylim(max(pos.min() * 0.5, 1e-12), pos.max() * 2.5)


# --------------------------------------------------------------------------
# lectura de CSV tolerante (sin pandas: solo stdlib)
# --------------------------------------------------------------------------
def _fnum(s) -> float:
    """Convierte a float; celda vacia / basura -> NaN."""
    if s is None:
        return np.nan
    if isinstance(s, (int, float, np.floating, np.integer)):
        v = float(s)
        return v if np.isfinite(v) else np.nan
    t = str(s).strip().replace(",", ".")
    if t == "" or t.lower() in ("nan", "none", "null", "na", "n/a", "-", "--"):
        return np.nan
    try:
        v = float(t)
    except ValueError:
        return np.nan
    return v if np.isfinite(v) else np.nan


def _find_col(cols, names):
    low = {}
    for c in cols or ():
        low.setdefault(str(c).strip().lower(), c)
    for n in names:
        c = low.get(str(n).strip().lower())
        if c is not None:
            return c
    return None


class _Table:
    """CSV leido a memoria: columnas + filas como dicts de texto."""

    def __init__(self, cols, rows):
        self.cols = list(cols or [])
        self.rows = list(rows or [])

    def __len__(self):
        return len(self.rows)

    def has(self, *names) -> bool:
        return _find_col(self.cols, names) is not None

    def num(self, *names):
        c = _find_col(self.cols, names)
        if c is None or not self.rows:
            return np.full(len(self.rows), np.nan)
        return np.array([_fnum(r.get(c)) for r in self.rows], float)

    def txt(self, *names, default: str = ""):
        c = _find_col(self.cols, names)
        if c is None:
            return [default] * len(self.rows)
        out = []
        for r in self.rows:
            v = r.get(c)
            out.append(default if v is None else str(v).strip())
        return out

    def step(self):
        s = self.num("step", "global_step", "iter", "iteration", "it", "steps")
        if not np.isfinite(s).any():
            s = np.arange(len(self.rows), dtype=float)
        return s


def _read_table(path) -> _Table:
    """Acepta ruta a CSV, un _Table ya construido, un DataFrame o una lista
    de dicts (registry.py puede pasar el leaderboard ya cargado)."""
    if isinstance(path, _Table):
        return path
    if hasattr(path, "columns") and hasattr(path, "to_dict"):      # DataFrame
        try:
            return _Table([str(c) for c in path.columns],
                          list(path.to_dict("records")))
        except Exception:
            return _Table([], [])
    if isinstance(path, (list, tuple)) and path and isinstance(path[0], dict):
        cols = []
        for r in path:
            for k in r:
                if k not in cols:
                    cols.append(str(k))
        return _Table(cols, list(path))
    try:
        p = Path(path)
    except TypeError:
        return _Table([], [])
    if not p.exists() or p.is_dir():
        return _Table([], [])
    try:
        with open(p, "r", newline="", encoding="utf-8-sig", errors="replace") as f:
            rd = csv.DictReader(f)
            rows = [r for r in rd if any(
                (v is not None and str(v).strip() != "") for v in r.values())]
            cols = list(rd.fieldnames or [])
    except Exception:
        return _Table([], [])
    return _Table(cols, rows)


#: alias de la columna split: el trainer escribe "eval" donde otros ponen "val"
_SPLIT_ALIAS = {"train": ("train", "tr", "fit"),
                "val": ("val", "eval", "valid", "validation", "dev"),
                "test": ("test", "tst")}


def _split_mask(t: _Table, want: str):
    """Mascara de filas cuyo campo split corresponde a `want` (train/val/test)."""
    c = _find_col(t.cols, ("split", "phase", "stage", "mode", "subset"))
    if c is None:
        return None
    pref = _SPLIT_ALIAS.get(want, (want,))
    vals = [str(r.get(c, "")).strip().lower() for r in t.rows]
    return np.array([v.startswith(pref) for v in vals], bool)


def _xy(x, y, mask=None):
    """Empareja x,y quedandose con lo finito y ordenando por x."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    n = min(x.size, y.size)
    if n == 0:
        return np.array([]), np.array([])
    x, y = x[:n], y[:n]
    m = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        mk = np.asarray(mask, bool).ravel()
        m &= mk[:n] if mk.size >= n else np.r_[mk, np.zeros(n - mk.size, bool)]
    if not m.any():
        return np.array([]), np.array([])
    xs, ys = x[m], y[m]
    o = np.argsort(xs, kind="stable")
    return xs[o], ys[o]


def _series(t: _Table, explicit, generic=(), split=None):
    """Serie (step, valor) probando nombres explicitos y luego genericos."""
    step = t.step()
    y = t.num(*explicit)
    if np.isfinite(y).any():
        return _xy(step, y)
    if generic:
        y = t.num(*generic)
        if np.isfinite(y).any():
            return _xy(step, y, _split_mask(t, split) if split else None)
    return np.array([]), np.array([])


# ==========================================================================
# 1. curvas de entrenamiento
# ==========================================================================
@_robust
def plot_training_curves(csv_path, out_png, title=None) -> str:
    """Panel 2x3 con la historia de un experimento a partir de su CSV.

    Columnas esperadas (todas opcionales, se aceptan sinonimos y NaN):
      step, split, train_loss, val_loss, val_bpt, val_ppl, lr, gen_score,
      tokens_seen, time_s.
    Marca con linea vertical verde el step del mejor checkpoint (min val_bpt).
    """
    t = _read_table(csv_path)
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.0))
    ax_loss, ax_bpt, ax_ppl, ax_lr, ax_gen, ax_spd = axes.ravel()

    sub = f"{len(t)} filas" if len(t) else "CSV vacio o ausente"
    name = str(title) if title else Path(str(csv_path)).parent.name or "experimento"

    # --- series ---
    xtr, ytr = _series(t, ("train_loss", "loss_train", "tr_loss"),
                       ("loss", "nll", "ce"), split="train")
    xva, yva = _series(t, ("val_loss", "loss_val", "valid_loss", "validation_loss"),
                       ("loss", "nll", "ce"), split="val")
    xb, yb = _series(t, ("val_bpt", "bpt_val", "val_bits_per_timestep"),
                     ("bpt", "bits_per_timestep"), split="val")
    xp, yp = _series(t, ("val_ppl", "ppl_val", "val_perplexity"),
                     ("ppl", "perplexity"), split="val")
    ppl_derived = False
    if yp.size == 0 and yva.size:
        xp, yp = xva, np.exp(np.clip(yva, -20, 20))
        ppl_derived = True
    xlr, ylr = _series(t, ("lr", "learning_rate", "lr_now"))
    xg, yg = _series(t, ("gen_score", "genscore", "gen_score_mean"))

    # mejor checkpoint = min val_bpt (si no hay bpt, min val_loss)
    best_step = best_val = None
    best_kind = "val_bpt"
    if yb.size:
        i = int(np.argmin(yb))
        best_step, best_val = float(xb[i]), float(yb[i])
    elif yva.size:
        i = int(np.argmin(yva))
        best_step, best_val = float(xva[i]), float(yva[i])
        best_kind = "val_loss"

    # --- (a) perdidas ---
    if xtr.size or xva.size:
        if xtr.size:
            _plot_trend(ax_loss, xtr, ytr, C_TRAIN,
                        label=f"train (n={xtr.size})")
        if xva.size:
            ax_loss.plot(xva, yva, color=C_VAL, lw=1.5, marker="o", ms=3.0,
                         label=f"val (n={xva.size})")
        ax_loss.set_ylabel("perdida (nats/token)")
        ax_loss.legend(loc="upper right")
        fin = np.r_[ytr, yva]
        fin = fin[np.isfinite(fin)]
        if fin.size and fin.min() > 0 and fin.max() / max(fin.min(), 1e-9) > 30:
            ax_loss.set_yscale("log")
    else:
        _msg(ax_loss, "sin perdidas registradas")
    ax_loss.set_title("(a) Perdida train / val")

    # --- (b) bits por paso de validacion ---
    if xb.size:
        ax_bpt.plot(xb, yb, color=C_VAL, lw=1.6, marker="o", ms=3.2)
        i = int(np.argmin(yb))
        ax_bpt.scatter([xb[i]], [yb[i]], s=70, facecolor="none",
                       edgecolor=C_BEST, lw=1.6, zorder=5)
        ax_bpt.annotate(f"min {yb[i]:.4f} bits/paso\nstep {xb[i]:.0f}",
                        xy=(xb[i], yb[i]), xytext=_ann_offset(xb[i], xb)[:2],
                        textcoords="offset points", va="bottom",
                        ha=_ann_offset(xb[i], xb)[2],
                        fontsize=7.8, color=C_BEST,
                        arrowprops=dict(arrowstyle="->", color=C_BEST, lw=0.9,
                                        shrinkA=1.0, shrinkB=4.0))
        ax_bpt.set_ylabel("bits / paso de tiempo")
        if yb.size > 2:
            lo, hi = float(np.min(yb)), float(np.percentile(yb, 95))
            if hi > lo:
                ax_bpt.set_ylim(lo - 0.05 * (hi - lo), hi + 0.35 * (hi - lo))
    else:
        _msg(ax_bpt, "sin val_bpt todavia")
    ax_bpt.set_title("(b) Bits por paso (validacion)")

    # --- (c) perplejidad ---
    if xp.size:
        ax_ppl.plot(xp, yp, color=C_GEN, lw=1.6, marker="o", ms=3.0)
        pos = yp[np.isfinite(yp) & (yp > 0)]
        if pos.size and pos.max() / max(pos.min(), 1e-9) > 20:
            ax_ppl.set_yscale("log")
        j = int(np.argmin(yp))
        ax_ppl.annotate(f"min {yp[j]:.2f}", xy=(xp[j], yp[j]),
                        xytext=_ann_offset(xp[j], xp)[:2],
                        textcoords="offset points", va="bottom",
                        ha=_ann_offset(xp[j], xp)[2],
                        fontsize=7.8, color=C_GEN)
        ax_ppl.set_ylabel("perplejidad por token")
        if ppl_derived:
            _box(ax_ppl, "derivada: exp(val_loss)", loc="tr")
    else:
        _msg(ax_ppl, "sin perplejidad")
    ax_ppl.set_title("(c) Perplejidad de validacion")

    # --- (d) learning rate ---
    if xlr.size:
        ax_lr.plot(xlr, ylr, color=C_LR, lw=1.6)
        pos = ylr[ylr > 0]
        if pos.size and pos.max() / max(pos.min(), 1e-12) > 50:
            ax_lr.set_yscale("log")
        ax_lr.set_ylabel("learning rate")
        _box(ax_lr, f"max={np.nanmax(ylr):.2e}\nfinal={ylr[-1]:.2e}", loc="tr")
    else:
        _msg(ax_lr, "sin learning rate")
    ax_lr.set_title("(d) Learning rate")

    # --- (e) gen_score ---
    if xg.size:
        ax_gen.plot(xg, yg, color=C_GEN, lw=1.6, marker="s", ms=3.6)
        k = int(np.argmax(yg))
        ax_gen.scatter([xg[k]], [yg[k]], s=70, facecolor="none",
                       edgecolor=C_BEST, lw=1.6, zorder=5)
        ax_gen.annotate(f"max {yg[k]:.2f}\nstep {xg[k]:.0f}",
                        xy=(xg[k], yg[k]),
                        xytext=_ann_offset(xg[k], xg, up=False)[:2],
                        textcoords="offset points", va="top",
                        ha=_ann_offset(xg[k], xg, up=False)[2],
                        fontsize=7.8, color=C_BEST,
                        arrowprops=dict(arrowstyle="->", color=C_BEST, lw=0.9,
                                        shrinkA=1.0, shrinkB=4.0))
        ax_gen.set_ylabel("gen_score (0-100)")
        ax_gen.set_ylim(0, max(100.0, float(np.nanmax(yg)) * 1.15))
    else:
        _msg(ax_gen, "sin generaciones evaluadas\n(gen_score aparece cada gen_every)")
    ax_gen.set_title("(e) Calidad generativa")

    # --- (f) velocidad ---
    sx, sy, slab = np.array([]), np.array([]), ""
    for names, lab in ((("tokens_per_s", "tokens_s", "tok_per_s", "tps",
                         "throughput"), "tokens/s"),
                       (("it_per_s", "iters_per_s", "steps_per_s", "sps"),
                        "iteraciones/s"),
                       (("s_per_step", "sec_per_step", "step_time_s",
                         "time_per_step"), "s/step")):
        sx, sy = _series(t, names)
        if sy.size:
            slab = lab
            break
    if sy.size == 0:
        st = t.step()
        tok = t.num("tokens_seen", "tokens", "n_tokens", "tokens_total")
        tm = t.num("time_s", "time", "elapsed_s", "wall_s", "elapsed", "secs")
        m = np.isfinite(st) & np.isfinite(tm)
        if m.sum() >= 2:
            o = np.argsort(st[m], kind="stable")
            s_ok, t_ok = st[m][o], tm[m][o]
            k_ok = tok[m][o] if np.isfinite(tok[m]).all() else None
            dt = np.diff(t_ok)
            if k_ok is not None:
                dk = np.diff(k_ok)
                g = (dt > 0) & (dk > 0)
                if g.any():
                    sx, sy, slab = s_ok[1:][g], dk[g] / dt[g], "tokens/s"
            if sy.size == 0:
                ds = np.diff(s_ok)
                g = (dt > 0) & (ds > 0)
                if g.any():
                    sx, sy, slab = s_ok[1:][g], dt[g] / ds[g], "s/step"
    if sy.size:
        _plot_trend(ax_spd, sx, sy, C_SPD, lw=1.6)
        ax_spd.set_ylabel(slab)
        med = float(np.nanmedian(sy))
        ax_spd.axhline(med, color=C_INK, ls=":", lw=0.9, alpha=0.7)
        _box(ax_spd, f"mediana={med:,.1f} {slab}", loc="br")
    else:
        _msg(ax_spd, "sin tokens_seen / time_s\npara estimar velocidad")
    ax_spd.set_title("(f) Velocidad de entrenamiento")

    # --- comun ---
    # Eje superior en EPOCAS EQUIVALENTES = tokens procesados / tamano del corpus.
    # NO son epocas en sentido estricto: el muestreo es aleatorio CON REEMPLAZO,
    # no una pasada ordenada por el dataset. Medido: a 1.02 "epocas" el modelo
    # solo habia visto el 63.7% del corpus y 1149 piezas de 9544 seguian sin
    # tocarse. Es una medida de COMPUTO relativo al tamano del dataset, y sirve
    # para comparar experimentos con distinto batch (4900 pasos con batch 24
    # equivalen a 7300 con batch 16), que es para lo que esta el eje.
    TOK_TRAIN = 32_245_043
    tok_per_step = None
    try:
        if t.has("tokens_seen"):
            st = t.step()
            tk = t.num("tokens_seen")
            m = np.isfinite(st) & np.isfinite(tk) & (st > 0) & (tk > 0)
            if m.any():
                tok_per_step = float(np.nanmax(tk[m]) / np.nanmax(st[m]))
    except Exception:
        tok_per_step = None

    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xlabel("step")
        if best_step is not None and ax.has_data():
            ax.axvline(best_step, color=C_BEST, ls="--", lw=0.9, alpha=0.65,
                       zorder=1)
        if tok_per_step and ax.has_data():
            sec = ax.secondary_xaxis(
                "top",
                functions=(lambda x, k=tok_per_step: x * k / TOK_TRAIN,
                           lambda e, k=tok_per_step: e * TOK_TRAIN / k))
            sec.set_xlabel("epocas equivalentes (tokens / corpus)", fontsize=7.5, color="#5b6472")
            sec.tick_params(labelsize=7, colors="#5b6472")

    head = f"Curvas de entrenamiento: {name}"
    if best_step is not None:
        sub += f"   |   mejor checkpoint: step {best_step:.0f} ({best_kind}={best_val:.4f})"
    if tok_per_step:
        _st = t.step()
        _ep = float(np.nanmax(_st) * tok_per_step / TOK_TRAIN) if len(t) else 0.0
        sub += f"   |   {_ep:.2f} epocas equiv. ({tok_per_step:.0f} tokens/paso)"
    fig.suptitle(head, fontsize=12.5, fontweight="bold", y=0.995)
    fig.text(0.5, 0.958, sub, ha="center", fontsize=8.6, color="#5b6472")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return _finish(fig, out_png)


# ==========================================================================
# 2-3. piano-roll
# ==========================================================================
def _binarize(r):
    """Binariza tolerante: si llegan probabilidades usa umbral 0.5."""
    r = np.nan_to_num(np.asarray(r), nan=0.0, posinf=1.0, neginf=0.0)
    if r.dtype.kind == "f" and bool(((r > 1e-6) & (r < 1.0 - 1e-6)).any()):
        return (r >= 0.5).astype(np.float32)
    return (r > 0).astype(np.float32)


def _as_roll(roll):
    """Normaliza la entrada a (roll [T,P] binario float32, note_min).

    Acepta numpy, tensores de torch (incluso en GPU), [1,T,88] y [88,T].
    Devuelve (None, note_min) si la forma no es utilizable.
    """
    r = roll
    if hasattr(r, "detach"):                     # tensor de torch
        try:
            r = r.detach().to("cpu")
        except Exception:
            return None, NOTE_MIN_MIDI
    try:
        r = np.asarray(r)
    except Exception:
        # bfloat16 (lo que devuelve el muestreo en amp) no tiene equivalente en
        # numpy: np.asarray lanza TypeError y el roll se perdia como "forma
        # invalida". Se reintenta pasando por float32 dentro de torch.
        try:
            r = np.asarray(r.float())
        except Exception:
            return None, NOTE_MIN_MIDI
    if r.ndim == 3 and r.shape[0] == 1:
        r = r[0]
    if r.ndim != 2 or r.size == 0:
        return None, NOTE_MIN_MIDI
    if r.shape[1] != N_PITCH and r.shape[0] == N_PITCH:
        r = r.T                                  # llego transpuesto [88,T]
    if r.shape[1] == 128:
        return _binarize(r), 0                   # rango MIDI completo
    if r.shape[1] != N_PITCH:
        return None, NOTE_MIN_MIDI
    return _binarize(r), NOTE_MIN_MIDI


def _roll_stats(roll):
    """Estadisticos basicos sin depender de metrics (evita coste extra)."""
    per = roll.sum(1)
    act = per > 0
    T = int(roll.shape[0])
    return dict(T=T, n_onsets=int(per.sum()),
                density=float(per.sum() / max(T, 1)),
                mean_poly=float(per[act].mean()) if act.any() else 0.0,
                empty=float(1.0 - act.mean()) if T else 1.0,
                n_pitch=int((roll.sum(0) > 0).sum()))


def _draw_pianoroll(ax, roll, title=None, prime_steps=None,
                    step_sec=STEP_SEC, annotate=True, legend=True,
                    compact=False):
    """Dibuja onsets en un eje. Devuelve True si habia datos.

    prime_steps = numero de pasos INICIALES DE ESTE roll que son prefijo real
    (lo que devuelve generate.split_prime_continuation junto al roll completo).
    Si el valor no cae dentro de (0, T) no puede marcar ninguna costura: se
    ignora y se avisa en la caja, porque entonces el roll suele contener solo
    la continuacion y prime_steps refiere a la pieza fuente, no a este roll.

    compact=True comprime la caja de estadisticos a una linea (para rejillas,
    donde una caja de 3 lineas taparia el registro agudo).
    """
    r, note_min = _as_roll(roll)
    if r is None:
        _msg(ax, "roll vacio o con forma invalida\n(se espera [T,88] binario)")
        if title:
            ax.set_title(title)
        return False

    T, P = r.shape
    step_sec = float(step_sec) if step_sec else STEP_SEC
    dur = T * step_sec
    lo, hi = note_min, note_min + P - 1
    ps, ps_warn = None, ""
    if prime_steps is not None:
        try:
            ps = int(prime_steps)
        except (TypeError, ValueError):
            ps = None
        if ps is not None and not (0 < ps < T):
            ps_warn = f"prime_steps={ps} fuera de (0,{T}): sin costura que marcar"
            ps = None

    ti, pi = np.nonzero(r > 0)
    drew_labels = False
    if ti.size > 250_000:                        # demasiados puntos: imshow
        # el prefijo se distingue por color tambien aqui: se pintan dos
        # imagenes con mapas distintos en lugar de una sola en gris, para no
        # perder la costura justo en los rolls mas largos.
        if ps:
            ax.imshow(r[:ps].T, aspect="auto", origin="lower", cmap="Greys",
                      vmin=0, vmax=1, interpolation="nearest",
                      extent=(0.0, ps * step_sec, lo - 0.5, hi + 0.5))
            ax.imshow(r[ps:].T, aspect="auto", origin="lower", cmap="Purples",
                      vmin=0, vmax=1, interpolation="nearest",
                      extent=(ps * step_sec, dur, lo - 0.5, hi + 0.5))
        else:
            ax.imshow(r.T, aspect="auto", origin="lower", cmap="Greys",
                      vmin=0, vmax=1, interpolation="nearest",
                      extent=(0.0, dur, lo - 0.5, hi + 0.5))
    else:
        s = float(np.clip(9000.0 / max(T, 1), 0.5, 13.0))
        if ps:
            m = ti < ps
            ax.scatter(ti[m] * step_sec, pi[m] + note_min, s=s, marker="s",
                       c=C_REF, linewidths=0,
                       label=f"prefijo real ({ps} pasos)")
            ax.scatter(ti[~m] * step_sec, pi[~m] + note_min, s=s, marker="s",
                       c=C_GEN, linewidths=0, label="continuacion generada")
            drew_labels = True
        else:
            ax.scatter(ti * step_sec, pi + note_min, s=s, marker="s",
                       c=C_TRAIN, linewidths=0)

    if ps:
        ax.axvspan(0.0, ps * step_sec, color=C_REF, alpha=0.13, zorder=0)
        ax.axvline(ps * step_sec, color=C_BAD, ls="--", lw=1.3, zorder=4)
        # etiqueta pegada a la costura, abajo (arriba vive la caja de stats)
        ax.annotate("inicio de lo generado", xy=(ps * step_sec, 0.025),
                    xycoords=ax.get_xaxis_transform(), xytext=(5, 0),
                    textcoords="offset points", fontsize=7.2, color=C_BAD,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="square,pad=0.15", fc="white",
                              ec="none", alpha=0.7))
        if legend and drew_labels:
            ax.legend(loc="lower right", markerscale=2.2)

    tk, tl = _c_ticks(lo, hi)
    if tk:
        ax.set_yticks(tk)
        ax.set_yticklabels([f"{n} ({m})" for n, m in zip(tl, tk)])
        for m in tk:
            ax.axhline(m, color=C_GRID, lw=0.4, alpha=0.55, zorder=0)
    ax.set_xlim(0.0, max(dur, step_sec))
    ax.set_ylim(lo - 1.0, hi + 1.0)
    ax.set_xlabel("tiempo (s)")
    ax.set_ylabel("nota MIDI")
    ax.grid(axis="y", visible=False)

    if annotate:
        st = _roll_stats(r)
        if compact:
            txt = (f"{st['n_onsets']} onsets   dens {st['density']:.3f}/paso   "
                   f"poly {st['mean_poly']:.2f}   "
                   f"vacios {100 * st['empty']:.0f}%   notas {st['n_pitch']}")
        else:
            txt = (f"{st['T']} pasos = {dur:.1f} s   |   {st['n_onsets']} onsets\n"
                   f"densidad {st['density']:.3f} onsets/paso "
                   f"({st['n_onsets'] / max(dur, 1e-9):.2f} notas/s)\n"
                   f"polifonia media {st['mean_poly']:.2f}   "
                   f"pasos vacios {100 * st['empty']:.1f}%   "
                   f"notas distintas {st['n_pitch']}")
        if ps:
            a, b = _roll_stats(r[:ps]), _roll_stats(r[ps:])
            txt += (f"\nprefijo: dens {a['density']:.3f} poly {a['mean_poly']:.2f}"
                    f"   |   generado: dens {b['density']:.3f} poly {b['mean_poly']:.2f}")
        if ps_warn:
            txt += "\n" + ps_warn
        _box(ax, txt, loc="tl", fontsize=7.0 if compact else 7.4)
    if title:
        ax.set_title(str(title))
    return True


@_robust
def plot_pianoroll(roll, out_png, title=None, prime_steps=None,
                   step_sec=STEP_SEC) -> str:
    """Piano-roll de onsets [T,88]: eje X en segundos, eje Y en nota MIDI.

    Si prime_steps se indica, sombrea el prefijo real y traza una linea
    vertical donde arranca la continuacion generada (para ver la costura).
    """
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    _draw_pianoroll(ax, roll, title=None, prime_steps=prime_steps,
                    step_sec=step_sec)
    ax.set_title(str(title) if title else "Piano-roll de onsets")
    fig.tight_layout()
    return _finish(fig, out_png)


@_robust
def plot_pianoroll_grid(rolls, out_png, titles=None, prime_steps=None,
                        ncols=2, step_sec=STEP_SEC) -> str:
    """Rejilla con varias generaciones. prime_steps: int comun o lista."""
    if rolls is None:
        rolls = []
    if isinstance(rolls, np.ndarray) and rolls.ndim == 3:
        rolls = [rolls[i] for i in range(rolls.shape[0])]
    rolls = list(rolls)
    n = len(rolls)
    if n == 0:
        fig, ax = plt.subplots(figsize=(10.0, 4.5))
        _msg(ax, "sin generaciones que mostrar")
        ax.set_title("Muestras generadas")
        return _finish(fig, out_png)

    ncols = int(max(1, min(int(ncols) if ncols else 2, n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(7.3 * ncols, 3.15 * nrows))
    flat = axes.ravel()

    if prime_steps is None or isinstance(prime_steps, (int, np.integer, float)):
        prim = [prime_steps] * n
    else:
        prim = list(prime_steps) + [None] * max(0, n - len(list(prime_steps)))
    tits = list(titles) if titles is not None else []
    tits += [f"muestra {i + 1}" for i in range(len(tits), n)]

    for i in range(n):
        _draw_pianoroll(flat[i], rolls[i], title=str(tits[i]),
                        prime_steps=prim[i], step_sec=step_sec,
                        annotate=True, legend=(i == 0), compact=True)
    for j in range(n, flat.size):
        flat[j].set_visible(False)

    fig.suptitle(f"Continuaciones generadas ({n} muestras)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return _finish(fig, out_png)


# ==========================================================================
# 4. comparacion de distribuciones generado vs referencia
# ==========================================================================
def _pair_bars_step(ax, centers, gen, ref, width=0.92):
    """Referencia como barras grises + generado como perfil escalonado."""
    ref_n, gen_n = _norm(ref), _norm(gen)
    if ref_n.size:
        ax.bar(centers[:ref_n.size], ref_n, width=width, color=C_REF,
               alpha=0.55, linewidth=0, label="corpus real")
    if gen_n.size:
        c = centers[:gen_n.size]
        ax.step(c, gen_n, where="mid", color=C_GEN, lw=1.4, label="generado")
        ax.fill_between(c, 0, gen_n, step="mid", color=C_GEN, alpha=0.18)
    ax.set_ylabel("fraccion")


def _pair_bars_side(ax, centers, gen, ref, width=0.40):
    """Barras lado a lado (pocos bins)."""
    ref_n, gen_n = _norm(ref), _norm(gen)
    if ref_n.size:
        ax.bar(centers[:ref_n.size] - width / 2, ref_n, width=width,
               color=C_REF, alpha=0.85, linewidth=0, label="corpus real")
    if gen_n.size:
        ax.bar(centers[:gen_n.size] + width / 2, gen_n, width=width,
               color=C_GEN, alpha=0.9, linewidth=0, label="generado")
    ax.set_ylabel("fraccion")


@_robust
def plot_distribution_comparison(gen_agg, ref_agg, out_png, title=None) -> str:
    """Compara los agregados de metrics.aggregate_features (generado vs real).

    Panel 2x3: pitch (88), clase de pitch (12), polifonia, inter-onset
    interval (y log), intervalo melodico y resumen de OA + gen_score.
    """
    gen_agg = gen_agg if isinstance(gen_agg, dict) else {}
    ref_agg = ref_agg if isinstance(ref_agg, dict) else {}

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.0))
    ax_p, ax_pc, ax_poly, ax_ioi, ax_int, ax_oa = axes.ravel()

    def hist(d, k):
        v = d.get(k) if isinstance(d, dict) else None
        return np.asarray(v, float).ravel() if v is not None else np.array([])

    # OA por histograma (solo donde hay las dos series con masa).
    # Se recorren TODOS los componentes de metrics.HIST_KEYS, harm_hist
    # incluido: es el de mayor peso calibrado y antes no aparecia.
    oas = {}
    for k in _HIST_KEYS:
        g, r = hist(gen_agg, k), hist(ref_agg, k)
        if g.size and r.size and g.sum() > 0 and r.sum() > 0:
            try:
                oas[k] = float(_overlap(g, r))
            except Exception:
                oas[k] = float(_fallback_overlap(g, r))

    def ttl(base, k):
        return base + (f"  (OA={oas[k]:.3f})" if k in oas else "  (sin OA)")

    # --- (a) pitch ---
    g, r = hist(gen_agg, "pitch_hist"), hist(ref_agg, "pitch_hist")
    if g.size or r.size:
        centers = np.arange(max(g.size, r.size), dtype=float) + NOTE_MIN_MIDI
        _pair_bars_step(ax_p, centers, g, r)
        tk, tl = _c_ticks(NOTE_MIN_MIDI, NOTE_MIN_MIDI + max(g.size, r.size) - 1)
        ax_p.set_xticks(tk)
        ax_p.set_xticklabels(tl)
        ax_p.set_xlabel("nota (Do marcados)")
        ax_p.legend(loc="upper right")
    else:
        _msg(ax_p, "sin histograma de pitch")
    ax_p.set_title(ttl("(a) Distribucion de pitch", "pitch_hist"))

    # --- (b) clase de pitch ---
    g, r = hist(gen_agg, "pitch_class"), hist(ref_agg, "pitch_class")
    if g.size or r.size:
        centers = np.arange(12, dtype=float)
        _pair_bars_side(ax_pc, centers, g[:12], r[:12], width=0.40)
        ax_pc.set_xticks(centers)
        ax_pc.set_xticklabels(PC_NAMES)
        ax_pc.set_xlabel("clase de pitch")
        ax_pc.legend(loc="upper right")
    else:
        _msg(ax_pc, "sin clase de pitch")
    ax_pc.set_title(ttl("(b) Clase de pitch", "pitch_class"))

    # --- (c) polifonia ---
    g, r = hist(gen_agg, "poly_hist"), hist(ref_agg, "poly_hist")
    if g.size or r.size:
        centers = np.arange(max(g.size, r.size), dtype=float)
        _pair_bars_side(ax_poly, centers, g, r, width=0.40)
        ax_poly.set_xticks(centers[::max(1, len(centers) // 13)])
        ax_poly.set_xlabel("notas simultaneas en un paso activo")
        ax_poly.legend(loc="upper right")
        mg = gen_agg.get("mean_poly")
        mr = ref_agg.get("mean_poly")
        if mg is not None or mr is not None:
            _box(ax_poly, f"media gen {float(mg or 0):.2f}\n"
                          f"media real {float(mr or 0):.2f}", loc="br")
    else:
        _msg(ax_poly, "sin polifonia")
    ax_poly.set_title(ttl("(c) Polifonia", "poly_hist"))

    # --- (d) inter-onset interval ---
    g, r = hist(gen_agg, "ioi_hist"), hist(ref_agg, "ioi_hist")
    if g.size or r.size:
        centers = np.arange(max(g.size, r.size), dtype=float)
        gn, rn = _norm(g), _norm(r)
        if rn.size:
            ax_ioi.bar(centers[:rn.size], np.maximum(rn, 1e-12), width=0.92,
                       color=C_REF, alpha=0.55, linewidth=0, label="corpus real")
        if gn.size:
            ax_ioi.step(centers[:gn.size], np.maximum(gn, 1e-12), where="mid",
                        color=C_GEN, lw=1.4, label="generado")
        pos = np.r_[gn[gn > 0], rn[rn > 0]]
        if pos.size:
            ax_ioi.set_yscale("log")
            ax_ioi.set_ylim(max(pos.min() * 0.6, 1e-7), pos.max() * 2.2)
        ax_ioi.set_xlabel("pasos entre onsets consecutivos (50 ms/paso)")
        ax_ioi.set_ylabel("fraccion (log)")
        ax_ioi.legend(loc="upper right")
    else:
        _msg(ax_ioi, "sin inter-onset interval")
    ax_ioi.set_title(ttl("(d) Inter-onset interval", "ioi_hist"))

    # --- (e) intervalo melodico ---
    g, r = hist(gen_agg, "interval_hist"), hist(ref_agg, "interval_hist")
    if g.size or r.size:
        n = max(g.size, r.size)
        half = (n - 1) // 2
        centers = np.arange(n, dtype=float) - half
        _pair_bars_step(ax_int, centers, g, r)
        ax_int.set_xlabel("semitonos respecto a la nota aguda previa")
        ax_int.axvline(0, color=C_INK, lw=0.7, ls=":", alpha=0.7)
        ax_int.legend(loc="upper right")
    else:
        _msg(ax_int, "sin intervalos melodicos")
    ax_int.set_title(ttl("(e) Intervalo melodico", "interval_hist"))

    # --- (f) resumen OA + gen_score ---
    labels = {"pitch_hist": "pitch", "pitch_class": "clase\npitch",
              "poly_hist": "polifonia", "ioi_hist": "IOI",
              "interval_hist": "intervalo", "harm_hist": "armonico"}
    keys = [k for k in _HIST_KEYS if k in oas]

    # El score se calcula CON calibracion, igual que evaluate.evaluate_generations:
    # sin ella metrics.gen_score usa pesos fijos y da otro numero (medido: 68.3
    # frente a 54.5 en la misma generacion), asi que la figura contradiria al CSV.
    res, cal = None, _calib()
    if _gen_score is not None:
        try:
            res = _gen_score(gen_agg, ref_agg, cal)
        except Exception:
            try:
                res = _gen_score(gen_agg, ref_agg)
            except Exception:
                res = None

    # pesos REALMENTE en vigor: con calibracion los devuelve gen_score en w_*
    # (derivados del poder discriminativo medido); sin ella, _HIST_W normalizado.
    wshow, wsrc = {}, "reserva"
    if res:
        wshow = {k[2:]: float(v) for k, v in res.items()
                 if k.startswith("w_") and k[2:] in _HIST_KEYS}
        if wshow:
            wsrc = "calibrado"
    if not wshow:
        tot = float(sum(_HIST_W.get(k, 0.0) for k in _HIST_KEYS)) or 1.0
        wshow = {k: _HIST_W.get(k, 0.0) / tot for k in _HIST_KEYS}

    if keys:
        vals = np.array([oas[k] for k in keys], float)
        cols = [C_BEST if v >= 0.80 else (C_SPD if v >= 0.60 else C_BAD)
                for v in vals]
        xs = np.arange(len(keys), dtype=float)
        ax_oa.bar(xs, vals, width=0.62, color=cols, linewidth=0)
        for x, v, k in zip(xs, vals, keys):
            ax_oa.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.8)
            w = wshow.get(k)
            ax_oa.text(x, 0.02, "w=0*" if not w else f"w={w:.2f}", ha="center",
                       fontsize=6.8, color="white" if v > 0.12 else C_INK)
        ax_oa.set_xticks(xs)
        ax_oa.set_xticklabels([labels.get(k, k) for k in keys], fontsize=7.4)
        ax_oa.set_ylim(0, 1.58)          # hueco superior para la ficha
        ax_oa.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_oa.set_ylabel("Overlapping Area (1 = identico)")
        ax_oa.set_xlabel(f"w = peso {wsrc}"
                         + ("  (0* = descartado)"
                            if any(not wshow.get(k) for k in keys) else ""))

        info = []
        if res and "oa_weighted" in res:
            ax_oa.axhline(res["oa_weighted"], color=C_INK, ls="--", lw=0.9,
                          alpha=0.75)
            ax_oa.text(len(keys) - 0.45, res["oa_weighted"] + 0.02,
                       f"OA ponderado {res['oa_weighted']:.3f}", ha="right",
                       fontsize=7.4, color=C_INK, zorder=7,
                       bbox=dict(boxstyle="square,pad=0.18", fc="white",
                                 ec="none", alpha=0.85))
            info += [f"gen_score = {res['gen_score']:.2f} / 100"
                     + ("  (calibrado)" if res.get("calibrated")
                        else "  (SIN calibrar)"),
                     f"OA ponderado  {res['oa_weighted']:.4f}",
                     f"penal densidad {res['pen_density']:.3f}"
                     f"  (ratio {res['density_ratio']:.2f})",
                     f"penal bucle   {res['pen_loop']:.3f}"]
        else:
            info += [f"OA medio {vals.mean():.4f}",
                     "gen_score no disponible"]
        for tag, key, fmt in (("densidad", "density", "{:.4f}"),
                              ("polifonia", "mean_poly", "{:.2f}"),
                              ("pasos vacios", "empty_ratio", "{:.3f}"),
                              ("repeat8", "repeat8", "{:.4f}")):
            gv, rv = gen_agg.get(key), ref_agg.get(key)
            if gv is not None or rv is not None:
                info.append(f"{tag}: gen " + fmt.format(float(gv or 0.0)) +
                            " / real " + fmt.format(float(rv or 0.0)))
        info.append(f"piezas: gen {gen_agg.get('n_pieces', 0)} / "
                    f"real {ref_agg.get('n_pieces', 0)}")
        _box(ax_oa, "\n".join(info), loc="tl", fontsize=7.2)
    else:
        _msg(ax_oa, "sin datos suficientes\npara calcular OA / gen_score")
    ax_oa.set_title("(f) Overlap por histograma y gen_score")

    fig.suptitle(str(title) if title else
                 "Comparacion de distribuciones: generado vs corpus real",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return _finish(fig, out_png)


# ==========================================================================
# 5. leaderboard
# ==========================================================================
def _barh_ranking(ax, names, vals, lower_is_better, unit, cmap_color):
    """Barras horizontales ordenadas, mejor arriba y destacado.

    Devuelve el nombre del mejor (o None si la columna no tiene datos) para
    que quien llama lo ponga en el titulo del panel.
    """
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    if not ok.any():
        _msg(ax, "columna ausente o vacia")
        return None
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(v[idx], kind="stable")]   # ascendente
    if lower_is_better:
        order = order[::-1]                # el menor (mejor) queda al final
    # barh dibuja y=0 abajo: el ultimo de `order` sale arriba
    y = np.arange(order.size, dtype=float)
    best_pos = order.size - 1
    cols = [C_BEST if i == best_pos else cmap_color for i in range(order.size)]
    ax.barh(y, v[order], height=0.62, color=cols, linewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=7.8)
    span = float(np.nanmax(v[idx]) - np.nanmin(v[idx])) or max(abs(v[idx]).max(), 1.0)
    for yy, i in zip(y, order):
        ax.text(v[i] + 0.015 * span, yy, f"{v[i]:.4g}", va="center",
                fontsize=7.6, color=C_INK)
    ax.set_xlim(0 if np.nanmin(v[idx]) >= 0 else None,
                float(np.nanmax(v[idx])) + 0.22 * span)
    ax.set_xlabel(unit + ("  (mejor = menor)" if lower_is_better
                          else "  (mejor = mayor)"))
    ax.grid(axis="y", visible=False)
    return names[order[best_pos]]


@_robust
def plot_leaderboard(csv_path, out_png) -> str:
    """Compara experimentos: gen_score (mayor mejor) y val_bpt (menor mejor).

    Espera reports/leaderboard.csv con una fila por experimento. Acepta
    sinonimos de columnas y filas incompletas.
    """
    t = _read_table(csv_path)
    if len(t) == 0:
        fig, ax = plt.subplots(figsize=(10.0, 4.2))
        _msg(ax, "leaderboard.csv vacio o ausente\n" + str(csv_path))
        ax.set_title("Leaderboard de experimentos")
        return _finish(fig, out_png)

    names = t.txt("name", "exp", "experiment", "run", "id")
    models = t.txt("model", "modelo", "arch", "family")
    if not any(n for n in names):
        names = models
    par = t.num("params_M", "params_m", "params_millions", "n_params_M")
    if not np.isfinite(par).any():
        raw = t.num("n_params", "params", "parametros", "n_parameters")
        par = np.where(np.isfinite(raw) & (raw > 1e5), raw / 1e6, raw)

    labels = []
    for i in range(len(t)):
        nm = names[i] or models[i] or f"exp_{i}"
        extra = []
        if models[i] and models[i] != nm:
            extra.append(models[i])
        if np.isfinite(par[i]):
            extra.append(f"{par[i]:.1f}M")
        labels.append(nm + (" (" + ", ".join(extra) + ")" if extra else ""))

    # el leaderboard de registry.py escribe best_gen_score / best_val_bpt
    gs = t.num("best_gen_score", "gen_score", "genscore", "gen_score_best",
               "best_genscore", "score")
    bpt = t.num("best_val_bpt", "val_bpt", "best_bpt", "bpt", "bpt_val",
                "best_test_bpt", "test_bpt", "val_bits_per_timestep")

    h = max(3.4, 0.44 * len(t) + 2.1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, h))
    b1 = _barh_ranking(ax1, labels, gs, lower_is_better=False,
                       unit="gen_score (0-100)", cmap_color=C_GEN)
    ax1.set_title("(a) Calidad generativa" +
                  (f"   -   mejor: {b1}" if b1 else ""))
    b2 = _barh_ranking(ax2, labels, bpt, lower_is_better=True,
                       unit="bits por paso de tiempo", cmap_color=C_TRAIN)
    ax2.set_title("(b) Verosimilitud de validacion" +
                  (f"   -   mejor: {b2}" if b2 else ""))

    # la altura de la figura depende del numero de filas: para que la cabecera
    # no se solape hay que reservar pulgadas, no fracciones fijas
    hf = float(fig.get_size_inches()[1])
    fig.suptitle(f"Leaderboard de experimentos ({len(t)})",
                 fontsize=12.5, fontweight="bold", y=1.0 - 0.28 / hf)
    fig.text(0.5, 1.0 - 0.56 / hf,
             "bpt es comparable entre familias: la tokenizacion evento<->roll es biyectiva",
             ha="center", va="top", fontsize=8.2, color="#5b6472")
    fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.80 / hf))
    return _finish(fig, out_png)


# ==========================================================================
# 6. interpretabilidad (TFT)
# ==========================================================================
@_robust
def plot_attention(attn, out_png, title=None) -> str:
    """Mapa de calor de atencion promediada + perfil por distancia relativa.

    attn admite [L,L], [H,L,L] o [B,H,L,L]: los ejes sobrantes se promedian.
    """
    a = attn
    if hasattr(a, "detach"):
        a = a.detach().to("cpu")
    a = np.asarray(a, dtype=np.float64)
    while a.ndim > 2:
        a = a.mean(axis=0)
    if a.ndim != 2 or a.size == 0:
        fig, ax = plt.subplots(figsize=(9.0, 4.5))
        _msg(ax, "matriz de atencion vacia o con forma invalida\n"
                 "(se espera [L,L], [H,L,L] o [B,H,L,L])")
        ax.set_title(str(title) if title else "Atencion promediada")
        return _finish(fig, out_png)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    Lq, Lk = a.shape

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.6, 5.2), gridspec_kw=dict(width_ratios=[3.0, 2.0]))

    vmax = float(np.percentile(a[a > 0], 99)) if (a > 0).any() else 1.0
    im = ax1.imshow(a, aspect="auto", origin="upper", cmap="magma",
                    vmin=0.0, vmax=max(vmax, 1e-9), interpolation="nearest")
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)
    cb.set_label("peso de atencion", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax1.set_xlabel("posicion atendida (key)")
    ax1.set_ylabel("posicion consulta (query)")
    ax1.set_title("(a) Atencion promediada")
    ax1.grid(False)
    ax1.plot([0, min(Lq, Lk) - 1], [0, min(Lq, Lk) - 1], color="#7fe3c0",
             lw=0.7, ls=":", alpha=0.8)

    # perfil por distancia relativa (con filas renormalizadas)
    rs = a.sum(axis=1, keepdims=True)
    an = np.divide(a, np.where(rs > 0, rs, 1.0))
    dmax = min(Lq, Lk)
    prof = np.full(dmax, np.nan)
    for d in range(dmax):
        q = np.arange(d, Lq)
        k = q - d
        m = (k >= 0) & (k < Lk)
        if m.any():
            prof[d] = float(an[q[m], k[m]].mean())
    ok = np.isfinite(prof) & (prof > 0)
    if ok.any():
        ax2.plot(np.flatnonzero(ok), prof[ok], color=C_GEN, lw=1.5)
        ax2.set_yscale("log")
        ax2.set_xlabel("distancia relativa query - key (pasos)")
        ax2.set_ylabel("peso medio (log)")
        w = prof[ok]
        d = np.flatnonzero(ok).astype(float)
        _box(ax2, f"L = {Lq} x {Lk}\n"
                  f"distancia media ponderada = {(d * w).sum() / w.sum():.1f} pasos\n"
                  f"masa causal (k<=q) = "
                  f"{np.tril(an).sum() / max(an.sum(), 1e-12):.3f}", loc="tr")
    else:
        _msg(ax2, "sin masa de atencion positiva")
    ax2.set_title("(b) Perfil por distancia")

    fig.suptitle(str(title) if title else "Interpretabilidad: atencion",
                 fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return _finish(fig, out_png)


@_robust
def plot_vsn_weights(weights, out_png, title=None) -> str:
    """Pesos de la Variable Selection Network del TFT.

    Admite dict {nombre: peso}, vector 1D [n_vars] o matriz 2D [T, n_vars]
    (tiempo en el eje 0), que se promedia sobre el tiempo y ademas se dibuja
    su evolucion.
    """
    names, W = None, None
    if isinstance(weights, dict):
        names = [str(k) for k in weights.keys()]
        vals = list(weights.values())
        try:
            W = np.asarray([np.asarray(
                v.detach().cpu() if hasattr(v, "detach") else v, float).ravel()
                for v in vals], float).T          # -> [T, n_vars]
        except Exception:
            W = np.asarray([[float(np.mean(np.asarray(v, float)))]
                            for v in vals], float).T
    else:
        w = weights
        if hasattr(w, "detach"):
            w = w.detach().to("cpu")
        W = np.asarray(w, float)
        while W.ndim > 2:                          # [B,T,n] -> [T,n]
            W = W.mean(axis=0)
        if W.ndim == 1:
            W = W[None, :]

    if W is None or W.size == 0 or W.ndim != 2:
        fig, ax = plt.subplots(figsize=(9.0, 4.5))
        _msg(ax, "pesos VSN vacios o con forma invalida")
        ax.set_title(str(title) if title else "Pesos de seleccion de variables")
        return _finish(fig, out_png)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    T, nv = W.shape
    if names is None or len(names) != nv:
        names = [f"var_{i}" for i in range(nv)]
    mean_w = W.mean(axis=0)

    two = T > 1
    if two:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13.6, max(3.6, 0.34 * nv + 2.4)),
            gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    else:
        fig, ax1 = plt.subplots(figsize=(9.2, max(3.4, 0.34 * nv + 2.2)))
        ax2 = None

    order = np.argsort(mean_w)                    # menor -> mayor (mejor arriba)
    y = np.arange(nv, dtype=float)
    sdev = W.std(axis=0) if T > 1 else np.zeros(nv)
    vmx = float(mean_w.max()) if mean_w.size else 1.0
    xmx = float((mean_w + sdev).max()) if mean_w.size else 1.0
    cols = [C_GEN if mean_w[i] >= 0.5 * vmx else C_TRAIN for i in order]
    ax1.barh(y, mean_w[order], height=0.66, color=cols, linewidth=0)
    if two:
        ax1.errorbar(mean_w[order], y, xerr=sdev[order], fmt="none",
                     ecolor=C_INK, elinewidth=0.7, capsize=2.0, alpha=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels([names[i] for i in order], fontsize=8.0)
    pct = 0.98 < mean_w.sum() < 1.02              # los pesos VSN suman 1
    for yy, i in zip(y, order):
        ax1.text(mean_w[i] + sdev[i] + 0.018 * max(xmx, 1e-9), yy,
                 f"{mean_w[i]:.3f}" + (f" ({100 * mean_w[i]:.1f}%)" if pct else ""),
                 va="center", fontsize=7.5, color=C_INK)
    ax1.set_xlim(0, xmx * 1.34 if xmx > 0 else 1.0)
    ax1.set_xlabel("peso medio de seleccion")
    ax1.grid(axis="y", visible=False)
    ax1.set_title("(a) Importancia media de variables")
    _box(ax1, f"{nv} variables, {T} pasos promediados\n"
              f"suma de pesos = {mean_w.sum():.3f}\n"
              f"top: {names[int(np.argmax(mean_w))]}", loc="br", fontsize=7.2)

    if ax2 is not None:
        top = order[::-1][:min(8, nv)]
        xs = np.arange(T, dtype=float)
        cmap = matplotlib.colormaps["viridis"]
        for j, i in enumerate(top):
            ax2.plot(xs, W[:, i], lw=1.2, label=names[i],
                     color=cmap(j / max(len(top) - 1, 1)) if len(top) > 1 else C_GEN)
        ax2.set_xlabel("paso de tiempo")
        ax2.set_ylabel("peso de seleccion")
        wmx = float(W[:, top].max()) if len(top) else 1.0
        if wmx > 0:
            ax2.set_ylim(min(0.0, float(W[:, top].min())), wmx * 1.42)
        ax2.set_title("(b) Evolucion temporal (top variables)")
        ax2.legend(loc="upper right", ncol=2, fontsize=7.0)

    fig.suptitle(str(title) if title else
                 "Interpretabilidad: seleccion de variables (VSN)",
                 fontsize=12.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, out_png)


# ==========================================================================
# 7. resumen del dataset
# ==========================================================================
@_robust
def plot_data_overview(stats_json, out_png) -> str:
    """Resumen del corpus a partir de data/processed/data_stats.json.

    Panel 2x3: duracion por pieza (minutos), pitch, polifonia, gaps entre
    onsets (y log), onsets por posicion en la grilla de 16 con la linea de
    1/16 (evidencia de que NO hay estructura de compas) y ficha numerica.
    """
    st = _load_json(stats_json)
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2))
    ax_dur, ax_pitch, ax_poly, ax_gap, ax_g16, ax_txt = axes.ravel()
    if not st:
        for ax in axes.ravel():
            _msg(ax, "data_stats.json ausente o ilegible")
        fig.suptitle("Resumen del dataset", fontsize=12.5, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return _finish(fig, out_png)

    step_sec = float(st.get("step_sec", STEP_SEC) or STEP_SEC)
    note_min = int(st.get("note_min", NOTE_MIN_MIDI) or NOTE_MIN_MIDI)

    # --- (a) duracion por pieza ---
    sl = st.get("seq_len") if isinstance(st.get("seq_len"), dict) else {}
    hist_len = st.get("seq_len_hist")
    if hist_len:
        ks, vs = _dict_hist(hist_len)
        if ks.size:
            ax_dur.bar(ks * step_sec / 60.0, vs, width=max(np.diff(ks).min()
                       if ks.size > 1 else 1, 1) * step_sec / 60.0,
                       color=C_TRAIN, linewidth=0)
            ax_dur.set_ylabel("piezas")
    if not hist_len and sl:
        def mn(k):
            v = sl.get(k)
            return _f(v) * step_sec / 60.0
        p1, p25, med, p75, p99 = mn("p1"), mn("p25"), mn("median"), mn("p75"), mn("p99")
        lo, hi, mean = mn("min"), mn("max"), mn("mean")
        if np.isfinite([p25, p75, med]).all():
            ax_dur.hlines(0, p1 if np.isfinite(p1) else p25,
                          p99 if np.isfinite(p99) else p75,
                          color=C_INK, lw=1.2)
            ax_dur.add_patch(Rectangle((p25, -0.20), max(p75 - p25, 1e-6), 0.40,
                                       facecolor=C_TRAIN, alpha=0.55,
                                       edgecolor=C_INK, lw=0.8))
            ax_dur.vlines(med, -0.24, 0.24, color=C_BAD, lw=2.0)
            for v in (lo, hi):
                if np.isfinite(v):
                    ax_dur.plot([v], [0], marker="|", ms=10, color=C_INK)
            if np.isfinite(mean):
                ax_dur.plot([mean], [0], marker="D", ms=6, color=C_BEST,
                            label="media")
            ax_dur.set_ylim(-0.75, 0.75)
            ax_dur.set_yticks([])
            ax_dur.set_xlabel("duracion por pieza (minutos)")
            _box(ax_dur, "percentiles (minutos)\n"
                 f"min {lo:.2f}  p1 {p1:.2f}  p25 {p25:.2f}\n"
                 f"mediana {med:.2f}  p75 {p75:.2f}\n"
                 f"p99 {p99:.2f}  max {hi:.2f}  media {mean:.2f}", loc="tl")
            ax_dur.legend(loc="lower right")
        else:
            _msg(ax_dur, "percentiles de longitud incompletos")
    elif not hist_len:
        _msg(ax_dur, "sin informacion de longitudes")
    ax_dur.set_title("(a) Duracion de las piezas")

    # --- (b) pitch ---
    ph = np.asarray(st.get("pitch_hist") or [], float).ravel()
    if ph.size:
        x = np.arange(ph.size) + note_min
        ax_pitch.bar(x, ph / max(ph.sum(), 1.0), width=0.92, color=C_TRAIN,
                     linewidth=0)
        tk, tl = _c_ticks(note_min, note_min + ph.size - 1)
        ax_pitch.set_xticks(tk)
        ax_pitch.set_xticklabels(tl)
        ax_pitch.set_xlabel("nota (Do marcados)")
        ax_pitch.set_ylabel("fraccion de onsets")
        top = int(np.argmax(ph)) + note_min
        ax_pitch.axvline(60, color=C_BAD, ls=":", lw=1.0, alpha=0.8)
        _box(ax_pitch, f"moda MIDI {top} (C4=60 marcado)\n"
                       f"rango {note_min}-{note_min + ph.size - 1}\n"
                       f"total {ph.sum():,.0f} onsets", loc="tr")
    else:
        _msg(ax_pitch, "sin pitch_hist")
    ax_pitch.set_title("(b) Distribucion de pitch")

    # --- (c) polifonia ---
    ks, vs = _dict_hist(st.get("polyphony_hist"), kmax=20)
    if ks.size:
        cols = [C_REF if k == 0 else C_TRAIN for k in ks]
        ax_poly.bar(ks, np.maximum(vs, 0), width=0.85, color=cols, linewidth=0)
        pos = vs[vs > 0]
        if pos.size:
            ax_poly.set_yscale("log")
            ax_poly.set_ylim(max(pos.min() * 0.5, 0.5), pos.max() * 3.0)
        ax_poly.set_xticks(ks[::max(1, len(ks) // 12)])
        ax_poly.set_xlabel("notas simultaneas por paso (0 = paso vacio)")
        ax_poly.set_ylabel("pasos (log)")
        fe = st.get("frac_empty_steps")
        mp = st.get("mean_onsets_per_active_step")
        _box(ax_poly, "gris = pasos vacios\n" +
             (f"vacios {100 * float(fe):.1f}%\n" if fe is not None else "") +
             (f"polifonia media activa {float(mp):.2f}" if mp is not None else ""),
             loc="tr")
    else:
        _msg(ax_poly, "sin polyphony_hist")
    ax_poly.set_title("(c) Polifonia por paso")

    # --- (d) gaps ---
    ks, vs = _dict_hist(st.get("gap_hist"))
    if ks.size:
        _log_bars(ax_gap, ks, vs, C_VAL, width=0.85)
        ax_gap.set_xlabel("gap entre onsets consecutivos (pasos de 50 ms)")
        ax_gap.set_ylabel("ocurrencias (log)")
        g = st.get("gap") if isinstance(st.get("gap"), dict) else {}
        for (key, col, lab), yf in zip((("median", C_BAD, "mediana"),
                                        ("p90", C_BEST, "p90"),
                                        ("p99", C_INK, "p99")),
                                       (0.965, 0.895, 0.825)):
            v = g.get(key)
            fv = _f(v)
            if np.isfinite(fv) and fv <= ks.max():
                ax_gap.axvline(fv, color=col, ls="--", lw=1.0, alpha=0.85)
                ax_gap.text(fv, yf, f" {lab}={fv:g}",
                            transform=ax_gap.get_xaxis_transform(),
                            fontsize=7.0, color=col, va="top", ha="left")
        if g:
            _box(ax_gap, f"media {_f(g.get('mean')):.2f}  "
                         f"mediana {g.get('median', '?')}\n"
                         f"p90 {g.get('p90', '?')}  p99 {g.get('p99', '?')}  "
                         f"max {g.get('max', '?')}\n"
                         f"<=16 pasos: "
                         f"{100 * _f(g.get('frac_le_16'), 0.0):.1f}%  "
                         f"(SHIFT max = 64)", loc="tr")
    else:
        _msg(ax_gap, "sin gap_hist")
    ax_gap.set_title("(d) Gaps entre onsets")

    # --- (e) grilla de 16 ---
    g16 = np.asarray(st.get("onsets_by_grid16") or [], float).ravel()
    if g16.size:
        frac = g16 / max(g16.sum(), 1.0)
        xs = np.arange(g16.size)
        ax_g16.bar(xs, frac, width=0.78, color=C_GEN, alpha=0.85, linewidth=0)
        unif = 1.0 / g16.size
        ax_g16.axhline(unif, color=C_BAD, ls="--", lw=1.3,
                       label=f"uniforme 1/{g16.size} = {unif:.4f}")
        dev = float(np.max(np.abs(frac - unif)) / unif * 100.0)
        ax_g16.set_ylim(0, max(frac.max() * 1.35, unif * 1.5))
        ax_g16.set_xticks(xs)
        ax_g16.set_xticklabels([str(i) for i in xs], fontsize=7.2)
        ax_g16.set_xlabel("posicion dentro de la grilla de 16 pasos")
        ax_g16.set_ylabel("fraccion de onsets")
        ax_g16.legend(loc="upper right")
        _box(ax_g16, f"desviacion maxima {dev:.2f}% del uniforme\n"
                     "la grilla es plana -> NO hay compas:\n"
                     "es piano transcrito, no partitura",
             loc="tl", fontsize=7.2)
    else:
        _msg(ax_g16, "sin onsets_by_grid16")
    ax_g16.set_title("(e) Onsets por posicion de grilla-16")

    # --- (f) ficha numerica ---
    def g(k, d=None):
        v = st.get(k, d)
        return v

    dur = st.get("duration_sec") if isinstance(st.get("duration_sec"), dict) else {}
    lines = [
        "FICHA DEL CORPUS",
        "",
        f"piezas                 {int(g('n_sequences', 0) or 0):,}",
        f"pasos de tiempo        {int(g('n_steps_total', 0) or 0):,}",
        f"onsets                 {int(g('total_onsets', 0) or 0):,}",
        f"horas de audio         {float(dur.get('total_hours', 0) or 0):.1f}",
        f"paso                   {step_sec * 1000:.1f} ms "
        f"({1.0 / step_sec:.0f} Hz)",
        f"notas MIDI             {note_min} - {int(g('note_max', 108) or 108)}"
        f"  ({int(g('n_positions', N_PITCH) or N_PITCH)} posiciones)",
        f"representacion         {g('representation', 'onset')}",
        "",
        f"densidad               {float(g('density', 0) or 0):.5f} onsets/(paso*nota)",
        f"pasos vacios           {100 * float(g('frac_empty_steps', 0) or 0):.2f}%",
        f"onsets / paso          {float(g('mean_onsets_per_step', 0) or 0):.3f}",
        f"onsets / paso activo   {float(g('mean_onsets_per_active_step', 0) or 0):.3f}",
        f"pasos activos          {int(g('n_active_steps', 0) or 0):,}",
        f"tokens NOTE+SHIFT est. {int(g('est_tokens_note_shift', 0) or 0):,}",
    ]
    _blank(ax_txt)
    ax_txt.text(0.02, 0.98, "\n".join(lines), transform=ax_txt.transAxes,
                va="top", ha="left", fontsize=8.0, family="DejaVu Sans Mono",
                linespacing=1.45, color=C_INK)
    ax_txt.set_title("(f) Ficha del dataset")

    fig.suptitle("Resumen del dataset de piano-roll de onsets",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return _finish(fig, out_png)
