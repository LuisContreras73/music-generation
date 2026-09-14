"""Registro de experimentos: CSV de metricas, mejor checkpoint y leaderboard global.

Que resuelve este modulo
------------------------
1. MetricLogger        -> experiments/<name>/logs/metrics.csv  (append robusto,
                          soporta que aparezcan columnas nuevas a mitad del
                          entrenamiento sin perder filas ni romper pandas).
2. save/load_checkpoint-> escritura ATOMICA (path.tmp + os.replace) para que una
                          interrupcion nunca corrompa el mejor checkpoint.
3. update_best         -> mantiene checkpoints/best.pt     (min val_bpt)
                          y      checkpoints/best_gen.pt   (max gen_score)
                          y      logs/best.json            (estado legible).
4. write_summary       -> logs/summary.json con el resumen final del experimento.
5. rebuild_leaderboard -> reports/leaderboard.csv (una fila por experimento).
6. promote_best        -> sincroniza reports/best/ con el ganador global.
                          IDEMPOTENTE: borra los ficheros del ganador anterior.
7. refresh_all         -> punto de entrada unico: rebuild + figura + promote.

Convenios
---------
* Todas las rutas salen de src/config.py (EXP_DIR, REPORTS, ROOT). Se leen en
  cada llamada (no se cachean) para que los tests puedan redirigirlas.
* Todo lo escrito a disco es JSON/CSV plano y ASCII: si el entrenamiento muere,
  lo que quedo en disco sigue siendo parseable.
* Nada de este modulo importa torch, pandas ni matplotlib al importarse; solo
  dentro de las funciones que los necesitan (arranque rapido de la CLI).

CLI
---
    python src/registry.py --refresh       rebuild + figuras + promote
    python src/registry.py --leaderboard   solo imprime la tabla
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# --- src/ en el path para poder importarse como "registry" o "src.registry" ---
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as _cfgmod                    # noqa: E402
from config import Config                   # noqa: E402

TMP = ".tmp"
BEST_JSON = "best.json"
METRICS_CSV = "metrics.csv"
SUMMARY_JSON = "summary.json"
LEADERBOARD_CSV = "leaderboard.csv"
CURVES_PNG = "training_curves.png"

# Nombres alternativos aceptados en los dicts de metricas (tolerancia entre agentes).
VAL_BPT_KEYS = ("val_bpt", "valid_bpt", "bpt_val", "val_bits_per_step", "bpt")
GEN_SCORE_KEYS = ("gen_score", "val_gen_score", "genscore", "score")
VAL_PPL_KEYS = ("val_ppl", "val_perplexity", "ppl_val", "ppl")
FRAME_F1_KEYS = ("frame_f1", "val_frame_f1", "f1_frame", "f1")
STEP_KEYS = ("step", "global_step", "iter", "iteration")
SECS_KEYS = ("elapsed_s", "elapsed_sec", "elapsed", "wall_s", "time_s", "seconds", "train_seconds")
HOURS_KEYS = ("train_hours", "elapsed_h", "hours")
TOKENS_KEYS = ("tokens_seen", "tokens", "n_tokens", "tokens_total")

LEADERBOARD_COLS = ["name", "model", "family", "params_M", "best_val_bpt", "best_val_ppl",
                    "best_gen_score", "frame_f1", "steps", "train_hours", "updated"]

# Columnas del leaderboard que SIEMPRE deben ser numericas. Coaccionarlas no es
# cosmetico: pandas 3 lee una columna con una sola celda de texto como dtype
# "str", y entonces sort_values ordena LEXICOGRAFICAMENTE ("9.5" > "80.1"), con
# lo que promote_best sincronizaria reports/best/ con el experimento equivocado.
NUMERIC_COLS = ("params_M", "best_val_bpt", "best_val_ppl", "best_gen_score",
                "frame_f1", "steps", "train_hours")

# Ficheros que promote_best considera SUYOS dentro de reports/best/ y por tanto
# puede borrar antes de copiar un ganador nuevo (evita mezclar dos experimentos).
OWNED_IN_BEST = ("best_checkpoint.pt", "metrics.csv", CURVES_PNG, "BEST.md",
                 "best.json", "config.json", "summary.json", "distributions.png",
                 "generation_*.png", "generation_*.npz", "*" + TMP)

MAX_GEN_COPIES = 6

# Override opcional de rutas (solo para tests); None = usar src/config.py.
_EXP_OVERRIDE: Path | None = None
_REP_OVERRIDE: Path | None = None

__all__ = [
    "MetricLogger", "save_checkpoint", "load_checkpoint", "restore_rng",
    "update_best", "write_summary", "rebuild_leaderboard", "promote_best",
    "refresh_all", "format_leaderboard", "load_leaderboard", "checkpoint_paths",
    "experiment_rows", "load_experiment_config", "set_paths", "exp_root",
    "reports_root", "read_json", "write_json", "main",
]


# =============================================================== rutas basicas
def set_paths(exp_dir=None, reports=None) -> None:
    """Redirige experiments/ y reports/ (lo usan los tests; None = por defecto)."""
    global _EXP_OVERRIDE, _REP_OVERRIDE
    _EXP_OVERRIDE = Path(exp_dir) if exp_dir else None
    _REP_OVERRIDE = Path(reports) if reports else None


def exp_root() -> Path:
    return Path(_EXP_OVERRIDE) if _EXP_OVERRIDE else Path(_cfgmod.EXP_DIR)


def reports_root() -> Path:
    return Path(_REP_OVERRIDE) if _REP_OVERRIDE else Path(_cfgmod.REPORTS)


def _mkdir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _rel(p) -> str:
    """Ruta relativa a la raiz del repo con '/' (portable dentro del JSON)."""
    p = Path(p)
    try:
        return p.resolve().relative_to(Path(_cfgmod.ROOT).resolve()).as_posix()
    except (ValueError, OSError):
        return p.as_posix()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _mtime_iso(p):
    try:
        return datetime.fromtimestamp(Path(p).stat().st_mtime).astimezone().isoformat(timespec="seconds")
    except OSError:
        return None


def _isna(v) -> bool:
    """True para None, NaN, pd.NA y cadenas vacias (sin importar pandas)."""
    if v is None:
        return True
    if isinstance(v, float) and not math.isfinite(v):
        return True
    return str(v).strip().lower() in ("", "nan", "<na>", "nat", "none")


def _coalesce(*vals):
    """Primer valor no None (y no NaN)."""
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and not math.isfinite(v):
            continue
        return v
    return None


# ====================================================== conversiones seguras
def _scalar(v):
    """Normaliza a tipos nativos de Python (tensores/numpy -> float/int/list).

    Los float no finitos (NaN/Inf) pasan a None: asi el JSON es valido y una
    metrica invalida nunca se confunde con la 'mejor'.
    """
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): _scalar(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_scalar(x) for x in v]
    if isinstance(v, np.generic):
        return _scalar(v.item())
    if isinstance(v, np.ndarray):
        return _scalar(v.item()) if v.size == 1 else [_scalar(x) for x in v.reshape(-1).tolist()]
    if hasattr(v, "detach"):                      # torch.Tensor sin importar torch
        try:
            t = v.detach().to("cpu")
            return _scalar(t.item()) if t.numel() == 1 else [_scalar(x) for x in t.reshape(-1).tolist()]
        except Exception:
            return str(v)
    if hasattr(v, "item"):
        try:
            return _scalar(v.item())
        except Exception:
            pass
    return str(v)


def _json_safe(o):
    return _scalar(o)


def _cell(v) -> str:
    """Valor -> celda de CSV (None -> vacio, para que pandas lo lea como NaN)."""
    v = _scalar(v)
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return f"{v:.10g}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=True)
    return str(v)


def _num(s):
    """Celda de CSV -> int/float/bool/str/None."""
    if s is None or not isinstance(s, str):
        return s
    t = s.strip()
    if t == "":
        return None
    if t in ("True", "true"):
        return True
    if t in ("False", "false"):
        return False
    try:
        f = float(t)
    except ValueError:
        return s
    if not math.isfinite(f):
        return None
    if f.is_integer() and not any(c in t.lower() for c in ".e"):
        return int(f)
    return f


def _pick(d, keys):
    """Primer valor float finito entre varias claves alternativas."""
    if not d:
        return None
    for k in keys:
        if k in d:
            v = _scalar(d[k])
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                return float(v)
    return None


# ============================================== escritura atomica de ficheros
def _atomic_write_text(path, text: str) -> Path:
    path = Path(path)
    _mkdir(path.parent)
    tmp = path.with_name(path.name + TMP)
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


def write_json(path, obj) -> Path:
    return _atomic_write_text(path, json.dumps(_json_safe(obj), indent=2, ensure_ascii=True) + "\n")


def read_json(path, default=None):
    """Lee un JSON tolerando que no exista o que este truncado."""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, (dict, list)) else (default if default is not None else {})
    except (OSError, ValueError):
        return default if default is not None else {}


def _atomic_copy(src, dst) -> bool:
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        return False
    _mkdir(dst.parent)
    tmp = dst.with_name(dst.name + TMP)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return True


# ================================================================ 1. logger
class MetricLogger:
    """CSV incremental de metricas, tolerante a columnas nuevas.

    * append(row) escribe una fila y hace flush inmediato: si el proceso muere,
      el CSV en disco sigue siendo valido y parseable por pandas.
    * Si la fila trae una columna que no estaba en la cabecera, el CSV se
      REESCRIBE completo (a .tmp y luego os.replace) con la union de columnas y
      las filas antiguas rellenadas con vacio. No se pierde ninguna fila.
    """

    def __init__(self, csv_path, fsync: bool = False):
        self.path = Path(csv_path)
        _mkdir(self.path.parent)
        self.fsync = bool(fsync)          # fsync por fila: mas seguro, mas lento
        self._fields = self._read_header()

    # ---- lectura
    def _read_header(self) -> list:
        try:
            if self.path.stat().st_size == 0:
                return []
        except OSError:
            return []
        with open(self.path, newline="", encoding="utf-8") as f:
            for head in csv.reader(f):
                return [h for h in head if h != ""]
        return []

    def _read_rows(self) -> list:
        if not self._fields:
            return []
        with open(self.path, newline="", encoding="utf-8") as f:
            return [{k: v for k, v in r.items() if k is not None}
                    for r in csv.DictReader(f, restval="")]

    def _dangling_line(self) -> bool:
        """True si el fichero no acaba en salto de linea (muerte a mitad de fila)."""
        try:
            with open(self.path, "rb") as f:
                if f.seek(0, os.SEEK_END) == 0:
                    return False
                f.seek(-1, os.SEEK_END)
                return f.read(1) not in (b"\n", b"\r")
        except OSError:
            return False

    @property
    def columns(self) -> list:
        return list(self._fields)

    def history(self) -> list:
        """Historial completo como lista de dicts con los valores convertidos."""
        return [{k: _num(v) for k, v in r.items()} for r in self._read_rows()]

    def last(self) -> dict:
        h = self.history()
        return h[-1] if h else {}

    def __len__(self) -> int:
        return len(self._read_rows())

    # ---- escritura
    def append(self, row: dict) -> dict:
        """Anade una fila; devuelve la fila tal como se escribio (strings)."""
        row = {str(k): _cell(v) for k, v in dict(row).items()}
        if not row:
            return row
        # La cabecera del FICHERO es la verdad, no la cache de __init__: si el
        # CSV se creo (o se extendio) despues de construir este logger, fiarse
        # de la cache vacia dispararia _rewrite([], row) y BORRARIA todo el
        # historial ya escrito.
        self._fields = self._read_header()
        if not self._fields:                                  # CSV nuevo o vacio
            self._fields = list(row.keys())
            self._rewrite([], row)
            return row
        new_cols = [k for k in row if k not in self._fields]
        if new_cols:                                          # columna nueva -> reescribir
            old = self._read_rows()
            self._fields = self._fields + new_cols
            self._rewrite(old, row)
            return row
        dangling = self._dangling_line()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            if dangling:                    # cierra la fila que quedo a medias
                f.write("\r\n")
            w = csv.DictWriter(f, fieldnames=self._fields, restval="", extrasaction="ignore")
            w.writerow(row)
            f.flush()
            if self.fsync:
                os.fsync(f.fileno())
        return row

    def _rewrite(self, old_rows: list, new_row: dict) -> None:
        tmp = self.path.with_name(self.path.name + TMP)
        try:
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self._fields, restval="", extrasaction="ignore")
                w.writeheader()
                for r in old_rows:
                    w.writerow(r)
                w.writerow(new_row)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise


# =========================================================== 2. checkpoints
def _rng_state() -> dict:
    import torch
    st = {"python": random.getstate(),
          "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            st["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return st


def restore_rng(ck: dict) -> bool:
    """Restaura el estado de los generadores guardado en un checkpoint."""
    import torch
    st = (ck or {}).get("rng") or {}
    try:
        if "python" in st:
            v = st["python"]
            random.setstate(tuple(v) if isinstance(v, list) else v)
        if "numpy" in st:
            v = st["numpy"]
            np.random.set_state(tuple(v) if isinstance(v, list) else v)
        if "torch" in st:
            v = st["torch"]
            torch.set_rng_state(v.cpu() if hasattr(v, "cpu") else v)
        if "cuda" in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda"])
        return True
    except Exception:
        return False


def _cfg_dict(cfg):
    if cfg is None:
        return None
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return None


def _n_params(model):
    try:
        if hasattr(model, "n_params"):
            return int(model.n_params())
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception:
        return None


def save_checkpoint(path, model, optimizer=None, cfg=None, step: int = 0,
                    metrics=None, scaler=None, extra=None) -> Path:
    """Guarda un checkpoint completo de forma ATOMICA (path.tmp -> os.replace).

    El dict guardado lleva claves duplicadas por compatibilidad entre agentes:
    "model_state"/"model", "optimizer_state"/"optimizer" y "scaler_state"/"scaler"
    apuntan al MISMO objeto (pickle no duplica los tensores).
    """
    import torch
    path = Path(path)
    _mkdir(path.parent)

    sd = model.state_dict()
    sd = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}
    osd = optimizer.state_dict() if optimizer is not None else None
    ssd = scaler.state_dict() if scaler is not None else None

    payload = {
        "format": 1,
        "step": int(step),
        "model_state": sd, "model": sd,
        "optimizer_state": osd, "optimizer": osd,
        "scaler_state": ssd, "scaler": ssd,
        "config": _cfg_dict(cfg),
        "metrics": _json_safe(metrics or {}),
        "rng": _rng_state(),
        "model_name": getattr(model, "name", None),
        "family": getattr(model, "family", None),
        "n_params": _n_params(model),
        "saved_at": _now_iso(),
        "torch_version": torch.__version__,
    }
    if extra:
        payload["extra"] = _json_safe(extra)

    tmp = path.with_name(path.name + TMP)
    try:
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


def load_checkpoint(path, map_location="cpu") -> dict:
    """Carga un checkpoint escrito por save_checkpoint.

    weights_only=False es necesario porque el payload incluye el estado de los
    RNG de numpy/python (objetos, no tensores). Solo cargamos ficheros propios.
    """
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:                                   # torch antiguo sin weights_only
        return torch.load(path, map_location=map_location)


def checkpoint_paths(cfg) -> dict:
    """Rutas canonicas de los checkpoints (consulta pura: no crea directorios)."""
    ck = Path(cfg.dir()) / "checkpoints"
    return {"best": ck / "best.pt", "best_gen": ck / "best_gen.pt", "last": ck / "last.pt"}


# ============================================================= 3. mejor local
def update_best(cfg, step: int, metrics: dict, model, optimizer=None, scaler=None) -> dict:
    """Actualiza best.pt (min val_bpt) y best_gen.pt (max gen_score).

    Devuelve {"new_best_bpt", "new_best_gen", "best_val_bpt", "best_val_bpt_step",
              "best_gen_score", "best_gen_score_step", rutas}.
    Escribe siempre logs/best.json con el estado acumulado.
    """
    sub = cfg.subdirs()
    logs, ckdir = sub["logs"], sub["ckpt"]
    st = read_json(logs / BEST_JSON, {})
    m = _json_safe(dict(metrics or {}))

    res = {"new_best_bpt": False, "new_best_gen": False,
           "best_val_bpt": st.get("best_val_bpt"),
           "best_val_bpt_step": st.get("best_val_bpt_step"),
           "best_gen_score": st.get("best_gen_score"),
           "best_gen_score_step": st.get("best_gen_score_step")}

    # --- criterio 1: verosimilitud (menor bpt es mejor)
    v = _pick(m, VAL_BPT_KEYS)
    if v is not None:
        prev = st.get("best_val_bpt")
        if prev is None or v < float(prev):
            save_checkpoint(ckdir / "best.pt", model, optimizer, cfg, step, m, scaler,
                            extra={"criterion": "min val_bpt", "value": v})
            st.update(best_val_bpt=v, best_val_bpt_step=int(step),
                      best_val_bpt_metrics=m, best_val_bpt_time=_now_iso())
            res.update(new_best_bpt=True, best_val_bpt=v, best_val_bpt_step=int(step))

    # --- criterio 2: parecido al corpus (mayor gen_score es mejor)
    g = _pick(m, GEN_SCORE_KEYS)
    if g is not None:
        prev = st.get("best_gen_score")
        if prev is None or g > float(prev):
            save_checkpoint(ckdir / "best_gen.pt", model, optimizer, cfg, step, m, scaler,
                            extra={"criterion": "max gen_score", "value": g})
            st.update(best_gen_score=g, best_gen_score_step=int(step),
                      best_gen_score_metrics=m, best_gen_score_time=_now_iso())
            res.update(new_best_gen=True, best_gen_score=g, best_gen_score_step=int(step))

    st.update(name=cfg.name, model=getattr(cfg, "model", None), family=getattr(cfg, "family", None),
              n_params=_coalesce(_n_params(model), st.get("n_params")),
              last_step=int(step), updated=_now_iso())
    write_json(logs / BEST_JSON, st)

    res["path_best"] = str(ckdir / "best.pt")
    res["path_best_gen"] = str(ckdir / "best_gen.pt")
    res["best_json"] = str(logs / BEST_JSON)
    return res


# ======================================================= lectura de metrics.csv
def _scan_metrics_csv(path) -> dict:
    """Deduce los mejores valores directamente del CSV (por si falta best.json)."""
    out = {"n_rows": 0}
    path = Path(path)
    if not path.is_file():
        return out
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = [{k: _num(v) for k, v in r.items() if k is not None}
                    for r in csv.DictReader(f, restval="")]
    except OSError:
        return out
    out["n_rows"] = len(rows)
    if not rows:
        return out

    best_bpt = best_gen = None
    for r in rows:
        stp = _pick(r, STEP_KEYS)
        v = _pick(r, VAL_BPT_KEYS)
        if v is not None and (best_bpt is None or v < best_bpt[0]):
            best_bpt = (v, stp, r)
        g = _pick(r, GEN_SCORE_KEYS)
        if g is not None and (best_gen is None or g > best_gen[0]):
            best_gen = (g, stp, r)

    if best_bpt:
        out["best_val_bpt"] = best_bpt[0]
        out["best_val_bpt_step"] = best_bpt[1]
        out["best_val_ppl"] = _pick(best_bpt[2], VAL_PPL_KEYS)
    if best_gen:
        out["best_gen_score"] = best_gen[0]
        out["best_gen_score_step"] = best_gen[1]

    ppl = [x for x in (_pick(r, VAL_PPL_KEYS) for r in rows) if x is not None]
    if ppl:
        out["best_val_ppl"] = _coalesce(out.get("best_val_ppl"), min(ppl))
    f1 = [x for x in (_pick(r, FRAME_F1_KEYS) for r in rows) if x is not None]
    if f1:
        out["frame_f1"] = max(f1)

    steps = [x for x in (_pick(r, STEP_KEYS) for r in rows) if x is not None]
    out["steps"] = int(max(steps)) if steps else len(rows)

    secs = [x for x in (_pick(r, SECS_KEYS) for r in rows) if x is not None]
    hrs = [x for x in (_pick(r, HOURS_KEYS) for r in rows) if x is not None]
    if secs:
        out["train_hours"] = horas_de_computo(secs)
    elif hrs:
        out["train_hours"] = float(max(hrs))

    toks = [x for x in (_pick(r, TOKENS_KEYS) for r in rows) if x is not None]
    if toks:
        out["tokens_seen"] = float(max(toks))
    return out


def horas_de_computo(secuencia, gap_min: float = 300.0) -> float:
    """Horas de computo reales a partir de la serie de tiempos de metrics.csv.

    No vale tomar el maximo, que es lo que se hacia antes, por dos motivos que se
    compensan en direcciones opuestas y ninguno da el numero correcto:

      * el contador se REINICIA en cada reanudacion, asi que el maximo solo mide
        el ultimo tramo (medido: 0.30 h en un entrenamiento de 1.95 h);
      * incluye las pausas en que el proceso estuvo parado, asi que en un run con
        un hueco largo el maximo SOBREestima (medido: 1.66 h por 0.84 h reales).

    Se suman los incrementos: un salto hacia atras abre tramo nuevo y los huecos
    anomalos se descartan. El umbral se deriva de la propia serie (12x el
    incremento mediano) porque la frecuencia de registro varia entre
    experimentos, con un minimo por si la serie es muy corta.
    """
    seq = [float(x) for x in secuencia]
    if not seq:
        return 0.0
    d = sorted(b - a for a, b in zip(seq, seq[1:]) if b > a)
    med = d[len(d) // 2] if d else 0.0
    gap = max(gap_min, 12.0 * med)
    total, prev = 0.0, None
    for t in seq:
        if prev is None or t < prev:        # inicio de tramo o reanudacion
            total += t
        elif t - prev <= gap:
            total += t - prev
        prev = t                            # hueco anomalo: no se cuenta
    return total / 3600.0


# =============================================================== 4. summary
def load_experiment_config(name: str):
    """Config de un experimento a partir de experiments/<name>/config.json."""
    p = exp_root() / str(name) / "config.json"
    try:
        return Config.load(p)
    except Exception:
        return None


def write_summary(cfg, extra=None) -> dict:
    """Escribe experiments/<name>/logs/summary.json y devuelve el dict."""
    sub = cfg.subdirs()
    logs = sub["logs"]
    best = read_json(logs / BEST_JSON, {})
    scan = _scan_metrics_csv(logs / METRICS_CSV)
    extra = _json_safe(dict(extra or {}))

    bm = best.get("best_val_bpt_metrics") or {}
    gm = best.get("best_gen_score_metrics") or {}

    n_params = _coalesce(extra.get("n_params"), best.get("n_params"))
    params_M = _coalesce(extra.get("params_M"),
                         round(float(n_params) / 1e6, 3) if n_params else None)

    figures = sorted(p.name for p in sub["figures"].glob("*.png"))
    gens = sorted(p.name for p in sub["gen"].glob("*") if p.is_file())
    cks = {k: _rel(v) for k, v in checkpoint_paths(cfg).items() if Path(v).is_file()}

    s = {
        "name": cfg.name,
        "model": getattr(cfg, "model", None),
        "family": getattr(cfg, "family", None),
        "seed": getattr(cfg, "seed", None),
        "notes": getattr(cfg, "notes", ""),
        "n_params": n_params,
        "params_M": params_M,
        "best_val_bpt": _coalesce(best.get("best_val_bpt"), scan.get("best_val_bpt")),
        "best_val_bpt_step": _coalesce(best.get("best_val_bpt_step"), scan.get("best_val_bpt_step")),
        "best_gen_score": _coalesce(best.get("best_gen_score"), scan.get("best_gen_score")),
        "best_gen_score_step": _coalesce(best.get("best_gen_score_step"),
                                         scan.get("best_gen_score_step")),
        "best_val_ppl": _coalesce(_pick(bm, VAL_PPL_KEYS), scan.get("best_val_ppl")),
        "frame_f1": _coalesce(_pick(gm, FRAME_F1_KEYS), _pick(bm, FRAME_F1_KEYS),
                              scan.get("frame_f1")),
        "steps": _coalesce(scan.get("steps"), best.get("last_step"), getattr(cfg, "steps", None)),
        "train_hours": scan.get("train_hours"),
        "tokens_seen": scan.get("tokens_seen"),
        "n_log_rows": scan.get("n_rows", 0),
        "seq_len": getattr(cfg, "seq_len", None),
        "batch_size": getattr(cfg, "batch_size", None),
        "paths": {"root": _rel(sub["root"]), "config": _rel(sub["root"] / "config.json"),
                  "metrics_csv": _rel(logs / METRICS_CSV), "best_json": _rel(logs / BEST_JSON),
                  "checkpoints": cks, "figures": figures, "generations": gens},
        "updated": _now_iso(),
    }
    s.update({k: v for k, v in extra.items() if k != "name"})
    if s.get("params_M") is None and s.get("n_params"):
        s["params_M"] = round(float(s["n_params"]) / 1e6, 3)

    if not (sub["root"] / "config.json").is_file():
        try:
            cfg.save()                          # deja el config a mano del leaderboard
        except Exception:
            pass
    write_json(logs / SUMMARY_JSON, s)
    return s


# ============================================================ 5. leaderboard
def _row_for_experiment(d) -> dict | None:
    """Fila del leaderboard para un directorio de experimento (tolera huecos)."""
    d = Path(d)
    cfgj = read_json(d / "config.json", {})
    best = read_json(d / "logs" / BEST_JSON, {})
    summ = read_json(d / "logs" / SUMMARY_JSON, {})
    scan = _scan_metrics_csv(d / "logs" / METRICS_CSV)
    if not (cfgj or best or summ or scan.get("n_rows")):
        return None

    bm = best.get("best_val_bpt_metrics") or {}
    gm = best.get("best_gen_score_metrics") or {}
    n_params = _coalesce(summ.get("n_params"), best.get("n_params"))
    params_M = _coalesce(summ.get("params_M"),
                         round(float(n_params) / 1e6, 3) if n_params else None)
    updated = _coalesce(summ.get("updated"), best.get("updated"),
                        _mtime_iso(d / "logs" / METRICS_CSV), _mtime_iso(d))

    return {
        "name": _coalesce(summ.get("name"), best.get("name"), cfgj.get("name"), d.name),
        "model": _coalesce(summ.get("model"), best.get("model"), cfgj.get("model")),
        "family": _coalesce(summ.get("family"), best.get("family"), cfgj.get("family")),
        "params_M": params_M,
        "best_val_bpt": _coalesce(summ.get("best_val_bpt"), best.get("best_val_bpt"),
                                  scan.get("best_val_bpt")),
        "best_val_ppl": _coalesce(summ.get("best_val_ppl"), _pick(bm, VAL_PPL_KEYS),
                                  scan.get("best_val_ppl")),
        "best_gen_score": _coalesce(summ.get("best_gen_score"), best.get("best_gen_score"),
                                    scan.get("best_gen_score")),
        "frame_f1": _coalesce(summ.get("frame_f1"), _pick(gm, FRAME_F1_KEYS),
                              _pick(bm, FRAME_F1_KEYS), scan.get("frame_f1")),
        "steps": _coalesce(summ.get("steps"), scan.get("steps"), best.get("last_step")),
        # el escaneo gana al summary: summary.json guarda el time_s del ultimo
        # tramo, mientras que el escaneo recorre TODO el historico de metrics.csv
        "train_hours": _coalesce(scan.get("train_hours"), summ.get("train_hours")),
        "updated": updated,
    }


def experiment_rows() -> list:
    """Una fila por subdirectorio de experiments/ con datos utilizables."""
    root = exp_root()
    rows = []
    if not root.is_dir():
        return rows
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.startswith("."):
            continue
        r = _row_for_experiment(d)
        if r:
            rows.append(r)
    return rows


def rebuild_leaderboard(write: bool = True):
    """Reconstruye reports/leaderboard.csv ordenado por gen_score DESC."""
    import pandas as pd
    df = _coerce_numeric(pd.DataFrame(experiment_rows(), columns=LEADERBOARD_COLS))
    if len(df):
        df = df.sort_values(["best_gen_score", "best_val_bpt"], ascending=[False, True],
                            na_position="last", kind="stable").reset_index(drop=True)
    if write:
        out = _mkdir(reports_root()) / LEADERBOARD_CSV
        tmp = out.with_name(out.name + TMP)
        df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(tmp, out)
    return df


def _coerce_numeric(df):
    """Fuerza a numero las columnas metricas (texto o vacio -> NaN).

    Sin esto una sola celda no numerica convierte toda la columna en texto y el
    orden del leaderboard pasa a ser lexicografico: se elegiria mal el ganador.
    """
    import pandas as pd
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_leaderboard(rebuild_if_missing: bool = True):
    """Lee reports/leaderboard.csv (o lo reconstruye si no existe/esta roto)."""
    import pandas as pd
    p = reports_root() / LEADERBOARD_CSV
    if p.is_file():
        try:
            df = pd.read_csv(p)
            for c in LEADERBOARD_COLS:
                if c not in df.columns:
                    df[c] = None
            return _coerce_numeric(df[LEADERBOARD_COLS].copy())
        except Exception:
            pass
    return rebuild_leaderboard() if rebuild_if_missing else pd.DataFrame(columns=LEADERBOARD_COLS)


def format_leaderboard(df=None) -> str:
    """Tabla legible en consola."""
    if df is None:
        df = load_leaderboard()
    if df is None or not len(df):
        return "(leaderboard vacio: no hay experimentos en " + _rel(exp_root()) + ")"
    import pandas as pd
    d = df.copy()
    for c, nd in (("params_M", 2), ("best_val_bpt", 4), ("best_val_ppl", 3),
                  ("best_gen_score", 2), ("frame_f1", 4), ("train_hours", 2)):
        if c in d.columns:
            # to_numeric antes de astype: una celda de texto haria fallar
            # astype("Float64") con ArrowInvalid y tumbaria la CLI.
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("Float64").round(nd)
    return d.to_string(index=False, na_rep="-")


# ============================================================ 6. promote_best
_METRICS = {
    "gen_score": ("best_gen_score", False),
    "best_gen_score": ("best_gen_score", False),
    "val_bpt": ("best_val_bpt", True),
    "best_val_bpt": ("best_val_bpt", True),
    "bpt": ("best_val_bpt", True),
    "val_ppl": ("best_val_ppl", True),
    "best_val_ppl": ("best_val_ppl", True),
    "frame_f1": ("frame_f1", False),
}


def _resolve_metric(metric: str):
    key = str(metric).strip().lower()
    if key not in _METRICS:
        raise ValueError("metrica desconocida: " + str(metric) +
                         " (usa una de: " + ", ".join(sorted(_METRICS)) + ")")
    return _METRICS[key]


def _clean_best_dir(d, manifest=(), keep=()) -> list:
    """Borra de reports/best/ los ficheros del ganador anterior. Idempotente.

    keep = nombres que ya son del ganador actual y estan al dia (no se tocan).
    """
    d = Path(d)
    if not d.is_dir():
        return []
    keep = {str(k) for k in (keep or ())}
    targets = set()
    for name in manifest or ():
        targets.add(d / Path(str(name)).name)
    for pat in OWNED_IN_BEST:
        targets.update(d.glob(pat))
    removed = []
    for p in sorted(targets):
        if p.name in keep:
            continue
        try:
            if p.is_file() and p.parent.resolve() == d.resolve():
                p.unlink()
                removed.append(p.name)
        except OSError:
            pass
    return removed


def _stat_tag(p) -> list:
    """Huella (ruta, tamano, mtime_ns) de un fichero fuente, serializable a JSON."""
    st = Path(p).stat()
    return [_rel(p), int(st.st_size), int(st.st_mtime_ns)]


def _gen_step(p: Path) -> int:
    ns = re.findall(r"(\d+)", p.stem)
    return int(ns[-1]) if ns else -1


def _pick_generations(exp_dir, k: int = MAX_GEN_COPIES) -> list:
    """Mejores generaciones: las de mayor step en el nombre (o las mas recientes).

    Mira en las DOS carpetas donde acaban las imagenes de generacion:
    generations/*.png (rolls guardados como PNG) y figures/generation*.png, que
    es donde src/train.py escribe generation_best.png y generation_grid.png.
    Mirando solo generations/ (que en la practica contiene .npz) reports/best/
    se quedaria sin ninguna imagen de generacion.
    """
    exp_dir = Path(exp_dir)
    cands = []
    for d, pat in ((exp_dir / "generations", "*.png"),
                   (exp_dir / "figures", "generation*.png")):
        if d.is_dir():
            cands += [p for p in d.glob(pat) if p.is_file()]
    seen, out = set(), []
    for p in sorted(cands, key=lambda q: (_gen_step(q), q.stat().st_mtime), reverse=True):
        if p.name in seen:
            continue
        seen.add(p.name)
        out.append(p)
    return out[:max(int(k), 0)]


def _safe_stem(s, n: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")
    return s[:n] if s else "gen"


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "-"
    if abs(f) >= 1e6:
        return f"{f:.3g}"
    s = f"{f:.{nd}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _md_table(df, cols, nd=None) -> str:
    nd = nd or {}
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = None if (c not in r or _isna(r[c])) else r[c]
            cells.append(_fmt(v, nd.get(c, 4)) if (v is None or isinstance(v, (int, float)))
                         else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def promote_best(metric: str = "val_bpt", df=None, top_k_gen: int = MAX_GEN_COPIES) -> dict:
    """Sincroniza reports/best/ con el mejor experimento del leaderboard.

    El criterio por defecto es val_bpt y NO gen_score, aunque gen_score mida lo
    que de verdad interesa. Motivo medido: gen_score se calcula sobre 16 muestras
    con prefijos sorteados, y el mismo modelo puntuo 79.3 y 62.7 en dos
    evaluaciones consecutivas con el val_bpt practicamente igual (1.5185 y
    1.5172). Elegir por una metrica con ese ruido y despues reportar esa misma
    metrica es seleccionar el ruido. val_bpt es determinista: teacher forcing
    sobre el split entero. Para comparar generacion esta la comparacion pareada
    de scripts/18_comparacion_pareada.py.

    Antes de copiar borra los ficheros del ganador anterior (manifest de
    best.json + patrones conocidos), de modo que reports/best/ nunca contenga
    artefactos de dos experimentos distintos. Es idempotente.
    """
    col, lower_is_better = _resolve_metric(metric)
    if df is None:
        df = load_leaderboard()
    best_dir = _mkdir(reports_root() / "best")
    prev = read_json(best_dir / BEST_JSON, {})

    if df is None or not len(df):
        removed = _clean_best_dir(best_dir, prev.get("files", []))
        return {"ok": False, "reason": "leaderboard vacio", "winner": None,
                "removed_from_previous": sorted(removed), "dir": _rel(best_dir)}

    used_col = col
    cand = df[df[col].notna()] if col in df.columns else df.iloc[0:0]
    if not len(cand):                                   # sin esa metrica: plan B
        alt = "best_val_bpt" if col != "best_val_bpt" else "best_gen_score"
        if alt in df.columns and df[alt].notna().any():
            cand, used_col, lower_is_better = df[df[alt].notna()], alt, (alt == "best_val_bpt")
        else:
            cand, used_col = df, None
    order = (cand.sort_values(used_col, ascending=lower_is_better, kind="stable")
             if used_col else cand)
    win = order.iloc[0]
    runner = order.iloc[1] if len(order) > 1 else None

    name = str(win["name"])
    src = exp_root() / name

    # --- plan de copia: (origen, nombre destino)
    ck_src = None
    for cand_name in ("best_gen.pt", "best.pt", "last.pt"):   # best_gen manda
        p = src / "checkpoints" / cand_name
        if p.is_file():
            ck_src = p
            break
    plan, missing = [], []
    if ck_src:
        plan.append((ck_src, "best_checkpoint.pt"))
    else:
        missing.append(_rel(src / "checkpoints") + "/best*.pt")
    for s, dst in ((src / "logs" / METRICS_CSV, METRICS_CSV),
                   (src / "figures" / CURVES_PNG, CURVES_PNG),
                   (src / "config.json", "config.json"),
                   (src / "logs" / SUMMARY_JSON, SUMMARY_JSON)):
        if s.is_file():
            plan.append((s, dst))
        else:
            missing.append(_rel(s))
    for s, dst in ((src / "figures" / "distributions.png", "distributions.png"),):
        if s.is_file():                      # figura opcional: no cuenta como "missing"
            plan.append((s, dst))
    for i, p in enumerate(_pick_generations(src, top_k_gen)):
        plan.append((p, f"generation_{i + 1:02d}_{_safe_stem(p.stem)}.png"))

    # --- lo que ya esta al dia no se vuelve a copiar (el checkpoint pesa cientos de MB
    #     y refresh_all se llama muchas veces durante el entrenamiento)
    prev_copied = prev.get("copied") or {}
    files, copied, reused = [], {}, []
    todo = []
    for s, dst in plan:
        tag = _stat_tag(s)
        d = best_dir / dst
        old = prev_copied.get(dst)
        if (isinstance(old, list) and list(old) == tag and d.is_file()
                and d.stat().st_size == tag[1]):
            files.append(dst)
            copied[dst] = tag
            reused.append(dst)
        else:
            todo.append((s, dst, tag))

    # Se copia PRIMERO y se limpia DESPUES: si una copia falla a mitad (disco
    # lleno, o el .pt bloqueado porque train.py lo esta reescribiendo) el
    # ganador anterior sigue completo en reports/best/ en vez de quedarse
    # mutilado. Cuando se limpia ya no queda ninguna operacion que pueda fallar,
    # asi que la carpeta nunca mezcla dos experimentos.
    for s, dst, tag in todo:
        if _atomic_copy(s, best_dir / dst):
            files.append(dst)
            copied[dst] = tag
        else:
            missing.append(_rel(s))

    removed = _clean_best_dir(best_dir, prev.get("files", []), keep=set(files))

    def _v(row, c):
        """Valor numerico de una celda del leaderboard (None si falta o es NaN)."""
        if row is None or c is None or c not in row or _isna(row[c]):
            return None
        try:
            f = float(row[c])
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return str(row[c])

    def _t(row, c):
        """Valor de texto de una celda (None si falta o es NaN)."""
        if row is None or c is None or c not in row or _isna(row[c]):
            return None
        return str(row[c]).strip()

    wv = _v(win, used_col)
    rv = _v(runner, used_col)
    meta = {
        "winner": name,
        "model": _t(win, "model"),
        "family": _t(win, "family"),
        "metric": metric,
        "metric_column": used_col,
        "lower_is_better": bool(lower_is_better) if used_col else None,
        "value": wv,
        "params_M": _v(win, "params_M"),
        "best_gen_score": _v(win, "best_gen_score"),
        "best_val_bpt": _v(win, "best_val_bpt"),
        "best_val_ppl": _v(win, "best_val_ppl"),
        "frame_f1": _v(win, "frame_f1"),
        "steps": (int(_v(win, "steps")) if isinstance(_v(win, "steps"), float)
                  and float(_v(win, "steps")).is_integer() else _v(win, "steps")),
        "train_hours": _v(win, "train_hours"),
        "experiment_updated": _t(win, "updated"),
        "runner_up": _t(runner, "name"),
        "runner_up_value": rv,
        "margin": (float(wv - rv) if isinstance(wv, float) and isinstance(rv, float) else None),
        "n_experiments": int(len(df)),
        "source_checkpoint": _rel(ck_src) if ck_src else None,
        "source_dir": _rel(src),
        "files": sorted(files),
        "copied": copied,
        "reused": sorted(reused),
        "missing": sorted(set(missing)),
        "removed_from_previous": sorted(removed),
        "promoted_at": _now_iso(),
        "ok": bool(files),
    }

    # --- BEST.md (informe para humanos)
    md = ["# Mejor modelo: " + name, "",
          f"Promovido el {meta['promoted_at']} usando **{metric}** "
          f"({'menor' if lower_is_better else 'mayor'} es mejor).", "",
          "## Metricas del ganador", "",
          f"- modelo `{meta['model']}`, familia `{meta['family']}`, "
          f"{_fmt(meta['params_M'], 2)} M parametros",
          f"- gen_score: **{_fmt(meta['best_gen_score'], 2)}** / 100",
          f"- val_bpt: **{_fmt(meta['best_val_bpt'], 4)}** bits/paso "
          f"(val_ppl {_fmt(meta['best_val_ppl'], 3)})",
          f"- frame_f1: {_fmt(meta['frame_f1'], 4)}",
          f"- pasos: {_fmt(meta['steps'], 0)}, horas de entrenamiento: "
          f"{_fmt(meta['train_hours'], 2)}",
          f"- ultima actualizacion del experimento: {meta['experiment_updated']}", "",
          "## Por que gano", ""]
    if meta["runner_up"]:
        md.append(f"Frente a `{meta['runner_up']}` ({used_col} = {_fmt(rv)}) el ganador "
                  f"alcanza {used_col} = {_fmt(wv)}; diferencia = {_fmt(meta['margin'])}.")
    else:
        md.append("Es el unico experimento con esta metrica disponible.")
    md += ["", "## Comparativa (top 5 del leaderboard)", "",
           _md_table(df.head(5), ["name", "model", "family", "params_M", "best_val_bpt",
                                  "best_gen_score", "steps"],
                     {"params_M": 2, "best_val_bpt": 4, "best_gen_score": 2, "steps": 0}),
           "", "## Ficheros de esta carpeta", ""]
    md += ["- `" + f + "`" for f in meta["files"]] or ["- (ninguno: faltaban artefactos)"]
    if meta["missing"]:
        md += ["", "Artefactos que el ganador no tenia: " +
               ", ".join("`" + m + "`" for m in meta["missing"])]
    md += ["", "---", "",
           "Regenerar con `python src/registry.py --refresh` "
           "(rebuild del leaderboard + figuras + promocion).", ""]
    _atomic_write_text(best_dir / "BEST.md", "\n".join(md))

    meta["files"] = sorted(set(meta["files"] + ["BEST.md", BEST_JSON]))
    write_json(best_dir / BEST_JSON, meta)
    meta["dir"] = _rel(best_dir)
    return meta


# ============================================================= 7. refresh_all
def _make_leaderboard_figure(df) -> dict:
    """Genera reports/figures/leaderboard.png via src/viz.py (import perezoso).

    viz.py lo escribe otro agente: si no existe, no expone plot_leaderboard o
    falla, se anota el aviso y refresh_all continua sin romperse.

    La firma de viz.plot_leaderboard puede pedir la RUTA del CSV o el propio
    DataFrame; se decide por el nombre del primer parametro y, si el intento
    deja constancia en viz.FAILURES, se reintenta con la otra forma.
    """
    out = {"path": None, "error": None}
    if df is None or not len(df):
        out["error"] = "leaderboard vacio"
        return out
    viz = None
    try:
        import viz as _viz
        viz = _viz
    except Exception as e:
        try:
            from src import viz as _viz2              # por si se usa como paquete
            viz = _viz2
        except Exception:
            out["error"] = f"viz no disponible ({type(e).__name__}: {e})"
            return out
    fn = getattr(viz, "plot_leaderboard", None)
    if not callable(fn):
        out["error"] = "viz.plot_leaderboard no existe (aun)"
        return out

    path = _mkdir(reports_root() / "figures") / "leaderboard.png"
    csv_path = reports_root() / LEADERBOARD_CSV
    if not csv_path.is_file():                       # viz suele leer del CSV
        rebuild_leaderboard()

    npos, first = 2, ""
    try:
        import inspect
        ps = list(inspect.signature(fn).parameters.values())
        pos = [p for p in ps if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        npos = 2 if (len(pos) >= 2 or any(p.kind == p.VAR_POSITIONAL for p in ps)) else len(pos)
        first = pos[0].name.lower() if pos else ""
    except Exception:
        pass
    wants_path = any(k in first for k in ("csv", "path", "file", "table"))
    cands = [csv_path, df] if wants_path else [df, csv_path]

    err, res = None, None
    for a0 in (cands if npos >= 1 else [None]):
        n_before = len(getattr(viz, "FAILURES", ()) or ())
        try:
            res = fn(a0, path) if npos >= 2 else (fn(a0) if npos == 1 else fn())
        except Exception as e:
            err = f"plot_leaderboard fallo ({type(e).__name__}: {e})"
            continue
        fails = getattr(viz, "FAILURES", None)
        if fails is not None and len(fails) > n_before:
            err = "viz dejo constancia de un fallo: " + str(fails[-1])
            continue
        err = None
        break

    p = Path(res) if isinstance(res, (str, Path)) and str(res) else path
    if p.is_file():
        out["path"] = _rel(p)
    if err:
        out["error"] = err
    elif not p.is_file():
        out["error"] = "plot_leaderboard no dejo el PNG en " + _rel(path)
    return out


def refresh_all(make_figures: bool = True, metric: str = "gen_score") -> dict:
    """Deja TODO actualizado: leaderboard.csv + figura comparativa + reports/best/."""
    df = rebuild_leaderboard()
    fig = _make_leaderboard_figure(df) if make_figures else {"path": None, "error": "desactivado"}
    best = promote_best(metric=metric, df=df)
    return {"n_experiments": int(len(df)),
            "leaderboard": _rel(reports_root() / LEADERBOARD_CSV),
            "figure": fig.get("path"), "figure_error": fig.get("error"),
            "best": best, "updated": _now_iso()}


# ==================================================================== CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Registro de experimentos: leaderboard y mejor modelo global.")
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild del leaderboard + figuras + promocion del mejor")
    ap.add_argument("--leaderboard", action="store_true", help="solo imprime la tabla")
    ap.add_argument("--promote", action="store_true", help="solo promociona el mejor")
    ap.add_argument("--metric", default="gen_score",
                    help="metrica de promocion: gen_score | val_bpt | val_ppl | frame_f1")
    ap.add_argument("--no-figures", action="store_true", help="no llamar a viz.plot_leaderboard")
    a = ap.parse_args(argv)

    if a.leaderboard and not (a.refresh or a.promote):
        print(format_leaderboard(rebuild_leaderboard()))
        return 0

    if a.promote and not a.refresh:
        df = rebuild_leaderboard()
        print(format_leaderboard(df))
        meta = promote_best(a.metric, df=df)
    else:                                               # por defecto: refresh completo
        res = refresh_all(make_figures=not a.no_figures, metric=a.metric)
        print(format_leaderboard(load_leaderboard(False)))
        print("")
        print(f"experimentos: {res['n_experiments']}   leaderboard: {res['leaderboard']}")
        print("figura: " + str(res["figure"]) +
              (f"   (aviso: {res['figure_error']})" if res["figure_error"] else ""))
        meta = res["best"]

    if meta.get("winner"):
        print(f"mejor: {meta['winner']} ({meta.get('metric_column')} = "
              f"{_fmt(meta.get('value'))})  -> {meta.get('dir')}")
        if meta.get("missing"):
            print("  faltaban: " + ", ".join(meta["missing"]))
    else:
        print("sin ganador: " + str(meta.get("reason")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
