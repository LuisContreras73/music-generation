"""Estrategias de muestreo anti-degeneracion para la familia token (V=155).

Por que existe este modulo
--------------------------
Con prefijos CORTOS (100 pasos = 5 s) y fuera de distribucion (melodias
monofonicas) los modelos del laboratorio degeneran en bucles: la fraccion de
8-gramas de frames repetidos sube a 0.63 frente a 0.217 del corpus real, y el
gen_score cae a 2.0-2.4 sobre una escala donde el ruido puntua 1.6. La causa no
es solo el modelo sino el DECODIFICADOR: el muestreo por nucleo (top-p) fijo
conserva una cola cuyo tamano depende de lo afilada que este la distribucion, y
en cuanto el modelo entra en un atractor de baja entropia (un motivo que se
predice a si mismo con p ~ 0.99) top-p recorta a un unico token y el bucle se
realimenta.

Este modulo NO toca src/generate.py (contrato estable del resto del
laboratorio). Ofrece filtros de logits como FUNCIONES PURAS componibles, un
constructor de muestreadores y una funcion de alto nivel que continua un
prefijo con el mismo contrato que src/generate.py::sample_tokens y ademas
devuelve la fraccion de 8-gramas de frames repetidos del resultado, que es la
medida directa del problema que se quiere resolver.

Convenios de los filtros
------------------------
* Trabajan sobre `logits` de forma [..., V] (la ultima dimension es el
  vocabulario) y devuelven un TENSOR NUEVO; nunca modifican la entrada.
* Solo pueden ANADIR -inf, jamas quitarlo: asi las prohibiciones duras del
  dominio (PAD/BOS/EOS, gramatica, tope de duracion) sobreviven a cualquier
  composicion posterior de filtros.
* Nunca dejan una fila entera en -inf. Cada filtro conserva al menos el argmax
  y, si por un valor de parametro absurdo la mascara vaciara la fila, se
  descarta esa mascara y se devuelve la fila intacta. Un muestreador que recibe
  una distribucion vacia produce NaN al hacer softmax y torch.multinomial
  aborta el proceso: eso no puede pasar a mitad de una generacion larga.
* Una fila que YA llega muerta (todo -inf, error del llamante) se deja tal cual
  en vez de propagar NaN a traves de log_softmax.

El dominio importa
------------------
El vocabulario mezcla dos cosas de naturaleza distinta: NOTE_ON(p) (que evento
ocurre) y SHIFT(d) (cuanto tiempo pasa). Las tecnicas anti-repeticion de texto
suponen que repetir un simbolo es sospechoso, y eso es FALSO para SHIFT: el
tiempo siempre avanza, el 35.0% de los tokens del corpus son SHIFT y su
distribucion esta muy concentrada. Penalizarlos no rompe bucles, rompe el ritmo
(ver el argumento en repetition_penalty y la medicion en la seccion C de
_selftest).

Referencias
-----------
Keskar et al. 2019 (CTRL, penalizacion de repeticion), Meister et al. 2023
(locally typical sampling, TACL), Holtzman et al. 2020 (nucleus), y las
heuristicas top-a / min-p de la practica de LLMs locales (umbral RELATIVO al
maximo en vez de un tamano de nucleo fijo).

Uso
---
    python src/sampling.py              # verificaciones sinteticas (CPU, rapido)
    python src/sampling.py --real       # ademas continua prefijos reales con lstm
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:                 # permite "python src/sampling.py"
    sys.path.insert(0, str(_HERE))

from data.tokenizer import (PAD, BOS, EOS, NOTE_OFF_ID, SHIFT_OFF_ID,       # noqa: E402
                            MAX_SHIFT, N_PITCH, VOCAB_SIZE, decode_tokens)

NEG_INF = float("-inf")
_EMPTY = np.zeros(0, dtype=np.int64)

__all__ = [
    "repetition_penalty", "no_repeat_ngram", "typical_sampling", "top_a", "min_p",
    "top_k_filter", "top_p_filter", "NGramBlocker", "build_sampler", "STRATEGIES",
    "strategy_opts", "sample_with_strategy", "repeat8_frames",
]


# =========================================================== utilidades internas
def _as_ids(seq) -> np.ndarray:
    """Cualquier secuencia de tokens -> np.int64 1-D."""
    if isinstance(seq, torch.Tensor):
        seq = seq.detach().cpu().numpy()
    arr = np.asarray(seq)
    if arr.size == 0:
        return _EMPTY
    return arr.astype(np.int64, copy=False).ravel()


def _histories(generated_ids, n_rows: int) -> list:
    """Normaliza el historial a una lista de n_rows secuencias 1-D.

    Acepta None, una secuencia unica (se comparte con todas las filas, que es lo
    natural cuando se filtra un solo vector de logits) o una lista/array 2-D con
    una secuencia por fila. Las historias por fila pueden tener longitudes
    distintas, asi que NO se convierten a un tensor rectangular.
    """
    if generated_ids is None:
        return [_EMPTY] * n_rows
    if isinstance(generated_ids, (torch.Tensor, np.ndarray)):
        arr = (generated_ids.detach().cpu().numpy()
               if isinstance(generated_ids, torch.Tensor) else generated_ids)
        rows = [_as_ids(r) for r in arr] if arr.ndim == 2 else [_as_ids(arr)] * n_rows
    else:
        lst = list(generated_ids)
        if lst and isinstance(lst[0], (list, tuple, np.ndarray, torch.Tensor)):
            rows = [_as_ids(s) for s in lst]
        else:
            rows = [_as_ids(lst)] * n_rows
    if len(rows) == 1 and n_rows > 1:
        rows = rows * n_rows
    if len(rows) != n_rows:
        raise ValueError("generated_ids tiene %d secuencias y logits %d filas"
                         % (len(rows), n_rows))
    return rows


def _flat(logits: torch.Tensor):
    """Vista [N,V] de un tensor [...,V] + su forma original."""
    if logits.dim() == 0:
        raise ValueError("logits debe tener al menos una dimension")
    return logits.reshape(-1, logits.shape[-1]), logits.shape


def _alive(flat: torch.Tensor) -> torch.Tensor:
    """[N] bool: filas con al menos un logit finito (las demas ya venian muertas)."""
    return torch.isfinite(flat).any(dim=-1)


def _apply_keep(flat: torch.Tensor, keep: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """Enmascara con -inf lo que `keep` descarta, con las dos redes de seguridad.

    1. Una fila que llegaba muerta se devuelve intacta (no se inventa soporte).
    2. Una fila viva que se quedaria sin ningun token conserva su mascara previa
       (se ignora este filtro en esa fila) en vez de emitir una fila vacia.
    """
    keep = keep & torch.isfinite(flat)              # nunca resucita un -inf
    empty = alive & ~keep.any(dim=-1)
    if bool(empty.any()):
        keep[empty] = torch.isfinite(flat[empty])
    keep[~alive] = True
    return flat.masked_fill(~keep, NEG_INF)


def _keep_argmax(keep: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Fuerza a conservar el token mas probable de cada fila (defensa barata)."""
    return keep | (ref >= ref.max(dim=-1, keepdim=True).values)


# ================================================================ 1. penalizacion
def repetition_penalty(logits: torch.Tensor, generated_ids, penalty: float = 1.15,
                       window: int = 64, exclude_shift: bool = True) -> torch.Tensor:
    """Penalizacion de repeticion de CTRL (Keskar et al. 2019) con ventana.

    Para cada token ya emitido en la ventana reciente:
        logit > 0  ->  logit / penalty
        logit <= 0 ->  logit * penalty
    El caso partido es lo que hace que la operacion baje la probabilidad a ambos
    lados del cero; dividir siempre PREMIARIA a los logits negativos.

    Ventana
    -------
    CTRL penaliza todo el historial. Aqui el historial son decenas de miles de
    tokens y el vocabulario tiene 155: penalizarlo entero deja fuera de juego
    todo el vocabulario util en pocos segundos. `window` limita el castigo a los
    ultimos tokens (por defecto 64, del orden de 2-4 s de musica), que es la
    escala a la que aparecen los bucles medidos.

    exclude_shift=True POR DEFECTO, y no es un detalle cosmetico
    ------------------------------------------------------------
    En esta tokenizacion el 35.0% de los tokens del corpus son SHIFT y su
    distribucion esta concentrada: P(d=1)=0.217, P(d=2)=0.215, P(d=3)=0.167,
    media 4.08 pasos. El tiempo SIEMPRE avanza, asi que un SHIFT repetido no es
    degeneracion sino la gramatica del formato. Penalizarlos desplaza masa de
    los shifts cortos (los frecuentes) a los largos, alarga la duracion media
    del salto y produce musica con silencios que el corpus no tiene: eso no baja
    la repeticion de frames, y ademas dispara la penalizacion por densidad del
    gen_score, que compara la densidad generada con la real. Los bucles viven en
    la secuencia de NOTAS; ahi es donde hay que castigar. La seccion C de
    _selftest lo mide sobre la distribucion real de shifts del corpus.

    penalty <= 1 desactiva el filtro (devuelve una copia).
    """
    flat, shape = _flat(logits)
    out = flat.clone().float()
    if penalty is None or penalty <= 1.0:
        return out.reshape(shape)
    hist = _histories(generated_ids, out.shape[0])
    w = int(window) if window and window > 0 else 0
    for i, ids in enumerate(hist):
        if ids.size == 0:
            continue
        recent = ids[-w:] if w else ids
        uniq = np.unique(recent)
        if exclude_shift:
            uniq = uniq[uniq < SHIFT_OFF_ID]
        uniq = uniq[(uniq >= 0) & (uniq < out.shape[1])]
        if uniq.size == 0:
            continue
        idx = torch.from_numpy(uniq).to(out.device)
        v = out[i].index_select(0, idx)
        out[i].index_copy_(0, idx, torch.where(v > 0, v / penalty, v * penalty))
    return out.reshape(shape)


# =============================================================== 2. n-gramas
def _ngram_table(ids: np.ndarray, n: int) -> dict:
    """dict (n-1 tokens) -> set de tokens que ya completaron ese prefijo."""
    tbl = {}
    lst = ids.tolist()
    for i in range(len(lst) - n + 1):
        tbl.setdefault(tuple(lst[i:i + n - 1]), set()).add(lst[i + n - 1])
    return tbl


def no_repeat_ngram(logits: torch.Tensor, generated_ids, n: int = 4) -> torch.Tensor:
    """Prohibe completar cualquier n-grama de tokens que ya haya aparecido.

    Es la unica restriccion DURA contra el bucle exacto: mientras la
    penalizacion solo desplaza probabilidad, esto hace imposible repetir por
    segunda vez la misma secuencia de n tokens, que es la firma del fallo
    medido (8-gramas de frames repetidos 0.63 frente a 0.217).

    Coste
    -----
    Version PURA (esta): construye el diccionario de prefijos de todo el
    historial, O(L) por llamada. Es la que se verifica por su simplicidad. Para
    generar de verdad, build_sampler acepta un NGramBlocker que mantiene el
    mismo diccionario de forma incremental y cuesta O(n) por token en vez de
    O(L*n) (medido en la seccion F de _selftest).

    n < 2 desactiva el filtro. Si prohibir vaciara la fila, no se prohibe nada
    en esa fila (ver _apply_keep): mas vale repetir que abortar por NaN.
    """
    flat, shape = _flat(logits)
    out = flat.clone().float()
    if n is None or n < 2:
        return out.reshape(shape)
    hist = _histories(generated_ids, out.shape[0])
    keep = torch.ones_like(out, dtype=torch.bool)
    for i, ids in enumerate(hist):
        if len(ids) < n:
            continue
        suffix = tuple(ids[len(ids) - n + 1:].tolist())
        for t in _ngram_table(ids, n).get(suffix, ()):
            if 0 <= t < out.shape[1]:
                keep[i, t] = False
    return _apply_keep(out, keep, _alive(out)).reshape(shape)


class NGramBlocker:
    """Version incremental de no_repeat_ngram: O(n) por token, no O(L).

    Mantiene el diccionario prefijo -> conjunto de continuaciones ya vistas y la
    cola de los ultimos n-1 tokens, de modo que banned() devuelve el conjunto
    prohibido para el sufijo actual sin recorrer el historial. Da exactamente el
    mismo conjunto que la funcion pura (se comprueba en la seccion B).
    """

    __slots__ = ("n", "tbl", "buf")

    def __init__(self, n: int = 4, prime=None):
        self.n = int(n)
        self.tbl = {}
        self.buf = []
        if prime is not None:
            for t in _as_ids(prime).tolist():
                self.update(t)

    def update(self, token: int) -> None:
        if self.n < 2:
            return
        if len(self.buf) == self.n - 1:                     # cierra un n-grama
            self.tbl.setdefault(tuple(self.buf), set()).add(int(token))
        self.buf.append(int(token))
        if len(self.buf) > self.n - 1:
            self.buf.pop(0)

    def banned(self) -> set:
        if self.n < 2 or len(self.buf) < self.n - 1:
            return set()
        return self.tbl.get(tuple(self.buf), set())


# ============================================================== 3. typical
def typical_sampling(logits: torch.Tensor, mass: float = 0.95) -> torch.Tensor:
    """Muestreo tipico (Meister et al. 2023, "locally typical sampling").

    Ordena los tokens por |-log p(x) - H[p]| (distancia entre su sorpresa y la
    entropia condicional) y conserva los mas cercanos hasta acumular `mass`.

    Por que aqui: top-p corta por la COLA, luego cuando la distribucion esta
    afilada (justo lo que ocurre dentro de un bucle: p_max ~ 0.99) deja un unico
    token y el bucle se cierra sobre si mismo. El criterio tipico descarta
    tambien los tokens DEMASIADO probables cuando su sorpresa se aleja de la
    entropia esperada, asi que en esos estados de baja entropia obliga a mirar
    alternativas: es la unica de las cinco piezas capaz de sacar al modelo del
    atractor sin usar conocimiento del dominio ni memoria de lo generado.

    mass fuera de (0,1) desactiva el filtro.
    """
    flat, shape = _flat(logits)
    out = flat.clone().float()
    if mass is None or not (0.0 < mass < 1.0):
        return out.reshape(shape)
    alive = _alive(out)
    safe = torch.where(alive.unsqueeze(-1), out, torch.zeros_like(out))
    logp = torch.log_softmax(safe, dim=-1)
    p = logp.exp()
    # p*log p con p=0 (logit -inf) da 0*(-inf)=NaN: hay que forzarlo a 0
    ent = -torch.where(p > 0, p * logp, torch.zeros_like(p)).sum(-1, keepdim=True)
    shifted = (-logp - ent).abs()                    # -inf -> +inf, van al final
    order = torch.argsort(shifted, dim=-1)
    ps = p.gather(-1, order)
    keep_sorted = (ps.cumsum(-1) - ps) < mass        # deja siempre el primero
    keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
    keep = _keep_argmax(keep, p)
    return _apply_keep(out, keep, alive).reshape(shape)


# =========================================================== 4. umbrales relativos
def top_a(logits: torch.Tensor, a: float = 0.2) -> torch.Tensor:
    """Umbral CUADRATICO relativo al maximo: se conserva p >= a * p_max^2.

    Al depender de p_max al cuadrado el corte se adapta solo a la forma de la
    distribucion: si el modelo esta seguro (p_max=0.9, a=0.2) el umbral es 0.162
    y sobrevive poco mas que el pico; si duda (p_max=0.05) el umbral cae a
    0.0005 y deja pasar casi todo. Top-p no hace eso: con p=0.95 fijo, en un
    estado seguro deja 1 token (bucle) y en uno inseguro deja cientos (ruido).

    a <= 0 desactiva el filtro. Siempre sobrevive el argmax.
    """
    flat, shape = _flat(logits)
    out = flat.clone().float()
    if a is None or a <= 0.0:
        return out.reshape(shape)
    alive = _alive(out)
    safe = torch.where(alive.unsqueeze(-1), out, torch.zeros_like(out))
    p = torch.softmax(safe, dim=-1)
    pmax = p.max(dim=-1, keepdim=True).values
    keep = _keep_argmax(p >= (a * pmax * pmax), p)
    return _apply_keep(out, keep, alive).reshape(shape)


def min_p(logits: torch.Tensor, p: float = 0.05) -> torch.Tensor:
    """Umbral LINEAL relativo al maximo: se conserva prob >= p * p_max.

    Misma idea adaptativa que top_a pero con dependencia lineal, asi que recorta
    menos en estados seguros: con p=0.05 y p_max=0.99 sigue admitiendo cualquier
    token con probabilidad >= 0.0495. Util como filtro de seguridad barato por
    debajo de typical, para eliminar la cola absurda sin volver a fijar un
    tamano de nucleo.

    p <= 0 desactiva el filtro. Siempre sobrevive el argmax.
    """
    flat, shape = _flat(logits)
    out = flat.clone().float()
    if p is None or p <= 0.0:
        return out.reshape(shape)
    alive = _alive(out)
    safe = torch.where(alive.unsqueeze(-1), out, torch.zeros_like(out))
    prob = torch.softmax(safe, dim=-1)
    pmax = prob.max(dim=-1, keepdim=True).values
    keep = _keep_argmax(prob >= (p * pmax), prob)
    return _apply_keep(out, keep, alive).reshape(shape)


# ------------- top-k / top-p locales (para reproducir la linea base aqui) ------
def top_k_filter(logits: torch.Tensor, k: int = 0) -> torch.Tensor:
    """Top-k clasico. Esta aqui solo para poder medir la linea base sin importar
    src/generate.py: este modulo es autonomo a proposito."""
    flat, shape = _flat(logits)
    out = flat.clone().float()
    if not k or k <= 0:
        return out.reshape(shape)
    k = min(int(k), out.shape[-1])
    kth = out.topk(k, dim=-1).values[..., -1:]
    return _apply_keep(out, out >= kth, _alive(out)).reshape(shape)


def top_p_filter(logits: torch.Tensor, p: float = 0.0) -> torch.Tensor:
    """Nucleus (Holtzman et al. 2020), misma semantica que src/generate.py."""
    flat, shape = _flat(logits)
    out = flat.clone().float()
    if not p or not (0.0 < p < 1.0):
        return out.reshape(shape)
    alive = _alive(out)
    safe = torch.where(alive.unsqueeze(-1), out, torch.zeros_like(out))
    srt, idx = torch.sort(safe, descending=True, dim=-1)
    ps = srt.softmax(-1)
    keep_sorted = (ps.cumsum(-1) - ps) < p                # deja siempre el primero
    keep = torch.zeros_like(keep_sorted).scatter(-1, idx, keep_sorted)
    return _apply_keep(out, keep, alive).reshape(shape)


# Alias privados: build_sampler recibe opciones que se llaman igual que los
# filtros (min_p, top_a, ...) y las sombrearian dentro de su cuerpo.
_F_REP, _F_NGRAM, _F_TYP, _F_TOPA, _F_MINP = (
    repetition_penalty, no_repeat_ngram, typical_sampling, top_a, min_p)


# ================================================================ 5. muestreador
DEFAULT_OPTS = dict(
    temperature=1.0,
    penalty=1.0, penalty_window=64, exclude_shift=True,
    ngram=0,
    typical=0.0, min_p=0.0, top_a=0.0, top_k=0, top_p=0.0,
    greedy=False,
)


def build_sampler(**opts):
    """Compone los filtros y devuelve sampler(logits, generated_ids) -> int.

    Opciones (todas desactivadas por defecto salvo temperature=1):
        temperature      escala de los logits (<=0 equivale a greedy)
        penalty          factor de repetition_penalty (1.0 = off)
        penalty_window   ventana de la penalizacion, en tokens
        exclude_shift    no penalizar SHIFT (por defecto True, ver el filtro)
        ngram            tamano de n-grama prohibido (0/1 = off)
        typical, min_p, top_a, top_k, top_p   truncados (0 = off)
        greedy           argmax en vez de muestrear
        generator        torch.Generator para reproducibilidad
        seed             atajo: crea un generator de CPU con esa semilla

    El sampler acepta ademas dos argumentos opcionales por palabra clave:
        banned   mascara bool [V] con las prohibiciones DURAS del dominio
        blocker  NGramBlocker ya actualizado (evita el coste O(L) del n-grama)

    ORDEN DE APLICACION y su porque
    -------------------------------
    1. `banned` (PAD/BOS/EOS, gramatica del paso, tope de duracion). Va PRIMERO
       porque top_a/min_p/typical umbralizan RELATIVO al maximo: si un token
       imposible fuera el argmax, fijaria el umbral y borraria tokens validos.
    2. repetition_penalty, sobre los logits CRUDOS y antes de la temperatura.
       Dividir por T despues de dividir por el factor conserva la fuerza del
       castigo; hacerlo al reves la convierte en penalty^(1/T), es decir el
       usuario dejaria de controlar lo que cree controlar.
    3. no_repeat_ngram (prohibicion dura). Antes del truncado para que el
       truncado vea ya el soporte definitivo y no gaste su presupuesto de masa
       en tokens que van a desaparecer.
    4. temperatura.
    5. truncados relativos, en orden typical -> min_p -> top_a -> top_k -> top_p.
       Van al FINAL porque estan definidos sobre la distribucion de la que
       realmente se va a muestrear; aplicarlos antes de la temperatura cortaria
       una distribucion distinta de la usada. Componer varios es intersectar sus
       conjuntos: typical elige el nucleo "tipico", los umbrales relativos
       barren la cola residual, y top_k/top_p estan solo para reproducir lineas
       base dentro de este modulo.
    6. red de seguridad: si tras todo la fila quedase vacia (no deberia, cada
       filtro se protege) se muestrea el argmax de los logits ya prohibidos.
    """
    bad = set(opts) - set(DEFAULT_OPTS) - {"generator", "seed"}
    if bad:
        raise ValueError("opciones desconocidas: %s" % sorted(bad))
    o = dict(DEFAULT_OPTS)
    o.update({k: v for k, v in opts.items() if k in DEFAULT_OPTS})
    gen_holder = [None]
    gen = opts.get("generator")
    if gen is None and opts.get("seed") is not None:
        gen = torch.Generator().manual_seed(int(opts["seed"]))
    gen_holder[0] = gen

    temp = float(o["temperature"])
    greedy = bool(o["greedy"]) or temp <= 0.0

    def sampler(logits: torch.Tensor, generated_ids=None, *, banned=None,
                blocker: NGramBlocker | None = None) -> int:
        """logits [V] (o [1,V]) -> id de token muestreado (int de Python)."""
        x = logits.detach().reshape(-1).float()
        if banned is not None:
            x = x.masked_fill(banned if isinstance(banned, torch.Tensor)
                              else torch.as_tensor(banned, dtype=torch.bool), NEG_INF)
        base = x                                            # referencia de rescate
        if o["penalty"] and o["penalty"] > 1.0:
            x = _F_REP(x, generated_ids, o["penalty"], o["penalty_window"],
                       o["exclude_shift"])
        if blocker is not None:
            ban = blocker.banned()
            if ban:
                idx = torch.tensor(sorted(ban), dtype=torch.long)
                idx = idx[(idx >= 0) & (idx < x.numel())]
                if idx.numel():
                    cand = x.clone()
                    cand[idx] = NEG_INF
                    x = cand if torch.isfinite(cand).any() else x
        elif o["ngram"] and o["ngram"] >= 2:
            x = _F_NGRAM(x, generated_ids, o["ngram"])
        if temp != 1.0 and not greedy:
            x = x / max(temp, 1e-5)
        if o["typical"]:
            x = _F_TYP(x, o["typical"])
        if o["min_p"]:
            x = _F_MINP(x, o["min_p"])
        if o["top_a"]:
            x = _F_TOPA(x, o["top_a"])
        if o["top_k"]:
            x = top_k_filter(x, o["top_k"])
        if o["top_p"]:
            x = top_p_filter(x, o["top_p"])
        if not torch.isfinite(x).any():                     # nunca deberia pasar
            x = torch.full_like(base, NEG_INF)
            x[int(torch.argmax(base))] = 0.0
        if greedy:
            return int(torch.argmax(x))
        probs = torch.softmax(x, dim=-1)
        g = gen_holder[0] if gen_holder[0] is not None else gen
        if g is not None and g.device.type != probs.device.type:
            # torch.multinomial exige que el generador este en el mismo device que
            # las probabilidades. Se re-crea en el device correcto conservando la
            # semilla, para no perder la reproducibilidad.
            g = torch.Generator(device=probs.device).manual_seed(int(gen.initial_seed()))
            gen_holder[0] = g
        return int(torch.multinomial(probs, 1, generator=g))

    sampler.opts = o                       # introspeccion (se guarda en el informe)
    return sampler


# Presets. "baseline" reproduce lo que usa hoy el laboratorio (Config.top_p=0.95)
# para que las comparaciones sean contra el decodificador realmente empleado.
STRATEGIES = {
    "greedy":        dict(greedy=True),
    "baseline":      dict(temperature=1.0, top_p=0.95),
    "typical":       dict(temperature=1.0, typical=0.95),
    "min_p":         dict(temperature=1.0, min_p=0.05),
    "top_a":         dict(temperature=1.0, top_a=0.2),
    "penalty":       dict(temperature=1.0, penalty=1.15, penalty_window=64,
                          exclude_shift=True, top_p=0.95),
    "ngram":         dict(temperature=1.0, ngram=6, top_p=0.95),
    "antiloop":      dict(temperature=1.0, penalty=1.15, penalty_window=64,
                          exclude_shift=True, ngram=6, typical=0.95),
    "antiloop_hard": dict(temperature=1.05, penalty=1.30, penalty_window=128,
                          exclude_shift=True, ngram=4, typical=0.92, min_p=0.02),
    # Control negativo del experimento del docstring de repetition_penalty:
    # identico a "antiloop" pero castigando tambien los SHIFT.
    "antiloop_shiftpen": dict(temperature=1.0, penalty=1.15, penalty_window=64,
                              exclude_shift=False, ngram=6, typical=0.95),
}


def strategy_opts(strategy, **overrides) -> dict:
    """Nombre de preset o dict -> dict de opciones (con sobrescrituras)."""
    if isinstance(strategy, str):
        if strategy not in STRATEGIES:
            raise ValueError("estrategia desconocida: %r. Opciones: %s"
                             % (strategy, sorted(STRATEGIES)))
        o = dict(STRATEGIES[strategy])
    elif isinstance(strategy, dict):
        o = dict(strategy)
    else:
        raise TypeError("strategy debe ser str o dict, llego %r" % type(strategy))
    o.update({k: v for k, v in overrides.items() if v is not None})
    return o


# ====================================================== metrica de bucle (frames)
def repeat8_frames(roll: np.ndarray, n: int = 8) -> float:
    """Fraccion de n-gramas de FRAMES repetidos de un piano-roll [T,88].

    Reimplementa (mismo algoritmo y mismo submuestreo) el campo `repeat8` de
    src/metrics.py::roll_features para que este modulo no dependa de metrics y
    se pueda usar dentro del bucle de muestreo. La identidad numerica con
    metrics se comprueba en la seccion E de _selftest.

    0 = ningun bloque de 8 frames se repite; 1 = todos son copias. Corpus real
    0.217, modelos degenerados con prefijo corto 0.63.
    """
    roll = np.asarray(roll)
    if roll.ndim != 2 or roll.shape[0] < 2 * n:
        return 0.0
    step = max(1, roll.shape[0] // 4000)             # techo de coste, igual que metrics
    w = roll[::step]
    if len(w) < 2 * n:
        return 0.0
    v = np.packbits(w.astype(np.uint8), axis=1)
    g = np.lib.stride_tricks.sliding_window_view(v, (n, v.shape[1]))[:, 0]
    g = g.reshape(len(g), -1)
    return float(1.0 - len(np.unique(g, axis=0)) / len(g))


# ============================================================ muestreo de alto nivel
@torch.no_grad()
def sample_with_strategy(model, prefix_tokens, n_steps: int, strategy="antiloop", *,
                         seq_len: int = 1024, device: str = "cpu",
                         max_tokens: int | None = None, allow_eos: bool = False,
                         enforce_grammar: bool = True, use_state: bool = True,
                         prefix_in_history: bool = True, seed: int | None = None,
                         generator: torch.Generator | None = None, **overrides) -> dict:
    """Continua un prefijo hasta acumular n_steps pasos de 50 ms, con estrategia.

    Mismo contrato de dominio que src/generate.py::sample_tokens:
      * PAD y BOS siempre prohibidos; EOS prohibido salvo allow_eos.
      * se prohiben los SHIFT(d) con d > pasos que faltan, para que ninguna
        muestra se pase del objetivo de duracion y todas sean comparables;
      * con enforce_grammar se prohibe repetir una nota dentro del mismo paso
        (imposible en el corpus, la tokenizacion emite un conjunto ordenado);
      * se mide grammar_prob_mass ANTES de enmascarar, que es la medida valida
        de cuanto ha aprendido el modelo de la gramatica cuando se enmascara.

    Diferencias con sample_tokens: el muestreo lo hace un sampler compuesto por
    build_sampler (no top-k/top-p fijos), el historial de repeticion y los
    n-gramas se llevan por fila, y se devuelve la fraccion de 8-gramas de frames
    repetidos de la CONTINUACION (no del prefijo + continuacion), que es lo que
    se quiere comparar entre estrategias.

    Devuelve dict con: tokens, new, rolls, steps, repeat8, repeat8_mean,
    density, grammar_violation_rate, grammar_prob_mass, n_tokens_generated,
    strategy, opts, elapsed_s, tokens_per_second.
    """
    opts = strategy_opts(strategy, **overrides)
    if generator is None and seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    sampler = build_sampler(generator=generator, **opts)
    ngram_n = int(opts.get("ngram", 0) or 0)

    x = torch.as_tensor(np.asarray(prefix_tokens), dtype=torch.long)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.to(device)
    B, L0 = x.shape
    model.eval().to(device)
    max_tokens = int(max_tokens or (n_steps * 3 + 64))

    steps = np.zeros(B, dtype=np.int64)
    done = np.zeros(B, dtype=bool)
    emitted = [set() for _ in range(B)]                  # notas ya puestas en el paso
    new_tokens = [[] for _ in range(B)]
    hist = [list(x[b].tolist()) if prefix_in_history else [] for b in range(B)]
    blockers = ([NGramBlocker(ngram_n, prime=h) for h in hist] if ngram_n >= 2
                else [None] * B)
    n_note_tok = n_violations = 0
    gram_mass_sum = 0.0
    gram_mass_n = 0

    base_ban = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    base_ban[PAD] = True
    base_ban[BOS] = True
    if not allow_eos:
        base_ban[EOS] = True

    stateful = bool(use_state and getattr(model, "supports_state", lambda: False)())
    state = None
    t_start = time.perf_counter()
    if stateful:                                          # cebado O(L) una sola vez
        if hasattr(model, "forward_state"):
            logits_all, state = model.forward_state(x)
            logits = logits_all[:, -1, :].float()
        else:
            for j in range(L0):
                out, state = model.step(x[:, j:j + 1], state)
            logits = out[:, -1, :].float()

    for _ in range(max_tokens):
        if done.all():
            break
        if not stateful:
            logits = model(x[:, -seq_len:])[:, -1, :].float()

        masked = logits.masked_fill(base_ban, NEG_INF)
        probs_raw = masked.softmax(-1)                    # antes de la gramatica
        nxt = torch.full((B, 1), PAD, dtype=torch.long, device=device)
        for b in range(B):
            if done[b]:
                continue
            ban = base_ban.clone()
            r = int(n_steps - steps[b])
            if r < MAX_SHIFT:                             # no pasarse del objetivo
                lo = SHIFT_OFF_ID + max(r, 0)
                if lo < VOCAB_SIZE:
                    ban[lo:] = True
            if emitted[b]:
                idxs = [NOTE_OFF_ID + p for p in emitted[b]]
                gram_mass_sum += float(probs_raw[b, idxs].sum())
                gram_mass_n += 1
                if enforce_grammar:
                    ban[idxs] = True
            t = sampler(logits[b], hist[b], banned=ban, blocker=blockers[b])
            nxt[b, 0] = t
            new_tokens[b].append(t)
            hist[b].append(t)
            if blockers[b] is not None:
                blockers[b].update(t)
            if t >= SHIFT_OFF_ID:                         # SHIFT
                steps[b] += t - SHIFT_OFF_ID + 1
                emitted[b] = set()
                if steps[b] >= n_steps:
                    done[b] = True
            elif t >= NOTE_OFF_ID:                        # NOTE_ON
                p = t - NOTE_OFF_ID
                n_note_tok += 1
                if p in emitted[b]:
                    n_violations += 1
                emitted[b].add(p)
            else:                                         # EOS
                done[b] = True
        if stateful:
            logits, state = model.step(nxt, state)
            logits = logits[:, -1, :].float()
        else:
            x = torch.cat([x, nxt], dim=1)
    elapsed = time.perf_counter() - t_start

    pref = x[:, :L0].cpu().numpy()
    new = [np.asarray(t, np.int64) for t in new_tokens]
    rolls = [decode_tokens(t, n_positions=N_PITCH, max_steps=n_steps) for t in new]
    rep8 = np.array([repeat8_frames(r) for r in rolls], dtype=np.float64)
    dens = np.array([float(r.sum()) / max(len(r), 1) for r in rolls], dtype=np.float64)
    n_gen = int(sum(len(t) for t in new))
    return dict(
        tokens=[np.concatenate([pref[b], new[b]]) for b in range(B)],
        new=new, rolls=rolls, steps=steps.copy(),
        repeat8=rep8, repeat8_mean=float(rep8.mean()),
        density=dens, density_mean=float(dens.mean()),
        grammar_violation_rate=float(n_violations / max(n_note_tok, 1)),
        grammar_prob_mass=float(gram_mass_sum / max(gram_mass_n, 1)),
        n_tokens_generated=n_gen,
        strategy=strategy if isinstance(strategy, str) else "custom",
        opts=dict(sampler.opts), elapsed_s=float(elapsed),
        tokens_per_second=float(n_gen / max(elapsed, 1e-9)),
    )


# ===========================================================================
# verificaciones propias del modulo   (python src/sampling.py [--real])
# ===========================================================================
def _title(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


def _corpus_shift_probs():
    """P(SHIFT(d)) empirica del corpus; None si no esta tokens.bin."""
    p = _HERE.parent / "data" / "processed" / "tokens.bin"
    if not p.exists():
        return None
    a = np.memmap(p, dtype=np.uint8, mode="r")
    c = np.bincount(np.asarray(a[:8_000_000]), minlength=VOCAB_SIZE)[SHIFT_OFF_ID:]
    return c.astype(np.float64) / c.sum()


def _check_a():
    """Cada filtro por separado, con logits donde la respuesta se sabe a mano."""
    _title("A. FILTROS AISLADOS (logits sinteticos con respuesta conocida)")
    ok = True

    # --- top_a / min_p sobre una distribucion disenada a mano ------------------
    # probs objetivo: 0.5, 0.3, 0.15, 0.04, 0.01  -> logits = log(p)
    p = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
    lg = torch.log(torch.tensor(p, dtype=torch.float32))
    out = min_p(lg, 0.1)                       # umbral = 0.1*0.5 = 0.05 -> deja 3
    keep = torch.isfinite(out).tolist()
    exp = [True, True, True, False, False]
    print("min_p(p=0.1)  umbral=%.4f  keep=%s  esperado=%s" % (0.1 * 0.5, keep, exp))
    ok &= keep == exp
    out = top_a(lg, 0.2)                       # umbral = 0.2*0.25 = 0.05 -> deja 3
    keep = torch.isfinite(out).tolist()
    print("top_a(a=0.2)  umbral=%.4f  keep=%s  esperado=%s" % (0.2 * 0.25, keep, exp))
    ok &= keep == exp
    # adaptatividad: la MISMA a con una distribucion plana no debe recortar nada
    flat5 = torch.zeros(5)
    keep_flat = torch.isfinite(top_a(flat5, 0.2)).tolist()
    print("top_a(a=0.2) sobre uniforme(5): umbral=%.4f keep=%s (no recorta: adaptativo)"
          % (0.2 * 0.2 ** 2, keep_flat))
    ok &= all(keep_flat)
    # y top_p=0.95 sobre la MISMA uniforme recorta, aunque no haya nada que quitar
    keep_tp = torch.isfinite(top_p_filter(flat5, 0.95)).tolist()
    print("top_p(0.95)  sobre uniforme(5): keep=%s  <- corte por cola, no adaptativo"
          % keep_tp)

    # --- typical: dos casos calculados a mano ---------------------------------
    # uniforme(4): H=log4 y todas las sorpresas valen log4 -> distancia 0 para
    # todas; con mass=0.9 se acumulan 0.25+0.25+0.25 = 0.75 < 0.9 y entra la 4a.
    u = torch.zeros(4)
    keep_u = torch.isfinite(typical_sampling(u, 0.9)).tolist()
    print("typical(0.9) sobre uniforme(4): keep=%s esperado=[T,T,T,T]" % keep_u)
    ok &= all(keep_u)
    # distribucion afilada: p=[0.99, 0.005, 0.0033, 0.0017]; H = 0.0752 nats.
    # sorpresas: 0.0101, 5.298, 5.714, 6.377 -> el pico es el MAS cercano a H,
    # entra primero (0.99 >= mass) y typical se queda solo con el.
    ps = torch.tensor([0.99, 0.005, 0.0033, 0.0017])
    lg2 = torch.log(ps)
    H = float(-(ps * lg2).sum())
    d = (-lg2 - H).abs()
    keep_s = torch.isfinite(typical_sampling(lg2, 0.9)).tolist()
    print("typical(0.9) sobre p=[.99,.005,.0033,.0017]: H=%.4f nats, |sorpresa-H|=%s"
          % (H, np.round(d.numpy(), 3).tolist()))
    print("             keep=%s  (con mass=0.9 el pico ya cubre la masa)" % keep_s)
    ok &= keep_s == [True, False, False, False]
    # ...pero con mass=0.995 typical ANADE alternativas, mientras top_p(0.95)
    # se queda con el pico solo: esa es la diferencia que rompe el bucle.
    k995 = torch.isfinite(typical_sampling(lg2, 0.995)).tolist()
    ktp = torch.isfinite(top_p_filter(lg2, 0.95)).tolist()
    print("typical(0.995) keep=%s   vs   top_p(0.95) keep=%s" % (k995, ktp))
    ok &= sum(k995) > sum(ktp)

    # --- ningun filtro deja la fila entera en -inf, ni produce NaN ------------
    _title("A.2 NINGUN FILTRO VACIA LA FILA NI PRODUCE NaN (casos extremos)")
    cases = {
        "uniforme": torch.zeros(VOCAB_SIZE),
        "un_pico": torch.full((VOCAB_SIZE,), -30.0).index_fill_(0, torch.tensor([7]), 30.0),
        "casi_toda_prohibida": torch.full((VOCAB_SIZE,), NEG_INF).index_fill_(
            0, torch.tensor([3, 100]), 1.0),
        "una_sola_viva": torch.full((VOCAB_SIZE,), NEG_INF).index_fill_(
            0, torch.tensor([42]), 0.0),
        "fila_muerta": torch.full((VOCAB_SIZE,), NEG_INF),
        "logits_enormes": torch.linspace(-1e4, 1e4, VOCAB_SIZE),
    }
    filters = {
        "repetition_penalty": lambda t: repetition_penalty(t, list(range(0, 155)), 2.0, 64, False),
        "no_repeat_ngram(2)": lambda t: no_repeat_ngram(t, list(range(155)) * 2, 2),
        "typical(0.9)": lambda t: typical_sampling(t, 0.9),
        "top_a(0.9)": lambda t: top_a(t, 0.9),
        "min_p(0.99)": lambda t: min_p(t, 0.99),
        "top_k(1)": lambda t: top_k_filter(t, 1),
        "top_p(0.01)": lambda t: top_p_filter(t, 0.01),
    }
    print("%-22s %s" % ("filtro", "  ".join("%-20s" % c for c in cases)))
    for fname, f in filters.items():
        row = []
        for cname, t in cases.items():
            y = f(t.clone())
            n_alive = int(torch.isfinite(y).sum())
            nan = bool(torch.isnan(torch.softmax(y, -1)).any())
            dead_in = not bool(torch.isfinite(t).any())
            good = (n_alive >= 1 and not nan) or (dead_in and not nan)
            ok &= good
            row.append("%-20s" % ("vivos=%d nan=%s" % (n_alive, nan)))
        print("%-22s %s" % (fname, "  ".join(row)))
    print("\n(la fila_muerta llega ya vacia: el contrato es NO propagar NaN, "
          "no inventar soporte)")
    print("RESULTADO A: %s" % ("OK" if ok else "FALLO"))
    return ok


def _check_b():
    """no_repeat_ngram: prohibe exactamente el token que cerraria el n-grama."""
    _title("B. no_repeat_ngram: SOLO SE PROHIBE LO QUE DEBE")
    ok = True
    # secuencia con el 3-grama (60, 61, 62) ya visto, y el sufijo actual (60,61)
    seq = [91, 60, 61, 62, 95, 70, 91, 60, 61]
    n = 3
    lg = torch.zeros(VOCAB_SIZE)
    out = no_repeat_ngram(lg, seq, n)
    banned = torch.nonzero(~torch.isfinite(out)).ravel().tolist()
    print("secuencia      : %s" % seq)
    print("n              : %d   sufijo actual: %s" % (n, seq[-(n - 1):]))
    print("prohibidos     : %s   (esperado [62]: cerraria 60,61,62 que ya aparecio)"
          % banned)
    ok &= banned == [62]
    print("logit de 62    : %s  (prohibido)" % out[62].item())
    print("logit de 63    : %s  (permitido, nunca siguio a 60,61)" % out[63].item())
    ok &= (not np.isfinite(out[62].item())) and np.isfinite(out[63].item())
    # el mismo n-grama con n=4 no esta cerrado -> no debe prohibir nada
    out4 = no_repeat_ngram(lg, seq, 4)
    b4 = torch.nonzero(~torch.isfinite(out4)).ravel().tolist()
    print("con n=4        : prohibidos=%s (esperado []: (91,60,61) solo aparece 1 vez)" % b4)
    ok &= b4 == []
    # n<2 desactiva
    ok &= torch.isfinite(no_repeat_ngram(lg, seq, 1)).all().item()
    print("con n=1        : filtro desactivado, todos finitos -> OK")

    # equivalencia con la version incremental sobre una secuencia larga aleatoria
    rng = np.random.default_rng(0)
    long_seq = rng.integers(3, VOCAB_SIZE, size=4000).tolist()
    for nn in (3, 4, 6, 8):
        blk = NGramBlocker(nn, prime=long_seq)
        pure = torch.nonzero(~torch.isfinite(no_repeat_ngram(lg, long_seq, nn))).ravel().tolist()
        inc = sorted(blk.banned())
        same = pure == inc
        ok &= same
        print("n=%d  pura=%s  incremental=%s  iguales=%s" % (nn, pure, inc, same))

    # caso patologico: bucle perfecto. La version pura prohibe la continuacion
    # del bucle, y como quedaria algo vivo, la prohibicion se aplica de verdad.
    loop = [3, 95, 5, 95] * 40
    out_l = no_repeat_ngram(torch.zeros(VOCAB_SIZE), loop, 4)
    bl = torch.nonzero(~torch.isfinite(out_l)).ravel().tolist()
    print("bucle [3,95,5,95]x40, n=4 -> prohibidos %s (rompe la iteracion 41)" % bl)
    ok &= len(bl) >= 1
    print("RESULTADO B: %s" % ("OK" if ok else "FALLO"))
    return ok


def _check_c():
    """repetition_penalty: que le pasa a las DURACIONES si se penalizan los SHIFT."""
    _title("C. repetition_penalty CON Y SIN EXCLUSION DE SHIFT")
    ok = True
    lg = torch.zeros(VOCAB_SIZE)
    lg[10] = 2.0                                  # logit positivo
    lg[100] = -2.0                                # logit negativo (un SHIFT)
    hist = [10, 100]
    a = repetition_penalty(lg, hist, 2.0, 64, exclude_shift=True)
    b = repetition_penalty(lg, hist, 2.0, 64, exclude_shift=False)
    print("logit 10 (nota, +2.0) -> excl=%+.3f  incl=%+.3f  (2.0/2 = 1.0 en ambos)"
          % (a[10], b[10]))
    print("logit 100 (SHIFT, -2.0) -> excl=%+.3f  incl=%+.3f  (-2.0*2 = -4.0 solo si se incluye)"
          % (a[100], b[100]))
    ok &= abs(float(a[10]) - 1.0) < 1e-6 and abs(float(b[10]) - 1.0) < 1e-6
    ok &= abs(float(a[100]) + 2.0) < 1e-6 and abs(float(b[100]) + 4.0) < 1e-6

    # --- efecto acumulado sobre la distribucion REAL de duraciones ------------
    sp = _corpus_shift_probs()
    src = "corpus (tokens.bin)"
    if sp is None:                                 # respaldo sintetico si no hay datos
        d = np.arange(1, MAX_SHIFT + 1)
        sp = np.exp(-d / 4.0); sp /= sp.sum(); src = "geometrica sintetica"
    d = np.arange(1, MAX_SHIFT + 1)
    print("\ndistribucion de SHIFT usada: %s ; P(1)=%.4f P(2)=%.4f P(3)=%.4f media=%.3f pasos"
          % (src, sp[0], sp[1], sp[2], float((d * sp).sum())))

    # Simulacion: un "modelo" que siempre propone exactamente la distribucion de
    # duraciones del corpus. Se muestrean 4000 shifts con y sin exclusion,
    # manteniendo la ventana de penalizacion sobre lo ya emitido. Si el filtro
    # respeta el dominio, la distribucion muestreada debe seguir siendo la del
    # corpus; si penaliza los SHIFT, se deforma.
    base = torch.log(torch.tensor(sp, dtype=torch.float64)).float()
    full = torch.full((VOCAB_SIZE,), NEG_INF)
    full[SHIFT_OFF_ID:] = base
    res = {}
    for excl in (True, False):
        g = torch.Generator().manual_seed(7)
        hist_ids, drawn = [], []
        for _ in range(4000):
            x = repetition_penalty(full, hist_ids, 1.15, 64, exclude_shift=excl)
            t = int(torch.multinomial(torch.softmax(x, -1), 1, generator=g))
            hist_ids.append(t)
            drawn.append(t - SHIFT_OFF_ID + 1)
        drawn = np.asarray(drawn)
        h = np.bincount(drawn, minlength=MAX_SHIFT + 1)[1:].astype(np.float64)
        h /= h.sum()
        oa = float(np.minimum(h, sp).sum())
        res[excl] = (float(drawn.mean()), oa, h)
        print("  exclude_shift=%-5s  duracion media=%.3f pasos (%.0f ms)  "
              "P(d=1)=%.4f  P(d>=16)=%.4f  OA vs corpus=%.4f"
              % (excl, drawn.mean(), drawn.mean() * 50, h[0], h[15:].sum(), oa))
    m_ex, oa_ex, _ = res[True]
    m_in, oa_in, _ = res[False]
    print("\n  penalizar los SHIFT alarga la duracion media un %+.1f%% (%.3f -> %.3f "
          "pasos) y baja el solapamiento con el corpus de %.4f a %.4f."
          % (100 * (m_in / m_ex - 1), m_ex, m_in, oa_ex, oa_in))
    print("  Consecuencia directa: menos onsets por segundo -> la penalizacion por")
    print("  densidad del gen_score, exp(-|log(d_gen/d_ref)|), castiga el resultado")
    print("  aunque el bucle no haya mejorado. Por eso el defecto es EXCLUIRLOS.")
    ok &= (m_in > m_ex) and (oa_in < oa_ex)
    print("RESULTADO C: %s" % ("OK" if ok else "FALLO"))
    return ok


def _check_d():
    """Composicion de todos los filtros: 1000 pasos aleatorios sin fila vacia."""
    _title("D. COMPOSICION DE TODOS LOS FILTROS, 1000 PASOS ALEATORIOS")
    rng = np.random.default_rng(1234)
    g = torch.Generator().manual_seed(1234)
    sampler = build_sampler(temperature=1.0, penalty=1.3, penalty_window=128,
                            exclude_shift=True, ngram=4, typical=0.92,
                            min_p=0.02, top_a=0.1, top_k=40, top_p=0.9,
                            generator=g)
    hist = []
    blk = NGramBlocker(4)
    n_alive_min = VOCAB_SIZE
    worst = None
    for i in range(1000):
        scale = float(rng.choice([0.1, 1.0, 8.0, 40.0]))      # de plana a degenerada
        lg = torch.from_numpy(rng.normal(0, scale, VOCAB_SIZE)).float()
        ban = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
        ban[PAD] = ban[BOS] = ban[EOS] = True
        if rng.random() < 0.5:                                 # tope de duracion duro
            ban[SHIFT_OFF_ID + int(rng.integers(0, MAX_SHIFT)):] = True
        for p in rng.choice(N_PITCH, size=int(rng.integers(0, 40)), replace=False):
            ban[NOTE_OFF_ID + int(p)] = True                   # gramatica del paso
        t = sampler(lg, hist, banned=ban, blocker=blk)
        assert not ban[t], "se muestreo un token prohibido en el paso %d" % i
        hist.append(t)
        blk.update(t)
        x = lg.masked_fill(ban, NEG_INF)
        y = top_p_filter(top_k_filter(min_p(top_a(typical_sampling(
            repetition_penalty(x, hist, 1.3, 128, True), 0.92), 0.1), 0.02), 40), 0.9)
        na = int(torch.isfinite(y).sum())
        if na < n_alive_min:
            n_alive_min, worst = na, i
        assert na >= 1, "fila vacia en el paso %d" % i
        assert not bool(torch.isnan(torch.softmax(y, -1)).any()), "NaN en el paso %d" % i
    uniq = len(set(hist))
    print("1000 pasos con banned aleatorio (PAD/BOS/EOS + tope de shift + gramatica)")
    print("  tokens distintos muestreados : %d de %d" % (uniq, VOCAB_SIZE))
    print("  minimo de tokens vivos tras los 7 filtros encadenados: %d (paso %d)"
          % (n_alive_min, worst))
    print("  ningun NaN, ningun token prohibido muestreado, ninguna fila vacia")
    print("RESULTADO D: OK")
    return True


def _check_e():
    """repeat8_frames identico a metrics.roll_features()['repeat8']."""
    _title("E. repeat8_frames COINCIDE CON src/metrics.py")
    import metrics
    rng = np.random.default_rng(3)
    ok = True
    cases = {}
    cases["aleatorio"] = (rng.random((400, N_PITCH)) < 0.005).astype(np.uint8)
    motif = (rng.random((8, N_PITCH)) < 0.02).astype(np.uint8)
    cases["bucle_puro"] = np.tile(motif, (50, 1))
    cases["mitad_bucle"] = np.concatenate(
        [cases["aleatorio"][:200], np.tile(motif, (25, 1))])
    cases["silencio"] = np.zeros((400, N_PITCH), np.uint8)
    p = _HERE.parent / "data" / "processed" / "rolls_packed.bin"
    if p.exists():
        mm = np.memmap(p, dtype=np.uint8, mode="r").reshape(-1, 11)
        cases["corpus_real(2000 pasos)"] = np.unpackbits(
            np.asarray(mm[1_000_000:1_002_000]), axis=1)[:, :N_PITCH]
    for k, r in cases.items():
        a = repeat8_frames(r)
        b = metrics.roll_features(r)["repeat8"]
        ok &= abs(a - b) < 1e-12
        print("  %-24s repeat8_frames=%.4f  metrics=%.4f  iguales=%s"
              % (k, a, b, abs(a - b) < 1e-12))
    print("RESULTADO E: %s" % ("OK" if ok else "FALLO"))
    return ok


def _check_f():
    """Sobrecoste por token de cada filtro (V=155, CPU)."""
    _title("F. SOBRECOSTE POR TOKEN DE CADA FILTRO (CPU, V=155)")
    rng = np.random.default_rng(5)
    lg = torch.from_numpy(rng.normal(0, 3, VOCAB_SIZE)).float()
    hist_short = rng.integers(3, VOCAB_SIZE, 128).tolist()
    hist_long = rng.integers(3, VOCAB_SIZE, 8192).tolist()
    blk = NGramBlocker(6, prime=hist_long)

    def bench(fn, reps=2000):
        fn()                                                   # calentamiento
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e6         # microsegundos

    rows = [
        ("repetition_penalty (win=64)", lambda: repetition_penalty(lg, hist_short, 1.15, 64, True)),
        ("repetition_penalty (win=256)", lambda: repetition_penalty(lg, hist_long, 1.15, 256, True)),
        ("no_repeat_ngram n=6, L=128", lambda: no_repeat_ngram(lg, hist_short, 6)),
        ("no_repeat_ngram n=6, L=8192", lambda: no_repeat_ngram(lg, hist_long, 6)),
        ("NGramBlocker.banned+update", lambda: (blk.banned(), blk.update(7))),
        ("typical_sampling(0.95)", lambda: typical_sampling(lg, 0.95)),
        ("top_a(0.2)", lambda: top_a(lg, 0.2)),
        ("min_p(0.05)", lambda: min_p(lg, 0.05)),
        ("top_k(40)", lambda: top_k_filter(lg, 40)),
        ("top_p(0.95)", lambda: top_p_filter(lg, 0.95)),
    ]
    print("%-32s %12s" % ("filtro", "us/llamada"))
    for name, fn in rows:
        print("%-32s %12.2f" % (name, bench(fn)))
    print("\n%-32s %12s" % ("estrategia completa", "us/token"))
    for name in ("baseline", "typical", "antiloop", "antiloop_hard"):
        s = build_sampler(seed=0, **STRATEGIES[name])
        ban = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
        ban[PAD] = ban[BOS] = ban[EOS] = True
        b = NGramBlocker(int(STRATEGIES[name].get("ngram", 0) or 0), prime=hist_long)
        use_b = b if b.n >= 2 else None
        print("%-32s %12.2f" % (name, bench(
            lambda s=s, use_b=use_b: s(lg, hist_long, banned=ban, blocker=use_b))))
    print("\nReferencia: un paso del LSTM base en CPU cuesta ~7000 us/token, y en")
    print("GPU ~2000 us; el filtrado mas caro esta tres ordenes por debajo.")
    return True


def _check_g():
    """Muestreo de extremo a extremo con un modelo pequeno aleatorio (sin GPU)."""
    _title("G. sample_with_strategy DE EXTREMO A EXTREMO (LSTM pequeno aleatorio)")
    from config import Config
    from models.lstm_baseline import LSTMBaseline
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)        # fuera de presupuesto
        cfg = Config(name="tmp", model="lstm", family="token", d_model=64,
                     hidden=96, n_layers=1, dropout=0.0, tie_weights=True)
        model = LSTMBaseline(cfg)
    torch.manual_seed(0)
    model.eval()
    print("modelo: %s (CPU, pesos aleatorios: solo comprueba el CONTRATO)"
          % model.param_report())
    prefix = np.array([[BOS] + [NOTE_OFF_ID + 40, SHIFT_OFF_ID + 3] * 12], np.int64)
    ok = True
    print("\n%-16s %6s %7s %8s %9s %9s %9s" % ("estrategia", "steps", "tokens",
                                               "repeat8", "densidad", "gram_viol", "s"))
    for name in ("greedy", "baseline", "typical", "min_p", "top_a", "antiloop",
                 "antiloop_hard"):
        r = sample_with_strategy(model, prefix, 100, name, device="cpu", seed=11)
        exact = int(r["steps"][0]) == 100
        ok &= exact and r["grammar_violation_rate"] == 0.0
        ok &= len(r["rolls"][0]) <= 100 and r["tokens"][0].shape[0] == \
            prefix.shape[1] + len(r["new"][0])
        print("%-16s %6d %7d %8.4f %9.4f %9.3f %9.2f"
              % (name, r["steps"][0], r["n_tokens_generated"], r["repeat8_mean"],
                 r["density_mean"], r["grammar_violation_rate"], r["elapsed_s"]))
    print("\ncontrato: steps == 100 exactos en todas, gramatica sin violaciones,")
    print("tokens == prefijo + nuevos, roll de la continuacion <= 100 pasos")
    # lote de 3 filas con historiales distintos
    pref3 = np.repeat(prefix, 3, axis=0)
    r3 = sample_with_strategy(model, pref3, 60, "antiloop", device="cpu", seed=3)
    print("lote B=3 -> steps=%s repeat8=%s" % (r3["steps"].tolist(),
                                               np.round(r3["repeat8"], 3).tolist()))
    ok &= (r3["steps"] == 60).all()
    # el sampler es determinista con la misma semilla
    r_a = sample_with_strategy(model, prefix, 60, "antiloop", device="cpu", seed=5)
    r_b = sample_with_strategy(model, prefix, 60, "antiloop", device="cpu", seed=5)
    same = np.array_equal(r_a["new"][0], r_b["new"][0])
    print("reproducible con seed=5: %s" % same)
    ok &= same
    print("RESULTADO G: %s" % ("OK" if ok else "FALLO"))
    return ok


def _check_real(n_steps: int = 300):
    """Prueba con el checkpoint real de lstm sobre los prefijos externos."""
    _title("H. MODELO REAL lstm EN CPU SOBRE LOS PREFIJOS DE 5 s")
    from config import Config
    from models import build_model
    from data.tokenizer import encode_roll_fast
    root = _HERE.parent
    ck_path = root / "experiments" / "lstm" / "checkpoints" / "best_gen.pt"
    if not ck_path.exists():
        print("no hay checkpoint en %s -> se omite" % ck_path)
        return True
    torch.set_num_threads(4)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in ck["config"].items()
                    if k in Config.__dataclass_fields__})
    model = build_model(cfg)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print("modelo %s  paso %s" % (model.param_report(), ck.get("step")))

    z = np.load(root / "external_eval_prefix_5s.npz", allow_pickle=True)
    off = z["offsets"]
    rolls = [z["rolls_flat"][off[i]:off[i + 1]] for i in range(len(off) - 1)]
    ids = [str(s) for s in z["ids"]]
    prefixes = [encode_roll_fast(r, add_bos=True, add_eos=False).astype(np.int64)
                for r in rolls]
    L = max(len(p) for p in prefixes)
    X = np.full((len(prefixes), L), PAD, np.int64)
    for i, p in enumerate(prefixes):                 # relleno por la IZQUIERDA
        X[i, L - len(p):] = p                        # (el ultimo token es el real)
    print("prefijos: %s  (%d pasos cada uno, %d tokens tras el relleno)"
          % (ids, len(rolls[0]), L))
    ref_rep8 = 0.217
    print("\n%-20s %8s %8s %9s %9s %8s" % ("estrategia", "repeat8", "vs 0.217",
                                           "densidad", "tokens", "s"))
    out = {}
    for name in ("baseline", "typical", "min_p", "top_a", "penalty", "ngram",
                 "antiloop", "antiloop_shiftpen", "antiloop_hard"):
        r = sample_with_strategy(model, X, n_steps, name, device="cpu", seed=2024)
        out[name] = r
        print("%-20s %8.4f %8.4f %9.4f %9d %8.1f"
              % (name, r["repeat8_mean"], r["repeat8_mean"] - ref_rep8,
                 r["density_mean"], r["n_tokens_generated"], r["elapsed_s"]))
    print("\nrepeat8 por muestra (baseline -> antiloop):")
    for i, s in enumerate(ids):
        print("  %-28s %.4f -> %.4f" % (s[:28], out["baseline"]["repeat8"][i],
                                        out["antiloop"]["repeat8"][i]))
    print("\nDensidad del corpus: 0.4526 onsets/paso. repeat8 del corpus: 0.217.")
    return True


def _selftest(real: bool = False) -> bool:
    ok = True
    ok &= _check_a()
    ok &= _check_b()
    ok &= _check_c()
    ok &= _check_d()
    ok &= _check_e()
    ok &= _check_f()
    ok &= _check_g()
    if real:
        ok &= _check_real()
    _title("RESUMEN: %s" % ("TODAS LAS VERIFICACIONES OK" if ok else "HAY FALLOS"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest(real="--real" in sys.argv) else 1)
