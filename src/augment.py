"""
Augmentacion de datos para piano-roll de ONSETS tokenizado (NOTE_ON + SHIFT).

Por que existe este modulo
--------------------------
El corpus es piano polifonico DENSO (polifonia media 1.86 en los pasos activos,
75.6% de pasos vacios) y los modelos entrenados sobre el degeneran en bucles
cuando se les da un prefijo corto, monofonico y ralo: la fraccion de 8-gramas de
frames repetidos sube a 0.63 (corpus real 0.217) y gen_score cae a 2.0-2.4,
mientras que con prefijos del propio corpus los MISMOS pesos puntuan 70-83.

Eso no es solo falta de datos: es un MISMATCH DE DOMINIO. El modelo nunca ha
visto una textura monofonica, asi que al condicionarlo con una cae en el
atractor mas cercano que si conoce. Por eso la augmentacion de aqui no busca
solo multiplicar ejemplos sino AMPLIAR EL RANGO DE TEXTURAS:

  transpose      x ~12 datos efectivos e invariancia a la tonalidad. Barato y
                 EXACTAMENTE invertible.
  time_stretch   variacion de tempo/agogica; ademas descorrelaciona la rejilla
                 de 50 ms de la metrica de la pieza.
  thin_voices    LA CLAVE PARA ESTE PROBLEMA: borra notas de los acordes
                 conservando siempre una por paso, es decir FABRICA la textura
                 monofonica-rala con la que el modelo falla, y la fabrica
                 dentro de la distribucion armonica del corpus, sin datos
                 nuevos.
  onset_jitter   desincroniza notas de un mismo acorde -> arpegios y rubato,
                 texturas intermedias entre acorde y linea.

Diseno
------
Las transformaciones no triviales trabajan sobre EVENTOS (times, pitches), no
sobre el roll [T,88]: una ventana de 1024 tokens son ~350 notas frente a 88*T
celdas, asi que cuesta un orden de magnitud menos y no hay que materializar el
roll. La conversion tokens <-> eventos esta vectorizada y es exacta; el
codificador de eventos reproduce BYTE A BYTE la salida de
data.tokenizer.encode_roll_fast (se verifica en __main__ sobre piezas reales).

transpose es la excepcion: se aplica DIRECTAMENTE sobre los ids de token porque
es una suma con mascara (~5 us) y porque asi es exactamente invertible aunque
la ventana empiece a media pieza, sin BOS ni EOS y con cadenas de SHIFT que no
tienen por que ser canonicas.

Invariantes que garantiza toda salida de este modulo (validate_tokens las
comprueba): ids en [0,155), SHIFT en [1,64] por construccion, y notas de un
mismo paso en orden ASCENDENTE y SIN REPETIR. Esta ultima no es gratis: al
comprimir el tiempo o al mover onsets se pueden colapsar dos pasos en uno, y si
comparten una nota hay que deduplicar porque la tokenizacion NO puede
representar dos onsets del mismo pitch en el mismo paso.

Uso en el DataLoader (TokenWindows.__getitem__, antes de construir x/y):

    from augment import augment_tokens, TEXTURE_CFG
    chunk = augment_tokens(chunk, rng, TEXTURE_CFG)     # ~85 us por ventana

El rng debe derivarse del indice de la ventana (como ya hace TokenWindows) para
que la augmentacion sea reproducible y distinta en cada worker.

Verificacion: `python src/augment.py` ejecuta 34 comprobaciones sobre piezas
reales del corpus e imprime la tabla de invariantes y tiempos.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:                      # permite "python src/augment.py"
    sys.path.insert(0, _SRC)                  # y "sys.path.insert(0,'src')"

from data.tokenizer import (                                        # noqa: E402
    PAD, BOS, EOS, NOTE_OFF_ID, SHIFT_OFF_ID, MAX_SHIFT, VOCAB_SIZE, N_PITCH,
)

__all__ = [
    "AugmentConfig", "DEFAULT_CFG", "TEXTURE_CFG",
    "Events", "tokens_to_events", "events_to_tokens", "canonicalize_events",
    "validate_tokens", "valid_transpositions", "transpose", "time_stretch",
    "thin_voices", "onset_jitter", "augment_tokens",
]

# Clave entera (t, p) -> t * 128 + p. 128 es potencia de 2 (shift en vez de
# division) y 88 < 128, asi que la codificacion es biyectiva y su orden natural
# es exactamente el orden canonico de la tokenizacion.
_PSHIFT = 7
_PMASK = (1 << _PSHIFT) - 1


# --------------------------------------------------------------------------- #
# configuracion
# --------------------------------------------------------------------------- #
@dataclass
class AugmentConfig:
    """Probabilidades y rangos de cada transformacion.

    Los valores por defecto son conservadores salvo en thin_voices: se aplica
    en 1 de cada 4 ventanas porque es la transformacion que ataca el fallo
    observado (prefijos monofonicos). Subirla mas empieza a sesgar la densidad
    global del entrenamiento, que es una de las cosas que gen_score penaliza.
    """

    # transposicion: casi siempre, es la mas barata y la mas segura
    p_transpose: float = 0.9
    max_semitones: int = 6                 # rango seguro +/-6 (media octava)

    # estiramiento temporal
    p_stretch: float = 0.3
    stretch_lo: float = 0.9
    stretch_hi: float = 1.1

    # adelgazado de voces (el que ataca el mismatch de textura)
    p_thin: float = 0.25
    thin_lo: float = 0.3                   # p_drop se sortea en [lo, hi]
    thin_hi: float = 1.0                   # hi=1.0 -> a veces monofonico puro
    thin_keep: str = "high"                # high | low | random

    # jitter de onsets
    p_jitter: float = 0.10
    jitter_max_shift: int = 1              # +/- pasos de 50 ms
    jitter_frac: float = 0.15              # fraccion de onsets movidos

    # El DataLoader necesita forma fija: al recortar/rellenar aqui, augment
    # es un drop-in en TokenWindows.__getitem__. Cualquier PREFIJO de una
    # secuencia valida sigue siendo valido, asi que truncar es seguro.
    keep_len: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_CFG = AugmentConfig()

# Preset agresivo en textura: pensado para el experimento que intenta cerrar el
# hueco con prefijos monofonicos. Casi la mitad de las ventanas se adelgazan y
# el sorteo de p_drop esta cargado hacia el extremo monofonico.
TEXTURE_CFG = AugmentConfig(p_thin=0.45, thin_lo=0.5, thin_hi=1.0,
                            thin_keep="random", p_jitter=0.2)


def _as_cfg(cfg) -> AugmentConfig:
    """Acepta None, dict o dataclass. En el bucle caliente pasa la dataclass."""
    if cfg is None:
        return DEFAULT_CFG
    if isinstance(cfg, AugmentConfig):
        return cfg
    if isinstance(cfg, dict):
        return AugmentConfig(**cfg)
    raise TypeError(f"cfg debe ser None, dict o AugmentConfig, no {type(cfg)}")


# --------------------------------------------------------------------------- #
# conversion tokens <-> eventos
# --------------------------------------------------------------------------- #
class Events(NamedTuple):
    """Representacion intermedia: notas (t, p) ordenadas + duracion total.

    times/pitches son int64 y estan en orden canonico (t creciente, y dentro de
    cada t pitch estrictamente creciente) si la secuencia de origen era valida.
    n_steps replica la semantica de decode_tokens: max(suma de SHIFT,
    ultimo onset + 1), de modo que el ida y vuelta conserva T exactamente.
    """

    times: np.ndarray
    pitches: np.ndarray
    n_steps: int
    has_bos: bool
    has_eos: bool


def _chain_shift(out: list, g: int) -> None:
    """Codifica un gap g >= 1 encadenando SHIFT(<=64), igual que el tokenizador.

    Se reimplementa (4 lineas) en vez de importar el _emit_shift privado de
    data.tokenizer para no depender de un nombre no publico; en __main__ se
    verifica que la salida completa coincide con encode_roll_fast.
    """
    while g > MAX_SHIFT:
        out.append(SHIFT_OFF_ID + MAX_SHIFT - 1)
        g -= MAX_SHIFT
    if g > 0:
        out.append(SHIFT_OFF_ID + g - 1)


def tokens_to_events(tokens) -> Events:
    """Secuencia de tokens -> Events. Vectorizado, sin bucles de Python.

    PAD y BOS no aportan tiempo ni notas; el primer EOS corta la secuencia (lo
    que sigue es relleno del DataLoader). El tiempo de cada nota es la suma de
    los SHIFT que la preceden, que sale de un unico cumsum.
    """
    tk = np.asarray(tokens).ravel()
    if tk.dtype != np.int64:
        tk = tk.astype(np.int64)             # uint8 + semitonos desbordaria
    has_bos = bool(tk.size and tk[0] == BOS)
    eos = np.flatnonzero(tk == EOS)
    has_eos = bool(eos.size)
    body = tk[: int(eos[0])] if has_eos else tk
    if body.size == 0:
        z = np.zeros(0, np.int64)
        return Events(z, z, 0, has_bos, has_eos)

    is_sh = body >= SHIFT_OFF_ID
    is_nt = (body >= NOTE_OFF_ID) & ~is_sh
    d = np.where(is_sh, body - (SHIFT_OFF_ID - 1), 0)
    cum = np.cumsum(d)                       # tiempo DESPUES de cada token
    ts = cum[is_nt]                          # en una nota d=0 -> cum = tiempo
    ps = body[is_nt] - NOTE_OFF_ID
    T = int(cum[-1])
    if ts.size:
        T = max(T, int(ts[-1]) + 1)
    return Events(ts, ps, T, has_bos, has_eos)


def events_to_tokens(times, pitches, n_steps: int,
                     add_bos: bool = True, add_eos: bool = True) -> np.ndarray:
    """Events -> tokens uint8. Espejo vectorizado de encode_roll_fast.

    Exige eventos CANONICOS (usa canonicalize_events antes si una
    transformacion pudo desordenar o duplicar). El silencio de cola se emite
    como un ultimo SHIFT para conservar n_steps, igual que el tokenizador.

    Efecto de borde conocido: una ventana cortada a media pieza que TERMINA en
    una nota no lleva SHIFT final, pero su duracion es ultimo_onset+1, asi que
    al recodificar aparece un SHIFT(1) de mas (1024 -> 1025 tokens). Es la
    misma secuencia que produciria el tokenizador para ese roll y decode la
    interpreta igual; con cfg.keep_len ese token sobrante se trunca y el ida y
    vuelta vuelve a ser la identidad exacta.
    """
    ts = np.asarray(times, np.int64)
    ps = np.asarray(pitches, np.int64)
    n = ts.size
    T = int(n_steps)
    head = [BOS] if add_bos else []

    if n == 0:
        out = list(head)
        if T > 0:
            _chain_shift(out, T)
        if add_eos:
            out.append(EOS)
        return np.asarray(out, np.uint8)

    T = max(T, int(ts[-1]) + 1)              # coherencia: T cubre el ultimo onset
    new = np.empty(n, bool)
    new[0] = True
    np.not_equal(ts[1:], ts[:-1], out=new[1:])
    grp = np.flatnonzero(new)                # primer indice de cada paso activo
    G = grp.size
    uniq = ts[grp]

    cnt = np.empty(G, np.int64)              # notas por paso activo
    cnt[:-1] = grp[1:] - grp[:-1]
    cnt[-1] = n - grp[-1]
    gaps = np.empty(G, np.int64)             # distancia al paso activo anterior
    gaps[0] = uniq[0]
    np.subtract(uniq[1:], uniq[:-1], out=gaps[1:])

    k = np.where(gaps > 0, (gaps + MAX_SHIFT - 1) // MAX_SHIFT, 0)   # SHIFT por gap
    tot = k + cnt
    starts = np.cumsum(tot) - tot
    ks = int(k.sum())
    body = np.empty(ks + n, np.uint8)

    if ks:                                   # tokens de tiempo (gaps > 64 en cadena)
        gi = np.repeat(np.arange(G), k)
        j = np.arange(ks) - np.repeat(np.cumsum(k) - k, k)
        last = j == (k[gi] - 1)
        val = np.where(last, gaps[gi] - (k[gi] - 1) * MAX_SHIFT, MAX_SHIFT)
        body[starts[gi] + j] = SHIFT_OFF_ID + val - 1
    ni = np.repeat(np.arange(G), cnt)        # tokens de nota
    jn = np.arange(n) - np.repeat(grp, cnt)
    body[starts[ni] + k[ni] + jn] = NOTE_OFF_ID + ps

    tail: list[int] = []
    if T > int(uniq[-1]):
        _chain_shift(tail, T - int(uniq[-1]))
    if add_eos:
        tail.append(EOS)
    pieces = [np.asarray(head, np.uint8), body, np.asarray(tail, np.uint8)]
    return np.concatenate([p for p in pieces if p.size])


def canonicalize_events(times, pitches):
    """Reordena por (t, pitch) y elimina duplicados exactos.

    Necesario tras time_stretch y onset_jitter: al mover onsets dos pasos
    distintos pueden caer en el mismo instante, y entonces sus notas quedan
    intercaladas (no ascendentes) y pueden repetir pitch. Repetir no es
    representable en esta tokenizacion (dos onsets del mismo pitch en el mismo
    paso serian un unico bit del roll), asi que la deduplicacion es obligatoria
    y no una decision estetica.

    Atajo: si la clave t*128+p ya es estrictamente creciente no hay nada que
    hacer, que es el caso comun; el sort solo se paga cuando hubo colision.
    """
    ts = np.asarray(times, np.int64)
    ps = np.asarray(pitches, np.int64)
    if ts.size <= 1:
        return ts, ps
    key = (ts << _PSHIFT) | ps
    dif = np.diff(key)
    if (dif > 0).all():
        return ts, ps
    if (dif < 0).any():                      # hubo desorden: ordenar
        key = np.sort(key, kind="stable")
    keep = np.empty(key.size, bool)          # y quitar duplicados adyacentes
    keep[0] = True
    np.not_equal(key[1:], key[:-1], out=keep[1:])
    key = key[keep]
    return key >> _PSHIFT, key & _PMASK


# --------------------------------------------------------------------------- #
# validacion
# --------------------------------------------------------------------------- #
def validate_tokens(tokens, vocab: int = VOCAB_SIZE) -> tuple[bool, str]:
    """Comprueba las invariantes duras -> (ok, mensaje).

    1. ids enteros en [0, vocab)  -> implica SHIFT en [1,64] y pitch en [0,88)
    2. dentro de un paso, pitches ascendentes y sin repetir

    (2) se verifica con la clave t*128+p: como el tiempo nunca retrocede en un
    stream de tokens, exigir que la clave crezca ESTRICTAMENTE es exactamente
    exigir orden ascendente y unicidad dentro de cada paso.
    """
    tk = np.asarray(tokens).ravel()
    if not np.issubdtype(tk.dtype, np.integer):
        return False, f"dtype no entero ({tk.dtype}): NaN/float no es un id"
    if tk.size == 0:
        return True, "vacia (valida)"
    lo, hi = int(tk.min()), int(tk.max())
    if lo < 0 or hi >= vocab:
        return False, f"id fuera de [0,{vocab}): min={lo} max={hi}"
    ev = tokens_to_events(tk)
    if ev.times.size > 1:
        key = (ev.times << _PSHIFT) | ev.pitches
        bad = np.flatnonzero(np.diff(key) <= 0)
        if bad.size:
            i = int(bad[0])
            return False, (f"nota {i + 1}: (t={int(ev.times[i + 1])},"
                           f"p={int(ev.pitches[i + 1])}) no sigue a "
                           f"(t={int(ev.times[i])},p={int(ev.pitches[i])})")
    return True, "ok"


# --------------------------------------------------------------------------- #
# 1. transposicion
# --------------------------------------------------------------------------- #
def valid_transpositions(tokens, max_semitones: int = 6) -> tuple[int, int]:
    """Rango [lo, hi] de semitonos que NO saca ninguna nota de [0, 88).

    Se deriva de la nota minima y maxima de la secuencia y se intersecta con
    +/-max_semitones (rango "seguro": mas de media octava empieza a mover el
    registro fuera de lo que el corpus tiene para ese tipo de textura).
    Siempre contiene el 0, y con una secuencia sin notas devuelve el rango
    completo.
    """
    tk = np.asarray(tokens).ravel()
    m = int(max_semitones)
    if tk.size == 0:
        return -m, m
    nt = tk[(tk >= NOTE_OFF_ID) & (tk < SHIFT_OFF_ID)]
    if nt.size == 0:
        return -m, m
    pmin = int(nt.min()) - NOTE_OFF_ID
    pmax = int(nt.max()) - NOTE_OFF_ID
    return max(-m, -pmin), min(m, N_PITCH - 1 - pmax)


def transpose(tokens, semitones: int) -> np.ndarray:
    """Desplaza todos los NOTE_ON y deja intactos SHIFT/BOS/EOS/PAD.

    Opera sobre los ids: sumar una constante a un bloque contiguo del
    vocabulario preserva el orden y la unicidad dentro de cada paso, asi que la
    salida es canonica sin recanonizar, y transpose(transpose(x, s), -s) es x
    EXACTAMENTE (mismos bytes, mismo dtype), tambien en ventanas cortadas a
    media pieza. Si alguna nota se saliera de [0,88) la transposicion es
    invalida y se lanza ValueError: usa valid_transpositions para elegir s.
    """
    tk = np.asarray(tokens)
    s = int(semitones)
    out = tk.copy()
    if s == 0 or tk.size == 0:
        return out
    m = (tk >= NOTE_OFF_ID) & (tk < SHIFT_OFF_ID)
    if not m.any():
        return out
    v = tk[m].astype(np.int64) + s
    if int(v.min()) < NOTE_OFF_ID or int(v.max()) >= SHIFT_OFF_ID:
        raise ValueError(f"transposicion {s:+d} saca notas de [0,{N_PITCH})")
    out[m] = v.astype(out.dtype, copy=False)
    return out


# --------------------------------------------------------------------------- #
# 2. estiramiento temporal
# --------------------------------------------------------------------------- #
def _stretch_events(ev: Events, factor: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Escala los tiempos ABSOLUTOS, no cada SHIFT por separado.

    Escalar y redondear cada gap acumularia error (100 gaps de 1 paso con
    factor 1.1 darian 100 pasos en vez de 110). Escalando el tiempo absoluto el
    error queda acotado a +/-0.5 pasos en todo momento.

    Redondear puede COLAPSAR dos pasos activos vecinos en el mismo instante
    (factor < 1) o dejar un gap 0: no se emite SHIFT(0) -que no existe en el
    vocabulario- sino que las dos notas pasan a formar un acorde, y si
    compartian pitch canonicalize_events lo deduplica. Por eso el numero de
    onsets tras time_stretch es "aproximadamente" igual, no exactamente igual.
    """
    f = float(factor)
    T = int(np.floor(ev.n_steps * f + 0.5))
    if ev.times.size == 0:
        return ev.times, ev.pitches, max(T, 0)
    # floor(x+0.5) y no np.rint: rint redondea al par ("banker's rounding") y
    # partiria los empates de forma no monotona respecto al tiempo.
    ts = np.floor(ev.times * f + 0.5).astype(np.int64)
    np.maximum(ts, 0, out=ts)
    return ts, ev.pitches, max(T, int(ts[-1]) + 1)


def time_stretch(tokens, factor: float) -> np.ndarray:
    """Escala la duracion de la pieza por factor (0.9-1.1 tipico).

    Reconstruye SHIFT validos (d en [1,64], encadenados si el gap crece por
    encima de 64). Ver _stretch_events para el manejo del colapso por redondeo.
    """
    ev = tokens_to_events(tokens)
    ts, ps, T = _stretch_events(ev, factor)
    ts, ps = canonicalize_events(ts, ps)
    out = events_to_tokens(ts, ps, T, ev.has_bos, ev.has_eos)
    return out.astype(np.asarray(tokens).dtype, copy=False)


# --------------------------------------------------------------------------- #
# 3. adelgazado de voces  (la transformacion clave para el mismatch)
# --------------------------------------------------------------------------- #
def _thin_mask(times: np.ndarray, p_drop: float, rng, keep: str) -> np.ndarray:
    """Mascara de notas conservadas: >= 1 por paso activo, garantizado.

    Se marca un "superviviente" por grupo (paso activo) ANTES de sortear, y se
    le fuerza a True. Asi con p_drop=1.0 sobrevive exactamente una nota por
    paso -> polifonia media 1.00 exacta, y nunca se crea silencio donde habia
    sonido (el numero de pasos activos no cambia).
    """
    n = times.size
    if n == 0:
        return np.zeros(0, bool)
    new = np.empty(n, bool)
    new[0] = True
    np.not_equal(times[1:], times[:-1], out=new[1:])
    grp = np.flatnonzero(new)
    G = grp.size
    cnt = np.empty(G, np.int64)
    cnt[:-1] = grp[1:] - grp[:-1]
    cnt[-1] = n - grp[-1]

    if keep == "low":                        # las notas ya vienen ascendentes:
        keeper = grp                         # primera del grupo = mas grave
    elif keep == "random":
        keeper = grp + (rng.random(G) * cnt).astype(np.int64)   # random()<1 -> < cnt
    else:                                    # "high" (por defecto): la melodia
        keeper = grp + cnt - 1               # suele estar en la voz superior
    m = rng.random(n) >= float(p_drop)       # p_drop=1.0 -> ninguna sobrevive aqui
    m[keeper] = True
    return m


def thin_voices(x, p_drop: float, rng=None, keep: str = "high"):
    """Borra notas de los acordes conservando SIEMPRE una por paso activo.

    Acepta un piano-roll [T,88] (devuelve roll) o una secuencia de tokens
    (devuelve tokens); es la unica transformacion que se expresa igual de bien
    en los dos dominios, porque solo es una mascara sobre las notas.

    Por que importa: con p_drop alto una pieza polifonica se convierte en una
    linea casi monofonica que conserva la armonia, el fraseo y la distribucion
    de IOI del corpus. Es exactamente la textura de los prefijos externos con
    los que el modelo degenera, y aqui sale gratis a partir de datos reales.

    Garantias: nunca aumenta el numero de onsets y nunca deja vacio un paso que
    tenia sonido.
    """
    if rng is None:
        rng = np.random.default_rng()
    a = np.asarray(x)
    if a.ndim == 2 and a.shape[1] == N_PITCH:                  # via roll
        ts, ps = np.nonzero(a)                                 # ya ordenado (t,p)
        m = _thin_mask(ts.astype(np.int64), p_drop, rng, keep)
        out = np.zeros_like(a)
        out[ts[m], ps[m]] = 1
        return out
    ev = tokens_to_events(a)                                   # via tokens
    m = _thin_mask(ev.times, p_drop, rng, keep)
    # La mascara no puede desordenar ni duplicar, pero recanonizar cuesta ~2 us
    # y hace que la salida sea canonica AUNQUE la entrada no lo fuera.
    ts, ps = canonicalize_events(ev.times[m], ev.pitches[m])
    out = events_to_tokens(ts, ps, ev.n_steps, ev.has_bos, ev.has_eos)
    return out.astype(a.dtype, copy=False)


# --------------------------------------------------------------------------- #
# 4. jitter de onsets
# --------------------------------------------------------------------------- #
def onset_jitter(tokens, max_shift: int = 1, rng=None, frac: float = 0.15) -> np.ndarray:
    """Mueve una fraccion de onsets +/-max_shift pasos (variacion de ejecucion).

    El jitter es POR NOTA y no por paso: mover el acorde entero solo seria un
    temblor de tempo, mientras que mover notas sueltas rompe la sincronia
    perfecta del acorde y produce arpegios/rubato, que es diversidad de textura
    util. Los tiempos se recortan a [0, T-1] para no alargar la ventana, y
    despues se recanoniza porque una nota puede aterrizar en un paso donde ya
    sonaba ese mismo pitch (colision -> se deduplica, el onset se pierde).
    """
    if rng is None:
        rng = np.random.default_rng()
    ev = tokens_to_events(tokens)
    n = ev.times.size
    tk = np.asarray(tokens)
    if n == 0 or max_shift <= 0 or frac <= 0.0:
        return tk.copy()
    delta = rng.integers(-int(max_shift), int(max_shift) + 1, size=n)
    delta[rng.random(n) >= float(frac)] = 0
    ts = np.clip(ev.times + delta, 0, max(ev.n_steps - 1, 0))
    ts, ps = canonicalize_events(ts, ev.pitches)
    out = events_to_tokens(ts, ps, ev.n_steps, ev.has_bos, ev.has_eos)
    return out.astype(tk.dtype, copy=False)


# --------------------------------------------------------------------------- #
# API principal
# --------------------------------------------------------------------------- #
def augment_tokens(tokens, rng=None, cfg=None) -> np.ndarray:
    """Aplica la cadena de augmentacion a una ventana de tokens.

    Orden: time_stretch -> thin_voices -> onset_jitter -> (recanonizar y
    recodificar) -> transpose. Las tres primeras comparten UNA sola conversion
    a eventos, que es donde esta el coste; transpose va al final y sobre los
    ids porque conmuta con las demas (solo toca el pitch, y no cambia cual es
    la nota mas aguda o mas grave de un acorde) y asi se evita una segunda
    pasada por eventos.

    Devuelve SIEMPRE un array nuevo (nunca una vista del memmap) del mismo
    dtype que la entrada. Con cfg.keep_len la longitud se conserva rellenando
    con PAD o truncando: truncar es seguro porque cualquier prefijo de una
    secuencia valida es valido.

    La conversion a eventos se hace SIEMPRE, aunque no se sortee ninguna
    transformacion de ese dominio: cuesta ~35 us sobre los 1024 tokens (nada
    frente al paso de entrenamiento) y a cambio la salida es canonica por
    construccion, tambien si la ventana de entrada no lo fuera. Con entrada
    canonica el ida y vuelta es la identidad exacta (verificado en __main__).
    """
    cfg = _as_cfg(cfg)
    if rng is None:
        rng = np.random.default_rng()
    tk = np.asarray(tokens)
    n_in = tk.size
    u = rng.random(4)                        # un solo sorteo para las 4 decisiones

    ev = tokens_to_events(tk)
    ts, ps, T = ev.times, ev.pitches, ev.n_steps
    if u[0] < cfg.p_stretch:
        f = float(rng.uniform(cfg.stretch_lo, cfg.stretch_hi))
        ts, ps, T = _stretch_events(Events(ts, ps, T, False, False), f)
    if u[1] < cfg.p_thin and ts.size:
        p_drop = float(rng.uniform(cfg.thin_lo, cfg.thin_hi))
        m = _thin_mask(ts, p_drop, rng, cfg.thin_keep)
        ts, ps = ts[m], ps[m]
    if u[2] < cfg.p_jitter and ts.size and cfg.jitter_max_shift > 0:
        d = rng.integers(-cfg.jitter_max_shift, cfg.jitter_max_shift + 1, ts.size)
        d[rng.random(ts.size) >= cfg.jitter_frac] = 0
        ts = np.clip(ts + d, 0, max(T - 1, 0))
    ts, ps = canonicalize_events(ts, ps)
    out = events_to_tokens(ts, ps, T, ev.has_bos, ev.has_eos)
    out = out.astype(tk.dtype, copy=False)

    if u[3] < cfg.p_transpose:
        lo, hi = valid_transpositions(out, cfg.max_semitones)
        s = int(rng.integers(lo, hi + 1))
        out = transpose(out, s)              # transpose ya devuelve copia

    if cfg.keep_len and out.size != n_in:
        if out.size > n_in:
            out = out[:n_in]
        else:
            pad = np.full(n_in, PAD, out.dtype)
            pad[: out.size] = out
            out = pad
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------- #
# verificacion (se ejecuta de verdad sobre piezas del corpus)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time
    import json

    from data.tokenizer import encode_roll_fast, decode_tokens
    from data.datasets import get_tokens, get_roll, PROC

    ROWS: list[tuple] = []

    def row(trans, inv, ok, detail="", us=None):
        ROWS.append((trans, inv, "OK" if ok else "FALLA", detail,
                     "" if us is None else f"{us:8.1f}"))

    def bench(fn, reps=200):
        fn()                                                   # calentamiento
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e6         # us/llamada

    rng = np.random.default_rng(0)
    splits = json.load(open(PROC / "splits.json"))
    pieces = [int(i) for i in splits["train"][:8]]
    TOKS = [get_tokens(i) for i in pieces]
    ROLLS = [get_roll(i) for i in pieces]
    print(f"piezas reales usadas: {pieces}")
    print("tokens/pieza:", [len(t) for t in TOKS])
    print("onsets/pieza:", [int(r.sum()) for r in ROLLS])

    def n_on(tok):
        return int(decode_tokens(tok).sum())

    # --- 0. el codificador de eventos es el del tokenizador ------------------
    ok = True
    for r in ROLLS:
        ts, ps = np.nonzero(r)
        ok &= np.array_equal(events_to_tokens(ts, ps, r.shape[0]), encode_roll_fast(r))
    row("events_to_tokens", "identico a encode_roll_fast", ok, f"{len(ROLLS)} piezas")

    # --- 0b. ida y vuelta tokens <-> eventos ---------------------------------
    ok = True
    for tk in TOKS:
        ev = tokens_to_events(tk)
        rt = events_to_tokens(ev.times, ev.pitches, ev.n_steps, ev.has_bos, ev.has_eos)
        ok &= np.array_equal(rt, np.asarray(tk, np.uint8))
    row("tokens<->eventos", "ida y vuelta exacta", ok, f"{len(TOKS)} piezas")

    # --- 1. transpose ---------------------------------------------------------
    ok_v, ok_n, ok_inv, worst = True, True, True, 0
    for tk in TOKS:
        lo, hi = valid_transpositions(tk, 6)
        base = n_on(tk)
        for s in range(lo, hi + 1):
            y = transpose(tk, s)
            good, msg = validate_tokens(y)
            ok_v &= good
            ok_n &= (n_on(y) == base)
            ok_inv &= np.array_equal(transpose(y, -s), np.asarray(tk))
        worst = max(worst, hi - lo + 1)
    us = bench(lambda: transpose(TOKS[0][:1024], 3))
    row("transpose", "tokens validos", ok_v, f"{worst} transposiciones/pieza", us)
    row("transpose", "mismo numero de onsets", ok_n)
    row("transpose", "inversa exacta (bit a bit)", ok_inv)

    ok_lim = True
    for tk in TOKS:
        lo, hi = valid_transpositions(tk, 60)      # limite real, sin recorte a 6
        for s in (lo - 1, hi + 1):
            try:
                transpose(tk, s)
                ok_lim = False                     # deberia haber lanzado
            except ValueError:
                pass
        transpose(tk, lo); transpose(tk, hi)       # los extremos si deben valer
    row("valid_transpositions", "lo/hi validos, lo-1 y hi+1 lanzan", ok_lim)
    # Cuanto multiplica de verdad la transposicion: una PIEZA entera suele
    # tocar los extremos del teclado y deja poco margen, pero el modelo se
    # entrena con VENTANAS, cuyo ambito es mucho menor.
    npz = [valid_transpositions(t, 6) for t in TOKS]
    nwin = [valid_transpositions(np.asarray(t[j:j + 1024]), 6)
            for t in TOKS for j in range(0, max(len(t) - 1024, 1), 1024)]
    row("valid_transpositions", "margen real de datos efectivos", True,
        f"pieza {np.mean([h - l + 1 for l, h in npz]):.1f}x, "
        f"ventana 1024 {np.mean([h - l + 1 for l, h in nwin]):.1f}x de 13 posibles")

    # --- 2. time_stretch ------------------------------------------------------
    ok_v, ok_T, ok_n = True, True, True
    dT, dN = [], []
    for tk in TOKS:
        r0 = decode_tokens(tk)
        for f in (0.9, 0.95, 1.05, 1.1, 0.5, 2.0):
            y = time_stretch(tk, f)
            good, msg = validate_tokens(y)
            ok_v &= good
            r1 = decode_tokens(y)
            dT.append(r1.shape[0] / max(r0.shape[0] * f, 1))
            dN.append(int(r1.sum()) / max(int(r0.sum()), 1))
            ok_n &= int(r1.sum()) <= int(r0.sum())
    ok_T = abs(np.mean(dT) - 1.0) < 0.01
    us = bench(lambda: time_stretch(TOKS[0][:1024], 1.07))
    row("time_stretch", "tokens validos", ok_v, f"6 factores x {len(TOKS)} piezas", us)
    row("time_stretch", "T escala con el factor", ok_T,
        f"T_obt/T_esp en [{min(dT):.4f},{max(dT):.4f}]")
    row("time_stretch", "onsets ~ iguales (colapso)", ok_n,
        f"n_obt/n_orig en [{min(dN):.4f},{max(dN):.4f}]")

    # --- 3. thin_voices -------------------------------------------------------
    ok_v = ok_le = ok_act = ok_roll = True
    ratios = []
    for tk, rl in zip(TOKS, ROLLS):
        act0 = int((rl.sum(1) > 0).sum())
        for p in (0.25, 0.5, 0.9, 1.0):
            y = thin_voices(tk, p, np.random.default_rng(1))
            good, _ = validate_tokens(y)
            ok_v &= good
            r1 = decode_tokens(y)
            ok_le &= int(r1.sum()) <= int(rl.sum())
            ok_act &= int((r1.sum(1) > 0).sum()) == act0
            ratios.append(int(r1.sum()) / max(int(rl.sum()), 1))
            rr = thin_voices(rl, p, np.random.default_rng(1))
            ok_roll &= (rr.shape == rl.shape and int(rr.sum()) <= int(rl.sum())
                        and int((rr.sum(1) > 0).sum()) == act0
                        and bool(((rr == 1) & (rl == 0)).sum() == 0))
    us = bench(lambda: thin_voices(TOKS[0][:1024], 0.8, rng))
    row("thin_voices(tokens)", "tokens validos", ok_v, "4 p_drop x 8 piezas", us)
    row("thin_voices(tokens)", "nunca aumenta onsets", ok_le,
        f"n/n_orig en [{min(ratios):.3f},{max(ratios):.3f}]")
    row("thin_voices(tokens)", "no crea silencio (pasos activos =)", ok_act)
    us = bench(lambda: thin_voices(ROLLS[0][:2048], 0.8, rng), reps=50)
    row("thin_voices(roll)", "misma forma, subconjunto, activos =", ok_roll, "", us)

    polys, ok_mono = [], True
    for tk, rl in zip(TOKS, ROLLS):
        r1 = decode_tokens(thin_voices(tk, 1.0, np.random.default_rng(2)))
        per = r1.sum(1)
        mp = float(per[per > 0].mean())
        polys.append(mp)
        ok_mono &= (mp == 1.0) and (per.max() == 1)
    mp0 = float(np.mean([r.sum(1)[r.sum(1) > 0].mean() for r in ROLLS]))
    row("thin_voices p_drop=1", "polifonia media == 1.000000", ok_mono,
        f"orig {mp0:.3f} -> {np.mean(polys):.6f}")
    for keep in ("high", "low", "random"):
        y = thin_voices(TOKS[0], 1.0, np.random.default_rng(3), keep=keep)
        good, _ = validate_tokens(y)
        r1 = decode_tokens(y)
        mp = float(np.nonzero(r1)[1].mean()) + 21          # pitch medio en MIDI
        row(f"thin_voices keep={keep}", "valida y monofonica",
            good and r1.sum(1).max() == 1,
            f"{int(r1.sum())} onsets, pitch medio MIDI {mp:.1f}")

    # --- 4. onset_jitter ------------------------------------------------------
    ok_v = ok_le = ok_T = True
    for tk in TOKS:
        r0 = decode_tokens(tk)
        for ms in (1, 2):
            y = onset_jitter(tk, ms, np.random.default_rng(4), frac=0.3)
            good, _ = validate_tokens(y)
            ok_v &= good
            r1 = decode_tokens(y)
            ok_le &= int(r1.sum()) <= int(r0.sum())
            ok_T &= abs(r1.shape[0] - r0.shape[0]) <= ms
    us = bench(lambda: onset_jitter(TOKS[0][:1024], 1, rng, 0.15))
    row("onset_jitter", "tokens validos", ok_v, "max_shift 1 y 2", us)
    row("onset_jitter", "no aumenta onsets, T estable", ok_le and ok_T)

    # --- 5. entradas extremas -------------------------------------------------
    extremes = {
        "vacia": np.zeros(0, np.uint8),
        "1 token (BOS)": np.array([BOS], np.uint8),
        "1 token (nota)": np.array([NOTE_OFF_ID + 40], np.uint8),
        "todo SHIFT (1024)": np.full(1024, SHIFT_OFF_ID + MAX_SHIFT - 1, np.uint8),
        "todo NOTE_ON (1024)": np.full(1024, NOTE_OFF_ID + 40, np.uint8),
        "notas aleatorias": rng.integers(NOTE_OFF_ID, SHIFT_OFF_ID, 1024).astype(np.uint8),
        "solo PAD": np.zeros(1024, np.uint8),
        "ventana real 1024": np.asarray(TOKS[1][500:1524], np.uint8),
    }
    # Ojo: "todo NOTE_ON (1024)" y "notas aleatorias" son entradas NO CANONICAS
    # (repiten pitch en el mismo paso, cosa que el roll no puede representar).
    # Se exige a todas: no lanzar y devolver ids en rango. Se exige ademas
    # salida canonica a las que pasan por eventos; transpose es un
    # desplazamiento de ids y preserva la entrada tal cual, no la repara.
    TRANSFORMS = (
        (lambda a: transpose(a, min(3, valid_transpositions(a, 6)[1])), "transpose", False),
        (lambda a: time_stretch(a, 0.93), "stretch", True),
        (lambda a: thin_voices(a, 0.7, rng), "thin", True),
        (lambda a: onset_jitter(a, 1, rng), "jitter", True),
        (lambda a: augment_tokens(a, rng, TEXTURE_CFG), "augment", True),
    )
    ok_ex, ok_can, det, detc = True, True, [], []
    for name, arr in extremes.items():
        for fn, lbl, must_canon in TRANSFORMS:
            try:
                y = np.asarray(fn(arr))
            except Exception as e:                      # noqa: BLE001
                ok_ex = False
                det.append(f"{name}/{lbl}: {type(e).__name__}: {e}")
                continue
            if y.size and (int(y.min()) < 0 or int(y.max()) >= VOCAB_SIZE):
                ok_ex = False
                det.append(f"{name}/{lbl}: id fuera de rango")
            good, msg = validate_tokens(y)
            if must_canon and not good:
                ok_can = False
                detc.append(f"{name}/{lbl}: {msg}")
    row("entradas extremas", "8 casos x 5 transf: sin excepcion, ids en rango", ok_ex,
        "; ".join(det) if det else "8 casos incluyendo 2 no canonicos")
    row("entradas extremas", "salida canonica (stretch/thin/jitter/augment)", ok_can,
        "; ".join(detc) if detc else "tambien con entrada no canonica")

    # --- 6. augment_tokens: validez masiva y coste ----------------------------
    W = [np.asarray(TOKS[i % len(TOKS)][j: j + 1024], np.uint8)
         for i in range(len(TOKS)) for j in range(0, 4000, 512)]
    W = [w for w in W if w.size == 1024]
    ok_all, ok_len, bad_msg = True, True, ""
    rng2 = np.random.default_rng(7)
    for w in W:
        for cfg in (DEFAULT_CFG, TEXTURE_CFG):
            y = augment_tokens(w, rng2, cfg)
            good, msg = validate_tokens(y)
            if not good:
                ok_all, bad_msg = False, msg
            ok_len &= (y.size == w.size and y.dtype == w.dtype)
    row("augment_tokens", f"{2 * len(W)} ventanas 1024 validas", ok_all, bad_msg)
    row("augment_tokens", "keep_len y dtype preservados", ok_len, "1024 uint8")

    # cfg como dict, tal como puede venir de un config.json
    y = augment_tokens(W[0], np.random.default_rng(0), DEFAULT_CFG.as_dict())
    row("augment_tokens", "acepta cfg dict", validate_tokens(y)[0])

    off = AugmentConfig(p_transpose=0.0, p_stretch=0.0, p_thin=0.0, p_jitter=0.0)
    ok_id = all(np.array_equal(augment_tokens(w, rng2, off), w) for w in W)
    row("augment_tokens", "con todo a p=0 es la identidad", ok_id,
        "el ida y vuelta por eventos no pierde nada")

    # --- 6b. cadena literal del enunciado: decode(augment(encode(roll))) ------
    ok_tr = ok_th = ok_st = True
    for r in ROLLS:
        e = encode_roll_fast(r)
        n0 = int(r.sum())
        vlo, vhi = valid_transpositions(e, 6)
        d_tr = decode_tokens(transpose(e, vhi if vhi > 0 else vlo))
        d_th = decode_tokens(thin_voices(e, 0.6, np.random.default_rng(11)))
        d_st = decode_tokens(time_stretch(e, 1.05))
        ok_tr &= (int(d_tr.sum()) == n0 and d_tr.shape == r.shape
                  and set(np.unique(d_tr)).issubset({0, 1}))
        ok_th &= (int(d_th.sum()) < n0 and d_th.shape == r.shape)
        ok_st &= (abs(int(d_st.sum()) - n0) / n0 < 0.01
                  and abs(d_st.shape[0] - 1.05 * r.shape[0]) <= 1)
    row("decode(aug(encode(roll)))", "transpose: mismo n de onsets, roll binario", ok_tr)
    row("decode(aug(encode(roll)))", "thin_voices: menos onsets, misma T", ok_th)
    row("decode(aug(encode(roll)))", "time_stretch: onsets ~=, T x factor", ok_st)

    # --- 6c. integracion con el DataLoader -----------------------------------
    w64 = np.asarray(W[0], np.int64)                      # lo que da TokenWindows
    y64 = augment_tokens(w64, rng2, TEXTURE_CFG)
    ro = np.asarray(W[0]).copy()
    ro.flags.writeable = False                            # memmap de solo lectura
    yro = augment_tokens(ro, rng2, TEXTURE_CFG)
    row("integracion DataLoader", "int64 y array de solo lectura",
        validate_tokens(y64)[0] and y64.dtype == np.int64 and y64.size == 1024
        and validate_tokens(yro)[0] and yro.flags.writeable,
        "salida escribible y contigua")
    lens = [augment_tokens(w, rng2, AugmentConfig(keep_len=False)).size for w in W]
    row("integracion DataLoader", "keep_len=False deja variar la longitud", True,
        f"1024 -> [{min(lens)},{max(lens)}] tok (1025 = SHIFT(1) de cola)")

    w = W[0]
    us_d = bench(lambda: augment_tokens(w, rng2, DEFAULT_CFG), reps=2000)
    us_t = bench(lambda: augment_tokens(w, rng2, TEXTURE_CFG), reps=2000)
    row("augment_tokens", "coste < 1 ms/ventana (DEFAULT)", us_d < 1000.0,
        f"{us_d:.0f} us", us_d)
    row("augment_tokens", "coste < 1 ms/ventana (TEXTURE)", us_t < 1000.0,
        f"{us_t:.0f} us", us_t)
    us_all = bench(lambda: augment_tokens(w, rng2, AugmentConfig(
        p_transpose=1.0, p_stretch=1.0, p_thin=1.0, p_jitter=1.0)), reps=2000)
    row("augment_tokens", "coste con las 4 activas", us_all < 1000.0,
        f"{us_all:.0f} us", us_all)

    # --- 7. efecto sobre la textura (el objetivo del modulo) ------------------
    from metrics import roll_features
    base = [roll_features(r) for r in ROLLS]
    mono = [roll_features(thin_voices(r, 1.0, np.random.default_rng(k)))
            for k, r in enumerate(ROLLS)]
    cfgT = AugmentConfig(p_transpose=1.0, p_stretch=0.5, p_thin=1.0,
                         thin_lo=0.5, thin_hi=1.0, p_jitter=0.3, keep_len=False)
    aug = [roll_features(decode_tokens(augment_tokens(t, np.random.default_rng(k), cfgT)))
           for k, t in enumerate(TOKS)]
    print()
    print(f"efecto sobre la TEXTURA ({len(TOKS)} piezas reales)")
    print(f"  {'':18s} {'original':>10} {'thin p=1.0':>11} {'preset agresivo':>16}")
    for k in ("mean_poly", "density", "empty_ratio", "n_distinct_pitch",
              "pitch_range", "repeat8"):
        print(f"  {k:18s} {np.mean([b[k] for b in base]):10.4f}"
              f" {np.mean([m[k] for m in mono]):11.4f}"
              f" {np.mean([a[k] for a in aug]):16.4f}")
    print("  (repeat8 del corpus completo = 0.217; el modelo degenerado = 0.63)")

    # --- tabla ----------------------------------------------------------------
    print()
    hdr = ("transformacion", "invariante", "res", "detalle", "us/llamada")
    w0 = max(len(r[0]) for r in ROWS + [hdr])
    w1 = max(len(r[1]) for r in ROWS + [hdr])
    w3 = max(len(str(r[3])) for r in ROWS + [hdr])
    line = "-" * (w0 + w1 + w3 + 26)
    print(line)
    print(f"{hdr[0]:<{w0}}  {hdr[1]:<{w1}}  {hdr[2]:<5}  {hdr[3]:<{w3}}  {hdr[4]:>10}")
    print(line)
    for a, b, c, d, e in ROWS:
        print(f"{a:<{w0}}  {b:<{w1}}  {c:<5}  {str(d):<{w3}}  {e:>10}")
    print(line)
    n_bad = sum(1 for r in ROWS if r[2] != "OK")
    print(f"{len(ROWS)} comprobaciones, {n_bad} fallidas")
