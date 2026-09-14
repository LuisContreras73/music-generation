"""Evaluacion CONDICIONAL AL PREFIJO de continuaciones de piano-roll [T,88].

Por que existe este fichero
---------------------------
gen_score (src/metrics.py) compara la continuacion contra el corpus GLOBAL. Eso
es correcto cuando el prefijo es tipico y ENGANOSO cuando no lo es. Caso medido:
con las 5 melodias monofonicas de external_eval_prefix_5s (densidad 0.10
onsets/paso, polifonia 1.00) music_transformer genero densidad 0.108 y polifonia 1.02, es
decir FUE FIEL AL PREFIJO, y aun asi pen_density = 0.41 porque el corpus tiene
densidad 0.4526. El modelo perdio el 59% del score por hacer lo correcto.

Fidelidad al corpus y fidelidad al prefijo son objetivos DISTINTOS y aqui se
miden por separado, en tres bloques:

  (a) PLAUSIBILIDAD: la continuacion tiene la FORMA de la musica del corpus
      (histogramas de altura, clase de altura, polifonia, IOI, intervalo
      melodico e intervalo armonico), reutilizando gen_score pero QUITANDO su
      penalizacion de densidad, que es justo la que rompe el caso atipico.
  (b) CONSISTENCIA CON EL PREFIJO: densidad, polifonia, tonalidad, ritmo,
      reutilizacion de alturas y continuidad de registro, todo medido contra el
      PREFIJO, no contra el corpus.
  (c) DEGENERACION: deteccion de bucle por autocorrelacion de frames, mas fina
      que el repeat8 de metrics.py (que solo cuenta 8-gramas exactos y no dice
      ni el periodo, ni la fuerza, ni cuando empieza el colapso).

El score final es una media geometrica ponderada de (a) y (b) multiplicada por
la penalizacion (c): las tres son condiciones NECESARIAS, no intercambiables.

Convenciones
------------
Todos los rolls son [T,88] con 1 en el onset, notas MIDI 21..108, paso 50 ms.
Las constantes de reescalado (_CONS_FLOOR / _CONS_CEIL / LOOP_TOL) estan MEDIDAS
sobre el corpus; el bloque CONSTANTES CALIBRADAS documenta con que protocolo.
"""
from __future__ import annotations

import numpy as np

from metrics import (N_PITCH, NOTE_MIN_MIDI, overlap, roll_features,
                     aggregate_features, gen_score)

# Densidad minima representable con un fragmento de ~600 pasos es 1/600 = 0.0017;
# EPS_DENS queda un orden por debajo, asi que un roll vacio da cociente ~0 sin
# generar inf ni nan al pasar por el logaritmo.
EPS_DENS = 1e-4

TAIL_STEPS = 40          # 2 s finales del prefijo para medir el registro
HEAD_STEPS = 40          # 2 s iniciales de la continuacion
NOVELTY_WINDOW = 50      # 2.5 s por ventana en la curva de novedad
# Ventana colapsada = no aporta NI UN frame nuevo (1/50 = 0.02, umbral estricto).
# Medido sobre 200 fragmentos reales de 600 pasos: con 0.02 solo el 4% de la
# musica real se marca como colapsada; con 0.05 se marcaria el 23.5%.
NOVELTY_COLLAPSE = 0.02
MAX_LAG_CAP = 256        # 12.8 s: periodo de bucle mas largo que se busca
MAX_STEPS_SS = 3000      # techo de coste de la matriz de Gram (T x T)


# =============================================================================
# CONSTANTES CALIBRADAS
# =============================================================================
# Las tres medidas de consistencia que no son cocientes (solapamientos y masa
# de alturas reutilizadas) no tienen una escala natural: hay que reescalarlas
# entre lo que da una continuacion SIN relacion con el prefijo y lo que da la
# continuacion REAL. Misma filosofia que la calibracion de gen_score.
#
# Protocolo medido (split val, 200 pares, prefijo 100 pasos / continuacion 600,
# semilla 7; reproducible con self_check(..., recalibrate=True)):
#   ceiling = mediana con la continuacion REAL que sigue al prefijo
#   floor   = mediana agrupando las dos condiciones SIN relacion, fragmento de
#             OTRA pieza y ruido i.i.d. con la densidad del prefijo
# Se agrupan las dos porque no coinciden y ninguna domina: otra pieza baja mas
# el solapamiento de clase de altura (0.509 frente a 0.554 del ruido, que al ser
# uniforme solapa un poco con todo) mientras que el ruido hunde la masa de
# alturas reutilizadas (0.188 frente a 0.377). El suelo honesto de "sin
# relacion" es el conjunto de ambas.
#
#   medida               real(techo)   otra pieza   ruido   SUELO agrupado
#   oa_pitch_class          0.6949       0.5088     0.5540      0.5313
#   oa_ioi                  0.6588       0.3433     0.4219      0.3800
#   pitch_mass_reused       0.6000       0.3768     0.1882      0.2374
_CONS_FLOOR = dict(oa_pitch_class=0.5313, oa_ioi=0.3800, pitch_mass_reused=0.2374)
_CONS_CEIL = dict(oa_pitch_class=0.6949, oa_ioi=0.6588, pitch_mass_reused=0.6000)

# Fuerza de autocorrelacion que la musica REAL ya alcanza: medido sobre 200
# fragmentos de 600 pasos del corpus, p50=0.127, p90=0.243, p95=0.314,
# p99=0.524. Por debajo de LOOP_TOL no se penaliza nada, porque un
# acompanamiento metronomico o una frase repetida producen picos legitimos.
LOOP_TOL = 0.30

# Gravedad segun el periodo del bucle: repetir un motivo de 8 pasos (0.4 s) es
# degeneracion pura; repetir una frase de 128 pasos (6.4 s) puede ser forma
# musical (A-A'), asi que se penaliza algo menos, pero nunca por debajo de
# LOOP_MIN_SEVERITY: repetir literalmente 5 s hasta el final tambien es colapso,
# no forma. Con estos valores un bucle PERFECTO (fuerza 1.0) da pen_loop = 0.00
# si el periodo es 8 y 0.25 si es 32 o mas.
LOOP_SHORT_PERIOD = 8
LOOP_LONG_PERIOD = 128
LOOP_MIN_SEVERITY = 0.75

# Pesos de las seis medidas de consistencia. Densidad y polifonia pesan mas
# porque son las dos que el score global castigaba al reves; el registro pesa
# menos porque un salto de octava es estilisticamente legitimo.
CONS_WEIGHTS = dict(c_density=0.22, c_poly=0.15, c_pitch_class=0.18,
                    c_ioi=0.15, c_pitch_reuse=0.18, c_register=0.12)

# Reparto plausibilidad / consistencia dentro del score condicional.
# La plausibilidad pesa mas porque la consistencia es una condicion BARATA: un
# bucle construido con las notas del prefijo la satura sin esfuerzo (por eso
# ademas se multiplica por la penalizacion de bucle). Reproducir la forma de los
# histogramas de intervalo armonico y melodico del corpus, en cambio, exige
# modelar musica.
W_PLAUS = 0.55
W_CONS = 0.45


# =============================================================================
# UTILIDADES
# =============================================================================
def _as_roll(x) -> np.ndarray:
    """Valida y normaliza a [T,88] uint8 binario."""
    r = np.asarray(x)
    if r.ndim != 2 or r.shape[1] != N_PITCH:
        raise ValueError("roll debe ser [T,88], recibido %s" % (r.shape,))
    return (r > 0).astype(np.uint8)


def _ratio_score(a: float, b: float) -> tuple:
    """Cociente a/b y su similitud exp(-|log|) en [0,1].

    Se usa la misma forma funcional que pen_density de metrics.py para que los
    numeros sean comparables: 1.0 si son iguales, 0.5 si uno dobla al otro,
    0.14 si hay un factor 8. Es simetrica en el logaritmo, que es lo correcto
    para una magnitud positiva: pasarse al doble y quedarse a la mitad son
    errores del mismo tamano.
    """
    a = max(float(a), EPS_DENS)
    b = max(float(b), EPS_DENS)
    q = a / b
    return float(q), float(np.exp(-abs(np.log(q))))


def _rescale(x: float, floor: float, ceil: float) -> float:
    """Lleva una medida cruda a [0,1] entre el suelo y el techo medidos."""
    if ceil - floor < 1e-9:
        return 0.0
    return float(np.clip((x - floor) / (ceil - floor), 0.0, 1.0))


def _mean_pitch(roll: np.ndarray) -> float:
    """Altura MIDI media ponderada por onsets. nan si no hay ninguno."""
    w = roll.sum(0).astype(np.float64)
    if w.sum() <= 0:
        return float("nan")
    return float((w * (np.arange(N_PITCH) + NOTE_MIN_MIDI)).sum() / w.sum())


def _frame_ids(roll: np.ndarray) -> np.ndarray:
    """Id entero por frame, igual id <=> frame identico bit a bit."""
    packed = np.packbits(roll, axis=1)                 # [T,11], igualdad exacta
    _, inv = np.unique(packed, axis=0, return_inverse=True)
    return np.asarray(inv, np.int64).ravel()


# =============================================================================
# 1. CONSISTENCIA CON EL PREFIJO
# =============================================================================
def prefix_consistency(prefix_roll, cont_roll,
                       tail_steps: int = TAIL_STEPS,
                       head_steps: int = HEAD_STEPS) -> dict:
    """Cuanto se parece la continuacion a SU PREFIJO (no al corpus).

    Devuelve las medidas crudas, su version reescalada a [0,1] (prefijo c_) y
    el escalar agregado 'consistency'.

    Guia de lectura. Las columnas son las MEDIANAS medidas sobre 200 pares del
    split val (prefijo 100 pasos, continuacion 600): "real" es el fragmento que
    de verdad sigue al prefijo, "otra" un fragmento de otra pieza al azar y
    "ruido" ruido i.i.d. con la densidad del prefijo.

      medida               real     otra    ruido   valor bueno
      density_ratio        1.083    1.173   0.993   1.0
      poly_ratio           1.049    1.032   0.692   1.0
      oa_pitch_class       0.695    0.509   0.554   >= 0.70
      oa_ioi               0.659    0.343   0.422   >= 0.66
      pitch_mass_reused    0.600    0.377   0.188   >= 0.60
      |register_delta|     2.58     6.64    7.12    <= 3 semitonos

    density_ratio / log_density_ratio
        Onsets por paso de la continuacion entre los del prefijo. 1.0 es
        perfecto; el log dice hacia que lado falla. c_density = exp(-|log|), asi
        que doblar la densidad cuesta la mitad de la nota. Ojo: el ruido de
        control se construye CON la densidad del prefijo, asi que aqui acierta
        por definicion; es la plausibilidad quien lo tumba, no esta medida.
    poly_ratio
        Polifonia media en pasos activos. Un prefijo monofonico (1.00)
        continuado con la textura media del corpus (1.86) daria 1.86.
    oa_pitch_class
        Solapamiento de las 12 clases de altura = TONALIDAD.
    oa_ioi
        Solapamiento del histograma de intervalos entre ataques = RITMO y
        velocidad de ejecucion.
    pitch_recall
        Fraccion de las alturas distintas del prefijo que reaparecen. Se reporta
        pero NO entra en el score: el ruido la satura porque toca las 88.
    pitch_mass_reused
        Fraccion de los ONSETS de la continuacion que caen sobre alturas ya
        presentes en el prefijo. Esta si discrimina, y es la unica medida en la
        que el ruido queda por debajo de una pieza ajena.
    new_pitch_mass = 1 - pitch_mass_reused, y new_pitch_count = alturas nuevas.
    register_delta
        Semitonos entre la altura media de los ultimos tail_steps del prefijo y
        la de los primeros head_steps de la continuacion, con signo. Se mide en
        la COSTURA, no entre medias globales, porque el salto es audible ahi.
        c_register = exp(-|delta|/12): una octava de salto vale 0.37.
    """
    p = _as_roll(prefix_roll)
    c = _as_roll(cont_roll)
    fp, fc = roll_features(p), roll_features(c)

    dens_ratio, c_density = _ratio_score(fc["density"], fp["density"])
    poly_ratio, c_poly = _ratio_score(fc["mean_poly"], fp["mean_poly"])

    oa_pc = overlap(fc["pitch_class"], fp["pitch_class"])
    oa_ioi = overlap(fc["ioi_hist"], fp["ioi_hist"])

    # --- reutilizacion de alturas -------------------------------------------
    p_used = p.sum(0) > 0
    c_hist = c.sum(0).astype(np.float64)
    c_used = c_hist > 0
    n_on_c = float(c_hist.sum())
    pitch_recall = float((p_used & c_used).sum() / max(int(p_used.sum()), 1))
    mass_reused = float(c_hist[p_used].sum() / n_on_c) if n_on_c > 0 else 0.0
    new_count = int((c_used & ~p_used).sum())

    # --- continuidad de registro --------------------------------------------
    # Se compara el FINAL del prefijo con el PRINCIPIO de la continuacion, no
    # las medias globales: el salto audible ocurre en la costura.
    mp_tail = _mean_pitch(p[-tail_steps:]) if tail_steps else _mean_pitch(p)
    if not np.isfinite(mp_tail):
        mp_tail = _mean_pitch(p)
    mp_head = _mean_pitch(c[:head_steps]) if head_steps else _mean_pitch(c)
    if not np.isfinite(mp_head):
        mp_head = _mean_pitch(c)
    if np.isfinite(mp_tail) and np.isfinite(mp_head):
        reg_delta = float(mp_head - mp_tail)
        c_register = float(np.exp(-abs(reg_delta) / 12.0))   # 1 octava -> 0.37
    else:
        reg_delta = float("nan")
        c_register = 0.0        # sin notas no hay registro que continuar

    unit = dict(
        c_density=c_density,
        c_poly=c_poly,
        c_pitch_class=_rescale(oa_pc, _CONS_FLOOR["oa_pitch_class"],
                               _CONS_CEIL["oa_pitch_class"]),
        c_ioi=_rescale(oa_ioi, _CONS_FLOOR["oa_ioi"], _CONS_CEIL["oa_ioi"]),
        c_pitch_reuse=_rescale(mass_reused, _CONS_FLOOR["pitch_mass_reused"],
                               _CONS_CEIL["pitch_mass_reused"]),
        c_register=c_register,
    )
    cons = float(sum(CONS_WEIGHTS[k] * unit[k] for k in CONS_WEIGHTS))

    out = dict(
        consistency=cons,
        density_ratio=dens_ratio,
        log_density_ratio=float(np.log(max(dens_ratio, EPS_DENS))),
        poly_ratio=poly_ratio,
        oa_pitch_class=float(oa_pc),
        oa_ioi=float(oa_ioi),
        pitch_recall=pitch_recall,
        pitch_mass_reused=mass_reused,
        new_pitch_mass=float(1.0 - mass_reused),
        new_pitch_count=new_count,
        register_delta=reg_delta,
        prefix_density=float(fp["density"]),
        cont_density=float(fc["density"]),
        prefix_mean_poly=float(fp["mean_poly"]),
        cont_mean_poly=float(fc["mean_poly"]),
    )
    out.update(unit)
    return out


# =============================================================================
# 2. AUTOSIMILITUD / DEGENERACION
# =============================================================================
def _lag_pearson(roll: np.ndarray, max_lag: int) -> np.ndarray:
    """Correlacion de Pearson entre el roll y el mismo roll desplazado lag.

    Se calcula sobre las 88*T celdas del solape, con media y norma recalculadas
    en cada ventana (Pearson exacto, no la version sesgada que decae con el
    lag). El truco de coste es la matriz de Gram G = X X^T: la suma de su
    diagonal 'lag' es exactamente sum_t <x_t, x_{t+lag}>, asi que un solo
    producto matricial da todos los lags. Como X es binaria, la suma de
    cuadrados de una ventana coincide con su suma, y las sumas parciales salen
    de un cumsum.
    """
    X = np.asarray(roll, np.float32)
    T = X.shape[0]
    G = X @ X.T
    cs = np.concatenate([[0.0], np.cumsum(X.sum(1, dtype=np.float64))])
    r = np.zeros(max_lag + 1)
    for lag in range(1, max_lag + 1):
        m = T - lag
        n = float(m * N_PITCH)
        dot = float(np.diagonal(G, offset=lag).sum(dtype=np.float64))
        sa = float(cs[m] - cs[0])            # onsets en X[:m]
        sb = float(cs[T] - cs[lag])          # onsets en X[lag:]
        va = sa - sa * sa / n                # binaria: sum(x^2) == sum(x)
        vb = sb - sb * sb / n
        if va > 1e-9 and vb > 1e-9:
            r[lag] = (dot - sa * sb / n) / np.sqrt(va * vb)
    return r


def _longest_repeat(ids: np.ndarray) -> int:
    """Longitud del fragmento contiguo mas largo que aparece dos veces.

    Busqueda binaria sobre la longitud + np.unique sobre las ventanas
    deslizantes: si dos ventanas de longitud L son identicas, existe repeticion
    de longitud L. Se permiten solapamientos, que es lo correcto aqui: un bucle
    de periodo 8 en 600 pasos tiene un fragmento repetido de 592.
    """
    T = len(ids)
    lo, hi, best = 1, T - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        w = np.lib.stride_tricks.sliding_window_view(ids, mid)
        # copia contigua: np.unique(axis=0) necesita ordenar filas
        u = np.unique(np.ascontiguousarray(w), axis=0)
        if len(u) < len(w):                  # alguna ventana esta duplicada
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return int(best)


def self_similarity(roll, max_lag: int = 0, window: int = NOVELTY_WINDOW,
                    collapse_thresh: float = NOVELTY_COLLAPSE,
                    max_steps: int = MAX_STEPS_SS) -> dict:
    """Medidas de degeneracion mas finas que repeat8.

    repeat8 solo dice que fraccion de los 8-gramas de frames esta duplicada: no
    distingue un ostinato de 8 pasos de una frase repetida de 5 s, no dice
    cuando empieza el colapso y satura tanto con silencio como con bucle.

    Claves devueltas:
      loop_period      periodo dominante en pasos (0 si no hay). Bucle perfecto
                       de motivo de 8 frames -> 8.
      loop_strength    altura del pico de autocorrelacion, ~[0,1]. Bucle
                       perfecto -> ~1.0; musica real del corpus -> ~0.1-0.3;
                       ruido i.i.d. -> ~0.0.
      frame_entropy_bits / frame_entropy_norm
                       entropia de la distribucion de frames ACTIVOS unicos, y
                       la misma normalizada por log2(numero de frames activos).
                       Se excluyen los frames vacios A PROPOSITO: son el 75% del
                       corpus y dominarian la distribucion, con el efecto
                       perverso de que un bucle denso puntuaria MAS entropia que
                       un fragmento disperso pero variado. Bucle -> ~0.2;
                       fragmento real -> ~0.9; silencio -> 0.
      longest_repeat / longest_repeat_ratio
                       fragmento literal mas largo repetido, en frames y en
                       fraccion de T. Bucle o silencio -> ~1.0; real -> ~0.05.
      unique_ratio     frames distintos / T.
      novelty_windows  fraccion de frames nunca vistos antes, por ventana.
      novelty_mean / novelty_first / novelty_last
                       resumen de esa curva. El colapso se ve en novelty_last.
      collapse_step    primer paso a partir del cual TODAS las ventanas quedan
                       por debajo de collapse_thresh, o -1 si nunca colapsa.
                       Localiza CUANDO se rompe el modelo.
    """
    r = _as_roll(roll)
    T = r.shape[0]
    if T < 4:
        return dict(loop_period=0, loop_strength=0.0, frame_entropy_bits=0.0,
                    frame_entropy_norm=0.0, longest_repeat=0,
                    longest_repeat_ratio=0.0, n_unique_frames=int(T > 0),
                    unique_ratio=1.0, novelty_windows=np.ones(1),
                    novelty_mean=1.0, novelty_first=1.0, novelty_last=1.0,
                    collapse_step=-1, n_steps=T)

    core = r[:max_steps]                     # techo de coste de la Gram
    ml = max_lag or min(len(core) // 2, MAX_LAG_CAP)
    ml = max(1, min(ml, len(core) - 2))
    ac = _lag_pearson(core, ml)
    peak = float(ac[1:].max())
    if peak > 0:
        lags = np.arange(1, ml + 1)
        # El pico se repite en los armonicos (16, 24, ...); el periodo es el
        # MENOR lag que casi alcanza el maximo.
        cand = lags[ac[1:] >= 0.95 * peak]
        period = int(cand.min())
        strength = float(np.clip(peak, 0.0, 1.0))
    else:
        period, strength = 0, 0.0

    ids = _frame_ids(r)
    n_uniq = int(ids.max()) + 1

    # --- entropia sobre frames ACTIVOS --------------------------------------
    active = r.sum(1) > 0
    if active.any():
        cnt = np.bincount(ids[active])
        cnt = cnt[cnt > 0].astype(np.float64)
        pr = cnt / cnt.sum()
        h = float(-(pr * np.log2(pr)).sum())
        denom = np.log2(max(int(active.sum()), 2))
        h_norm = float(np.clip(h / denom, 0.0, 1.0))
    else:
        h, h_norm = 0.0, 0.0

    lrep = _longest_repeat(ids)

    # --- novedad por ventanas ------------------------------------------------
    # Un frame es nuevo si es su PRIMERA aparicion en toda la secuencia:
    # return_index de np.unique da justo el indice de esa primera aparicion.
    is_new = np.zeros(T, bool)
    is_new[np.unique(ids, return_index=True)[1]] = True
    nw = max(1, int(window))
    n_win = int(np.ceil(T / nw))
    nov = np.array([is_new[i * nw:(i + 1) * nw].mean() for i in range(n_win)])

    collapse = -1
    below = nov < collapse_thresh
    if below.any():
        # primer indice desde el que TODAS las ventanas siguientes estan bajas
        k = n_win
        while k > 0 and below[k - 1]:
            k -= 1
        if k < n_win:
            collapse = int(k * nw)

    return dict(
        loop_period=period, loop_strength=strength,
        frame_entropy_bits=h, frame_entropy_norm=h_norm,
        longest_repeat=int(lrep), longest_repeat_ratio=float(lrep / T),
        n_unique_frames=n_uniq, unique_ratio=float(n_uniq / T),
        novelty_windows=nov, novelty_mean=float(nov.mean()),
        novelty_first=float(nov[0]), novelty_last=float(nov[-1]),
        collapse_step=collapse, n_steps=T,
    )


def loop_penalty(loop_strength: float, loop_period: int) -> float:
    """Penalizacion [0,1] a partir del bucle detectado.

    Dos ideas:
    * Solo se penaliza el EXCESO sobre lo que la musica real ya hace
      (LOOP_TOL): un acompanamiento regular produce picos de autocorrelacion
      legitimos y no debe costar puntos.
    * Un bucle corto es mas grave que uno largo, pero nunca menos de
      LOOP_MIN_SEVERITY, porque repetir literalmente una frase de 6 s hasta el
      final tambien es colapso.
    """
    s = float(np.clip(loop_strength, 0.0, 1.0))
    excess = max(0.0, s - LOOP_TOL) / (1.0 - LOOP_TOL)
    if loop_period <= 0:
        return 1.0
    span = np.log2(LOOP_LONG_PERIOD / LOOP_SHORT_PERIOD)
    sev = 1.0 - np.log2(max(loop_period, 1) / LOOP_SHORT_PERIOD) / span
    sev = float(np.clip(sev, LOOP_MIN_SEVERITY, 1.0))
    return float(np.clip(1.0 - excess * sev, 0.0, 1.0))


# =============================================================================
# 3. SCORE CONDICIONAL
# =============================================================================
def conditional_score(prefix_roll, cont_roll, ref_agg: dict,
                      calib: dict | None = None) -> dict:
    """Score 0-100 de una continuacion DADO su prefijo.

        cond_score = 100 * plaus^W_PLAUS * cons^W_CONS * pen_loop

    (a) plaus  = oa_weighted de metrics.gen_score, es decir la parte de forma de
        los histogramas ya reescalada entre ruido (0) y corpus real (1) con la
        calibracion. Se toma SIN pen_density (la densidad la manda el prefijo,
        no el corpus) y SIN su pen_loop (aqui se sustituye por una deteccion de
        bucle mejor). Dividir la densidad fuera es todo el arreglo del sesgo
        descrito en la cabecera del fichero.
    (b) cons   = escalar de prefix_consistency, en [0,1].
    (c) pen_loop = loop_penalty(fuerza, periodo) del detector de bucles.

    Por que media GEOMETRICA y no suma ponderada: plausibilidad y consistencia
    son condiciones necesarias e independientes, no compensables. Una suma
    permitiria que un ruido con la densidad del prefijo (consistencia media,
    plausibilidad nula) sacara la mitad de la nota; el producto lo manda al
    suelo, que es la respuesta correcta. Los exponentes suman 1, asi que el
    score se queda en 0-100 y es interpretable en la misma escala que gen_score.

    ref_agg se pasa explicito (metrics.aggregate_features del corpus, p.ej.
    evaluate.load_ref_agg("val_excerpt")) para no acoplar este modulo al disco.
    Aviso: la calibracion de data/processed/calibration.json se hizo con 16
    fragmentos de 800 pasos; evaluar UN fragmento de 600 baja el OA por error de
    muestreo, asi que plaus queda algo pesimista en terminos absolutos. Como el
    sesgo es el mismo para todos los modelos, la comparacion sigue siendo valida.
    """
    p = _as_roll(prefix_roll)
    c = _as_roll(cont_roll)

    g = gen_score(aggregate_features([c]), ref_agg, calib)
    plaus = float(np.clip(g.get("oa_weighted", 0.0), 0.0, 1.0))

    pc = prefix_consistency(p, c)
    cons = float(np.clip(pc["consistency"], 0.0, 1.0))

    ss = self_similarity(c)
    pen = loop_penalty(ss["loop_strength"], ss["loop_period"])

    score = 100.0 * (plaus ** W_PLAUS) * (cons ** W_CONS) * pen
    return dict(
        cond_score=float(score),
        plausibility=float(100.0 * plaus),
        plaus_unit=plaus,
        consistency=cons,
        pen_loop=float(pen),
        loop_period=int(ss["loop_period"]),
        loop_strength=float(ss["loop_strength"]),
        gen_score_global=float(g.get("gen_score", 0.0)),
        pen_density_global=float(g.get("pen_density", 1.0)),
        calibrated=bool(g.get("calibrated", False)),
    )


# =============================================================================
# 4. FILA DE CSV
# =============================================================================
def _load_ref_default(ref_key: str = "val_excerpt"):
    """Carga perezosa de la referencia del corpus.

    La importacion se hace DENTRO de la funcion porque src/evaluate.py arrastra
    torch: quien pase ref_agg explicito puede usar este modulo sin pagarlo.
    """
    from evaluate import load_ref_agg, load_calibration
    return load_ref_agg(ref_key), load_calibration()


def evaluate_pair(prefix_roll, cont_roll, ref_agg: dict | None = None,
                  calib: dict | None = None, piece_id: str = "",
                  model: str = "", ref_key: str = "val_excerpt") -> dict:
    """Todas las metricas de un par (prefijo, continuacion) en un dict PLANO.

    Pensado para una fila de CSV: solo escalares (la curva de novedad se
    resume en sus estadisticos). Si no se pasa ref_agg se carga la referencia
    del corpus del disco.
    """
    p = _as_roll(prefix_roll)
    c = _as_roll(cont_roll)
    if ref_agg is None:
        ref_agg, calib_disk = _load_ref_default(ref_key)
        calib = calib if calib is not None else calib_disk

    row = dict(piece_id=piece_id, model=model,
               prefix_steps=int(p.shape[0]), cont_steps=int(c.shape[0]),
               prefix_onsets=int(p.sum()), cont_onsets=int(c.sum()))
    row.update(prefix_consistency(p, c))
    ss = self_similarity(c)
    row.update({"ss_" + k: v for k, v in ss.items() if k != "novelty_windows"})
    row.update(conditional_score(p, c, ref_agg, calib))
    row["cont_repeat8"] = float(roll_features(c)["repeat8"])
    return row


# =============================================================================
# AUTOCOMPROBACION: casos control con respuesta conocida de antemano
# =============================================================================
def make_controls(prefix, real_cont, other_cont, n_steps, seed=0) -> dict:
    """Continuaciones de control para un prefijo dado.

    real   fragmento REAL que sigue al prefijo         -> debe puntuar lo mas alto
    other  fragmento real de OTRA pieza                -> plausible, poco consistente
    noise  ruido i.i.d. con la densidad DEL PREFIJO    -> minimo
    loop   motivo de 8 frames repetido hasta el final  -> bucle periodo 8
    silence                                            -> minimo
    prefix_loop  el propio prefijo repetido            -> consistente y degenerado
    """
    rng = np.random.default_rng(seed)
    p = _as_roll(prefix)
    dens = p.sum() / (p.shape[0] * N_PITCH)
    noise = (rng.random((n_steps, N_PITCH)) < dens).astype(np.uint8)
    motif = _as_roll(real_cont)[:8]
    loop = np.tile(motif, (n_steps // 8 + 1, 1))[:n_steps]
    silence = np.zeros((n_steps, N_PITCH), np.uint8)
    ploop = np.tile(p, (n_steps // p.shape[0] + 1, 1))[:n_steps]
    return dict(real=_as_roll(real_cont)[:n_steps],
                other=_as_roll(other_cont)[:n_steps],
                noise=noise, loop=loop, silence=silence, prefix_loop=ploop)
