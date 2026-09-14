"""Generacion jerarquica coarse-to-fine para musica simbolica (estilo AudioLM).

Referencia
----------
Borsos et al., "AudioLM: a Language Modeling Approach to Audio Generation"
(arXiv:2209.03143). AudioLM genera primero tokens SEMANTICOS (w2v-BERT), que
fijan la estructura a largo plazo y son faciles de modelar porque tiran casi
todo el detalle, y despues tokens ACUSTICOS (SoundStream) condicionados a ellos.
La leccion que se traslada no es el codec sino la ESTRUCTURA DEL PROBLEMA:
separar "que cancion es" de "como suena exactamente", y modelar lo primero en
una secuencia mas corta y mas limpia.

Traduccion a piano-roll de onsets tokenizado (NOTE_ON + SHIFT)
--------------------------------------------------------------
NIVEL GRUESO (semantico) = CONTORNO MELODICO. Un evento grueso por PASO ACTIVO
    cerrado:  (nota mas aguda del paso, duracion hasta el siguiente ataque,
    cuantas notas tenia el paso). Es lo que un oyente reconoce como "la
    cancion". En el corpus hay ~2.9 tokens por evento grueso, asi que la
    secuencia gruesa es ~3x mas corta; y, sobre todo, su ESTADISTICA NO DEPENDE
    DE LA TEXTURA: un coral a 4 voces y una melodia monofonica producen la misma
    clase de secuencia gruesa (una nota + una duracion por paso).

NIVEL FINO = el resto de notas de cada paso (armonia/acompanamiento) mas el
    orden exacto de emision, condicionado al contorno.

Por que esto ataca el problema real del laboratorio
---------------------------------------------------
La prueba final da prefijos de 100 pasos (~70 tokens) de melodias MONOFONICAS y
los modelos planos degeneran en bucles (gen_score 2.0-2.4 con 0.63 de 8-gramas
de frames repetidos, frente a 70-83 con prefijos del corpus). Las dos sospechas
del laboratorio son mismatch de LONGITUD DE CONTEXTO y mismatch de DOMINIO.
Este modelo ataca las dos por construccion:

  (dominio)   un prefijo monofonico es, literalmente, PURO NIVEL GRUESO: cada
              paso tiene exactamente una nota, luego la descomposicion es sin
              perdida y la secuencia gruesa del prefijo esta DENTRO de la
              distribucion que el decoder grueso ve en entrenamiento (donde se
              queda con la voz superior de cada acorde, que es justo lo que
              hace thin_voices en src/augment.py, pero gratis y siempre). El
              numero de notas del paso entra como variable de condicionamiento
              EXPLICITA (tex), asi que "esto es monofonico" es una entrada del
              modelo, no algo que tenga que inferir de una textura que no vio.
  (contexto)  la prediccion que importa --cual es la siguiente nota de la
              melodia-- la hace el decoder GRUESO, cuyo estado en el prefijo de
              prueba son ~35 eventos musicales bien formados en vez de 70
              tokens de tipos mezclados; y el modelo entrena con ventanas de
              contexto CORTAS al azar (ver "Regularizador de contexto corto"),
              de modo que L=70 no es un regimen que nunca haya visto.

Como se combinan los dos niveles para dar los logits de 155 clases
-------------------------------------------------------------------
El contrato es forward(x: Long[B,L]) -> Float[B,L,V]: la jerarquia es INTERNA.
En cada posicion de token t:

  1. se deriva causalmente k(t) = (numero de SHIFT en x[:, :t+1]) - 1, el indice
     del ULTIMO PASO YA CERRADO;
  2. el decoder grueso corre sobre la secuencia de eventos gruesos y da h_c[k];
  3. el decoder fino recibe  emb(x_t) + W_c h_c[k(t)] + emb(intra_t)  y produce
     fine_logits[t] sobre las 155 clases;
  4. el decoder grueso ademas PREDICE el siguiente evento grueso desde h_c[k(t)]
     -- 88 logits de altura y 64 de duracion -- y esa prediccion se inyecta como
     SESGO ADITIVO sobre los bloques NOTE_ON y SHIFT del vocabulario, con una
     compuerta sigmoide por posicion:

         logits[t] = fine_logits[t] + g_note[t] * pitch_bias[t] (bloque NOTE_ON)
                                    + g_shift[t] * dur_bias[t]  (bloque SHIFT)

     El paso k(t)+1 esta ABIERTO en t, asi que "el siguiente evento grueso" es
     exactamente el paso en curso: su nota mas aguda y su duracion. En una
     textura monofonica ese sesgo ES la respuesta (la unica nota del paso es la
     mas aguda), que es el caso de la prueba. El sesgo es ademas lo que ENTRENA
     las cabezas gruesas con senal directa, y por eso generate_melody_only()
     funciona con los mismos pesos sin ninguna perdida auxiliar (el contrato
     TokenARModel no admite terminos "aux", solo los FrameARModel).

CAUSALIDAD: donde esta la trampa y como se evita
------------------------------------------------
El riesgo principal de esta arquitectura es obvio y hay que decirlo: la nota mas
aguda de un paso NO se conoce hasta que el paso termina. Dentro de un acorde,
en la posicion t, todavia pueden llegar notas mas agudas. Usar mel[seg(t)] --el
agudo del paso al que PERTENECE t-- seria una fuga de futuro de libro.

Por eso el nivel grueso NO se deriva del paso en curso sino SOLO DE PASOS YA
CERRADOS. Formalmente, con is_shift = (x >= 91):

    closed[t] = cumsum(is_shift)[t]          # SHIFT en x[:, :t+1]
    seg[t]    = closed[t] - is_shift[t]      # paso al que pertenece t
    k(t)      = closed[t] - 1                # ULTIMO paso CERRADO

El evento grueso k se "emite" en la posicion del (k+1)-esimo SHIFT, y en ese
instante todo el paso ya esta en x[:, :t+1]: sus notas van ANTES del SHIFT que
lo cierra. Luego h_c[k(t)] es funcion de x[:, :t+1] y nada mas. El codigo
NUNCA indexa por seg[t]; solo por closed[t]-1. Si t es un SHIFT entonces
k(t) = seg[t] y se usa el paso que se acaba de cerrar: es legal y es la
informacion mas fresca posible, porque el SHIFT esta en x[:, :t+1].

Una ultima sutileza, medida y no supuesta: aunque las ENTRADAS gruesas que lee
el pasado son identicas bit a bit ante cualquier perturbacion de x[:, t],
cambiar el TIPO de ese token (NOTE <-> SHIFT) cambia el NUMERO de eventos
gruesos K y por tanto la LONGITUD del tensor sobre el que corre el stack
grueso. Eso no mueve informacion, pero si el orden de reduccion de los matmul,
y aparecen diferencias de ~1e-6 en float32 (y ~1e-7 entre filas del mismo batch,
porque K era el maximo del batch). Por eso cfg.coarse_pad (True por defecto)
rellena la secuencia gruesa hasta L: su forma pasa a depender SOLO de L y la
salida es identica BIT A BIT, tambien entre filas del batch. Con
coarse_pad=False el modelo es mas rapido y sigue pasando tests/test_causality.py
(tolerancia 1e-4), pero la igualdad exacta se pierde por redondeo. Ambas
variantes se miden en __main__.

AVISO (defecto real encontrado en auditoria y CORREGIDO; no lo repitas): K
depende del contenido COMPLETO de x, futuro incluido. Usarlo solo para
dimensionar un tensor cuesta redondeo; usarlo para calcular un VALOR que entre
en los logits es fuga causal de verdad. La version anterior derivaba la ventana
de contexto corto del nivel grueso como Wc = ceil(Wf * K / L): en modo train con
coarse_pad=False, perturbar un solo token del futuro movia los logits del PASADO
hasta 1.6e-02 (no 1e-6: informacion). Ahora la conversion tokens -> eventos usa
una CONSTANTE (coarse_window(), 2.9 tokens por evento), es causal con las dos
variantes de coarse_pad y hace que coarse_pad deje de cambiar la semantica del
regularizador; _selftest lo comprueba en modo train para ambas.

Ademas: (a) el decoder grueso es causal sobre k; (b) las posiciones de relleno
de la secuencia gruesa van todas DESPUES de las reales (empaquetado por
elemento del batch), asi que la mascara causal ya impide que una posicion real
mire relleno; (c) el decoder fino es causal sobre t y su entrada en la posicion
s solo depende de x[:, :s+1]; la composicion de las tres cosas es causal.
La caracteristica intra_t (cuantas notas lleva emitidas el paso ABIERTO,
incluida la de t) tambien es un escaneo prefijo, no mira al futuro.

Esto es la "alternativa honesta" que pide el enunciado, y no es un apano: es la
unica definicion del contorno que un generador autoregresivo puede usar, porque
en muestreo el paso en curso tampoco existe todavia.

Reversibilidad de la descomposicion
-----------------------------------
decompose_tokens / recompose_tokens son inversas EXACTAS (byte a byte) sobre
cualquier secuencia canonica del tokenizador, y se verifica en __main__ sobre
piezas reales del corpus. La reconstruccion es: para cada evento k, emitir
sorted(extra_k + [mel_k]) ascendente y despues SHIFT(dur_k).
Que se pierde y que no:
  * NADA en las piezas del corpus: encode_roll siempre cierra el ultimo paso con
    un SHIFT de cola, asi que no queda ningun paso abierto.
  * Un gap > 64 se tokeniza como varios SHIFT encadenados; eso produce eventos
    gruesos de tipo REST (mel = -1, sin notas). Se reconstruyen exactos.
  * Un paso abierto al final de una VENTANA (las ventanas de entrenamiento
    cortan a media pieza) no es un evento grueso; se guarda aparte en
    "open_notes" y se reemite tal cual, asi que tambien es exacto.
  * Tokens especiales: se conservan como cabecera/cola. Si aparecieran EN MEDIO
    (no ocurre en el corpus) se reportan en "mid_specials" y la reconstruccion
    ya no es exacta; decompose lo dice, no lo esconde.
  * La descomposicion es exacta; lo LOSSY a proposito es el CONDICIONAMIENTO:
    al decoder grueso solo le llegan (mel, dur, tex) --no que notas concretas
    formaban el acorde-- que es precisamente el cuello de botella semantico que
    se busca. El decoder fino si ve el flujo de tokens completo, asi que el
    modelo en conjunto no pierde informacion.

Regularizador de contexto corto (ataca la sospecha 1 del laboratorio)
---------------------------------------------------------------------
Durante el ENTRENAMIENTO, con probabilidad cfg.short_ctx_p (0.25 por defecto),
cada elemento del batch recibe una ventana de atencion deslizante W tomada
log-uniforme en [cfg.short_ctx_min, L]. Asi el modelo ve regularmente contextos
de 32-200 tokens dentro de la misma ventana de 1024 y L=70 en inferencia deja de
ser un regimen extrapolado. Se implementa como mascara ADITIVA extra sobre el
sesgo relativo (columna j del layout sin desplazar <-> distancia L-j, ver
music_transformer.rel_bias), asi que no cuesta ni un parametro, no altera el
camino de evaluacion (solo actua en self.training) y solo PODA pasado. La misma
poda se aplica al decoder grueso en unidades de EVENTO, ceil(W / 2.9) con la
constante medida del corpus; la conversion NO puede usar K (ver el AVISO de la
seccion de causalidad). Requiere cfg.rel_attn=True; con atencion absoluta se
desactiva solo. cfg.fine_window > 0 fija ademas una ventana FIJA al decoder fino
en todo momento (estilo AudioLM, que limita la etapa acustica y deja el largo
alcance a la semantica) y SOLO a el, igual en train que en eval; por defecto
esta apagada para no penalizar la verosimilitud.

Rendimiento medido (RTX 4070 SUPER 12.9 GiB, bf16 autocast, fwd+bwd+clip+step,
B=16, L=1024, config por defecto del laboratorio). La GPU la comparten varios
agentes: los tres modelos se midieron EN LA MISMA PASADA, por rondas
intercaladas, y se reporta el MINIMO sobre 30 rondas (min y p25 coinciden dentro
del 1%, asi que la ventana estaba libre). La linea base reproduce su cifra
documentada --162 ms frente a los 165 ms de music_transformer.py-- lo que valida
la medida:
    hierarchical coarse_pad=True   25.99 M   171.7 ms   5.82 it/s   pico 5.96 GiB
    hierarchical coarse_pad=False  25.99 M   144.9 ms   6.90 it/s   pico 5.14 GiB
    music_transformer (linea base) 25.56 M   162.4 ms   6.16 it/s   pico 5.79 GiB
(los picos de esa tabla incluyen los estados de Adam de los TRES modelos a la
vez; medidos por separado son 5.38 / 4.56 / 5.19 GiB). Es decir: con la
causalidad bit a bit activada el modelo cuesta un 6% mas que la linea base, y
sin ella es un 12% mas barato, porque el stack grueso corre sobre una secuencia
~2.4x mas corta. Cabe de sobra en 12.9 GiB con B=16 y L=1024.
En CPU (fp32, B=2, L=256): 2.86 it/s.

Parametros con la config por defecto (d_model=512, 8 capas, 8 cabezas,
d_ff=2048): 25.99 M = 6.53 M del nivel grueso + 19.46 M del nivel fino.

Knobs propios de este modelo (todos con getattr y valor por defecto, asi que el
modelo funciona con la Config tal cual; OJO: igual que cfg.attn_dropout en
music_transformer.py, NO son campos de src/config.py, luego no sobreviven a
Config.save()/load() -- para usarlos hay que fijarlos sobre la instancia o
anadir el campo):
    n_coarse_layers  int    capas del decoder grueso          (max(2, n_layers//4) = 2)
                            se recorta a [1, n_layers-1] y el nivel fino se
                            queda siempre con al menos un bloque
    coarse_pad       bool   rellena la secuencia gruesa a L   (True)
    short_ctx_p      float  prob. de ventana corta por ejemplo (0.25)
    short_ctx_min    int    ventana minima en tokens           (32)
    fine_window      int    ventana FIJA del decoder fino, 0=off (0)

Reutilizacion
-------------
Los bloques Transformer y el sesgo relativo con skewing se IMPORTAN de
music_transformer.py (codigo ya auditado y con el camino rapido de SDPA); este
fichero no lo modifica ni lo copia.

Contrato: TokenARModel, forward(x: Long[B,L]) -> Float[B,L,V] (logits de x[t+1]).
Causalidad estricta: logits[:, t] depende solo de x[:, :t+1].
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__" and __package__ in (None, ""):
    # Permite ejecutar la autoverificacion con:  python src/models/hierarchical.py
    import sys as _sys_boot
    from pathlib import Path as _Path_boot

    _root_boot = _Path_boot(__file__).resolve().parents[2]
    for _p_boot in (str(_root_boot), str(_root_boot / "src")):
        if _p_boot not in _sys_boot.path:
            _sys_boot.path.insert(0, _p_boot)

# --- importaciones robustas: el modulo se usa dentro del paquete models pero
# tambien debe poder cargarse suelto teniendo src/ en sys.path ---
try:
    from .base import TokenARModel
except ImportError:                                      # pragma: no cover
    try:
        from models.base import TokenARModel
    except ImportError:
        from src.models.base import TokenARModel

try:
    from .music_transformer import Block, skew_mask, sinusoidal_pe
except ImportError:                                      # pragma: no cover
    try:
        from models.music_transformer import Block, skew_mask, sinusoidal_pe
    except ImportError:
        from src.models.music_transformer import Block, skew_mask, sinusoidal_pe

try:
    from data.tokenizer import (VOCAB_SIZE, NOTE_OFF_ID, SHIFT_OFF_ID, N_PITCH,
                                MAX_SHIFT, PAD, BOS, EOS)
except ImportError:                                      # pragma: no cover
    try:
        from src.data.tokenizer import (VOCAB_SIZE, NOTE_OFF_ID, SHIFT_OFF_ID,
                                        N_PITCH, MAX_SHIFT, PAD, BOS, EOS)
    except ImportError:
        PAD, BOS, EOS = 0, 1, 2
        NOTE_OFF_ID, N_PITCH = 3, 88
        SHIFT_OFF_ID, MAX_SHIFT = 91, 64
        VOCAB_SIZE = 155

# Vocabularios del nivel grueso (entradas del decoder grueso).
# altura:  0 = PAD, 1 = REST (paso sin notas), 2 + p = NOTE(p)
MEL_PAD, MEL_REST, MEL_OFF = 0, 1, 2
N_MEL_IN = MEL_OFF + N_PITCH                 # 90
# duracion: 0 = PAD, 1 + (d-1) = SHIFT(d)
N_DUR_IN = 1 + MAX_SHIFT                     # 65
# textura (numero de notas del paso): 0 = PAD, 1 + min(n, 7)
N_TEX_IN = 1 + 8                             # 9
N_INTRA = 8                                  # notas ya emitidas en el paso abierto


# =============================================================================
# 1. Descomposicion en niveles: funciones PURAS sobre numpy, reversibles
# =============================================================================
def decompose_tokens(tokens) -> dict:
    """Separa una secuencia de tokens en nivel GRUESO + nivel FINO.

    Devuelve un dict con:
      mel   int64[K]   altura mas aguda del paso k, o -1 si el paso no tiene
                       notas (evento REST: un SHIFT encadenado de un gap > 64)
      dur   int64[K]   duracion SHIFT(d) que cierra el paso k, d in [1,64]
      cnt   int64[K]   numero de notas del paso k (0 para REST)
      pos   int64[K]   indice del token SHIFT que cierra el paso k
      extra list[K]    las notas NO agudas del paso k, ascendentes (nivel fino)
      head  int64[]    tokens especiales iniciales (BOS/PAD)
      tail  int64[]    tokens especiales finales (EOS/PAD)
      open_notes int64[]  notas tras el ultimo SHIFT (paso ABIERTO, sin cerrar)
      mid_specials int  especiales encontrados en medio (0 en el corpus)

    recompose_tokens(decompose_tokens(t)) == t byte a byte si mid_specials == 0
    y las notas de cada paso venian ascendentes (invariante del tokenizador).
    """
    t = np.asarray(tokens, dtype=np.int64).ravel()
    is_sp = t < NOTE_OFF_ID
    musical = np.flatnonzero(~is_sp)
    empty = np.empty(0, np.int64)
    if musical.size == 0:
        return dict(mel=empty, dur=empty, cnt=empty, pos=empty, extra=[],
                    head=t.copy(), tail=empty, open_notes=empty, mid_specials=0)
    a, b = int(musical[0]), int(musical[-1])
    head, tail = t[:a].copy(), t[b + 1:].copy()
    body_raw = t[a:b + 1]
    mid = int((body_raw < NOTE_OFF_ID).sum())
    body = body_raw[body_raw >= NOTE_OFF_ID]
    idx = np.flatnonzero(body_raw >= NOTE_OFF_ID) + a     # indice original

    mel, dur, cnt, pos, extra = [], [], [], [], []
    cur: list[int] = []
    for j, tok in enumerate(body):
        tok = int(tok)
        if tok < SHIFT_OFF_ID:                            # NOTE_ON
            cur.append(tok - NOTE_OFF_ID)
        else:                                             # SHIFT -> cierra paso
            if cur:
                top = max(cur)
                rest = sorted(cur)
                rest.remove(top)                          # quita UNA ocurrencia
                mel.append(top)
                extra.append(np.asarray(rest, np.int64))
            else:
                mel.append(-1)
                extra.append(empty.copy())
            cnt.append(len(cur))
            dur.append(tok - SHIFT_OFF_ID + 1)
            pos.append(int(idx[j]))
            cur = []
    return dict(mel=np.asarray(mel, np.int64), dur=np.asarray(dur, np.int64),
                cnt=np.asarray(cnt, np.int64), pos=np.asarray(pos, np.int64),
                extra=extra, head=head, tail=tail,
                open_notes=np.asarray(sorted(cur), np.int64), mid_specials=mid)


def recompose_tokens(dec: dict) -> np.ndarray:
    """Inversa de decompose_tokens: niveles grueso+fino -> secuencia de tokens."""
    out: list[int] = [int(v) for v in np.asarray(dec["head"], np.int64).ravel()]
    mel, dur, extra = dec["mel"], dec["dur"], dec["extra"]
    for k in range(len(mel)):
        m = int(mel[k])
        notes = list(np.asarray(extra[k], np.int64).ravel())
        if m >= 0:
            notes.append(m)
        for p in sorted(int(v) for v in notes):
            out.append(NOTE_OFF_ID + p)
        out.append(SHIFT_OFF_ID + int(dur[k]) - 1)
    for p in np.asarray(dec["open_notes"], np.int64).ravel():
        out.append(NOTE_OFF_ID + int(p))
    out.extend(int(v) for v in np.asarray(dec["tail"], np.int64).ravel())
    return np.asarray(out, np.int64)


def coarse_to_tokens(mel, dur) -> np.ndarray:
    """Contorno (mel, dur) -> tokens MONOFONICOS: NOTE_ON(mel) + SHIFT(dur).

    Es la realizacion del nivel grueso sin nivel fino, es decir exactamente la
    textura de los prefijos de la prueba (una nota por paso activo).
    """
    out: list[int] = []
    for m, d in zip(np.asarray(mel, np.int64).ravel(), np.asarray(dur, np.int64).ravel()):
        if int(m) >= 0:
            out.append(NOTE_OFF_ID + int(m))
        out.append(SHIFT_OFF_ID + int(np.clip(int(d), 1, MAX_SHIFT)) - 1)
    return np.asarray(out, np.int64)


# =============================================================================
# 2. Derivacion CAUSAL del nivel grueso dentro del forward (torch, vectorizada)
# =============================================================================
def coarse_index(x: torch.Tensor, pad_to: int = 0) -> dict:
    """Indices y rasgos del nivel grueso a partir de x [B,L], SIN mirar al futuro.

    Todo lo que se devuelve es un escaneo PREFIJO sobre x. En particular
    k_at_token[:, t] = closed[t] - 1 apunta al ultimo paso CERRADO, nunca al
    paso al que pertenece t (que aun podria recibir notas mas agudas): esa es la
    fuga que hay que evitar y aqui no puede ocurrir porque seg[] no se usa para
    indexar nada que se lea en la posicion t.

    Claves:
      mel_idx, dur_idx, tex_idx  Long[B,K]  entradas del decoder grueso
      valid      Bool[B,K]   posiciones gruesas reales (el relleno va al final)
      k_at_token Long[B,L]   indice grueso a leer en la posicion t (>=0)
      has_coarse Bool[B,L]   False mientras no haya ningun paso cerrado
      intra      Long[B,L]   notas ya emitidas en el paso ABIERTO, incluida t
      K          int         longitud de la secuencia gruesa (>= pad_to)

    pad_to fuerza la longitud de la secuencia gruesa (relleno al FINAL, que la
    mascara causal ya hacia invisible para las posiciones reales). Con
    pad_to = L la forma del tensor grueso depende SOLO de L y no del contenido
    de x, que es lo que hace la salida BIT A BIT estable al perturbar un token
    (ver la nota sobre coarse_pad en la cabecera del modulo).
    """
    B, L = x.shape
    dev = x.device
    is_note = (x >= NOTE_OFF_ID) & (x < SHIFT_OFF_ID)
    is_shift = x >= SHIFT_OFF_ID
    pitch = (x - NOTE_OFF_ID).clamp_(0, N_PITCH - 1)
    dstep = (x - SHIFT_OFF_ID).clamp_(0, MAX_SHIFT - 1)

    closed = torch.cumsum(is_shift.long(), dim=1)              # [B,L]
    seg = closed - is_shift.long()                             # paso de t
    # closed <= L siempre, luego con pad_to >= L la longitud gruesa es pad_to sin
    # mirar el contenido: se evita la sincronizacion host-GPU de .item(), que en
    # entrenamiento mete un stall por forward.
    if int(pad_to) >= L:
        K = int(pad_to)
    else:
        K = int(closed[:, -1].max().item()) if L > 0 else 0
        K = max(K, int(pad_to))
    Kb = K + 1                                                 # +1 = cubo basura

    trash = torch.full_like(seg, K)
    idx_note = torch.where(is_note, seg, trash)
    idx_shift = torch.where(is_shift, seg, trash)

    mel = torch.full((B, Kb), -1, dtype=torch.long, device=dev)
    mel.scatter_reduce_(1, idx_note, pitch, reduce="amax")
    cnt = torch.zeros((B, Kb), dtype=torch.long, device=dev)
    cnt.scatter_add_(1, idx_note, is_note.long())
    dmat = torch.zeros((B, Kb), dtype=torch.long, device=dev)
    dmat.scatter_add_(1, idx_shift, dstep * is_shift.long())

    # notas acumuladas ANTES del paso seg[t] -> posicion dentro del paso abierto
    prefix = torch.cumsum(cnt, dim=1) - cnt                    # [B,Kb]
    intra = (torch.cumsum(is_note.long(), dim=1) - prefix.gather(1, seg)).clamp_(0, N_INTRA - 1)

    n_ev = closed[:, -1]                                       # [B] eventos reales
    ar = torch.arange(K, device=dev).unsqueeze(0)
    valid = ar < n_ev.unsqueeze(1) if K > 0 else torch.zeros((B, 0), dtype=torch.bool, device=dev)

    mel, cnt, dmat = mel[:, :K], cnt[:, :K], dmat[:, :K]
    z = torch.zeros_like(mel)
    mel_idx = torch.where(valid, torch.where(mel < 0, z + MEL_REST, mel + MEL_OFF), z)
    dur_idx = torch.where(valid, dmat + 1, z)
    tex_idx = torch.where(valid, cnt.clamp(0, N_TEX_IN - 2) + 1, z)

    return dict(mel_idx=mel_idx, dur_idx=dur_idx, tex_idx=tex_idx, valid=valid,
                k_at_token=(closed - 1).clamp_(min=0), has_coarse=closed >= 1,
                intra=intra, K=K)


# =============================================================================
# 3. Regularizador de contexto corto (mascara deslizante por elemento del batch)
# =============================================================================
def draw_windows(B: int, L: int, p: float, lo: int, device) -> torch.Tensor | None:
    """Ventana W por elemento del batch: log-uniforme en [lo, L] con prob. p."""
    if p <= 0.0 or L <= lo:
        return None
    take = torch.rand(B, device=device) < p
    if not bool(take.any()):
        return None
    u = torch.rand(B, device=device)
    w = torch.exp(math.log(lo) + u * (math.log(L) - math.log(lo))).long().clamp_(lo, L)
    return torch.where(take, w, torch.full_like(w, L))


# Tokens por evento grueso en el corpus (medido pieza a pieza: 2.65-3.30, media
# ~2.9). Se usa como CONSTANTE para traducir una ventana en tokens a la ventana
# equivalente en eventos gruesos. TIENE que ser una constante y no K/L: K es el
# numero de SHIFT de la secuencia, es decir depende del CONTENIDO COMPLETO de x
# --incluido el futuro-- y una ventana de atencion cuyo tamano dependa del futuro
# es una FUGA CAUSAL real (no de redondeo) en modo train. Medido antes de
# corregirlo: max|delta| en los logits del pasado = 1.6e-02.
TOK_PER_EVENT_NUM, TOK_PER_EVENT_DEN = 10, 29        # 2.9 tokens por evento


def coarse_window(W_tok: torch.Tensor) -> torch.Tensor:
    """Ventana en TOKENS -> ventana equivalente en EVENTOS gruesos.

    ceil(W_tok / 2.9) con minimo 1: una fila nunca puede quedarse sin ninguna
    clave visible (si se enmascarase tambien la diagonal, el softmax daria NaN).
    El valor NO depende de x, ver la nota de TOK_PER_EVENT_*.
    """
    return ((W_tok * TOK_PER_EVENT_NUM + TOK_PER_EVENT_DEN - 1)
            // TOK_PER_EVENT_DEN).clamp_(min=1)


def window_mask(base: torch.Tensor, W: torch.Tensor, L: int) -> torch.Tensor:
    """Anade una ventana deslizante de W tokens a la mascara del skewing.

    En el layout SIN desplazar que consume rel_bias, la columna j corresponde a
    la distancia relativa  d = L - j  (ver music_transformer.skew_padded), asi
    que limitar la memoria a W es enmascarar  j <= L - W, independiente de la
    fila. Solo PODA pasado (el futuro ya estaba enmascarado), luego no puede
    crear fuga causal POR SI MISMA. W = L es un no-op: solo marca la columna 0,
    que es el relleno de ceros y nunca se lee. W > L tampoco enmascara nada.

    OJO: la condicion j <= L - W equivale a "distancia L - j >= W", es decir la
    regla efectiva no depende de L; pero el VALOR de W si tiene que ser
    independiente del contenido de x, porque entra en los logits de todas las
    posiciones. Ver coarse_window().
    """
    j = torch.arange(L + 1, device=base.device).view(1, 1, 1, L + 1)
    return base | (j <= (L - W).view(-1, 1, 1, 1))


# =============================================================================
# 4. Modelo
# =============================================================================
class HierarchicalMusicLM(TokenARModel):
    """Decoder grueso (contorno melodico) + decoder fino condicionado a el."""

    name = "hierarchical"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_model)
        n_layers = int(cfg.n_layers)
        n_heads = int(cfg.n_heads)
        d_ff = int(getattr(cfg, "d_ff", 4 * d))
        dropout = float(getattr(cfg, "dropout", 0.1))
        attn_dropout = float(getattr(cfg, "attn_dropout", 0.0))
        self.rel_attn = bool(getattr(cfg, "rel_attn", True))
        self.max_rel_dist = int(getattr(cfg, "max_rel_dist", 512))
        self.tie_weights = bool(getattr(cfg, "tie_weights", True))
        self.max_seq_len = int(getattr(cfg, "seq_len", 1024))
        self.vocab_size = int(VOCAB_SIZE)
        self.d_model = d

        # Reparto de capas: ~1/4 al nivel grueso (su secuencia es ~3x mas corta).
        # Los DOS stacks necesitan al menos un bloque: con n_layers=1 la version
        # anterior dejaba el decoder fino VACIO (solo embeddings + LayerNorm) y
        # el forward seguia devolviendo logits, en silencio.
        want = int(getattr(cfg, "n_coarse_layers", max(2, n_layers // 4)))
        self.n_coarse = max(1, min(want, max(1, n_layers - 1)))
        self.n_fine = max(1, n_layers - self.n_coarse)

        # contexto corto: solo en entrenamiento y solo con atencion relativa
        self.short_ctx_p = float(getattr(cfg, "short_ctx_p", 0.25))
        self.short_ctx_min = int(getattr(cfg, "short_ctx_min", 32))
        self.fine_window = int(getattr(cfg, "fine_window", 0))   # 0 = sin limite
        # Rellena la secuencia gruesa hasta L, de modo que su FORMA dependa solo
        # de L y no del contenido de x. Es lo que hace la causalidad exacta BIT
        # A BIT: sin esto, cambiar un token de tipo (NOTE <-> SHIFT) cambia el
        # numero de eventos gruesos K, y aunque las ENTRADAS gruesas que lee el
        # pasado son identicas, el stack grueso corre sobre un tensor de otra
        # longitud y el orden de reduccion de los matmul cambia: aparecen
        # diferencias de ~1e-6 (redondeo float32 puro, no informacion).
        # Cuesta ~14% de tiempo (el stack grueso corre a L en vez de a ~0.43 L).
        # Ponerlo a False recupera ese tiempo y sigue pasando el test del
        # laboratorio (tolerancia 1e-4), pero ya no es exacto bit a bit.
        self.coarse_pad = bool(getattr(cfg, "coarse_pad", True))

        blk = lambda: Block(d, n_heads, d_ff, dropout, self.rel_attn,
                            self.max_rel_dist, attn_dropout)

        # --- nivel grueso ---
        self.mel_emb = nn.Embedding(N_MEL_IN, d)
        self.dur_emb = nn.Embedding(N_DUR_IN, d)
        self.tex_emb = nn.Embedding(N_TEX_IN, d)
        self.coarse_drop = nn.Dropout(dropout)
        self.coarse_blocks = nn.ModuleList([blk() for _ in range(self.n_coarse)])
        self.coarse_ln = nn.LayerNorm(d)
        self.head_pitch = nn.Linear(d, N_PITCH)        # siguiente nota aguda
        self.head_dur = nn.Linear(d, MAX_SHIFT)        # siguiente duracion
        # estado "aun no hay ningun paso cerrado"
        self.bos_coarse = nn.Parameter(torch.zeros(d))

        # --- puente grueso -> fino ---
        self.cond_ln = nn.LayerNorm(d)
        self.cond_proj = nn.Linear(d, d)

        # --- nivel fino ---
        self.tok_emb = nn.Embedding(self.vocab_size, d)
        self.intra_emb = nn.Embedding(N_INTRA, d)
        self.drop = nn.Dropout(dropout)
        self.fine_blocks = nn.ModuleList([blk() for _ in range(self.n_fine)])
        self.ln_f = nn.LayerNorm(d)
        self.head = None if self.tie_weights else nn.Linear(d, self.vocab_size, bias=False)
        # compuerta del sesgo grueso: [note_gate, shift_gate] por posicion
        self.gate = nn.Linear(d, 2)

        self._pe_cache: torch.Tensor | None = None       # solo rama absoluta

        self.apply(self._init_weights)
        std_res = 0.02 / math.sqrt(2 * max(1, n_layers))
        for m in self.modules():
            if getattr(m, "_is_residual_out", False):
                nn.init.normal_(m.weight, mean=0.0, std=std_res)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.bos_coarse)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)          # sigmoid(-2) = 0.12
        nn.init.zeros_(self.cond_proj.bias)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ---------------------------------------------------------------- utilidades
    def _abs_pe(self, L: int, device, dtype) -> torch.Tensor:
        pe = self._pe_cache
        if pe is None or pe.shape[0] < L or pe.device != device or pe.dtype != dtype:
            pe = sinusoidal_pe(max(L, self.max_seq_len), self.d_model, device, dtype)
            self._pe_cache = pe
        return pe[:L]

    def _mask(self, L: int, device, W: torch.Tensor | None) -> torch.Tensor | None:
        """Mascara del skewing (+ ventana opcional). None si atencion absoluta."""
        if not self.rel_attn:
            return None
        m = skew_mask(L, device)
        if W is not None:
            m = window_mask(m, W, L)
        return m

    def _run_stack(self, h: torch.Tensor, blocks, mask) -> torch.Tensor:
        for b in blocks:
            h = b(h, mask)
        return h

    # ------------------------------------------------------------ nivel grueso
    def coarse_states(self, mel_idx: torch.Tensor, dur_idx: torch.Tensor,
                      tex_idx: torch.Tensor, W: torch.Tensor | None = None) -> torch.Tensor:
        """Eventos gruesos [B,K] -> estados causales h_c [B,K,d]."""
        B, K = mel_idx.shape
        h = self.mel_emb(mel_idx) + self.dur_emb(dur_idx) + self.tex_emb(tex_idx)
        if not self.rel_attn:
            h = h + self._abs_pe(K, h.device, h.dtype).unsqueeze(0)
        h = self.coarse_drop(h)
        h = self._run_stack(h, self.coarse_blocks, self._mask(K, mel_idx.device, W))
        return self.coarse_ln(h)

    def coarse_next_logits(self, h_c: torch.Tensor):
        """h_c [...,d] -> (logits de altura [...,88], logits de duracion [...,64])
        del SIGUIENTE evento grueso, es decir del paso que esta ABIERTO."""
        return self.head_pitch(h_c), self.head_dur(h_c)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: Long[B,L] -> logits Float[B,L,V] del token siguiente."""
        assert x.dim() == 2, f"se esperaba [B,L], llego {tuple(x.shape)}"
        B, L = x.shape
        assert L >= 1, "secuencia vacia"
        dev = x.device
        ci = coarse_index(x, pad_to=L if self.coarse_pad else 0)
        K = ci["K"]

        # ventanas de contexto corto (solo entrenamiento, solo atencion relativa)
        Wf = Wc = None
        if self.rel_attn:
            Ws = (draw_windows(B, L, self.short_ctx_p, max(1, self.short_ctx_min), dev)
                  if self.training else None)
            # El regularizador de contexto corto poda LOS DOS niveles por igual:
            # Ws tokens arriba, coarse_window(Ws) eventos abajo. La conversion usa
            # una CONSTANTE y nunca K (que depende del contenido: seria fuga).
            if Ws is not None and K > 0:
                Wc = coarse_window(Ws)
            Wf = Ws
            # fine_window limita SOLO el decoder fino (estilo AudioLM: la etapa de
            # detalle ve poco contexto y el largo alcance queda en la semantica), y
            # lo hace IGUAL en train y en eval.
            if self.fine_window > 0:
                cap = torch.full((B,), min(self.fine_window, L), dtype=torch.long, device=dev)
                Wf = cap if Wf is None else torch.minimum(Wf, cap)

        # --- 1. estados gruesos y su lectura causal en cada posicion de token ---
        bos = self.bos_coarse.to(self.tok_emb.weight.dtype).view(1, 1, -1)
        if K > 0:
            h_c = self.coarse_states(ci["mel_idx"], ci["dur_idx"], ci["tex_idx"], Wc)
            gi = ci["k_at_token"].unsqueeze(-1).expand(B, L, self.d_model)
            hc_t = h_c.gather(1, gi)                                   # [B,L,d]
            hc_t = torch.where(ci["has_coarse"].unsqueeze(-1), hc_t, bos.to(hc_t.dtype))
        else:
            hc_t = bos.expand(B, L, self.d_model)

        # --- 2. decoder fino condicionado ---
        h = self.tok_emb(x) + self.intra_emb(ci["intra"]) + self.cond_proj(self.cond_ln(hc_t))
        if not self.rel_attn:
            h = h + self._abs_pe(L, h.device, h.dtype).unsqueeze(0)
        h = self.drop(h)
        h = self._run_stack(h, self.fine_blocks, self._mask(L, dev, Wf))
        h = self.ln_f(h)
        logits = F.linear(h, self.tok_emb.weight) if self.tie_weights else self.head(h)

        # --- 3. sesgo del nivel grueso sobre los bloques NOTE_ON y SHIFT ---
        p_log, d_log = self.coarse_next_logits(hc_t)                   # [B,L,88], [B,L,64]
        g = torch.sigmoid(self.gate(h))                                # [B,L,2]
        bias = torch.cat([torch.zeros(B, L, NOTE_OFF_ID, device=dev, dtype=logits.dtype),
                          (g[..., :1] * p_log).to(logits.dtype),
                          (g[..., 1:] * d_log).to(logits.dtype)], dim=-1)
        return logits + bias

    def supports_state(self) -> bool:
        return False

    # ------------------------------------------------- generacion del contorno
    @torch.no_grad()
    def generate_melody_only(self, prefix_tokens, n_events: int = 128, *,
                             temperature: float = 1.0, top_k: int = 0,
                             top_p: float = 0.0, penalty: float = 1.0,
                             penalty_window: int = 16, greedy: bool = False,
                             max_ctx: int = 0, return_full: bool = False,
                             generator: torch.Generator | None = None) -> np.ndarray:
        """Continua SOLO el contorno melodico: una nota (la mas aguda) por paso.

        Es el modo natural para los prefijos de la prueba, que son monofonicos:
        el prefijo se descompone en (mel, dur), el decoder grueso lo continua en
        su propio nivel --secuencia ~3x mas corta y sin ninguna decision de
        textura que tomar-- y el resultado se re-tokeniza como NOTE_ON+SHIFT.
        No interviene el decoder fino, asi que no puede inventar armonia ni
        caer en los bucles de acompanamiento del nivel de token.

        Devuelve los tokens de la CONTINUACION (o prefijo+continuacion si
        return_full=True). penalty aplica la penalizacion de repeticion de CTRL
        sobre las ULTIMAS penalty_window alturas generadas, para romper ciclos.
        """
        self.eval()
        dev = next(self.parameters()).device
        pre = np.asarray(prefix_tokens, np.int64).ravel()
        dec = decompose_tokens(pre)
        mel = [int(v) for v in dec["mel"]]
        dur = [int(v) for v in dec["dur"]]
        cnt = [int(v) for v in dec["cnt"]]
        n_pre = len(mel)

        for _ in range(int(n_events)):
            if mel:
                lo = 0 if max_ctx <= 0 else max(0, len(mel) - int(max_ctx))
                m = torch.tensor([[(MEL_REST if v < 0 else v + MEL_OFF) for v in mel[lo:]]],
                                 dtype=torch.long, device=dev)
                d = torch.tensor([[v for v in dur[lo:]]], dtype=torch.long, device=dev)
                tx = torch.tensor([[min(max(v, 0), N_TEX_IN - 2) + 1 for v in cnt[lo:]]],
                                  dtype=torch.long, device=dev)
                h = self.coarse_states(m, d, tx)[:, -1]                # [1,d]
            else:
                h = self.bos_coarse.view(1, -1)
            # dtype del stack, no float() a secas: con el modelo en bf16/fp16 un
            # h en float32 rompe el matmul de las cabezas gruesas.
            p_log, d_log = self.coarse_next_logits(h.to(self.head_pitch.weight.dtype))
            if penalty and penalty > 1.0:
                recent = [v for v in mel[max(0, len(mel) - int(penalty_window)):] if v >= 0]
                for v in set(recent):
                    p_log[0, v] = p_log[0, v] / penalty if p_log[0, v] > 0 else p_log[0, v] * penalty
            nxt_p = _pick(p_log, temperature, top_k, top_p, greedy, generator)
            nxt_d = _pick(d_log, temperature, top_k, top_p, greedy, generator)
            mel.append(int(nxt_p)); dur.append(int(nxt_d) + 1); cnt.append(1)

        toks = coarse_to_tokens(mel[n_pre:], dur[n_pre:])
        if return_full:
            return np.concatenate([pre, toks])
        return toks


def _pick(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
          greedy: bool, generator=None) -> int:
    """Muestreo de una sola categoria con temperatura / top-k / top-p."""
    lg = logits.reshape(-1).float()
    if greedy or temperature <= 0:
        return int(lg.argmax().item())
    lg = lg / float(temperature)
    try:                                          # reutiliza los filtros auditados
        from sampling import top_k_filter, top_p_filter
    except ImportError:                           # pragma: no cover
        try:
            from src.sampling import top_k_filter, top_p_filter
        except ImportError:
            top_k_filter = top_p_filter = None
    if top_k_filter is not None:
        if top_k:
            lg = top_k_filter(lg, int(top_k))
        if top_p:
            lg = top_p_filter(lg, float(top_p))
    probs = torch.softmax(lg, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


# =============================================================================
# 5. Verificacion (ejecutar:  python src/models/hierarchical.py)
# =============================================================================
def _cfg(**kw):
    try:
        from config import Config
    except ImportError:                                      # pragma: no cover
        from src.config import Config
    c = Config()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _real_tokens(n: int = 6):
    """Piezas reales del corpus (indices del split de test)."""
    try:
        from data.datasets import get_tokens, load_meta
    except ImportError:                                      # pragma: no cover
        from src.data.datasets import get_tokens, load_meta
    _, splits = load_meta()
    idx = list(splits["test"])[:n]
    return [get_tokens(int(i)) for i in idx]


def _test_reversibility(verbose: bool = True) -> bool:
    """Descomposicion grueso/fino exacta sobre piezas REALES del corpus."""
    ok = True
    try:
        pieces = _real_tokens(6)
    except Exception as e:                                   # pragma: no cover
        print(f"  reversibilidad: no se pudo leer el corpus ({e}) -> se usan sinteticos")
        rng = np.random.default_rng(0)
        pieces = []
        for _ in range(6):
            roll = (rng.random((400, 88)) < 0.01).astype(np.uint8)
            try:
                from data.tokenizer import encode_roll_fast
            except ImportError:
                from src.data.tokenizer import encode_roll_fast
            pieces.append(encode_roll_fast(roll).astype(np.int64))
    for i, t in enumerate(pieces):
        t = np.asarray(t, np.int64)
        dec = decompose_tokens(t)
        rec = recompose_tokens(dec)
        same = rec.shape == t.shape and bool((rec == t).all())
        ok = ok and same and dec["mid_specials"] == 0
        if verbose:
            n_mono = int((dec["cnt"] == 1).sum())
            print(f"  pieza {i}: {len(t):6d} tokens -> K={len(dec['mel']):5d} eventos gruesos "
                  f"({len(t)/max(1,len(dec['mel'])):.2f} tok/evento, {n_mono/max(1,len(dec['mel'])):.0%} monofonicos), "
                  f"extra={sum(len(e) for e in dec['extra']):5d} notas finas, "
                  f"especiales_en_medio={dec['mid_specials']} -> "
                  f"recompose {'EXACTO' if same else 'DISTINTO'}")
    # ventanas cortadas a media pieza (paso abierto al final)
    t = np.asarray(pieces[0], np.int64)
    sub_ok = True
    for a, b in [(37, 37 + 70), (100, 1124), (5, 6), (0, 1)]:
        w = t[a:b]
        if w.size == 0:
            continue
        r = recompose_tokens(decompose_tokens(w))
        sub_ok = sub_ok and r.shape == w.shape and bool((r == w).all())
    ok = ok and sub_ok
    if verbose:
        print(f"  ventanas cortadas a media pieza (paso abierto) -> "
              f"{'EXACTO' if sub_ok else 'FALLO'}")
    # coherencia entre la version numpy y la version torch usada en el forward
    t = np.asarray(pieces[0][:2048], np.int64)
    dec = decompose_tokens(t)
    ci = coarse_index(torch.from_numpy(t).unsqueeze(0))
    K = ci["K"]
    mel_ref = np.where(dec["mel"] < 0, MEL_REST, dec["mel"] + MEL_OFF)[:K]
    agree = bool((ci["mel_idx"][0, :K].numpy() == mel_ref).all()) and \
            bool((ci["dur_idx"][0, :K].numpy() == dec["dur"][:K]).all()) and K == len(dec["mel"])
    ok = ok and agree
    if verbose:
        print(f"  coarse_index (torch, causal) == decompose_tokens (numpy) en K={K} eventos "
              f"-> {'OK' if agree else 'FALLO'}")
    return ok


def _mono_seq(L: int) -> list:
    """Melodia monofonica de L tokens: BOS + (NOTE_ON, SHIFT) alternados.

    Es la textura de los prefijos de la prueba final, el caso que mas importa.
    """
    seq = [BOS]
    i = 0
    while len(seq) < L:
        seq.append(NOTE_OFF_ID + 40 + (i % 7))
        if len(seq) < L:
            seq.append(SHIFT_OFF_ID + 3)
        i += 1
    return seq[:L]


def _test_strict_causality(model, L: int = 48, verbose: bool = True) -> bool:
    """Perturba SOLO x[:, t] y exige logits IDENTICOS BIT A BIT en todo s < t.

    Mas estricto que tests/test_causality.py (que perturba todo el sufijo y
    compara con tolerancia 1e-4): aqui la comparacion es torch.equal, exacta,
    y se barre TODA posicion t (no una muestra) con tokens de los dos tipos.
    Los patrones incluyen un paso ABIERTO larguisimo: si el modelo mirase el
    agudo del paso en curso, cambiar una nota de ese paso cambiaria logits
    anteriores y se veria aqui.
    """
    model = model.eval()
    torch.manual_seed(7)
    V = VOCAB_SIZE
    ok = True
    worst = 0.0
    # (a) secuencia aleatoria; (b) un solo acorde larguisimo (paso abierto de 40
    # notas): el caso que rompe una derivacion no causal del contorno.
    xs = {
        "aleatoria": torch.randint(3, V, (2, L)),
        "acorde largo": torch.cat([
            torch.tensor([[BOS] + list(range(NOTE_OFF_ID, NOTE_OFF_ID + L - 2)) + [SHIFT_OFF_ID]]),
            torch.tensor([[BOS] + list(range(NOTE_OFF_ID + 20, NOTE_OFF_ID + 20 + L - 2)) + [SHIFT_OFF_ID + 3]]),
        ], 0),
        "monofonica": torch.tensor(_mono_seq(L)).view(1, -1).repeat(2, 1),
    }
    for tag, x in xs.items():
        x = x[:, :L].contiguous()
        with torch.no_grad():
            base = model(x)
        for t in range(1, L):                       # barrido exhaustivo
            x2 = x.clone()
            # cambia SOLO la posicion t por otro token del mismo y de otro tipo
            for new in (NOTE_OFF_ID + 87, SHIFT_OFF_ID + 63, NOTE_OFF_ID):
                if bool((x2[:, t] == new).all()):
                    continue
                x2[:, t] = new
                with torch.no_grad():
                    pert = model(x2)
                same = torch.equal(base[:, :t], pert[:, :t])
                diff = (base[:, t:] - pert[:, t:]).abs().max().item()
                bad = (base[:, :t] - pert[:, :t]).abs().max().item()
                worst = max(worst, bad)
                ok = ok and same
                if not same and verbose:
                    print(f"    DISTINTO {tag} t={t} new={new}: max|delta pasado|={bad:.3e}")
                if verbose and t in (L // 2,) and new == NOTE_OFF_ID + 87:
                    print(f"  [{tag}] t={t}: logits[:, :{t}] identicos bit a bit "
                          f"({'torch.equal=True' if same else 'FALLO'}), "
                          f"max|delta en t..L|={diff:.3e}")
    if verbose:
        print(f"  causalidad estricta bit a bit (3 patrones x {L - 1} posiciones x 3 tokens, "
              f"coarse_pad={model.coarse_pad}) -> {'OK' if ok else 'FUGA CAUSAL'}"
              + (f"   max|delta pasado|={worst:.3e}" if worst > 0 else ""))
    return ok


def _test_shapes(model, verbose: bool = True) -> bool:
    ok = True
    for L in (1, 17, 70, 1024):
        x = torch.randint(3, VOCAB_SIZE, (2, L))
        with torch.no_grad():
            y = model.eval()(x)
        good = tuple(y.shape) == (2, L, VOCAB_SIZE) and bool(torch.isfinite(y).all())
        ok = ok and good
        if verbose:
            print(f"  L={L:5d}: {tuple(y.shape)} esperado (2,{L},{VOCAB_SIZE}) "
                  f"finito={bool(torch.isfinite(y).all())} -> {'OK' if good else 'FALLO'}")
    # casos degenerados: sin ningun SHIFT (K=0) y todo SHIFT
    for tag, x in [("sin SHIFT (K=0)", torch.full((2, 9), NOTE_OFF_ID + 5)),
                   ("todo SHIFT", torch.full((2, 9), SHIFT_OFF_ID + 2)),
                   ("todo PAD", torch.zeros((2, 9), dtype=torch.long))]:
        with torch.no_grad():
            y = model(x)
        good = tuple(y.shape) == (2, 9, VOCAB_SIZE) and bool(torch.isfinite(y).all())
        ok = ok and good
        if verbose:
            print(f"  {tag}: {tuple(y.shape)} finito={bool(torch.isfinite(y).all())} "
                  f"-> {'OK' if good else 'FALLO'}")
    return ok


def _no_dropout(model):
    """Pone todo el dropout a 0 y devuelve la funcion que lo restaura."""
    ps = [(m, m.p) for m in model.modules() if isinstance(m, nn.Dropout)]
    for m, _ in ps:
        m.p = 0.0
    drops = [(m, m.attn.p_drop) for m in model.modules() if hasattr(m, "attn")]
    for m, _ in drops:
        m.attn.p_drop = 0.0

    def restore():
        for m, q in ps:
            m.p = q
        for m, q in drops:
            m.attn.p_drop = q
    return restore


def _test_train_mode_causality(model, L: int = 64, tag: str = "", n_t: int = 12,
                               exact: bool = True, verbose: bool = True) -> bool:
    """La ventana de contexto corto no puede crear fuga: se comprueba en train().

    Se pone todo el dropout a 0 y se fija la semilla antes de cada forward para
    que las ventanas sorteadas sean las mismas en ambas llamadas, y se barren
    varias posiciones t con tokens de los dos tipos (NOTE, SHIFT y PAD). Este es
    el test que destapo la fuga de Wc = ceil(Wf * K / L): la ventana del nivel
    grueso valia lo que valia el numero TOTAL de SHIFT de la secuencia, futuro
    incluido. Se ejecuta con short_ctx_p forzado a 1.0 para que la ventana este
    SIEMPRE activa (con 0.25 el fallo solo aparecia a ratos).
    """
    restore = _no_dropout(model)
    p_old = model.short_ctx_p
    model.short_ctx_p = 1.0
    model.train()
    torch.manual_seed(5)
    x = torch.randint(3, VOCAB_SIZE, (4, L))
    worst, nbad, ntot = 0.0, 0, 0
    for t in sorted(set(int(v) for v in torch.linspace(1, L - 1, n_t))):
        for new in (NOTE_OFF_ID + 87, SHIFT_OFF_ID + 63, NOTE_OFF_ID, PAD):
            x2 = x.clone()
            if bool((x2[:, t] == new).all()):
                continue
            x2[:, t] = new
            torch.manual_seed(11)
            with torch.no_grad():
                base = model(x)
            torch.manual_seed(11)
            with torch.no_grad():
                pert = model(x2)
            ntot += 1
            if not torch.equal(base[:, :t], pert[:, :t]):
                nbad += 1
                worst = max(worst, (base[:, :t] - pert[:, :t]).abs().max().item())
    model.short_ctx_p = p_old
    restore()
    model.eval()
    # exact=True exige igualdad BIT A BIT (es lo que promete coarse_pad=True).
    # Con coarse_pad=False la longitud del tensor grueso depende del contenido y
    # queda el mismo redondeo de ~1e-7 que en eval: se juzga con la tolerancia
    # 1e-4 del laboratorio. Lo que NO puede volver a pasar es un 1.6e-02.
    ok = (nbad == 0) if exact else (worst < 1e-4)
    if verbose:
        crit = "bit a bit" if exact else "tol 1e-4"
        print(f"  train {tag:22s} {ntot:3d} perturbaciones, no identicas={nbad:3d}, "
              f"max|delta pasado|={worst:.3e} ({crit}) -> "
              f"{'OK' if ok else 'FUGA CAUSAL'}")
    return ok


def _test_batch_independence(model, tag: str = "", verbose: bool = True) -> bool:
    """Cambiar la fila 0 del batch no puede tocar las demas (eval, bit a bit).

    Con coarse_pad=True la longitud gruesa vale L y no depende del batch, asi
    que la igualdad debe ser EXACTA. Con coarse_pad=False K es el maximo del
    batch y aparece redondeo (~1e-7): no es fuga causal, pero si acoplamiento
    entre ejemplos, y conviene saberlo.
    """
    model.eval()
    torch.manual_seed(2)
    x = torch.randint(3, VOCAB_SIZE, (4, 48))
    with torch.no_grad():
        base = model(x)
    x2 = x.clone()
    x2[0] = torch.randint(3, VOCAB_SIZE, (48,))
    with torch.no_grad():
        pert = model(x2)
    same = torch.equal(base[1:], pert[1:])
    d = (base[1:] - pert[1:]).abs().max().item()
    if verbose:
        print(f"  {tag:26s} filas 1..3 identicas bit a bit={same} max|delta|={d:.3e} "
              f"-> {'OK' if same else 'ACOPLAMIENTO (redondeo)'}")
    return same


def _test_short_ctx(model, verbose: bool = True) -> bool:
    """La ventana deslizante hace lo que dice: recorta memoria, no la inventa."""
    if not model.rel_attn:
        return True
    L = 64
    x = torch.randint(3, VOCAB_SIZE, (1, L))
    model.eval()
    W = torch.tensor([8])
    with torch.no_grad():
        h = model.tok_emb(x)
        m_full = skew_mask(L, x.device)
        m_win = window_mask(m_full, W, L)
        a = model.fine_blocks[0](h, m_full)
        b = model.fine_blocks[0](h, m_win)
        # con ventana 8, la posicion 3 ve el mismo pasado (0..3) en ambos casos
        same_early = torch.allclose(a[:, 3], b[:, 3], atol=1e-6)
        diff_late = (a[:, -1] - b[:, -1]).abs().max().item()
    ok = same_early and diff_late > 1e-6
    if verbose:
        print(f"  ventana W=8 sobre L=64: pos 3 intacta={same_early}, "
              f"pos 63 cambia (max|d|={diff_late:.3e}) -> {'OK' if ok else 'FALLO'}")
    return ok


def _test_coarse_pad_equiv(verbose: bool = True) -> bool:
    """Rellenar la secuencia gruesa hasta L no cambia el MODELO, solo el tensor.

    Es la comprobacion de que coarse_pad es puro apano numerico y no altera la
    funcion que implementa el modelo: las posiciones de relleno van al final y
    la mascara causal ya impedia que una posicion real las mirase.
    """
    torch.manual_seed(3)
    ma = HierarchicalMusicLM(_cfg(coarse_pad=True, short_ctx_p=1.0)).eval()
    mb = HierarchicalMusicLM(_cfg(coarse_pad=False, short_ctx_p=1.0)).eval()
    mb.load_state_dict(ma.state_dict())
    x = torch.randint(3, VOCAB_SIZE, (2, 96))
    with torch.no_grad():
        d = (ma(x) - mb(x)).abs().max().item()
    ok = d < 1e-4
    if verbose:
        print(f"  eval : logits coarse_pad=True vs False: max|delta|={d:.3e} "
              f"(solo redondeo) -> {'OK' if ok else 'FALLO'}")
    # Y en TRAIN, con la ventana corta activa: aqui es donde la version anterior
    # SI cambiaba de modelo, porque Wc valia ceil(Wf*K/L) y K vale L o ~0.43L
    # segun el relleno, o sea ventanas gruesas 2.4x distintas para la misma Wf.
    ra, rb = _no_dropout(ma), _no_dropout(mb)
    ma.train(); mb.train()
    torch.manual_seed(21)
    with torch.no_grad():
        ya = ma(x)
    torch.manual_seed(21)
    with torch.no_grad():
        yb = mb(x)
    dt = (ya - yb).abs().max().item()
    ok_t = dt < 1e-4
    ra(); rb(); ma.eval(); mb.eval()
    if verbose:
        print(f"  train: logits coarse_pad=True vs False: max|delta|={dt:.3e} "
              f"(la ventana corta debe valer lo mismo en las dos) -> "
              f"{'OK' if ok_t else 'FALLO'}")
    return ok and ok_t


def _test_eval_prefixes(model, verbose: bool = True) -> bool:
    """Los PREFIJOS REALES de la prueba final, que es para lo que existe esto.

    external_eval_prefix_5s.npz: 5 melodias conocidas, 100 pasos (5 s) cada una.
    Se comprueba (a) que la descomposicion en niveles es EXACTA sobre ellas,
    (b) que son puro nivel grueso --exactamente 1 nota por paso activo, o sea
    2 tokens por evento: NOTE_ON + SHIFT--, (c) que el forward acepta esas
    longitudes y (d) que generate_melody_only las continua sin salirse de la
    textura monofonica.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    f = root / "external_eval_prefix_5s.npz"
    if not f.exists():                                       # pragma: no cover
        print("  external_eval_prefix_5s.npz no encontrado: se omite")
        return True
    try:
        from data.tokenizer import encode_roll_fast, decode_tokens
    except ImportError:                                      # pragma: no cover
        from src.data.tokenizer import encode_roll_fast, decode_tokens
    z = np.load(f, allow_pickle=True)
    rolls, off, ids = z["rolls_flat"], z["offsets"], z["ids"]
    ok = True
    for i, name in enumerate(ids):
        r = rolls[off[i]:off[i + 1]]
        tk = encode_roll_fast(r, add_bos=True, add_eos=False).astype(np.int64)
        dec = decompose_tokens(tk)
        K = len(dec["mel"])
        exact = np.array_equal(recompose_tokens(dec), tk)
        mono = bool((dec["cnt"] <= 1).all())
        with torch.no_grad():
            lg = model.eval()(torch.from_numpy(tk).unsqueeze(0))
        shp = tuple(lg.shape) == (1, len(tk), VOCAB_SIZE) and bool(torch.isfinite(lg).all())
        cont = model.generate_melody_only(tk, n_events=60, temperature=1.0,
                                          top_p=0.95, penalty=1.2)
        roll2 = decode_tokens(np.concatenate([tk, cont]))
        poly_ok = int(roll2.sum(1).max()) <= 1
        good = exact and mono and shp and poly_ok
        ok = ok and good
        if verbose:
            print(f"  {str(name)[:24]:24s} {len(tk):3d} tok -> K={K:3d} eventos "
                  f"({len(tk)/max(1,K):.2f} tok/evento) monofonico={mono} "
                  f"reversible={exact} forward{tuple(lg.shape)} "
                  f"contorno+{len(cont)} tok polif_max={int(roll2.sum(1).max())} "
                  f"-> {'OK' if good else 'FALLO'}")
    if verbose:
        print("  (2.10 tok/evento = 1 NOTE_ON + 1 SHIFT por paso: el prefijo de la")
        print("   prueba es LITERALMENTE el nivel grueso, sin nivel fino ninguno)")
    return ok


def _test_melody_gen(model, verbose: bool = True) -> bool:
    """generate_melody_only produce tokens legales y estrictamente monofonicos."""
    try:
        from data.tokenizer import decode_tokens
    except ImportError:                                      # pragma: no cover
        from src.data.tokenizer import decode_tokens
    prefix = coarse_to_tokens([40, 40, 47, 47, 49, 49, 47], [8, 8, 8, 8, 8, 8, 16])
    out = model.generate_melody_only(prefix, n_events=40, temperature=1.0, top_p=0.95)
    full = np.concatenate([prefix, out])
    legal = bool(((out >= NOTE_OFF_ID) & (out < VOCAB_SIZE)).all())
    roll = decode_tokens(full)
    poly = roll.sum(1)
    mono = bool((poly <= 1).all())
    ok = legal and mono and len(out) > 0
    if verbose:
        act = int((poly > 0).sum())
        print(f"  generate_melody_only: {len(prefix)} tokens de prefijo -> {len(out)} tokens, "
              f"roll {roll.shape}, pasos activos={act}, polifonia max={int(poly.max()) if len(poly) else 0}, "
              f"legales={legal} -> {'OK' if ok else 'FALLO'}")
    return ok


def _has_cuda() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _gpu_free_gib() -> float:
    free, _ = torch.cuda.mem_get_info()
    return free / 2**30


def _bench(model, B: int = 16, L: int = 1024, device: str = "cpu", iters: int = 8):
    """fwd+bwd+clip+step; devuelve (it/s, ms, pico GiB). Aborta si la GPU esta
    saturada por otros procesos (la primera iteracion tarda mas de 2 s), porque
    entonces el numero no mide el modelo sino la contencion."""
    import time
    ts_warm = [0.0]
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(3, VOCAB_SIZE, (B, L + 1), device=device)
    xi, yi = x[:, :-1].contiguous(), x[:, 1:].contiguous()
    amp = device.startswith("cuda")
    ts = []
    for i in range(iters + 3):
        if device.startswith("cuda") and i == 1 and ts_warm[0] > 2.0:
            raise RuntimeError(f"GPU saturada por otros procesos: "
                               f"{ts_warm[0]*1e3:.0f} ms en la primera iteracion")
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lo = model(xi)
            loss = F.cross_entropy(lo.reshape(-1, VOCAB_SIZE).float(), yi.reshape(-1), ignore_index=PAD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i == 0:
            ts_warm[0] = dt
        if i >= 3:
            ts.append(dt)
    ts.sort()
    med = ts[len(ts) // 2]
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0.0
    return 1.0 / med, med * 1e3, peak


def _bench_gpu_ab(B: int = 16, L: int = 1024, rounds: int = 12) -> None:
    """A/B contra music_transformer por RONDAS INTERCALADAS.

    La GPU la comparten varios agentes del laboratorio. Medir un modelo detras
    de otro no es comparable (una racha de contencion cae entera sobre uno), y
    la MEDIANA queda inservible. Aqui cada ronda hace un paso de cada modelo y
    se reporta el MINIMO: es la ronda en la que la GPU estuvo libre, es decir la
    mejor estimacion del coste propio del modelo. Si min y p25 coinciden, la
    ventana estaba limpia y el numero es fiable; si difieren mucho, habia
    contencion y hay que repetir.
    """
    import time
    try:
        from models.music_transformer import MusicTransformer
    except ImportError:                                      # pragma: no cover
        from src.models.music_transformer import MusicTransformer
    specs = [("hierarchical coarse_pad=True ", lambda: HierarchicalMusicLM(_cfg())),
             ("hierarchical coarse_pad=False", lambda: HierarchicalMusicLM(_cfg(coarse_pad=False))),
             ("music_transformer (linea base)", lambda: MusicTransformer(_cfg()))]
    ms, opts, times, peaks = [], [], [], []
    for tag, mk in specs:
        m = mk().to("cuda").train()
        ms.append((tag, m))
        opts.append(torch.optim.AdamW(m.parameters(), lr=1e-4))
        times.append([]); peaks.append(0.0)
    x = torch.randint(3, VOCAB_SIZE, (B, L + 1), device="cuda")
    xi, yi = x[:, :-1].contiguous(), x[:, 1:].contiguous()
    for r in range(rounds + 2):
        for i, ((tag, m), opt) in enumerate(zip(ms, opts)):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(xi)
                loss = F.cross_entropy(lo.reshape(-1, VOCAB_SIZE).float(),
                                       yi.reshape(-1), ignore_index=PAD)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if r >= 2:
                times[i].append(time.perf_counter() - t0)
                peaks[i] = max(peaks[i], torch.cuda.max_memory_allocated() / 2**30)
    print(f"  B={B} L={L} bf16, fwd+bwd+clip+step, {rounds} rondas intercaladas:")
    for i, (tag, m) in enumerate(ms):
        t = sorted(times[i])
        print(f"    {tag}: {m.n_params()/1e6:5.2f} M  min {t[0]*1e3:6.1f} ms "
              f"({1/t[0]:5.2f} it/s)  p25 {t[len(t)//4]*1e3:6.1f} ms  "
              f"pico {peaks[i]:.2f} GiB")
    print("    (los picos incluyen los estados de Adam de los TRES modelos a la vez)")


def _selftest(bench_gpu: bool = True) -> bool:
    torch.manual_seed(0)
    cfg = _cfg()
    model = HierarchicalMusicLM(cfg)
    print("=" * 78)
    print(f"  {model.param_report()}   (grueso {model.n_coarse} capas / fino {model.n_fine} capas, "
          f"d_model={cfg.d_model}, heads={cfg.n_heads}, d_ff={cfg.d_ff}, "
          f"rel_attn={cfg.rel_attn}, R={cfg.max_rel_dist})")
    pre_c = ("coarse", "mel_emb", "dur_emb", "tex_emb", "head_pitch", "head_dur", "bos_coarse")
    pre_f = ("fine", "tok_emb", "intra_emb", "ln_f", "gate", "cond", "head.")
    n_c = sum(p.numel() for n, p in model.named_parameters() if n.startswith(pre_c))
    n_f = sum(p.numel() for n, p in model.named_parameters() if n.startswith(pre_f))
    print(f"  desglose: nivel grueso {n_c/1e6:.2f} M | nivel fino {n_f/1e6:.2f} M | "
          f"total {model.n_params()/1e6:.2f} M")
    print("=" * 78)

    print("\n[1] formas  (L = 1, 17, 70, 1024 + casos degenerados)")
    ok_shapes = _test_shapes(model)

    print("\n[2] causalidad ESTRICTA bit a bit (eval): se perturba SOLO x[:, t]")
    ok_strict = _test_strict_causality(model)
    print("     control: la misma prueba con coarse_pad=False (K depende del contenido)")
    torch.manual_seed(0)
    m_np = HierarchicalMusicLM(_cfg(coarse_pad=False))
    m_np.load_state_dict(model.state_dict())
    ok_nopad = _test_strict_causality(m_np, verbose=True)
    print(f"     (coarse_pad=False falla la igualdad EXACTA por redondeo, no por fuga: "
          f"{'esperado' if not ok_nopad else 'tambien exacto aqui'})")

    print("\n[3] causalidad en modo train (ventanas de contexto corto SIEMPRE")
    print("    activas: short_ctx_p forzado a 1.0. Aqui estaba la fuga de Wc)")
    ok_train = _test_train_mode_causality(model, tag="coarse_pad=True")
    ok_train = _test_train_mode_causality(m_np, tag="coarse_pad=False",
                                          exact=False) and ok_train

    print("\n[3b] independencia entre elementos del batch (eval)")
    ok_batch = _test_batch_independence(model, "coarse_pad=True")
    _test_batch_independence(m_np, "coarse_pad=False")

    print("\n[4] test del laboratorio: tests/test_causality.py")
    try:
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        for p in (str(root), str(root / "src")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from tests.test_causality import check_token_causality, check_shapes_token
        ok_lab = check_shapes_token(model, L=32)
        for L in (32, 70, 256, 1024):
            print(f"    L={L}:", end=" ")
            ok_lab = check_token_causality(model, L=L) and ok_lab
        print("    (con coarse_pad=False, tolerancia 1e-4 del laboratorio)")
        for L in (32, 70):
            print(f"    L={L}:", end=" ")
            ok_lab = check_token_causality(m_np, L=L) and ok_lab
    except Exception as e:                                   # pragma: no cover
        print(f"  no se pudo importar tests.test_causality: {e}")
        ok_lab = False

    print("\n[5] ventana de contexto corto y equivalencia de coarse_pad")
    ok_win = _test_short_ctx(model)
    ok_eq = _test_coarse_pad_equiv()

    print("\n[6] reversibilidad de la descomposicion en niveles (piezas REALES)")
    ok_rev = _test_reversibility()

    print("\n[7] generacion solo del contorno melodico")
    ok_gen = _test_melody_gen(model)

    print("\n[8] los PREFIJOS REALES de la prueba final (external_eval_prefix_5s.npz)")
    ok_pref = _test_eval_prefixes(model)

    print("\n[9] velocidad")
    it_cpu, ms_cpu, _ = _bench(model, B=2, L=256, device="cpu", iters=4)
    print(f"  CPU  B=2  L=256  fp32 : {it_cpu:.2f} it/s ({ms_cpu:.0f} ms)")
    if bench_gpu and _has_cuda():
        try:
            free = _gpu_free_gib()
            print(f"  GPU: {free:.2f} GiB libres de "
                  f"{torch.cuda.mem_get_info()[1]/2**30:.2f}")
            if free > 8.0:
                _bench_gpu_ab(B=16, L=1024, rounds=12)
            else:
                print("  GPU ocupada (<8 GiB libres): se omite el benchmark de")
                print("  entrenamiento; el numero de CPU de arriba si es valido.")
        except Exception as e:                               # pragma: no cover
            print(f"  benchmark GPU fallido: {e}")

    ok = all([ok_shapes, ok_strict, ok_train, ok_batch, ok_lab, ok_win, ok_eq,
              ok_rev, ok_gen, ok_pref])
    print("\n" + "=" * 78)
    print(f"  RESULTADO GLOBAL: {'TODO OK' if ok else 'HAY FALLOS'}")
    print("=" * 78)
    return ok


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                                        # pragma: no cover
        pass
    gpu = "--no-gpu" not in sys.argv
    sys.exit(0 if _selftest(bench_gpu=gpu) else 1)
