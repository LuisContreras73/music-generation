"""Perceiver AR: contexto largo con un unico cross-attend a N latentes.

Hawthorne, Jaegle, Cangea, Borgeaud, Nash, Malinowski, Dieleman, Vinyals,
Botvinick, Simon, Sheahan, Zeghidour, Alayrac, Carreira, Engel,
"General-purpose, long-context autoregressive modeling with Perceiver AR",
ICML 2022, PMLR 162:8535-8558 (arXiv:2202.07765).

IDEA DEL PAPER EN UNA FRASE
---------------------------
Un decoder causal normal paga O(M^2) en CADA una de sus L capas. Perceiver AR
paga O(M*N) UNA sola vez (el cross-attend) y luego O(N^2) en las L capas del
stack, con N < M. El contexto M puede crecer a decenas de miles de tokens sin
que crezca el coste del stack profundo; lo que se pierde es que solo hay N
salidas, no M.

LAS TRES COSAS QUE CASI TODO EL MUNDO IMPLEMENTA MAL
-----------------------------------------------------
(1) LAS QUERIES LATENTES NO SON PARAMETROS APRENDIDOS. Apendice C del paper,
    literal: "in Perceiver AR, it is typically constructed by taking the last N
    elements of the input array: X_Q = X_KV[-N:,:]". Es decir, las N queries
    SON los embeddings de las N ULTIMAS posiciones de la entrada. No hay ningun
    nn.Parameter de latentes en este fichero (eso es el Perceiver original /
    Perceiver IO, que NO es autoregresivo). Si las queries fuesen aprendidas no
    habria forma de saber que posicion predice cada latente y la causalidad
    seria imposible de definir.
(2) LA MASCARA DEL CROSS-ATTEND. El paper la da explicitamente: se enmascara el
    par (n, m), n en [0,N), m en [0,M), cuando

        m > n + M - N - 1

    o sea, el latente n solo ve los PRIMEROS n + M - N elementos de la entrada.
    Por que eso es exactamente la causalidad, ver la seccion siguiente.
(3) EL CROSS-ATTEND ES UNO SOLO. Despues van L capas de self-attention causal
    ESTANDAR (mascara triangular) sobre los N latentes. No se repite el
    cross-attend por capa.

POR QUE LA FORMULA m > n + M - N - 1 PRESERVA LA CAUSALIDAD
------------------------------------------------------------
El latente n es, por construccion (1), la posicion de entrada

        p(n) = M - N + n .

En un modelo autoregresivo la salida en la posicion p predice x[p+1] y por
tanto puede depender de x[0..p] y de nada mas. Hay dos caminos por los que la
informacion entra en el latente n:

  a) el CROSS-ATTEND, que con la formula del paper ve m en [0, n+M-N-1], es
     decir x[0 .. p(n)-1]: el pasado ESTRICTO de p(n). El "-1" no es un
     descuido: excluye la propia posicion.
  b) la CONEXION RESIDUAL de la query. Como la query es el embedding de
     x[p(n)] (punto 1), el residual del bloque cross-attend inyecta x[p(n)] sin
     pasar por la atencion.

Sumando a) y b), el latente n depende exactamente de x[0 .. p(n)] y de nada
posterior. Ni se pierde informacion (el token propio llega por el residual) ni
se filtra futuro. Esta es la razon de que el "-1" sea correcto Y no lesivo, y
es tambien la razon de que NO se pueda quitar el residual de la query de este
bloque sin romper el modelo: sin el, el latente n nunca veria su propio token.

Despues, las L capas de self-attention usan la mascara triangular normal
(latente n ve latentes 0..n). Como el latente n' <= n depende de x[0..p(n')] y
p(.) es creciente, la composicion sigue dependiendo solo de x[0..p(n)].
Induccion trivial sobre las capas.

CASO DEGENERADO (y aqui hay una trampa real). Si M == N, la fila n = 0 del
cross-attend queda COMPLETAMENTE enmascarada (ve los primeros 0 + M - N = 0
elementos). Un softmax sobre una fila entera de -inf da NaN. Este fichero lo
detecta, deja pasar una clave ficticia para que el kernel no produzca NaN y
despues ANULA la salida de esa fila con masked_fill, de modo que el resultado
es exactamente "el latente 0 no atiende a nada" y se queda con su residual, que
es su propio embedding: justo lo que debe hacer la posicion 0 de cualquier
decoder causal. Verificado abajo.

RoPE PARCIAL
-------------
El paper rota solo una FRACCION de las dimensiones de cada cabeza (25% en sus
modelos de musica; Apendice E). La intuicion es dejar canales "sin posicion"
para contenido puro. Aqui se importan Rotary/apply_rope de
src/models/modern_transformer.py (ya auditados) y se rota unicamente las
primeras rot_dim dimensiones de cada cabeza, con

    rot_dim = par mas cercano a d_head * cfg.rope_fraction   (64 * 0.25 = 16)

Detalle que importa: la query del latente n se rota con la posicion ABSOLUTA
p(n) = M - N + n y la clave m con la posicion ABSOLUTA m, de forma que el
producto q.k depende de p(n) - m, que es la distancia real en la secuencia de
entrada. Rotar las queries con su indice latente n en vez de con p(n) seria el
bug clasico: el latente 0 creeria estar en la posicion 0 de la pieza cuando en
realidad esta en la posicion M - N.

CROSS-ATTEND DROPOUT
---------------------
El paper usa cross-attend dropout 0.7 en TODOS sus modelos de musica
(Apendice E.2). Se implementa como dropout sobre las PROBABILIDADES de atencion
del cross-attend (dropout_p de SDPA), que es lo que el nombre dice y lo unico
compatible con un valor tan alto como 0.7. Solo actua en training; en eval es
0, por eso no interfiere con los tests de causalidad (y aunque actuase, el
dropout solo puede ANULAR conexiones ya permitidas: nunca crea una fuga).

VALOR POR DEFECTO QUE PONGO: 0.7, el del paper, via cfg.cross_attend_dropout
(getattr, NO es campo de Config).

Y AHORA LA PARTE HONESTA, porque 0.7 NO es inocuo en nuestro regimen. En el
paper M=8192 y N=1024: cada latente atiende sobre miles de claves y tirar el
70% deja aun cientos, asi que el resumen atendido apenas cambia. Con M = N
(nuestro caso) el latente n solo tiene n claves visibles, y para n pequeno el
dropout se las lleva TODAS con probabilidad 0.7^n. Medido en este fichero
(_test_cross_dropout, M = N = 1024): la fraccion esperada de latentes que se
quedan sin NINGUNA clave es 0.23%, pero concentrada entera en el arranque
(n=1: 70%, n=2: 49%, n=3: 34%, n=5: 17%). Es un regularizador cualitativamente
distinto al del paper. Ademas nuestro corpus (714 h, 10604 piezas) es mas
grande que MAESTRO, con menos riesgo de sobreajuste.
RECOMENDACION PRACTICA: para entrenar aqui, cfg.cross_attend_dropout = 0.1.
El defecto se queda en 0.7 porque este fichero es una reproduccion de Perceiver
AR y desviarse del paper debe ser una decision explicita del experimento, no un
valor escondido en el codigo.

EL PROBLEMA DE CONTRATO  <<<<<<<<<<<<  LEE ESTO ANTES DE ENTRENAR
------------------------------------------------------------------
El laboratorio exige forward(x: Long[B,L]) -> Float[B,L,V]: una prediccion por
posicion. Perceiver AR solo produce N. El paper lo asume explicitamente
(Seccion 5.1.2): "Changing the width of the self-attention stack ... alters the
number of training outputs for which loss can be computed per sequence".

Resolucion adoptada:
  * n_latents = getattr(cfg, "n_latents", 512). NO se anade ningun campo a
    src/config.py.
  * Si L <= n_latents: N = L, el modelo es un decoder causal completo (todas
    las posiciones son latentes) y las L salidas son validas enteras.
  * Si L > n_latents: las primeras L - N posiciones NO tienen prediccion. El
    forward devuelve igualmente [B, L, V] rellenando esas filas con CEROS
    (valor neutro: distribucion uniforme, finita, sin gradiente).

    *** QUIEN ENTRENE SIN loss_mask ESTARA APRENDIENDO DE LOGITS INVALIDOS. ***

    Un logit cero da cross-entropy ln(155) = 5.043 nats = 7.276 bits/paso en
    esas posiciones, CONSTANTE y sin gradiente. Con el defecto del laboratorio
    (seq_len=1024, n_latents=512) eso es la MITAD de las posiciones, y la
    bits/paso reportada saldria ~ (1.7 + 7.276)/2 = 4.5: PEOR que el modelo
    trivial (4.09), sin que nada falle ruidosamente. Por eso:

        mask = model.loss_mask(L, device=logits.device)   # Bool[L], True=cuenta
        loss = F.cross_entropy(logits[:, mask].reshape(-1, V),
                               target[:, mask].reshape(-1), ignore_index=PAD)

    Metodos de apoyo: loss_mask(L, device=None) -> Bool[L] y
    n_valid_outputs(L) -> int.

HONESTIDAD SOBRE SI ESTO APORTA ALGO EN NUESTROS DATOS
-------------------------------------------------------
La ventaja de Perceiver AR (coste del stack independiente de M) SOLO aparece
cuando M >> N. Con seq_len=1024 y n_latents=1024 este modelo es EXACTAMENTE un
decoder causal de n_layers capas cuya primera capa es un cross-attend en vez de
una self-attention: mismo coste, mismos parametros, cero ventaja, y ademas ese
cross-attend inicial es estrictamente MENOS expresivo que una self-attention
(una sola mezcla, con mascara estricta, sin ver el propio token salvo por el
residual). Nuestras piezas tienen ~3000 tokens de mediana, asi que el maximo M
util es ~3000: un factor 3 sobre el contexto actual, no un factor 8 ni 64 como
en el paper (MAESTRO con M=8192 sobre 32768 tokens). Con M=3072 y N=1024 el
ahorro real es el del stack a cambio de 3x menos salidas por ventana, es decir
3x menos senal de entrenamiento por paso de optimizador. En bits/paso val no
hay razon para esperar que bata a ModernTransformer (1.67). Ver el veredicto
medido al pie del fichero.

ARQUITECTURA CONCRETA Y PRESUPUESTO DE PARAMETROS
---------------------------------------------------
Para ser comparable con los demas modelos del laboratorio (~25 M) se reparte
cfg.n_layers como 1 cross-attend + (n_layers - 1) capas de self-attention. Un
bloque cross-attend tiene exactamente los mismos parametros que uno de
self-attention (q 1 matriz + kv 2 matrices + proj 1 = 4 D^2, igual que
qkv 3 + proj 1), asi que con la config por defecto salen 25 778 688 = 25.78 M.
No es EXACTAMENTE el mismo numero que ModernTransformer (25 779 200): hay 512
parametros de diferencia, y no son ruido sino dos decisiones concretas que se
cancelan casi del todo. ModernTransformer usa QK-norm (q_norm + k_norm, 2 x 512
= 1024 escalares) y este fichero no; este fichero tiene una norma de mas en el
cross-attend (norm_kv sobre X_KV, 512 escalares) porque query y clave vienen de
tensores distintos. 1024 - 512 = 512. A 25.78 M eso es el 0.002%: comparables a
todos los efectos, pero "identicos" seria falso.

Se reutilizan RMSNorm, SwiGLU y RoPE de modern_transformer.py A PROPOSITO: asi
la UNICA variable que cambia frente a ModernTransformer es la estructura
Perceiver AR (cross-attend + cuello de botella de latentes). El paper usa
LayerNorm + MLP GELU, pero comparar contra el modelo mas fuerte del laboratorio
con el resto de piezas identicas es mas informativo que reproducir el bloque de
2022.

Knobs extra, todos por getattr y ninguno campo de Config:
    cfg.n_latents            : int   (512)     N
    cfg.rope_fraction        : float (0.25)    fraccion de d_head que se rota
    cfg.rope_theta           : float (10000.)  base de RoPE
    cfg.cross_attend_dropout : float (0.7)     dropout del cross-attend (paper)
    cfg.attn_dropout         : float (0.0)     dropout de la self-attention

RESULTADOS MEDIDOS (reales, ejecutando este fichero; ninguno extrapolado)
--------------------------------------------------------------------------
Parametros con la config por defecto del laboratorio (d_model=512, n_layers=8
= 1 cross + 7 self, n_heads=8, d_ff=2048 -> d_hidden SwiGLU 1408, tie_weights):

    25 778 688  =  25.78 M   (contrastado contra una formula independiente en
                              _test_param_count; identico a ModernTransformer)
    rot_dim = 16 de 64 por cabeza (25%, el del paper)

Causalidad estricta (todas las comprobaciones dan 0.0 EXACTO, no "pequeno"):
    CPU fp32, perturbando solo x[:,t] para TODO t, L=48 con n_latents=512
    (regimen L<=N) y con n_latents=16 (regimen L>N), y L=12 con n_latents=1:
        max|delta logits en posiciones < t| = 0.0e+00
    GPU bf16 (kernel CUDA de SDPA con la mascara booleana), L=128 con
    n_latents=128 y 48:  max|delta pasado| = 0.0e+00
    Test generico del laboratorio (tests/test_causality.py) OK en los dos
    regimenes.

Velocidad (fwd+bwd+clip+step, mejor de varias repeticiones). LA GPU ESTA
COMPARTIDA con otro entrenamiento (7.8 GiB y 100% de uso durante la medida);
las cifras absolutas son por tanto una COTA INFERIOR.

AVISO DE METODO, aprendido rompiendolo: medir primero las N repeticiones de una
arquitectura y despues las de la otra NO sirve en una GPU compartida. La carga
externa cambia entre las dos mitades y el cociente sale lo que quiera: con esa
version el fichero llego a imprimir "perceiver_ar rinde 240% del decoder
causal" en una pasada (perceiver_ar 41.8 it/s justo cuando el otro
entrenamiento afloja, modern 17.4 it/s un minuto despues) y 95% en otra, con el
mismo codigo y los mismos pesos. Por eso _compare_with_modern INTERCALA rondas
A/B/A/B... y reporta la MEDIANA, no el mejor: asi las dos arquitecturas ven la
misma carga externa. Con ese metodo el cociente es estable (dispersion < 2
puntos entre rondas y ~5 entre pasadas). Las cifras de abajo son intercaladas y
se dan como RANGO sobre 4 pasadas, no como un unico numero bonito.

    GPU bf16, batch 2:
        M=1024, N=1024:  16.4 it/s,  pico 0.87 GiB,  1024 salidas validas
        M=3072, N=1024:  15.0 it/s,  pico 0.91 GiB,  1024 salidas validas
    CPU fp32, batch 2 (esta maquina esta ademas alimentando el otro
    entrenamiento, asi que las cifras absolutas oscilan bastante entre pasadas;
    lo que SI se repite es el cociente entre configuraciones):
        M=1024, N=1024:  0.59-1.00 it/s
        M=1024, N= 512:  1.08-2.03 it/s   (~2.0x mas rapido que N=1024: el
                                           stack procesa la mitad de posiciones)
        M=3072, N=1024:  0.46-0.71 it/s

Y LA COMPARACION QUE DECIDE (_compare_with_modern, batch 2, bf16, GPU, 6 rondas
INTERCALADAS, mediana). No se compara it/s, que favorece trivialmente a quien
produce menos salidas, sino SALIDAS VALIDAS POR SEGUNDO: predicciones con
gradiente real por unidad de tiempo, que es la moneda del entrenamiento.

    M=1024   perceiver_ar (N=1024)  15.7-16.6 it/s  0.87 GiB   32 000 validas/s
             modern (decoder causal) 17.2-17.5 it/s  0.87 GiB   35 500 validas/s
             -> perceiver_ar rinde 91-96% del decoder causal (4 pasadas)

    M=3072   perceiver_ar (N=1024)  14.4-15.0 it/s  0.91 GiB   30 000 validas/s
             modern (decoder causal)  6.4- 6.6 it/s  1.66 GiB   39 800 validas/s
             -> perceiver_ar rinde 75-77% del decoder causal (4 pasadas)

Es decir: a M=3072 Perceiver AR es 2.3x mas rapido POR PASO y usa un 45% menos
de memoria (0.91 GiB frente a 1.66), y aun asi produce un ~24% MENOS de senal de
entrenamiento por segundo, porque produce 3x menos salidas.
A M=1024 (M = N) ni siquiera ahorra memoria: los dos picos son 0.87 GiB.
Todos los picos estan medidos con UN SOLO modelo residente en la tarjeta.

POR QUE, EN UNA CUENTA
-----------------------
A nuestro ancho (d_model=512, d_hidden=1408) el trabajo denso por posicion y
capa es 4*D^2 + 3*D*hid = 3.21 M MACs, mientras que el trabajo de atencion
causal por posicion y capa es ~M*D MACs. Se igualan en

        M = (4*D^2 + 3*D*hid) / D  =  4*D + 3*hid  =  6 272 tokens.

Con M=1024 la atencion es el 14% del coste; con M=3072, el 33%. Perceiver AR
solo ataca ese sumando (y solo por el factor M/N). El sumando DENSO, que es el
dominante, escala con el numero de posiciones procesadas, y ahi Perceiver AR
gana exactamente el mismo factor M/N por el que pierde salidas: se cancela.
La cuenta de FLOPs predice paridad (~1.0x) y la medida da 0.75x; la diferencia
es de kernel: la mascara booleana explicita del cross-attend impide usar la
ruta is_causal=True de SDPA, que se salta los bloques enteramente futuros, asi
que se visitan tambien los ~N^2/2 pares enmascarados (17% del cross-attend a
M=3072).

VEREDICTO PARA NUESTRO LABORATORIO (sin adornos)
--------------------------------------------------
En ESTOS datos Perceiver AR no puede batir a un decoder causal normal, y la
razon no es de implementacion sino de escala:
  1. Su mecanismo ahorra en el termino de atencion, que aqui es el 14-33% del
     coste. El termino que domina (las matrices densas) no lo toca. El punto de
     equilibrio esta en M ~ 6 300 tokens y el corpus tiene piezas de ~3 000
     tokens de mediana: NO llegamos ni a una pieza entera.
  2. Con seq_len=1024 y n_latents=1024 (M = N) el modelo es literalmente un
     decoder causal de 8 capas cuya primera capa es un cross-attend en vez de
     una self-attention, y ese cross-attend es estrictamente MENOS expresivo
     (una sola mezcla, mascara estricta, el token propio solo llega por el
     residual). Medido (intercalado): 91-96% del rendimiento del decoder. Cero
     ventaja, algo de desventaja.
  3. Con M > N el coste real es la SENAL: 3x menos posiciones con gradiente por
     ventana. Medido (intercalado): 75-77%.
  4. Ni siquiera el argumento de memoria muerde aqui: modern a M=3072 pico 1.66
     GiB sobre una tarjeta de 12.9 GiB. No hay presion de memoria que resolver.
No espero que baje de 1.83 (Music Transformer) ni se acerque a 1.67
(ModernTransformer) en bits/paso val; lo razonable es esperar algo
LIGERAMENTE PEOR que ModernTransformer a igualdad de pasos, y bastante peor a
igualdad de tiempo si se entrena con M > N.
Lo unico que esta arquitectura si da, y que el decoder no da barato, es
contexto de PIEZA COMPLETA con coste de stack constante. Si en algun momento el
laboratorio quiere condicionar en 8 000-16 000 tokens (varias piezas, o audio a
mayor resolucion), este fichero ya esta listo y ahi la cuenta se invierte. Con
el corpus actual, no.
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__" and __package__ in (None, ""):
    # Permite:  python src/models/perceiver_ar.py
    import sys as _sys_boot
    from pathlib import Path as _Path_boot

    _root_boot = _Path_boot(__file__).resolve().parents[2]
    for _p_boot in (str(_root_boot), str(_root_boot / "src")):
        if _p_boot not in _sys_boot.path:
            _sys_boot.path.insert(0, _p_boot)

try:
    from .base import TokenARModel
except ImportError:                                      # pragma: no cover
    try:
        from models.base import TokenARModel
    except ImportError:
        from src.models.base import TokenARModel

# RoPE / RMSNorm / SwiGLU ya auditados: se IMPORTAN, no se reescriben.
try:
    from .modern_transformer import Rotary, apply_rope, RMSNorm, SwiGLU, swiglu_hidden
except ImportError:                                      # pragma: no cover
    try:
        from models.modern_transformer import (Rotary, apply_rope, RMSNorm,
                                               SwiGLU, swiglu_hidden)
    except ImportError:
        from src.models.modern_transformer import (Rotary, apply_rope, RMSNorm,
                                                   SwiGLU, swiglu_hidden)

try:
    from data.tokenizer import VOCAB_SIZE
except ImportError:                                      # pragma: no cover
    try:
        from src.data.tokenizer import VOCAB_SIZE
    except ImportError:
        VOCAB_SIZE = 155


# ---------------------------------------------------------------------------
# RoPE parcial
# ---------------------------------------------------------------------------
def rope_rot_dim(d_head: int, fraction: float) -> int:
    """Numero PAR de dimensiones por cabeza que se rotan.

    El paper rota el 25% en sus modelos de musica. Tiene que ser par porque
    rotate_half empareja la componente j con la j + rot_dim/2; se fuerza un
    minimo de 2 (una sola rotacion 2D) y un maximo de d_head (RoPE completo).
    """
    r = int(round(int(d_head) * float(fraction)))
    r = (r // 2) * 2
    return max(2, min(int(d_head), r))


def apply_rope_partial(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       rot_dim: int) -> torch.Tensor:
    """Rota SOLO las primeras rot_dim dimensiones de x [..., L, d_head].

    Las dimensiones restantes pasan intactas: son canales de contenido puro,
    sin informacion posicional. cos/sin deben venir de un Rotary(rot_dim).
    """
    if rot_dim >= x.shape[-1]:
        return apply_rope(x, cos, sin)
    return torch.cat([apply_rope(x[..., :rot_dim], cos, sin), x[..., rot_dim:]],
                     dim=-1)


# ---------------------------------------------------------------------------
# Cross-attend (el corazon de Perceiver AR)
# ---------------------------------------------------------------------------
def perceiver_cross_mask(N: int, M: int, device=None) -> torch.Tensor:
    """Mascara del cross-attend del paper. True = VISIBLE (convenio de SDPA).

    El paper enmascara (n, m) cuando  m > n + M - N - 1, luego lo VISIBLE es
    m <= n + M - N - 1: los primeros n + M - N elementos de la entrada.
    Devuelve Bool[N, M]; se difunde sobre [B, H, N, M] dentro de SDPA.
    """
    n = torch.arange(int(N), device=device).view(int(N), 1)
    m = torch.arange(int(M), device=device).view(1, int(M))
    return m <= (n + (int(M) - int(N)) - 1)


class PerceiverCrossAttention(nn.Module):
    """Un unico cross-attend de M entradas a N latentes, causal segun el paper.

    Las queries llegan ya recortadas por el modelo (las N ultimas posiciones de
    la entrada); este modulo no crea ningun latente aprendido, a proposito.

    Manejo de la fila vacia: con M == N el latente 0 no tiene ninguna clave
    visible. En la mascara cacheada se habilita la clave 0 para que el kernel no
    devuelva NaN y despues se ANULA la salida de esa fila, con lo que el
    resultado es exactamente "no atiende a nada". Causalmente seria incluso
    legal dejar la clave 0 (el latente 0 ocupa la posicion 0 cuando M == N),
    pero se anula para que la mascara EFECTIVA sea literalmente la del paper.
    """

    def __init__(self, d_model: int, n_heads: int, rot_dim: int,
                 dropout: float, cross_dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser multiplo de n_heads"
        self.n_heads = int(n_heads)
        self.d_head = d_model // self.n_heads
        self.rot_dim = int(rot_dim)
        self.p_cross = float(cross_dropout)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj._is_residual_out = True
        self.resid_drop = nn.Dropout(dropout)
        # Cache de la mascara: [N,M] bool son 1 MiB con 1024x1024 y no cambia
        # entre pasos con la misma forma. No es un buffer: no va al state_dict.
        self._mask_key = None
        self._mask = None          # mascara YA parcheada, lista para SDPA
        self._empty = None         # Bool[N] filas sin ninguna clave visible

    def _mask_for(self, N: int, M: int, device):
        """Devuelve (mascara lista para SDPA, filas vacias, hay_alguna_vacia).

        Que una fila este vacia NO depende de los datos: la fila n ve n + M - N
        claves, luego esta vacia si y solo si n <= N - M, y como N <= M eso solo
        ocurre con M == N y n == 0. Por eso el parcheo de la clave ficticia se
        hace UNA vez al construir el cache y 'hay_alguna_vacia' se decide en
        Python con (M == N). La version anterior lo deducia del tensor con
        ~keep.any(-1) y un .item() por forward: eso obliga a una sincronizacion
        host-device en CADA paso y rompe el grafo de torch.compile
        (cfg.compile_model existe y train.py lo usa), ademas de clonar la
        mascara de 1 MiB en cada forward justo en el regimen por defecto M == N.
        """
        key = (int(N), int(M), str(device))
        if self._mask_key != key:
            keep = perceiver_cross_mask(N, M, device)
            empty = torch.zeros(int(N), dtype=torch.bool, device=device)
            if int(M) == int(N):
                # fila 0 sin claves: se habilita la clave 0 SOLO para que el
                # kernel no haga softmax sobre una fila entera de -inf (NaN);
                # su salida se anula despues, asi que la mascara EFECTIVA sigue
                # siendo literalmente la del paper.
                keep[0, 0] = True
                empty[0] = True
            self._mask, self._empty = keep, empty
            self._mask_key = key
        return self._mask, self._empty, int(M) == int(N)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor,
                cos_q, sin_q, cos_k, sin_k) -> torch.Tensor:
        B, N, D = q_in.shape
        M = kv_in.shape[1]
        H, dh = self.n_heads, self.d_head

        q = self.q_proj(q_in).view(B, N, H, dh).transpose(1, 2)
        k, v = self.kv_proj(kv_in).split(D, dim=-1)
        k = k.view(B, M, H, dh).transpose(1, 2)
        v = v.view(B, M, H, dh).transpose(1, 2)
        # Queries rotadas con su posicion ABSOLUTA p(n) = M-N+n (cos_q ya viene
        # cortado a ese rango por el modelo), claves con su posicion m.
        q = apply_rope_partial(q, cos_q, sin_q, self.rot_dim)
        k = apply_rope_partial(k, cos_k, sin_k, self.rot_dim)

        keep, empty, has_empty = self._mask_for(N, M, q.device)
        p = self.p_cross if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, dropout_p=p)
        if has_empty:
            y = y.masked_fill(empty.view(1, 1, N, 1), 0.0)
        y = y.transpose(1, 2).reshape(B, N, D)
        return self.resid_drop(self.proj(y))


class CrossAttendBlock(nn.Module):
    """Pre-norma: z = X_Q + Cross(RMS(X_Q), RMS(X_KV)); z = z + SwiGLU(RMS(z)).

    El residual X_Q es la pieza que completa la causalidad (ver cabecera): sin
    el, el latente n nunca veria el token de su propia posicion.
    """

    def __init__(self, d_model: int, n_heads: int, d_hidden: int, rot_dim: int,
                 dropout: float, cross_dropout: float):
        super().__init__()
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.attn = PerceiverCrossAttention(d_model, n_heads, rot_dim, dropout,
                                            cross_dropout)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_hidden, dropout)

    def forward(self, x_q, x_kv, cos_q, sin_q, cos_k, sin_k):
        z = x_q + self.attn(self.norm_q(x_q), self.norm_kv(x_kv),
                            cos_q, sin_q, cos_k, sin_k)
        return z + self.ff(self.norm2(z))


# ---------------------------------------------------------------------------
# Stack de latentes: self-attention causal estandar
# ---------------------------------------------------------------------------
class LatentSelfAttention(nn.Module):
    """Self-attention causal (triangular) sobre los N latentes, RoPE parcial."""

    def __init__(self, d_model: int, n_heads: int, rot_dim: int,
                 dropout: float, attn_dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = int(n_heads)
        self.d_head = d_model // self.n_heads
        self.rot_dim = int(rot_dim)
        self.p_drop = float(attn_dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj._is_residual_out = True
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos, sin) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        q = apply_rope_partial(q, cos, sin, self.rot_dim)
        k = apply_rope_partial(k, cos, sin, self.rot_dim)
        p = self.p_drop if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(B, N, D)))


class LatentBlock(nn.Module):
    """x = x + Attn(RMS(x)); x = x + SwiGLU(RMS(x))."""

    def __init__(self, d_model: int, n_heads: int, d_hidden: int, rot_dim: int,
                 dropout: float, attn_dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = LatentSelfAttention(d_model, n_heads, rot_dim, dropout,
                                        attn_dropout)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_hidden, dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ff(self.norm2(x))


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
_WARNED_PARTIAL = False


def _warn_partial_outputs(M: int, N: int) -> None:
    """Avisa UNA vez por proceso de que M - N logits son relleno, no predicciones.

    Por que existe este aviso y no basta con documentarlo: src/train.py y
    src/evaluate.py calculan F.cross_entropy sobre TODAS las posiciones de la
    ventana; no conocen loss_mask ni pueden conocerla (el contrato del
    laboratorio no la tiene). Con los valores por defecto (seq_len=1024,
    n_latents=512) eso mezcla 512 posiciones de ceros a ln(155) nats en la
    media: el entrenamiento no falla, no salta ninguna excepcion, y la
    bits/paso reportada sale ~4.5 en vez de ~1.7. Un fallo silencioso que
    cuesta horas de GPU antes de que alguien sospeche. El aviso lo hace
    audible una sola vez y no cambia ningun numero.
    Si solo se usa el ULTIMO logit (muestreo con ventana deslizante) el aviso
    es inofensivo: esa posicion siempre es valida.
    """
    global _WARNED_PARTIAL
    if _WARNED_PARTIAL:
        return
    _WARNED_PARTIAL = True
    warnings.warn(
        f"perceiver_ar: L={M} > n_latents={N}; las primeras {M - N} posiciones "
        f"de los logits son CEROS de relleno, no predicciones. Usa "
        f"model.loss_mask(L) para la perdida y la evaluacion, o pon "
        f"cfg.n_latents >= seq_len. Sin la mascara, la bits/paso queda "
        f"contaminada con log2(155)=7.276 bits constantes en esas posiciones.",
        RuntimeWarning, stacklevel=3)


class PerceiverAR(TokenARModel):
    """Perceiver AR fiel al paper, adaptado al contrato TokenARModel.

    forward(x: Long[B,L]) -> Float[B,L,V].
    Si L > n_latents las primeras L - n_latents filas son CEROS y NO son
    predicciones: usa loss_mask(L). Lee la cabecera del fichero.
    """

    name = "perceiver_ar"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_model = int(cfg.d_model)
        n_layers = int(cfg.n_layers)
        n_heads = int(cfg.n_heads)
        d_ff = int(getattr(cfg, "d_ff", 4 * d_model))
        dropout = float(getattr(cfg, "dropout", 0.1))

        # --- knobs fuera de Config (getattr, sin tocar src/config.py) ---
        self.n_latents = int(getattr(cfg, "n_latents", 512))
        self.rope_fraction = float(getattr(cfg, "rope_fraction", 0.25))
        self.rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
        self.cross_attend_dropout = float(getattr(cfg, "cross_attend_dropout", 0.7))
        attn_dropout = float(getattr(cfg, "attn_dropout", 0.0))

        assert self.n_latents >= 1, "n_latents debe ser >= 1"
        assert n_layers >= 2, "hacen falta al menos 1 cross-attend + 1 self-attention"

        self.tie_weights = bool(getattr(cfg, "tie_weights", True))
        self.max_seq_len = int(getattr(cfg, "seq_len", 1024))
        self.vocab_size = int(VOCAB_SIZE)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_hidden = swiglu_hidden(d_ff)
        self.rot_dim = rope_rot_dim(self.d_head, self.rope_fraction)
        # 1 cross-attend + (n_layers - 1) self-attention: mismo presupuesto de
        # parametros que un decoder de n_layers capas (ver cabecera).
        self.n_self_layers = n_layers - 1
        self.n_layers = n_layers

        self.tok_emb = nn.Embedding(self.vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        # Un solo cache de cos/sin, dimensionado a rot_dim (RoPE parcial).
        self.rope = Rotary(self.rot_dim, self.rope_theta, init_len=self.max_seq_len)

        self.cross = CrossAttendBlock(d_model, n_heads, self.d_hidden, self.rot_dim,
                                      dropout, self.cross_attend_dropout)
        self.blocks = nn.ModuleList([
            LatentBlock(d_model, n_heads, self.d_hidden, self.rot_dim, dropout,
                        attn_dropout)
            for _ in range(self.n_self_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.head = None if self.tie_weights else nn.Linear(d_model, self.vocab_size,
                                                            bias=False)

        self.apply(self._init_weights)
        # 1/sqrt(2*n_layers) en las proyecciones que ESCRIBEN en el residual:
        # con 2 escrituras por bloque la varianza del stream crece ~linealmente
        # con la profundidad si no se corrige.
        std_res = 0.02 / math.sqrt(2 * max(1, n_layers))
        for m in self.modules():
            if getattr(m, "_is_residual_out", False):
                nn.init.normal_(m.weight, mean=0.0, std=std_res)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # RMSNorm.weight se queda en 1.

    # -- interfaz del contrato roto: quien entrena DEBE usar esto -----------
    def n_valid_outputs(self, L: int) -> int:
        """Cuantas de las L posiciones tienen prediccion real: min(L, n_latents)."""
        return int(min(int(L), self.n_latents))

    def loss_mask(self, L: int, device=None) -> torch.Tensor:
        """Bool[L]: True donde forward() devuelve un logit VALIDO.

        Regimen L <= n_latents -> todo True (decoder causal completo).
        Regimen L >  n_latents -> True solo en las ultimas n_latents posiciones;
        las primeras L - n_latents son ceros de relleno, NO predicciones.
        Ver la advertencia grande de la cabecera: entrenar sin esta mascara
        mezcla 7.276 bits/paso constantes en la media y arruina la metrica sin
        lanzar ningun error.
        """
        L = int(L)
        m = torch.zeros(L, dtype=torch.bool, device=device)
        m[L - self.n_valid_outputs(L):] = True
        return m

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """x: Long[B,M] -> logits Float[B,M,V] del token siguiente.

        Las N = min(M, n_latents) ULTIMAS filas son predicciones reales; si
        M > N las primeras M - N son ceros (ver loss_mask).
        pos_offset desplaza las posiciones de RoPE; existe para muestreo con
        ventana deslizante y no cambia el contrato.
        """
        assert x.dim() == 2, f"se esperaba [B,L], llego {tuple(x.shape)}"
        B, M = x.shape
        assert M >= 1, "secuencia vacia"
        N = self.n_valid_outputs(M)
        if M > N:
            _warn_partial_outputs(M, N)

        h_kv = self.drop(self.tok_emb(x))                   # X_KV: [B,M,D]
        # X_Q = X_KV[-N:] : las queries NO son parametros aprendidos (paper, Ap. C)
        h_q = h_kv[:, M - N:]                               # X_Q:  [B,N,D]

        # RoPE: posiciones absolutas 0..M-1 para las claves; las queries y los
        # latentes viven en las posiciones M-N..M-1, que es la MISMA rodaja.
        cos_k, sin_k = self.rope(M, int(pos_offset), device=h_kv.device,
                                 dtype=h_kv.dtype)
        cos_q, sin_q = cos_k[M - N:], sin_k[M - N:]

        z = self.cross(h_q, h_kv, cos_q, sin_q, cos_k, sin_k)   # [B,N,D]
        for blk in self.blocks:
            z = blk(z, cos_q, sin_q)
        z = self.norm_f(z)
        logits = F.linear(z, self.tok_emb.weight) if self.tie_weights else self.head(z)

        if N == M:
            return logits
        # Relleno neutro para cumplir la FORMA del contrato. No son predicciones.
        pad = logits.new_zeros(B, M - N, logits.shape[-1])
        return torch.cat([pad, logits], dim=1)

    def supports_state(self) -> bool:
        return False

    def extra_repr(self) -> str:
        return (f"n_latents={self.n_latents}, cross_attend_dropout="
                f"{self.cross_attend_dropout}, rot_dim={self.rot_dim}/{self.d_head}")


# ---------------------------------------------------------------------------
# Verificacion (ejecutar:  python src/models/perceiver_ar.py)
# ---------------------------------------------------------------------------
def _mk(n_latents=512, **kw):
    from config import Config
    base = dict(d_model=128, n_layers=3, n_heads=4, d_ff=512, dropout=0.0)
    base.update(kw)
    cfg = Config(**base)
    cfg.n_latents = int(n_latents)
    cfg.cross_attend_dropout = 0.0      # los tests corren en eval, pero explicito
    return PerceiverAR(cfg).eval()


def _test_cross_mask(verbose: bool = True) -> bool:
    """(b) La mascara del cross-attend es LITERALMENTE la del paper.

    Se imprime la matriz para M=8, N=4 y se comprueba fila a fila que el
    latente n ve exactamente los primeros n + M - N elementos.
    """
    M, N = 8, 4
    keep = perceiver_cross_mask(N, M)
    ok = True
    if verbose:
        print(f"  mascara del cross-attend para M={M}, N={N}  "
              f"(fila n = latente, columna m = entrada; 1 = VISIBLE)")
        print("        m: " + " ".join(f"{m}" for m in range(M)))
        for n in range(N):
            fila = " ".join("1" if bool(keep[n, m]) else "." for m in range(M))
            print(f"   n={n} p(n)={M-N+n}: {fila}   ve {int(keep[n].sum())} "
                  f"= n+M-N = {n + M - N}")
    for n in range(N):
        esperado = torch.zeros(M, dtype=torch.bool)
        esperado[:n + M - N] = True
        good = bool(torch.equal(keep[n], esperado))
        ok = ok and good
    # la formula del paper, aplicada tal cual, sobre una malla independiente
    ref = torch.tensor([[not (m > n + M - N - 1) for m in range(M)] for n in range(N)])
    same = bool(torch.equal(keep, ref))
    ok = ok and same
    # y el caso M == N (regimen decoder), donde la fila 0 queda vacia
    k2 = perceiver_cross_mask(4, 4)
    vacia = bool((~k2.any(-1))[0].item()) and int(k2.sum().item()) == 6
    ok = ok and vacia
    if verbose:
        print(f"   coincide con 'not (m > n + M - N - 1)' evaluado a mano: {same}")
        print(f"   M=N=4: triangular estricta, la fila n=0 queda VACIA "
              f"(total visibles {int(k2.sum())} = 4*3/2): {vacia}")
        print(f"  -> mascara del paper: {'OK' if ok else 'FALLO'}")
    return ok


def _test_no_learned_latents(verbose: bool = True) -> bool:
    """(1 del paper) No existe ningun nn.Parameter de queries latentes.

    Control positivo: cambiar el embedding de la posicion M-N+n cambia el
    latente n, luego las queries SALEN de la entrada y no de un parametro.
    """
    m = _mk(n_latents=8, d_model=64, n_layers=2, n_heads=4, d_ff=256)
    sospechosos = [n for n, p in m.named_parameters()
                   if p.dim() >= 2 and p.shape[0] == 8 and "emb" not in n]
    sin_latentes = len(sospechosos) == 0
    x = torch.randint(3, m.vocab_size, (1, 12))
    with torch.no_grad():
        a = m(x)
        x2 = x.clone()
        x2[0, 11] = 3 + (int(x[0, 11]) - 3 + 51) % (m.vocab_size - 3)
        b = m(x2)
    ultima_cambia = (a[0, 11] - b[0, 11]).abs().max().item() > 0
    ok = sin_latentes and ultima_cambia
    if verbose:
        print(f"  parametros con una dimension == N=8 que pudieran ser latentes "
              f"aprendidos: {sospechosos} -> {'ninguno' if sin_latentes else 'HAY'}")
        print(f"  la query del ultimo latente sale de la ENTRADA (cambiar x[-1] "
              f"cambia su logit): {ultima_cambia}")
        print(f"  -> queries = X_KV[-N:], no aprendidas: {'OK' if ok else 'FALLO'}")
    return ok


def _test_strict_causality(model, L: int, verbose: bool = True) -> bool:
    """(a) Comprobacion BIT A BIT, posicion a posicion.

    Para cada t se perturba UNICAMENTE x[:, t] y se exige diferencia
    EXACTAMENTE 0.0 en todos los logits de posiciones < t. Ademas, toda
    posicion VALIDA (>= L - N) debe reaccionar a su propio token, y toda
    posicion invalida debe ser exactamente cero siempre.
    """
    m = model.eval()
    N = m.n_valid_outputs(L)
    start = L - N                              # primera posicion valida
    torch.manual_seed(0)
    x = torch.randint(3, m.vocab_size, (2, L))
    with torch.no_grad():
        base = m(x)
    pad_cero = True if start == 0 else bool((base[:, :start] == 0).all().item())
    worst_past, min_self = 0.0, float("inf")
    for t in range(L):
        x2 = x.clone()
        x2[:, t] = 3 + (x[:, t] - 3 + 37) % (m.vocab_size - 3)
        with torch.no_grad():
            pert = m(x2)
        if t > 0:
            worst_past = max(worst_past, (base[:, :t] - pert[:, :t]).abs().max().item())
        if t >= start:
            min_self = min(min_self, (base[:, t] - pert[:, t]).abs().max().item())
    ok = worst_past == 0.0 and min_self > 0.0 and pad_cero
    if verbose:
        reg = "L <= N (decoder completo)" if start == 0 else "L > N (con relleno)"
        print(f"  L={L}, n_latents={m.n_latents} -> N={N}, {reg}")
        print(f"    perturbando solo x[:,t] para los {L} valores de t: "
              f"max|delta logits en posiciones < t| = {worst_past:.1e} "
              f"(exige == 0.0 exacto)")
        print(f"    min sobre las posiciones VALIDAS de max|delta en la propia "
              f"posicion| = {min_self:.3e} (debe ser > 0)")
        print(f"    las {start} posiciones de relleno son exactamente cero: {pad_cero}")
        print(f"    -> {'OK' if ok else 'FUGA CAUSAL'}")
    return ok


def _test_generic_causality(model, L: int, verbose: bool = True) -> bool:
    """(a) El test generico del laboratorio, tests/test_causality.py."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tests.test_causality import check_token_causality, check_shapes_token
    a = check_token_causality(model, L=L, V=model.vocab_size)
    b = check_shapes_token(model, L=L, V=model.vocab_size)
    return bool(a and b)


def _test_shapes(verbose: bool = True) -> bool:
    """(c) Formas para L = 1, 17, 70, 512, 1024, 2048 (ambos regimenes)."""
    m = _mk(n_latents=512, d_model=128, n_layers=3, n_heads=4, d_ff=512)
    ok = True
    for L in (1, 17, 70, 512, 1024, 2048):
        x = torch.randint(3, m.vocab_size, (2, L))
        with torch.no_grad():
            y = m(x)
        N = m.n_valid_outputs(L)
        good = (tuple(y.shape) == (2, L, m.vocab_size)
                and bool(torch.isfinite(y).all().item()))
        ok = ok and good
        if verbose:
            print(f"  L={L:5d}: salida {tuple(y.shape)} finita="
                  f"{bool(torch.isfinite(y).all().item())}  validas={N} "
                  f"relleno={L - N}  -> {'OK' if good else 'FALLO'}")
    return ok


def _test_loss_mask(verbose: bool = True) -> bool:
    """(d) loss_mask marca lo correcto en los dos regimenes."""
    m = _mk(n_latents=512, d_model=64, n_layers=2, n_heads=4, d_ff=256)
    ok = True
    for L in (1, 17, 70, 512, 1024, 2048):
        msk = m.loss_mask(L)
        N = m.n_valid_outputs(L)
        esperado = torch.zeros(L, dtype=torch.bool)
        esperado[L - N:] = True
        good = bool(torch.equal(msk, esperado)) and int(msk.sum()) == N
        # coherencia con el forward: donde la mascara es False, logits == 0
        x = torch.randint(3, m.vocab_size, (1, L))
        with torch.no_grad():
            y = m(x)
        cero_fuera = bool((y[0][~msk] == 0).all().item()) if int((~msk).sum()) else True
        no_cero_dentro = bool((y[0][msk].abs().sum(-1) > 0).all().item())
        good = good and cero_fuera and no_cero_dentro
        ok = ok and good
        if verbose:
            print(f"  L={L:5d}: loss_mask suma {int(msk.sum())} de {L} "
                  f"(esperado {N}), primer True en {int(msk.nonzero()[0])}, "
                  f"logits==0 fuera={cero_fuera} y !=0 dentro={no_cero_dentro} "
                  f"-> {'OK' if good else 'FALLO'}")
    # El coste de ignorarla, en numeros
    bits = math.log(155) / math.log(2)
    if verbose:
        print(f"  coste de NO usarla con L=1024, n_latents=512: la mitad de las "
              f"posiciones aportan {bits:.3f} bits/paso constantes; una bpt real "
              f"de 1.70 se reportaria como {(1.70 + bits) / 2:.2f} (trivial=4.09)")
    return ok


def _test_rope_relative(verbose: bool = True) -> bool:
    """RoPE parcial: las queries usan la posicion ABSOLUTA p(n) = M-N+n.

    Se comprueba con el modelo completo en float64: una MISMA subsecuencia
    colocada en offsets absolutos distintos da los mismos logits (equivariancia
    a traslacion). Si las queries se rotasen con el indice latente n en vez de
    con p(n), la distancia query-clave seria falsa y esto fallaria.
    """
    m = _mk(n_latents=16, d_model=64, n_layers=2, n_heads=4, d_ff=256).double()
    torch.manual_seed(0)
    x = torch.randint(3, m.vocab_size, (2, 24))
    with torch.no_grad():
        a = m(x, pos_offset=0)
        errs = [(m(x, pos_offset=off) - a).abs().max().item() for off in (1, 713, 4096)]
    ok = max(errs) < 1e-9
    if verbose:
        for off, e in zip((1, 713, 4096), errs):
            print(f"  L=24 N=16 con pos_offset={off:5d} vs 0: max|diff| = {e:.3e} "
                  f"(tol 1e-9)")
        print(f"  rot_dim = {m.rot_dim} de {m.d_head} dimensiones por cabeza "
              f"({100.0 * m.rot_dim / m.d_head:.0f}%, el paper usa 25%)")
        print(f"  -> RoPE parcial con posiciones absolutas correctas: "
              f"{'OK' if ok else 'FALLO'}")
    return ok


def _test_cross_dropout(verbose: bool = True) -> bool:
    """El cross-attend dropout existe, actua solo en train y no rompe causalidad.

    Ademas se cuantifica el problema del valor 0.7 en nuestro regimen M == N:
    el latente n solo tiene n claves visibles, asi que se queda SIN NINGUNA con
    probabilidad p^n.
    """
    from config import Config
    cfg = Config(d_model=64, n_layers=2, n_heads=4, d_ff=256, dropout=0.0)
    cfg.n_latents = 64
    cfg.cross_attend_dropout = 0.7
    m = PerceiverAR(cfg)
    x = torch.randint(3, m.vocab_size, (4, 64))
    m.train()
    with torch.no_grad():
        y1, y2 = m(x), m(x)
    cambia_en_train = (y1 - y2).abs().max().item() > 0
    m.eval()
    with torch.no_grad():
        y3, y4 = m(x), m(x)
    determinista_en_eval = (y3 - y4).abs().max().item() == 0.0
    # causalidad EN MODO TRAIN: el dropout solo puede anular conexiones ya
    # permitidas, nunca crear una nueva. Se fija la semilla para que las dos
    # llamadas usen la misma mascara de dropout.
    m.train()
    t = 40
    x2 = x.clone()
    x2[:, t] = 3 + (x[:, t] - 3 + 37) % (m.vocab_size - 3)
    torch.manual_seed(123)
    with torch.no_grad():
        a = m(x)
    torch.manual_seed(123)
    with torch.no_grad():
        b = m(x2)
    causal_train = (a[:, :t] - b[:, :t]).abs().max().item() == 0.0
    p = 0.7
    esperado = sum(p ** n for n in range(1, 1024)) / 1024.0
    ok = cambia_en_train and determinista_en_eval and causal_train
    if verbose:
        print(f"  p=0.7: dos forwards en train difieren ({cambia_en_train}), en eval "
              f"son identicos bit a bit ({determinista_en_eval})")
        print(f"  causalidad tambien en MODO TRAIN (misma semilla): "
              f"max|delta pasado| = 0.0 -> {causal_train}")
        print(f"  coste del 0.7 con M=N=1024: P(latente n se queda sin ninguna "
              f"clave) = 0.7^n -> n=1:{p:.2f} n=2:{p**2:.2f} n=3:{p**3:.2f} "
              f"n=5:{p**5:.2f}; media sobre los 1024 latentes: {100*esperado:.2f}%")
        print(f"  -> cross-attend dropout: {'OK' if ok else 'FALLO'}")
    return ok


def _test_param_count(verbose: bool = True) -> bool:
    """(e) Parametros contra una formula independiente."""
    from config import Config
    ok = True
    for kw in (dict(), dict(d_model=256, n_layers=4, n_heads=4, tie_weights=False)):
        cfg = Config(**kw)
        m = PerceiverAR(cfg)
        V, D, Ly, hid = m.vocab_size, m.d_model, m.n_layers, m.d_hidden
        exp = (V * D                       # embedding
               + Ly * 4 * D * D            # cross: q+kv+proj = 4D^2 ; self: qkv+proj = 4D^2
               + Ly * 3 * D * hid          # SwiGLU por bloque
               + (2 * Ly + 2) * D          # 3 normas en el cross + 2 por self + norm_f
               + (0 if m.tie_weights else V * D))
        got = m.n_params()
        good = exp == got
        ok = ok and good
        if verbose:
            print(f"  d_model={D} n_layers={Ly} (1 cross + {m.n_self_layers} self) "
                  f"tie={m.tie_weights}: esperado {exp} ({exp/1e6:.4f} M) "
                  f"real {got} ({got/1e6:.4f} M) -> {'OK' if good else 'FALLO'}")
            print(f"    {m.param_report()}   rot_dim={m.rot_dim}/{m.d_head}")
        del m
    return ok


def _test_train_sanity(verbose: bool = True) -> bool:
    """El modelo ENTRENA: todos los parametros reciben gradiente y memoriza.

    Se entrena SOLO sobre las posiciones validas (loss_mask), que es como hay
    que usarlo, con L=64 > n_latents=32 para ejercitar el regimen dificil.
    """
    from config import Config
    torch.manual_seed(0)
    cfg = Config(d_model=128, n_layers=3, n_heads=4, d_ff=512, dropout=0.0)
    cfg.n_latents = 32
    cfg.cross_attend_dropout = 0.0
    m = PerceiverAR(cfg).train()
    x = torch.randint(3, m.vocab_size, (2, 65))
    inp, tgt = x[:, :-1], x[:, 1:]
    msk = m.loss_mask(64)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    l0 = None
    for i in range(200):
        lg = m(inp)
        loss = F.cross_entropy(lg[:, msk].reshape(-1, m.vocab_size),
                               tgt[:, msk].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if i == 0:
            l0 = loss.detach().item()
            sin_grad = [n for n, p in m.named_parameters()
                        if p.grad is None or float(p.grad.abs().sum()) == 0.0]
        opt.step()
    l1 = loss.detach().item()
    ok = len(sin_grad) == 0 and l1 < 0.05 and l0 > 4.5
    if verbose:
        print(f"  L=64, n_latents=32 -> se entrena sobre {int(msk.sum())} posiciones")
        print(f"  tensores de parametros sin gradiente: {len(sin_grad)} "
              f"{sin_grad[:3] if sin_grad else ''}")
        print(f"  perdida (solo posiciones validas): {l0:.3f} -> {l1:.4f} en 200 "
              f"pasos (ln 155 = 5.043 es el uniforme) -> {'OK' if ok else 'FALLO'}")
    return ok


def _test_equivalence_decoder(verbose: bool = True) -> bool:
    """Con n_latents >= L el modelo es un decoder causal COMPLETO.

    Comprobacion util: los logits con n_latents=1024 y con n_latents=64 para
    L=64 son identicos bit a bit (mismos pesos), porque N = min(L, n_latents)
    y la arquitectura no depende de n_latents mas que por ese minimo.
    """
    from config import Config
    torch.manual_seed(0)
    cfg = Config(d_model=64, n_layers=2, n_heads=4, d_ff=256, dropout=0.0)
    cfg.n_latents = 1024
    cfg.cross_attend_dropout = 0.0
    a = PerceiverAR(cfg).eval()
    cfg2 = Config(d_model=64, n_layers=2, n_heads=4, d_ff=256, dropout=0.0)
    cfg2.n_latents = 64
    cfg2.cross_attend_dropout = 0.0
    b = PerceiverAR(cfg2).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randint(3, a.vocab_size, (2, 64))
    with torch.no_grad():
        d = (a(x) - b(x)).abs().max().item()
    ok = d == 0.0
    if verbose:
        print(f"  L=64 con n_latents=1024 vs n_latents=64 (mismos pesos): "
              f"max|diff| = {d!r} (exige 0.0) -> {'OK' if ok else 'FALLO'}")
    return ok


def _benchmark(batch: int, L: int, n_latents: int, device: str = "cpu",
               iters: int = 6, reps: int = 3) -> float:
    """fwd+bwd+clip+step, MEJOR de reps repeticiones de iters pasos."""
    import time
    from config import Config
    cfg = Config()
    cfg.n_latents = int(n_latents)
    m = PerceiverAR(cfg).to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(3, m.vocab_size, (batch, L + 1), device=device)
    inp, tgt = x[:, :-1], x[:, 1:]
    msk = m.loss_mask(L, device=device)
    import contextlib
    def ctx():
        return (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
                else contextlib.nullcontext())
    best = 0.0
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            with ctx():
                lg = m(inp)
                loss = F.cross_entropy(lg[:, msk].reshape(-1, m.vocab_size),
                                       tgt[:, msk].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        best = max(best, iters / (time.perf_counter() - t0))
    extra = ""
    if device == "cuda":
        extra = f", pico {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        torch.cuda.reset_peak_memory_stats()
    nv = int(msk.sum())
    print(f"  [{device}] batch {batch}, M={L}, n_latents={n_latents}: "
          f"{best:.2f} it/s ({1000.0 / best:.0f} ms/it), {nv} salidas validas de "
          f"{L} -> {best * batch * nv:.0f} salidas validas/s{extra}")
    del m, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return best


def _test_gpu_causality(verbose: bool = True) -> bool:
    """(a) Causalidad bit a bit tambien en el kernel CUDA de SDPA, en bf16.

    Importa porque la mascara booleana [N,M] la consume un kernel distinto del
    de CPU y con bf16 la acumulacion es otra; un fallo de causalidad podria ser
    especifico del kernel.
    """
    from config import Config
    ok = True
    for L, N in ((128, 128), (128, 48)):
        cfg = Config()
        cfg.n_latents = N
        m = PerceiverAR(cfg).cuda().eval()
        x = torch.randint(3, m.vocab_size, (2, L), device="cuda")
        t = 96
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            base = m(x)
            x2 = x.clone()
            x2[:, t] = 3 + (x[:, t] - 3 + 37) % (m.vocab_size - 3)
            pert = m(x2)
        dp = (base[:, :t] - pert[:, :t]).abs().max().item()
        ds = (base[:, t] - pert[:, t]).abs().max().item()
        good = dp == 0.0 and ds > 0.0
        ok = ok and good
        if verbose:
            print(f"  L={L} n_latents={N} ({base.dtype}): perturbando solo x[:,{t}] "
                  f"-> max|delta pasado| = {dp:.1e} (exige 0.0), propio {ds:.3e} "
                  f"-> {'OK' if good else 'FUGA CAUSAL'}")
        del m, base, pert
        torch.cuda.empty_cache()
    return ok


def _compare_with_modern() -> bool:
    """La medida que de verdad decide si Perceiver AR aporta algo AQUI.

    No compara it/s (eso favorece trivialmente a quien produce menos salidas)
    sino SALIDAS VALIDAS POR SEGUNDO: cuantas predicciones con gradiente real
    genera cada arquitectura por unidad de tiempo. Es la moneda del
    entrenamiento. Ver el veredicto al final del fichero.

    METODO (importa tanto como el resultado). Los dos modelos se miden en
    rondas INTERCALADAS A/B/A/B... y se reporta la MEDIANA de las rondas, no el
    mejor tiempo. Razon medida, no teorica: esta GPU esta compartida con otro
    entrenamiento y su carga varia en escala de segundos. Midiendo primero las
    3 repeticiones de A y luego las de B, este mismo codigo llego a imprimir
    "perceiver_ar rinde 240%" en una pasada (41.8 it/s para perceiver_ar
    mientras el otro trabajo aflojaba, 17.4 para modern un minuto despues) y
    "95%" en otra. Intercalando, las dos arquitecturas ven la misma carga
    externa en cada ronda y el cociente se repite dentro de 2 puntos.
    """
    import time
    from config import Config
    try:
        from .modern_transformer import ModernTransformer
    except ImportError:                                  # pragma: no cover
        from models.modern_transformer import ModernTransformer

    B, ROUNDS, ITERS = 2, 6, 5

    def round_(m, opt, inp, tgt, msk):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(inp)
                loss = F.cross_entropy(lg[:, msk].reshape(-1, m.vocab_size),
                                       tgt[:, msk].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        torch.cuda.synchronize()
        return ITERS / (time.perf_counter() - t0)

    def median(v):
        s = sorted(v)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    print(f"  perceiver_ar (n_latents=1024) vs modern (decoder causal), batch {B},"
          f" bf16, {ROUNDS} rondas INTERCALADAS, mediana:")
    for L in (1024, 3072):
        cfg = Config()
        cfg.n_latents = 1024
        models = {}
        ctors = (("perceiver_ar", lambda: PerceiverAR(cfg)),
                 ("modern", lambda: ModernTransformer(Config())))

        def build(key, ctor):
            m = ctor().cuda().train()
            msk = (m.loss_mask(L, device="cuda") if key == "perceiver_ar"
                   else torch.ones(L, dtype=torch.bool, device="cuda"))
            x = torch.randint(3, m.vocab_size, (B, L + 1), device="cuda")
            return dict(m=m, opt=torch.optim.AdamW(m.parameters(), lr=1e-4),
                        inp=x[:, :-1], tgt=x[:, 1:], msk=msk, s=[])

        # Pico de memoria: se mide con UN SOLO modelo residente. Con los dos
        # vivos a la vez el pico global no es atribuible a ninguno, y restar
        # "lo que habia antes" tampoco vale porque el asignador reutiliza
        # bloques liberados (se midio: daba 0.49 GiB para algo que solo ocupa
        # 0.91). Asi que la memoria se mide en pasadas aisladas y solo despues
        # se construyen los dos para cronometrar.
        pk = {}
        for key, ctor in ctors:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            d = build(key, ctor)
            round_(d["m"], d["opt"], d["inp"], d["tgt"], d["msk"])
            pk[key] = torch.cuda.max_memory_allocated() / 2 ** 30
            del d
            torch.cuda.empty_cache()
        for key, ctor in ctors:
            d = build(key, ctor)
            round_(d["m"], d["opt"], d["inp"], d["tgt"], d["msk"])   # calentamiento
            d["pk"] = pk[key]
            models[key] = d
        for _ in range(ROUNDS):                         # A/B/A/B/...
            for d in models.values():
                d["s"].append(round_(d["m"], d["opt"], d["inp"], d["tgt"], d["msk"]))
        vs = {}
        for key, d in models.items():
            it = median(d["s"])
            nv = int(d["msk"].sum())
            vs[key] = it * B * nv
            print(f"    {key:12s} M={L:5d}: {it:6.2f} it/s (rondas "
                  + " ".join(f"{v:.1f}" for v in d["s"]) + f")  pico {d['pk']:.2f} "
                  f"GiB  {nv:5d} validas -> {vs[key]:8.0f} salidas validas/s")
        print(f"    -> a M={L}, perceiver_ar rinde "
              f"{100.0 * vs['perceiver_ar'] / vs['modern']:.0f}% de las salidas "
              f"validas/s del decoder causal")
        del models
        torch.cuda.empty_cache()
    return True


def _selftest(bench: bool = True) -> bool:
    torch.manual_seed(0)
    ok = True
    print("== (b) mascara del cross-attend, la del paper ==")
    ok &= _test_cross_mask()
    print("== las queries NO son parametros aprendidos ==")
    ok &= _test_no_learned_latents()
    print("== (a) causalidad, test GENERICO del laboratorio ==")
    print("  -- regimen L <= N (L=32, n_latents=512) --")
    ok &= _test_generic_causality(_mk(n_latents=512), L=32)
    print("  -- regimen L >  N (L=64, n_latents=16) --")
    ok &= _test_generic_causality(_mk(n_latents=16), L=64)
    print("== (a) causalidad ESTRICTA bit a bit, posicion a posicion ==")
    ok &= _test_strict_causality(_mk(n_latents=512), L=48)
    ok &= _test_strict_causality(_mk(n_latents=16), L=48)
    ok &= _test_strict_causality(_mk(n_latents=1), L=12)
    print("== RoPE parcial y posiciones absolutas ==")
    ok &= _test_rope_relative()
    print("== cross-attend dropout (paper: 0.7) ==")
    ok &= _test_cross_dropout()
    print("== (c) formas ==")
    ok &= _test_shapes()
    print("== (d) loss_mask ==")
    ok &= _test_loss_mask()
    print("== n_latents >= L equivale a un decoder causal completo ==")
    ok &= _test_equivalence_decoder()
    print("== (e) parametros ==")
    ok &= _test_param_count()
    print("== entrenamiento de sanidad ==")
    ok &= _test_train_sanity()

    if bench:
        print("== (e) velocidad (CPU, fp32) ==")
        _benchmark(batch=2, L=1024, n_latents=1024, device="cpu", iters=3, reps=2)
        _benchmark(batch=2, L=1024, n_latents=512, device="cpu", iters=3, reps=2)
        _benchmark(batch=2, L=3072, n_latents=1024, device="cpu", iters=2, reps=2)
        if torch.cuda.is_available():
            print("== (a) causalidad en GPU con el kernel CUDA de SDPA, bf16 ==")
            try:
                ok &= _test_gpu_causality()
                print("== (e) velocidad en GPU (bf16) y COMPARATIVA HONESTA ==")
                torch.cuda.reset_peak_memory_stats()
                _benchmark(batch=2, L=1024, n_latents=1024, device="cuda",
                           iters=5, reps=3)
                _benchmark(batch=2, L=3072, n_latents=1024, device="cuda",
                           iters=5, reps=3)
                _compare_with_modern()
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                print(f"  SIN MEMORIA en la GPU (compartida): "
                      f"{str(e).splitlines()[0][:110]}")

    print("== resultado global ==")
    print("  TODO OK" if ok else "  HAY FALLOS")
    return bool(ok)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
