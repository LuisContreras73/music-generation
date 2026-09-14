"""
DeepLSTM: pila recurrente PROFUNDA con residuales, LayerNorm y dropout variacional.

La linea base recurrente del laboratorio (LSTMBaseline, 3 capas de 1024) es el
modelo con MEJOR gen_score (80.8) pese a ser el PEOR en verosimilitud
(test_bpt 2.02 frente a 1.81 del Music Transformer). Este modulo lleva esa misma
familia a 8 capas sin que la profundidad la degrade, que es lo que pasa si uno
se limita a apilar nn.LSTM.

Arquitectura (token-level, V = 155, ver src/data/tokenizer.py)

    x [B,L] Long
      e   = Embedding(V, d_model)                       [B,L,d]
      s   = LockedDropout(e)                            mascara UNICA en el tiempo
      u   = Linear(d_model -> H, sin sesgo)             solo si H != d_model
      h   = u
      repetir n_layers veces (bloque pre-norm residual):
            z = LayerNorm(h)
            z = LockedDropout(z)                        una mascara por capa
            y, (h_i,c_i) = LSTM_i(z)                    nn.LSTM de UNA capa, H->H
            h = h + y                                   <-- RESIDUAL
      h   = LockedDropout(LayerNorm(h))
      z   = Linear(H -> d_model, sin sesgo)             solo si H != d_model
      z   = LayerNorm(z + skip_gate * s)                <-- SKIP GLOBAL desde el embedding
      logits = Linear(d_model -> V)                     peso atado al embedding
    logits [B,L,V]  del token x[t+1]

Las cuatro piezas que piden las instrucciones, y por que cada una
-----------------------------------------------------------------
1. RESIDUAL + LAYERNORM ENTRE CAPAS.
   Una pila de LSTM sin residuales multiplica jacobianos entre capas igual que
   una MLP profunda sin skips: a 6-8 capas el gradiente que llega a la primera
   capa es ruido y el modelo entrena PEOR que con 3. El bloque es pre-norm
   (LayerNorm -> capa -> suma) y no post-norm porque el pre-norm deja un camino
   lineal e ininterrumpido desde el embedding hasta la cabeza (h = u + sum_i y_i):
   con eso la profundidad solo puede anadir, no destruir. La proyeccion de
   entrada (d_model -> H) hace que TODOS los bloques tengan la misma anchura,
   asi que el residual es la identidad en todos ellos y no hay ninguna capa con
   una proyeccion que rompa el camino (cuando H == d_model la proyeccion
   desaparece y es literalmente la identidad).
   Ojo al detalle: la LayerNorm es PUNTUAL en el tiempo (normaliza sobre el eje
   de caracteristicas, nunca sobre L), asi que no introduce dependencia con el
   futuro. Normalizar sobre el tiempo seria una fuga causal inmediata.
   Esto NO es un acto de fe: check_depth_helps() lo mide. Con 8 capas, misma
   anchura, mismos 4.26 M parametros y la misma semilla, memorizando un lote
   fijo durante 60 pasos (medido en CPU):
       con residual+LayerNorm   loss 5.109 -> 0.020
       pila pelada (ablacion)   loss 5.187 -> 2.612
   y el gradiente que llega a W_hh de la PRIMERA capa frente a la ultima:
       con residual   cociente ultima/primera = 0.2   (llega de sobra abajo)
       sin residual   cociente ultima/primera = 4.8   (se desvanece hacia abajo)
   Es exactamente el fallo que se predice al apilar recurrentes sin skips.

2. DROPOUT VARIACIONAL / LOCKED (Gal & Ghahramani 2016).
   LockedDropout muestrea UNA mascara [B,1,H] por secuencia y la reutiliza en
   todos los pasos temporales, en lugar de una mascara distinta por paso como
   hace nn.Dropout (y como hace el dropout interno de nn.LSTM). La diferencia no
   es cosmetica: con mascara nueva en cada paso, el ruido inyectado en la
   trayectoria del estado es blanco en el tiempo y el LSTM aprende a
   promediarlo, de modo que el dropout deja de regularizar la recurrencia y solo
   anade varianza al gradiente. Con mascara fija, el modelo tiene que resolver
   la secuencia ENTERA con el mismo subconjunto de unidades, que es lo que de
   verdad fuerza representaciones redundantes. Es la receta de AWD-LSTM y de
   Performance-RNN bien regularizado.
   Se implementa aqui (no se usa el dropout de nn.LSTM) por tres razones: es el
   unico modo de tener mascara compartida en el tiempo; deja el dropout bajo
   nuestro control entre bloques; y evita el fallo de entorno documentado en
   lstm_baseline.py (nn.LSTM con num_layers>1 y dropout>0 en CUDA corrompe el
   codigo de salida del proceso, 0xC0000409). Aqui la pila son nn.LSTM de UNA
   capa por necesidad arquitectonica (hay que meter norm+residual entre ellas),
   asi que el fallo de cuDNN no se puede dar.

3. CABEZA PROYECTADA CON PESOS ATADOS.
   H != d_model en general (la anchura recurrente conviene mayor que d_model),
   asi que la salida se proyecta a d_model y solo entonces se aplica la cabeza,
   cuyo peso es EL MISMO TENSOR que el embedding (tie_weights). Atar pesos en un
   vocabulario de 155 simbolos no ahorra parametros relevantes, pero fuerza que
   "escribir el token p" y "leer el token p" compartan geometria, que con
   vocabularios pequenos y muy estructurados (88 NOTE_ON contiguos en pitch +
   64 SHIFT contiguos en duracion) es un sesgo inductivo util: las notas vecinas
   quedan cerca en el espacio de embedding y la cabeza hereda esa vecindad.

4. SKIP GLOBAL EMBEDDING -> CABEZA (skip_gate, escalar aprendido, init 1.0).
   El embedding del token ACTUAL entra en la cabeza sin pasar por ocho capas
   recurrentes. Con contexto largo da igual (el estado ya codifica el token
   actual), pero con contexto CORTO el estado aun esta "frio" y la senal mas
   fiable que tiene el modelo es literalmente el ultimo token: tras un NOTE_ON
   grave viene casi siempre otro NOTE_ON o un SHIFT corto, y esa regla local no
   deberia tener que sobrevivir a ocho capas con dropout para llegar a los
   logits. El gate es aprendido, asi que el modelo puede apagarlo si estorba.

Por que ESTA arquitectura encaja con el caso que se evalua (prefijos de ~70 tokens)
-----------------------------------------------------------------------------------
El problema declarado es que los modelos se entrenan con ventanas de 1024 tokens
y en la prueba final reciben 70 (5 s de melodia monofonica), y degeneran en
bucles (fraccion de 8-gramas repetidos 0.63 frente a 0.217 del corpus).

  * UN LSTM NO TIENE MISMATCH DE LONGITUD DE CONTEXTO, POR CONSTRUCCION. La
    funcion que calcula en la posicion t es exactamente la misma recurrencia
    h_t = f(h_{t-1}, x_t) sea L = 70 o L = 1024: no hay codificacion posicional,
    ni sesgos relativos por distancia, ni una ventana de atencion cuyo tamano
    cambie la normalizacion del softmax. Un transformer con atencion relativa
    aprende un reparto de masa de atencion sobre "cuanto pasado hay"; con 70
    tokens ese reparto esta fuera de distribucion. Aqui no hay nada que
    desajustar: los 70 primeros pasos de una ventana de entrenamiento de 1024
    aplican LA MISMA FUNCION que los 70 pasos del prefijo de la prueba.
    Matiz MEDIDO (check_prefix_invariance), porque la version anterior de este
    docstring decia "bit a bit" y eso es FALSO: forward(x[:,:L]) y
    forward(x)[:,:L] difieren en ~2.6e-06 en float32 para L = 1, 17, 70, 200,
    512, porque el backend recurrente elige algoritmo segun la longitud. Es
    redondeo de float32, no un cambio de funcion (los logits estan en el rango
    +-2 y su separacion util es 1e-2), pero no es igualdad exacta y no se debe
    afirmar que lo sea.
  * ESTADO INICIAL APRENDIDO (h0, c0 como parametros, 10 k parametros). Todas
    las ventanas arrancan en el mismo estado, asi que el modelo aprende un prior
    explicito para "todavia no tengo contexto" en vez de que el cero sea un punto
    arbitrario del espacio de estados. Eso calibra justo el regimen t = 0..70,
    que es el que se evalua.
    CUIDADO con la lectura facil (la version anterior de este docstring la hacia):
    h0 NO es un prior de "principio de obra". Las ventanas de entrenamiento NO
    empiezan en el principio de la pieza; src/data/datasets.py::TokenWindows
    sortea el arranque DENTRO de la pieza para el split de train. Lo que aprende
    h0 es un prior de "contexto desconocido", que da la casualidad de ser
    exactamente lo que hace falta cuando llega un prefijo suelto de 5 s.
  * MEMORIA DE TAMANO FIJO = MALA COPISTA. Los bucles verbatim son, en un
    transformer, un efecto de la atencion: copiar un 8-grama de hace 200 tokens
    es una operacion barata y exacta (induction head). Un estado de tamano fijo
    comprime con perdidas y no dispone de esa operacion, por lo que es mucho mas
    dificil que caiga en repeticion exacta. Esto no es teoria: es la explicacion
    mas simple de por que la LSTM del laboratorio gana en gen_score aun perdiendo
    en bpt. Este modelo conserva esa propiedad y solo mejora la capacidad y la
    regularizacion.
  * EL DROPOUT VARIACIONAL ATACA EL MODO DE FALLO. Una trayectoria de estado que
    depende de unas pocas unidades muy agudas es exactamente la que se engancha
    en un ciclo limite. Forzar que la secuencia entera se resuelva con un
    subconjunto aleatorio de unidades aplana esos atractores.
  * MUESTREO O(1) CON step(). Con prefijo corto y continuacion larga, regenerar
    el prefijo entero en cada token es O(L^2); aqui es O(1) por token y el estado
    se ceba de una vez con forward_state(prefijo).

Receta de entrenamiento recomendada (no la impone este modulo)
--------------------------------------------------------------
El mismatch de DOMINIO (nunca vio texturas monofonicas ralas) no lo arregla la
arquitectura: usar src/augment.py::thin_voices para adelgazar acordes a casi
monofonico en una fraccion de los batches, y muestrear con repetition_penalty /
no_repeat_ngram de src/sampling.py. Este modulo no toca nada de eso.

Presupuesto de parametros
-------------------------
Con la config por defecto del laboratorio (d_model=512, n_layers=8, hidden=1024)
una pila de 8 LSTM de 1024 son 68.3 M parametros, mas del doble del techo de
30 M. La anchura recurrente H se AJUSTA AL PRESUPUESTO: se baja al mayor
multiplo de 64 que cabe en 30 M (con la config por defecto, H = 640) y se avisa
por warnings. El ajuste es una funcion determinista del cfg, asi que recargar un
checkpoint reconstruye exactamente la misma arquitectura. Se desactiva con
DeepLSTM.fit_budget = False. default_config() ya trae H = 640 explicito y por
tanto no dispara ningun aviso: 27.01 M parametros, 8 capas.

Coste medido (RTX 4070 SUPER 12.9 GB, default_config, 27.01 M parametros)
-------------------------------------------------------------------------
AVISO METODOLOGICO, LEER ANTES DE CITAR NINGUNA CIFRA DE GPU: esta maquina
comparte la GPU con el resto del laboratorio. Durante TODA la medicion habia mas
de 20 procesos ajenos y 100% de utilizacion permanente (la memoria ajena oscilo
entre 1.2 y 11.5 de los 12.0 GiB segun el momento, la utilizacion no). Las
cifras de abajo son por tanto COTAS INFERIORES del rendimiento real: miden el
modelo compitiendo con veinte procesos, no el modelo. Todas se toman con
torch.cuda.synchronize() a ambos lados del cronometro y se informa mediana,
mejor y el cociente mejor/mediana (si se aleja de 1, la tirada mide contencion).

CORRECCION DE AUDITORIA: la primera version de este fichero anunciaba
"mediana 0.51 it/s" como coste del modelo. Esa cifra era contencion pura. Vuelta
a medir con el mismo batch, la misma longitud y la misma precision, con la GPU
igual de saturada, sale 5.6x mejor:

    entrenamiento GPU, batch 16, L=1024, bf16, GPU COMPARTIDA AL 100%
        tirada estable (mejor/mediana 1.11x, la carga ajena era constante):
            mediana 2.88 it/s (348 ms/it), mejor 3.18 it/s
        tres tiradas mas con la carga ajena a rafagas (mejor/mediana 9.7x,
        11.2x y 28.4x: esas MEDIANAS no miden nada y no se citan):
            mejor 3.20, 3.08 y 2.72 it/s
        El MEJOR tiempo es el estimador robusto aqui, y se agrupa en
        2.7-3.2 it/s en las cuatro tiradas. Tomese 2.9 it/s como COTA INFERIOR;
        con la GPU libre sera mas.
        pico de VRAM 2.08 GiB de 12.9, identico en las cuatro tiradas  <-- esta
        cifra SI es exacta pese a la contencion (max_memory_allocated es por
        proceso). Cabe de sobra a batch 16 y el margen permite batch 24-32.
    entrenamiento CPU, batch 2, L=128, fp32
        La CPU de esta maquina TAMPOCO esta libre: el MISMO codigo, el mismo
        modelo y el mismo lote dan 2.77, 2.54 y 0.34 it/s en tres momentos del
        mismo dia (un factor 8), siempre con mejor/mediana ~1.06 dentro de cada
        tirada, o sea sin que la tirada se delate. La version anterior de este
        fichero citaba 0.92 it/s como si fuera el coste del modelo; no lo es,
        igual que no lo es 0.34. La cifra defendible es la MEJOR observada:
            2.77 it/s (362 ms/it), 708 tokens/s
        Moraleja operativa: en esta maquina ninguna cifra absoluta de coste, ni
        de GPU ni de CPU, significa nada sin decir con que carga de fondo se
        tomo, y solo los cocientes medidos alternando modelos son comparables.
    muestreo con step(), CPU, batch 1
        6.8 ms/token en la tirada mejor (148 tokens/s por secuencia), 10.6 en la
        peor. Lo que importa aqui no es la constante sino que el coste es O(1)
        por token y no O(L) como en un transformer sin cache.

Comparacion RELATIVA con la linea base, que es lo unico robusto a la contencion:
se cronometran deep_lstm y LSTMBaseline ALTERNANDOSE en el mismo proceso, de modo
que la carga de fondo afecta a los dos por igual.

    CPU, batch 2, L=128, fp32 (medido en la auditoria)
        deep_lstm     (8x640,  27.01 M)  mediana 277 ms/it
        lstm_baseline (3x1024, 23.70 M)  mediana 173 ms/it   cociente 1.60x

Ocho capas cuestan 1.6x por iteracion frente a las tres de la linea base, no el
2.7x que sugeriria contar capas: las capas son mas estrechas (640 frente a 1024)
y el total de FLOPs recurrentes es casi el mismo (8 x 8H^2 con H=640 = 26.2 M
frente a 3 x 8H^2 con H=1024 = 25.2 M); lo que se paga es lanzar 8 llamadas
secuenciales al backend recurrente en vez de 3. En GPU el cociente sera menor que
en CPU (mas paralelismo por llamada), pero mientras la GPU este ocupada no se
puede medir: con la GPU libre hay que volver a pasar bench_train.

Causalidad
----------
La recurrencia es causal por construccion y TODAS las demas operaciones del grafo
(embedding, proyecciones, LayerNorm sobre el eje de caracteristicas, dropout,
suma residual, skip global, cabeza) son puntuales en el tiempo. No existe
ninguna operacion que mezcle posiciones. Se verifica igualmente con el test del
laboratorio y con una prueba mas dura: check_strict_causality perturba UN SOLO
token x[:,t] y exige torch.equal (BIT A BIT, no una tolerancia) en todos los
logits < t, barriendo TODAS las posiciones t = 1..L-1 y no una muestra, y ademas
engancha un hook al embedding para exigir d logits[:,t] / d emb[:,t'>t] == 0.0
EXACTO. Como control de que la prueba no pasa por vacuidad se exige tambien que
el logit EN t SI cambie y que el gradiente EN t sea > 0. Ver el bloque __main__.

Pruebas del modulo (todas se ejecutan con  python src/models/deep_lstm.py)
--------------------------------------------------------------------------
    check_strict_causality   causalidad bit a bit + gradiente cero exacto
    check_step_equivalence   forward == step token a token == trozos irregulares
    check_prefix_invariance  forward(x[:,:L]) == forward(x)[:,:L]: la propiedad
                             que hace util a este modelo con prefijos de 70
    check_locked_dropout     el dropout es variacional, demostrado numericamente
    check_depth_helps        ablacion medida: sin residuales la pila no entrena
    check_shapes             L = 1, 17, 70, 1024 y entrada todo-PAD
    bench_train / bench_step coste, siempre con torch.cuda.synchronize()
Se ejercitan tres configuraciones: default_config(), la alternativa del contrato
(d_model=256, n_layers=4, tie_weights=False, que es OTRA rama de codigo: cambia
padding_idx, la cabeza deja de estar atada y cambia el recuento analitico) y una
minima (d_model=128, H=192, 3 capas).
"""
from __future__ import annotations

import time
import warnings

import torch
import torch.nn as nn

try:                                     # uso normal: sys.path.insert(0, "src")
    from config import Config
    from models.base import TokenARModel
    from data.tokenizer import VOCAB_SIZE, PAD
except ImportError:                      # ejecucion directa del archivo
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    for _p in (str(_ROOT), str(_ROOT / "src")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from config import Config
    from models.base import TokenARModel
    from data.tokenizer import VOCAB_SIZE, PAD


PARAM_BUDGET = (20e6, 30e6)              # techo del laboratorio para este modelo
WIDTH_MULTIPLE = 64                      # granularidad del ajuste de anchura


# ---------------------------------------------------------------------------
# dropout variacional (locked)
# ---------------------------------------------------------------------------
class LockedDropout(nn.Module):
    """Dropout con la MISMA mascara en todos los pasos temporales.

    Gal & Ghahramani (2016). La mascara se muestrea con forma [B,1,F] y se
    difunde sobre el eje temporal, de modo que una unidad apagada lo esta
    durante TODA la secuencia. nn.Dropout, en cambio, muestrea [B,L,F]: una
    mascara distinta por paso, que en una recurrencia es ruido blanco que el
    propio LSTM aprende a promediar (y por tanto regulariza mucho menos).

    Escalado inverso (1/(1-p)) como nn.Dropout, asi que en eval() es la
    identidad exacta y no hace falta reescalar nada al generar.

    `mask` permite reutilizar una mascara entre llamadas, que es lo que hace
    falta si se encadenan ventanas de BPTT truncado y se quiere que la mascara
    sea constante a lo largo de TODA la secuencia larga y no por ventana.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        assert 0.0 <= float(p) < 1.0, "p debe estar en [0,1), llego %r" % (p,)
        self.p = float(p)

    def make_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Mascara [B,1,F] con el dtype/device de x (lista para difundir)."""
        keep = 1.0 - self.p
        return x.new_empty(x.size(0), 1, x.size(-1)).bernoulli_(keep).div_(keep)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        m = self.make_mask(x) if mask is None else mask.to(dtype=x.dtype, device=x.device)
        return x * m

    def extra_repr(self) -> str:
        return "p=%.3f, mascara compartida en el tiempo" % self.p


# ---------------------------------------------------------------------------
# presupuesto de parametros
# ---------------------------------------------------------------------------
def analytic_params(V: int, d: int, H: int, n_layers: int, tie: bool,
                    learned_init: bool = True, skip: bool = True) -> int:
    """Numero exacto de parametros SIN construir el modelo.

    Se usa para ajustar la anchura al presupuesto sin instanciar ocho pilas
    recurrentes de prueba. El bloque __main__ comprueba que coincide con el
    recuento real de nn.Module (si alguien cambia la arquitectura y se olvida de
    esta funcion, salta ahi).
    """
    n = V * d                                   # embedding
    if H != d:
        n += d * H                              # in_proj  (sin sesgo)
    n += n_layers * (8 * H * H + 8 * H)         # nn.LSTM de 1 capa: W_ih,W_hh,b_ih,b_hh
    n += n_layers * 2 * H                       # LayerNorm pre-bloque
    n += 2 * H                                  # LayerNorm final
    if H != d:
        n += H * d                              # out_proj (sin sesgo)
    n += 2 * d                                  # LayerNorm de la cabeza
    n += V if tie else (d * V + V)              # cabeza (atada -> solo el sesgo)
    if learned_init:
        n += 2 * n_layers * H                   # h0, c0
    if skip:
        n += 1                                  # skip_gate
    return int(n)


def fit_width(V: int, d: int, H: int, n_layers: int, tie: bool,
              learned_init: bool = True, hi: float = PARAM_BUDGET[1],
              mult: int = WIDTH_MULTIPLE) -> int:
    """Mayor anchura recurrente <= H, multiplo de `mult`, que cabe en `hi`."""
    if analytic_params(V, d, H, n_layers, tie, learned_init) <= hi:
        return H
    w = (H // mult) * mult
    while w > mult and analytic_params(V, d, w, n_layers, tie, learned_init) > hi:
        w -= mult
    return min(w, H) if w >= mult else H


# ---------------------------------------------------------------------------
# modelo
# ---------------------------------------------------------------------------
class DeepLSTM(TokenARModel):
    """LSTM profundo residual con dropout variacional. forward([B,L]) -> [B,L,V]."""

    name = "deep_lstm"

    # La pila recurrente corre en float32 aunque haya autocast activo. Valvula de
    # escape: PyTorch registra _cudnn_rnn con politica de autocast fp16 FIJA, de
    # modo que dentro de autocast(bfloat16) las capas se ejecutan igualmente en
    # float16 (ver la nota de lstm_baseline.py). Solo hace falta si el
    # entrenamiento diverge; cuesta 2-3x en velocidad.
    fp32_rnn: bool = False
    # Ajustar la anchura recurrente al presupuesto de parametros (ver fit_width).
    fit_budget: bool = True
    # Estado inicial (h0,c0) como parametros en vez de ceros.
    learned_init: bool = True

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        V = int(VOCAB_SIZE)
        d = int(cfg.d_model)
        nl = int(cfg.n_layers)
        p = float(cfg.dropout)
        # `rnn_width` no existe en Config todavia; si algun dia se anade, manda
        # sobre `hidden` y permite fijar la anchura sin tocar este archivo.
        H_req = int(getattr(cfg, "rnn_width", 0) or cfg.hidden)
        assert nl >= 1 and d >= 1 and H_req >= 1, "n_layers/d_model/hidden deben ser >= 1"
        tie = bool(cfg.tie_weights)

        H = fit_width(V, d, H_req, nl, tie, self.learned_init) if self.fit_budget else H_req
        if H != H_req:
            warnings.warn(
                "DeepLSTM: anchura recurrente reducida de %d a %d para caber en el "
                "presupuesto (%d capas de %d serian %.1f M parametros, techo %.0f M). "
                "Es determinista a partir del cfg, asi que un checkpoint se recarga "
                "igual. Para evitar el aviso usa DeepLSTM.default_config(); para "
                "desactivarlo, DeepLSTM.fit_budget = False."
                % (H_req, H, nl, H_req,
                   analytic_params(V, d, H_req, nl, tie, self.learned_init) / 1e6,
                   PARAM_BUDGET[1] / 1e6),
                RuntimeWarning, stacklevel=2)

        self.vocab_size, self.d_model, self.hidden, self.n_layers = V, d, H, nl
        self.tie_weights = tie
        self.dropout_p = p

        # --- entrada ---
        # padding_idx solo si NO se atan pesos: con pesos atados la cabeza
        # inyectaria gradiente en la fila de PAD que el embedding mantiene a cero.
        self.emb = nn.Embedding(V, d, padding_idx=None if tie else PAD)
        self.drop_emb = LockedDropout(p)
        self.in_proj = nn.Linear(d, H, bias=False) if H != d else nn.Identity()

        # --- pila residual ---
        # nn.LSTM de UNA capa cada uno: hace falta meter LayerNorm y la suma
        # residual entre capas, cosa imposible dentro de un nn.LSTM fusionado.
        # Efecto colateral util: nunca se activa el dropout interno de cuDNN, que
        # en esta maquina corrompe el codigo de salida del proceso.
        self.layers = nn.ModuleList([
            nn.LSTM(input_size=H, hidden_size=H, num_layers=1, batch_first=True)
            for _ in range(nl)])
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(nl)])
        self.drops = nn.ModuleList([LockedDropout(p) for _ in range(nl)])

        # --- salida ---
        self.norm_f = nn.LayerNorm(H)
        self.drop_out = LockedDropout(p)
        self.out_proj = nn.Linear(H, d, bias=False) if H != d else nn.Identity()
        self.norm_head = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        if tie:
            self.head.weight = self.emb.weight          # mismo tensor, no una copia
        # Skip global embedding -> cabeza. Escalar aprendido: el modelo decide
        # cuanto pesa el token actual frente al estado recurrente.
        self.skip_gate = nn.Parameter(torch.ones(()))

        # --- estado inicial aprendido ---
        if self.learned_init:
            self.h0 = nn.Parameter(torch.zeros(nl, 1, H))
            self.c0 = nn.Parameter(torch.zeros(nl, 1, H))
        else:
            self.register_parameter("h0", None)
            self.register_parameter("c0", None)

        self._init_weights()

        n = self.n_params()
        lo, hi = PARAM_BUDGET
        if not (lo <= n <= hi):
            warnings.warn(
                "DeepLSTM: %.2f M parametros, fuera del presupuesto [%.0f, %.0f] M "
                "(n_layers=%d, H=%d, d_model=%d)."
                % (n / 1e6, lo / 1e6, hi / 1e6, nl, H, d),
                RuntimeWarning, stacklevel=2)

    def param_report(self) -> str:
        """Informe de parametros que DICE la anchura realmente construida.

        cfg.hidden puede no describir el modelo: fit_width baja la anchura
        recurrente para caber en el presupuesto y el cfg que se guarda en
        config.json conserva el valor pedido. Quien lea el log tiene que ver H.
        """
        txt = "%s  d_model=%d H=%d n_layers=%d" % (
            super().param_report(), self.d_model, self.hidden, self.n_layers)
        want = int(getattr(self.cfg, "rnn_width", 0) or self.cfg.hidden)
        if self.hidden != want:
            txt += ("  [OJO: cfg.hidden=%d, la anchura construida es H=%d por el "
                    "ajuste al presupuesto; cfg.hidden NO describe este modelo]"
                    % (want, self.hidden))
        return txt

    # ------------------------------------------------------------------ init
    def _init_weights(self) -> None:
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        if self.emb.padding_idx is not None:
            with torch.no_grad():
                self.emb.weight[self.emb.padding_idx].zero_()

        for layer in self.layers:
            for pname, prm in layer.named_parameters():
                if pname.startswith("weight_ih"):
                    nn.init.xavier_uniform_(prm)
                elif pname.startswith("weight_hh"):
                    # Ortogonal por bloque de compuerta (i,f,g,o): radio espectral 1,
                    # que retrasa explosion y desvanecimiento del gradiente en el
                    # eje temporal. Con 8 capas esto importa mas, no menos.
                    with torch.no_grad():
                        for k in range(4):
                            nn.init.orthogonal_(prm.data[k * self.hidden:(k + 1) * self.hidden])
                elif pname.startswith("bias_"):
                    nn.init.zeros_(prm)
            # Sesgo de la compuerta de olvido a 1 (Gers et al. 2000). Orden en
            # nn.LSTM: i, f, g, o.
            with torch.no_grad():
                layer.bias_ih_l0[self.hidden:2 * self.hidden].fill_(1.0)

        for proj in (self.in_proj, self.out_proj):
            if isinstance(proj, nn.Linear):
                nn.init.xavier_uniform_(proj.weight)
        nn.init.zeros_(self.head.bias)
        if not self.tie_weights:
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    # ------------------------------------- piezas compartidas forward / step
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.long, torch.int):
            x = x.long()
        return self.emb(x)

    def _init_hc(self, ref: torch.Tensor, batch: int):
        """(h0,c0) [n_layers,B,H] para state=None: aprendido si learned_init."""
        if self.learned_init:
            h = self.h0.to(device=ref.device, dtype=ref.dtype).expand(self.n_layers, batch, self.hidden)
            c = self.c0.to(device=ref.device, dtype=ref.dtype).expand(self.n_layers, batch, self.hidden)
            return h, c
        # Dos tensores DISTINTOS a proposito: devolver el mismo objeto para h y
        # para c es un alias que hoy no rompe nada (todos los consumidores copian
        # con .contiguous()) pero convierte cualquier escritura in-place futura
        # sobre h en una corrupcion silenciosa de c.
        shape = (self.n_layers, batch, self.hidden)
        return (torch.zeros(shape, device=ref.device, dtype=ref.dtype),
                torch.zeros(shape, device=ref.device, dtype=ref.dtype))

    def _check_state(self, state, x: torch.Tensor):
        """Valida la forma de (h,c) y lo lleva al device de la entrada.

        Un state con MAS filas que n_layers se aceptaria en silencio (la pila
        solo consume las primeras n_layers) y se generaria desde el estado
        equivocado; aqui es un AssertionError con la forma. El dtype NO se toca
        aqui: se ajusta capa a capa en _stack, contra el dtype real de la
        entrada de cada nn.LSTM (que bajo autocast no es el mismo que el del
        embedding).
        """
        assert isinstance(state, (tuple, list)) and len(state) == 2, \
            "state debe ser (h, c), llego %r" % (type(state),)
        h, c = state
        want = (self.n_layers, int(x.size(0)), self.hidden)
        for tag, t in (("h", h), ("c", c)):
            assert tuple(t.shape) == want, (
                "state[%s] tiene forma %s, se esperaba [n_layers,B,hidden]=%s"
                % (tag, tuple(t.shape), list(want)))
        if h.device != x.device:
            h = h.to(device=x.device)
        if c.device != x.device:
            c = c.to(device=x.device)
        return (h, c)

    def _stack(self, u: torch.Tensor, state):
        """Pila residual. state = (h,c) con forma [n_layers,B,H] (convenio nn.LSTM).

        Devuelve (h [B,L,H], nuevo state). Todo lo que hay aqui aparte de la
        recurrencia es puntual en el tiempo.
        """
        if state is not None:
            state = self._check_state(state, u)
            h0, c0 = state
        else:
            h0, c0 = self._init_hc(u, u.size(0))

        h = u
        hs, cs = [], []
        for i, layer in enumerate(self.layers):
            z = self.norms[i](h)                  # pre-norm, sobre el eje de features
            z = self.drops[i](z)                  # locked: misma mascara en todo t
            st_i = (h0[i:i + 1].to(dtype=z.dtype).contiguous(),
                    c0[i:i + 1].to(dtype=z.dtype).contiguous())
            y, (h_i, c_i) = layer(z, st_i)
            # Suma residual. Bajo autocast, y puede salir en float16 (politica
            # fija de _cudnn_rnn) y h venir en float32: la promocion de tipos de
            # torch deja el flujo residual en float32, que es justo lo que
            # conviene para acumular 8 ramas.
            h = h + y
            hs.append(h_i)
            cs.append(c_i)
        return h, (torch.cat(hs, dim=0), torch.cat(cs, dim=0))

    def _run_rnn(self, u: torch.Tensor, state):
        if self.fp32_rnn:
            dev = "cuda" if u.is_cuda else "cpu"
            with torch.autocast(device_type=dev, enabled=False):
                st = None if state is None else (state[0].float(), state[1].float())
                return self._stack(u.float(), st)
        return self._stack(u, state)

    def _head_from_stack(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """[B,L,H] + embedding [B,L,d] -> logits [B,L,V]. Puntual en el tiempo."""
        z = self.drop_out(self.norm_f(h))
        z = self.out_proj(z)
        z = self.norm_head(z + self.skip_gate * s)     # skip global
        return self.head(z)

    def _run(self, x: torch.Tensor, state):
        s = self.drop_emb(self._embed(x))              # [B,L,d]
        u = self.in_proj(s)                            # [B,L,H]
        h, st = self._run_rnn(u, state)
        return self._head_from_stack(h, s), st

    # ----------------------------------------------------------- API publica
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x Long[B,L] -> logits Float[B,L,V] del token x[t+1]."""
        assert x.dim() == 2, "se esperaba x [B,L], llego %s" % (tuple(x.shape),)
        return self._run(x, None)[0]

    def forward_state(self, x: torch.Tensor, state=None):
        """Como forward pero arrastrando estado: (logits, state).

        Sirve para BPTT truncado y para cebar el muestreo con un prefijo en una
        sola llamada (es lo que hace src/sampling.py). Nota sobre el dropout: en
        train() cada llamada muestrea mascaras nuevas, asi que encadenar ventanas
        equivale a remuestrear la mascara por ventana (lo estandar); si se quiere
        una mascara constante a lo largo de toda la secuencia larga, hay que
        pasarla a mano a los LockedDropout.
        """
        assert x.dim() == 2, "se esperaba x [B,L], llego %s" % (tuple(x.shape),)
        return self._run(x, state)

    def supports_state(self) -> bool:
        return True

    def step(self, x_t: torch.Tensor, state=None):
        """Un token: x_t Long[B,1] (o [B]) -> (logits [B,1,V], state). Coste O(1)."""
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(1)
        assert x_t.dim() == 2 and x_t.size(1) == 1, \
            "step espera x_t [B,1], llego %s" % (tuple(x_t.shape),)
        return self._run(x_t, state)

    def init_state(self, batch_size: int, device=None, dtype=torch.float32):
        """Estado inicial explicito; equivale exactamente a pasar state=None."""
        dev = device if device is not None else self.emb.weight.device
        ref = torch.zeros((), device=dev, dtype=dtype)
        h, c = self._init_hc(ref, int(batch_size))
        return (h.contiguous(), c.contiguous())

    @staticmethod
    def detach_state(state):
        """Corta el grafo entre ventanas de BPTT truncado."""
        if state is None:
            return None
        h, c = state
        return (h.detach(), c.detach())

    # -------------------------------------------------- config recomendada
    @staticmethod
    def default_config(**over) -> Config:
        """27.01 M parametros: 8 capas de 640, d_model=512, dropout 0.15.

        n_layers=8 es la profundidad por defecto del laboratorio y aqui SI se
        puede usar porque los residuales y la LayerNorm la hacen entrenable.
        H=640 y no 1024 es lo que impone el presupuesto: 8 capas de 1024 son
        68.3 M. A igualdad de parametros con la linea base (3x1024 = 23.7 M),
        este reparto cambia anchura por profundidad, que es lo que interesa
        cuando el contexto es corto: con 70 tokens el cuello de botella no es
        cuanta historia cabe en el estado (cabe toda) sino cuanta computacion no
        lineal se aplica a cada token.

        dropout 0.15 > 0.1 de la linea base: el dropout variacional con mascara
        fija es mas agresivo a igualdad de p (apaga la unidad durante toda la
        secuencia), pero tambien regulariza de verdad, y el modo de fallo que
        hay que combatir aqui es el sobreajuste a texturas densas del corpus.
        """
        base = dict(name="deep_lstm", model="deep_lstm", family="token",
                    d_model=512, hidden=640, n_layers=8, dropout=0.15,
                    tie_weights=True, seq_len=1024, batch_size=16)
        base.update(over)
        return Config(**base)


# ---------------------------------------------------------------------------
# verificaciones propias del modulo
# ---------------------------------------------------------------------------
def check_step_equivalence(model: DeepLSTM, batch: int = 3, L: int = 64,
                           device: str = "cpu", tol: float = 1e-4,
                           verbose: bool = True) -> bool:
    """forward(x) == recorrer step() token a token == trocear con forward_state.

    Es la prueba que legitima el muestreo O(1): si no coincidieran, generar con
    step() estaria muestreando de un modelo distinto del que se entreno con
    forward(). Se comprueban tres caminos:
      a) step() token a token desde state=None,
      b) forward_state por trozos IRREGULARES (7,1,20,3,...), que es lo que hace
         un muestreador que ceba con un prefijo y luego va de uno en uno,
      c) arrancar de init_state() explicito en vez de state=None (verifica que el
         estado inicial aprendido entra por los dos caminos).
    Modelo en eval(): con dropout activo la comparacion no tendria sentido.
    """
    was_training = model.training
    orig_device = next(model.parameters()).device
    model = model.to(device).eval()
    torch.manual_seed(0)
    # Desde 0, NO desde 3: PAD/BOS/EOS son entradas reales (TokenWindows rellena
    # con PAD la cola de las ventanas de val/test, y sampling.py alimenta PAD a
    # las secuencias ya terminadas del lote). Si step() y forward() divergieran
    # solo en esos tres simbolos, un test que empieza en 3 no lo veria.
    x = torch.randint(0, model.vocab_size, (batch, L), device=device)

    def _sweep():
        with torch.no_grad():
            full = model(x)                                     # [B,L,V]

            st, outs = None, []                                 # token a token
            for t in range(L):
                lg, st = model.step(x[:, t:t + 1], st)
                outs.append(lg)
            seq = torch.cat(outs, dim=1)

            st, outs, i = None, [], 0                           # trozos irregulares
            for n in (7, 1, 20, 3, L):
                n = min(n, L - i)
                if n <= 0:
                    break
                lg, st = model.forward_state(x[:, i:i + n], st)
                outs.append(lg)
                i += n
            chunk = torch.cat(outs, dim=1)

            init = model.init_state(batch, device=x.device)      # estado explicito
            expl, _ = model.forward_state(x, init)
        return (full, (full - seq).abs().max().item(),
                (full - chunk).abs().max().item(),
                (full - expl).abs().max().item())

    tf32_cudnn = torch.backends.cudnn.allow_tf32
    tf32_mm = torch.backends.cuda.matmul.allow_tf32
    try:
        # Con TF32 las GEMM de cuDNN usan 10 bits de mantisa y cuDNN elige
        # algoritmos distintos segun la longitud (pasada completa frente a pasos
        # de longitud 1), lo que sube la diferencia a ~1e-3 sin que haya ninguna
        # discrepancia de modelo. Se mide sin TF32 y se informa con TF32.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        full, d_step, d_chunk, d_init = _sweep()
        if device != "cpu" and (tf32_cudnn or tf32_mm):
            torch.backends.cudnn.allow_tf32 = tf32_cudnn
            torch.backends.cuda.matmul.allow_tf32 = tf32_mm
            _, s32, c32, i32 = _sweep()
            if verbose:
                print("  [equivalencia] (informativo, TF32 activado) step=%.3e "
                      "trozos=%.3e init=%.3e" % (s32, c32, i32))
    finally:
        torch.backends.cudnn.allow_tf32 = tf32_cudnn
        torch.backends.cuda.matmul.allow_tf32 = tf32_mm
        model.to(orig_device)
        model.train(was_training)

    ok = (d_step < tol and d_chunk < tol and d_init < tol
          and bool(torch.isfinite(full).all().item()))
    if verbose:
        print("  [equivalencia] device=%-4s max|forward-step|=%.3e  "
              "max|forward-trozos|=%.3e  max|forward-init_state|=%.3e  tol=%.0e -> %s"
              % (device, d_step, d_chunk, d_init, tol, "OK" if ok else "FALLO"))
    return ok


def check_strict_causality(model: DeepLSTM, batch: int = 2, L: int = 24,
                           device: str = "cpu", verbose: bool = True) -> bool:
    """Causalidad mas dura que check_token_causality, por dos vias.

    1) POSICION A POSICION. El test del laboratorio reescribe TODO el sufijo de
       golpe y admite < 1e-4 de diferencia en el pasado. Aqui se perturba UN
       SOLO token x[:,t] y se exige que los logits de todas las posiciones < t
       sean identicos BIT A BIT (torch.equal). Se mira ademas la posicion t-1,
       que es la primera que caeria con un desplazamiento de una casilla, y se
       exige que el logit EN t SI cambie (si no, el modelo estaria ignorando su
       entrada y pasaria el test por vacuidad).
    2) POR GRADIENTE, CON CERO EXACTO. Un hook sobre el embedding captura su
       salida como hoja derivable; se retropropaga desde logits[:,t] y se exige
       d logits[:,t] / d emb[:,t'] == 0.0 exacto para todo t' > t. El gradiente
       recorre TODOS los caminos del grafo (tambien el skip global), incluidos
       los que una perturbacion discreta podria no despertar.
    """
    assert L >= 4, "hacen falta L >= 4 posiciones para sondear"
    # Se barren TODAS las posiciones 1..L-1, no una muestra: el fichero anunciaba
    # "posicion a posicion" mientras sondeaba 5 posiciones de 24. Un off-by-one
    # que solo afectase a, por ejemplo, t=7 habria pasado desapercibido.
    was_training = model.training
    orig_device = next(model.parameters()).device
    model = model.to(device).eval()
    torch.manual_seed(0)
    V = model.vocab_size
    x = torch.randint(3, V, (batch, L), device=device)
    try:
        with torch.no_grad():
            base = model(x)
        probes = list(range(1, L))
        viol, worst_prev, min_at_t = 0, 0.0, float("inf")
        for t in probes:
            for _ in range(3):
                x2 = x.clone()
                new = torch.randint(3, V, (batch,), device=device)
                new = torch.where(new == x2[:, t], (new + 7) % V, new).clamp(3, V - 1)
                x2[:, t] = new
                with torch.no_grad():
                    pert = model(x2)
                if not torch.equal(base[:, :t], pert[:, :t]):
                    viol += 1
                worst_prev = max(worst_prev,
                                 (base[:, t - 1] - pert[:, t - 1]).abs().max().item())
                min_at_t = min(min_at_t, (base[:, t] - pert[:, t]).abs().max().item())

        holder = {}

        def _hook(_mod, _inp, out):
            e = out.detach().clone().requires_grad_(True)
            holder["e"] = e
            return e

        hk = model.emb.register_forward_hook(_hook)
        try:
            grads = []
            for t in sorted({t for t in (0, 1, 2, L // 3, L // 2, L - 2, L - 1)
                             if 0 <= t < L}):
                model.zero_grad(set_to_none=True)
                holder.clear()
                model(x)[:, t].sum().backward()
                g = holder["e"].grad.abs().sum(-1)              # [B,L]
                g_fut = g[:, t + 1:].max().item() if t + 1 < L else 0.0
                grads.append((t, g_fut, g[:, t].max().item()))
        finally:
            hk.remove()
            model.zero_grad(set_to_none=True)
    finally:
        model.to(orig_device)
        model.train(was_training)

    ok_grad = all(gf == 0.0 and gt > 0.0 for _, gf, gt in grads)
    ok = viol == 0 and min_at_t > 0.0 and ok_grad
    if verbose:
        print("  [estricto] TODAS las posiciones t=1..%d (%d sondas x 3 perturbaciones): "
              "logits<t identicos BIT A BIT en %d/%d pruebas"
              % (L - 1, len(probes), len(probes) * 3 - viol, len(probes) * 3))
        print("             max|delta| en t-1=%.3e (exige 0 exacto)  "
              "min|delta| en t=%.3e (exige >0)" % (worst_prev, min_at_t))
        print("             gradiente: max|dlogits[t]/demb[t'>t]| = %s  (exige 0.0 exacto)"
              % ", ".join("t=%d:%.1e" % (t, gf) for t, gf, _ in grads))
        print("             -> %s" % ("OK, causalidad estricta" if ok else "FUGA CAUSAL"))
    return ok


def check_locked_dropout(model: DeepLSTM, batch: int = 4, L: int = 32,
                         p: float = 0.5, verbose: bool = True) -> bool:
    """Demuestra NUMERICAMENTE que el dropout es variacional (locked).

    Tres comprobaciones, todas con el modelo en train():
      1) Sobre el modulo: con entrada de unos, la salida en t=0 es identica a la
         de cualquier otro t (misma mascara). Se contrasta con nn.Dropout con la
         misma p, que debe FALLAR esa igualdad (si no fallara, la comparacion no
         probaria nada).
      2) End-to-end: un hook sobre uno de los LockedDropout de la pila captura
         entrada y salida DURANTE UN forward() real. El patron de unidades
         anuladas (salida == 0) debe ser el mismo en los L pasos, y el factor de
         escala de las unidades vivas debe ser 1/(1-p).
      3) En eval() el modulo es la identidad exacta.
    """
    was_training = model.training
    ps = [m.p for m in model.modules() if isinstance(m, LockedDropout)]
    for m in model.modules():
        if isinstance(m, LockedDropout):
            m.p = float(p)
    try:
        model.train()
        torch.manual_seed(0)
        H = model.hidden
        dev = next(model.parameters()).device

        # --- 1) modulo: misma mascara en el tiempo, y contraste con nn.Dropout ---
        ld = model.drops[0]
        ones = torch.ones(batch, L, H, device=dev)
        y = ld(ones)
        same_t = all(torch.equal(y[:, 0], y[:, t]) for t in range(L))
        rows_differ = not torch.equal(y[0, 0], y[1, 0])         # mascara por secuencia
        nd = nn.Dropout(p).train()
        y2 = nd(ones)
        std_same_t = all(torch.equal(y2[:, 0], y2[:, t]) for t in range(L))

        # --- 2) end-to-end dentro de forward() ---
        cap = {}

        def _hook(_m, inp, out):
            cap["in"] = inp[0].detach().clone()
            cap["out"] = out.detach().clone()

        hk = model.drops[-1].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(torch.randint(3, model.vocab_size, (batch, L), device=dev))
        finally:
            hk.remove()
        a, b = cap["in"], cap["out"]
        zero_pat = (b == 0)                                     # [B,L,H]
        e2e_same_t = all(torch.equal(zero_pat[:, 0], zero_pat[:, t]) for t in range(L))
        alive = ~zero_pat & (a.abs() > 1e-6)
        scale = (b[alive] / a[alive])
        scale_ok = bool((scale - 1.0 / (1.0 - p)).abs().max().item() < 1e-5)
        frac_off = float(zero_pat.float().mean())
        n_pat = len({tuple(zero_pat[i, 0].tolist()) for i in range(batch)})

        # --- 3) eval() = identidad ---
        model.eval()
        ident = torch.equal(ld(ones), ones)
    finally:
        for m, q in zip([m for m in model.modules() if isinstance(m, LockedDropout)], ps):
            m.p = q
        model.train(was_training)

    ok = same_t and rows_differ and (not std_same_t) and e2e_same_t and scale_ok and ident
    if verbose:
        print("  [locked dropout] p=%.2f  B=%d L=%d H=%d" % (p, batch, L, H))
        print("    1) modulo  : salida(t=0)==salida(t) para los %d pasos -> %s   "
              "mascaras distintas entre secuencias del batch -> %s" % (L, same_t, rows_differ))
        print("       control : nn.Dropout(p=%.2f) da la misma mascara en t -> %s "
              "(debe ser False: es dropout por paso)" % (p, std_same_t))
        print("    2) forward(): patron de unidades anuladas identico en los %d pasos -> %s"
              % (L, e2e_same_t))
        print("       fraccion anulada=%.3f (esperada %.2f)  escala de las vivas=%.6f "
              "(esperada %.6f) -> %s  patrones distintos en el batch=%d/%d"
              % (frac_off, p, float(scale.mean()), 1.0 / (1.0 - p), scale_ok, n_pat, batch))
        print("    3) eval()  : identidad exacta -> %s" % ident)
        print("    -> %s" % ("OK, dropout variacional" if ok else "FALLO"))
    return ok


def check_depth_helps(n_layers: int = 8, width: int = 256, batch: int = 4, L: int = 96,
                      steps: int = 60, lr: float = 1e-3, device: str = "cpu",
                      verbose: bool = True) -> bool:
    """Mide que los residuales + LayerNorm son los que hacen entrenable la pila.

    Entrena DOS modelos con la misma semilla, la misma anchura y EXACTAMENTE los
    mismos parametros, memorizando un lote fijo:
      * DeepLSTM tal cual (pre-norm + residual),
      * una ablacion que apila las mismas capas a pelo (sin norm, sin residual).
    Se informa la curva de perdida y, sobre todo, la norma del gradiente que
    llega a W_hh de la PRIMERA capa frente a la ultima: si la profundidad
    estuviera rota, el gradiente de abajo seria mucho menor que el de arriba.
    Devuelve True si la version con residuales acaba con perdida menor.
    """

    class _NoResidual(DeepLSTM):
        """Ablacion: h = LSTM_i(dropout(h)), sin LayerNorm y sin suma residual."""

        name = "ablacion_sin_residual"

        def _stack(self, u, state):
            h0, c0 = self._init_hc(u, u.size(0)) if state is None else state
            h, hs, cs = u, [], []
            for i, layer in enumerate(self.layers):
                z = self.drops[i](h)
                y, (h_i, c_i) = layer(z, (h0[i:i + 1].to(z.dtype).contiguous(),
                                          c0[i:i + 1].to(z.dtype).contiguous()))
                h = y
                hs.append(h_i)
                cs.append(c_i)
            return h, (torch.cat(hs, dim=0), torch.cat(cs, dim=0))

    dev = torch.device(device)
    torch.manual_seed(0)
    x = torch.randint(3, int(VOCAB_SIZE), (batch, L + 1), device=dev)
    lossf = nn.CrossEntropyLoss()
    out = {}
    for cls in (DeepLSTM, _NoResidual):
        torch.manual_seed(1234)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = DeepLSTM.default_config(d_model=width, hidden=width,
                                          n_layers=n_layers, dropout=0.0)
            m = cls(cfg).to(dev).train()
        opt = torch.optim.AdamW(m.parameters(), lr=lr)
        curve = []
        for s in range(steps):
            loss = lossf(m(x[:, :-1]).reshape(-1, m.vocab_size), x[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if s in (0, steps // 3, 2 * steps // 3, steps - 1):
                curve.append((s, float(loss.detach())))
        lossf(m(x[:, :-1]).reshape(-1, m.vocab_size), x[:, 1:].reshape(-1)).backward()
        g0 = m.layers[0].weight_hh_l0.grad.norm().item()
        gN = m.layers[-1].weight_hh_l0.grad.norm().item()
        out[cls.name] = (curve, g0, gN, m.n_params())
        if verbose:
            print("    %-22s %.2f M par  loss: %s" % (
                cls.name, m.n_params() / 1e6,
                "  ".join("s=%d:%.3f" % (s, l) for s, l in curve)))
            print("    %-22s |grad| W_hh capa 0=%.3e  capa %d=%.3e  "
                  "cociente ultima/primera=%.1f"
                  % ("", g0, m.n_layers - 1, gN, gN / max(g0, 1e-30)))
        del m, opt
    ok = out["deep_lstm"][0][-1][1] < out["ablacion_sin_residual"][0][-1][1]
    if verbose:
        print("    -> %s" % ("OK: con residuales la pila de %d capas entrena, sin ellos no"
                             % n_layers if ok else "FALLO: la ablacion entrena igual o mejor"))
    return ok


def check_prefix_invariance(model: DeepLSTM, L_max: int = 1024,
                            lengths=(1, 17, 70, 200, 512), batch: int = 2,
                            device: str = "cpu", tol: float = 1e-4,
                            verbose: bool = True) -> bool:
    """Mide la propiedad que hace a este modelo candidato para prefijos cortos.

    El problema declarado del laboratorio es el mismatch entre entrenar con
    ventanas de 1024 tokens y evaluar con prefijos de ~70. Para un RNN la funcion
    que se aplica en la posicion t no depende de cuanto contexto venga DESPUES ni
    de cuanta ventana se haya reservado, asi que:

        forward(x[:, :L])  ==  forward(x)[:, :L]

    Se MIDE en vez de afirmarlo. NO es igualdad bit a bit: el backend recurrente
    elige algoritmo segun la longitud y queda un residuo de redondeo de float32
    del orden de 1e-06. Lo que se exige es que ese residuo sea despreciable
    frente a la escala de los logits (tol=1e-4, la misma con la que se acepta la
    equivalencia forward/step). Un transformer cuya normalizacion dependa de la
    longitud de la ventana no pasa esta prueba por construccion.
    """
    was_training = model.training
    orig_device = next(model.parameters()).device
    model = model.to(device).eval()
    torch.manual_seed(0)
    x = torch.randint(0, model.vocab_size, (batch, L_max), device=device)
    rows, ok = [], True
    try:
        with torch.no_grad():
            full = model(x)
            for L in lengths:
                if L > L_max:
                    continue
                part = model(x[:, :L])
                d = (full[:, :L] - part).abs().max().item()
                rows.append((L, d, bool(torch.equal(full[:, :L], part)),
                             full[:, :L].abs().max().item()))
                ok &= d < tol
    finally:
        model.to(orig_device)
        model.train(was_training)
    if verbose:
        for L, d, eq, sc in rows:
            print("    L=%-5d forward(x[:,:L]) vs forward(x[:,:%d])[:,:L]: max|delta|="
                  "%.3e  |logit|max=%.2f  igualdad bit a bit=%s" % (L, L_max, d, sc, eq))
        print("  [prefijo] el contexto largo no cambia la funcion en las posiciones "
              "iniciales, tol=%.0e -> %s" % (tol, "OK" if ok else "FALLO"))
    return ok


def check_shapes(model: DeepLSTM, lengths=(1, 17, 70, 1024), batch: int = 2,
                 device: str = "cpu", verbose: bool = True) -> bool:
    """Formas y finitud para varias longitudes, incluida la entrada todo-PAD."""
    was_training = model.training
    orig_device = next(model.parameters()).device
    model = model.to(device).eval()
    ok = True
    try:
        with torch.no_grad():
            for L in lengths:
                x = torch.randint(3, model.vocab_size, (batch, L), device=device)
                y = model(x)
                good = tuple(y.shape) == (batch, L, model.vocab_size) and bool(
                    torch.isfinite(y).all().item())
                ok &= good
                if verbose:
                    print("    L=%-5d -> %s  finito=%s  |logit|max=%.2f  %s"
                          % (L, tuple(y.shape), bool(torch.isfinite(y).all().item()),
                             y.abs().max().item(), "OK" if good else "FALLO"))
            yp = model(torch.zeros(1, 16, dtype=torch.long, device=device))
            good = bool(torch.isfinite(yp).all().item())
            ok &= good
            if verbose:
                print("    entrada todo-PAD (16 tokens) -> finito=%s  %s"
                      % (good, "OK" if good else "FALLO"))
    finally:
        model.to(orig_device)
        model.train(was_training)
    return ok


def bench_train(model: DeepLSTM, batch: int = 16, L: int = 1024, device: str = "cuda",
                amp: str = "bf16", iters: int = 20, warmup: int = 5) -> float:
    """Paso de entrenamiento completo (fwd + bwd + clip + optimizador).

    Se informa MEDIANA, MEJOR y p90, no la media: esta maquina comparte la GPU
    con otros procesos del laboratorio y la media se contamina con los picos de
    contencion. Si mejor/mediana se aleja de 1, la tirada mide la contencion y no
    el modelo. Se sincroniza CUDA en cada iteracion y se cronometra con
    perf_counter (time.time en Windows tiene ~15.6 ms de resolucion).
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("  [bench] sin GPU, omitido")
        return 0.0
    dev = torch.device(device)
    was_training = model.training
    orig_device = next(model.parameters()).device
    model = model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(3, model.vocab_size, (batch, L + 1), device=dev)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    use_amp = dtype is not None and device == "cuda"
    lossf = nn.CrossEntropyLoss(ignore_index=PAD)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    obs_dtype, gnorm, loss_val, times = None, float("nan"), float("nan"), []
    for i in range(warmup + iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
            logits = model(x[:, :-1])
            obs_dtype = logits.dtype
        loss = lossf(logits.reshape(-1, model.vocab_size).float(), x[:, 1:].reshape(-1))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
            gnorm, loss_val = float(gn.detach()), float(loss.detach())
    times.sort()
    med, best = times[len(times) // 2], times[0]
    p90 = times[min(len(times) - 1, int(0.9 * len(times)))]
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else 0.0
    print("  [bench %s] batch=%d L=%d amp=%s n=%d -> mediana %.3f it/s (%.1f ms/it)  "
          "mejor %.3f it/s  p90 %.3f it/s  pico VRAM=%.2f GiB"
          % (device, batch, L, amp if use_amp else "off", len(times), 1.0 / med,
             med * 1e3, 1.0 / best, 1.0 / p90, peak))
    print("             tokens/s=%.0f  dtype de logits=%s  loss=%.4f (ln 155=%.4f)  "
          "|grad|=%.3f finito=%s  contencion(mejor/mediana)=%.2fx"
          % (batch * L / med, obs_dtype, loss_val,
             float(torch.log(torch.tensor(155.0))), gnorm, gnorm == gnorm, med / best))
    model.to(orig_device)
    model.train(was_training)
    del opt, x
    if device == "cuda":
        torch.cuda.empty_cache()
    return 1.0 / med


def bench_step(model: DeepLSTM, batch: int = 16, n_tok: int = 200,
               device: str = "cpu", prefix: int = 70) -> float:
    """Muestreo con step() tras cebar un prefijo corto: tokens/s por secuencia."""
    dev = torch.device(device)
    if device == "cuda" and not torch.cuda.is_available():
        return 0.0
    was_training = model.training
    orig_device = next(model.parameters()).device
    model = model.to(dev).eval()
    with torch.no_grad():
        pre = torch.randint(3, model.vocab_size, (batch, prefix), device=dev)
        lg, st = model.forward_state(pre)                      # cebado O(L) una vez
        x = lg[:, -1].argmax(-1, keepdim=True)
        for _ in range(5):                                     # calentamiento
            lg, st2 = model.step(x, st)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        lg, st = model.forward_state(pre)
        x = lg[:, -1].argmax(-1, keepdim=True)
        for _ in range(n_tok):
            lg, st = model.step(x, st)
            x = lg[:, -1].argmax(-1, keepdim=True)
        if device == "cuda":
            torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print("  [muestreo %s] prefijo=%d + %d tokens, batch=%d -> %.0f tokens/s por "
          "secuencia (%.2f ms/token)" % (device, prefix, n_tok, batch, n_tok / dt,
                                         dt / n_tok * 1e3))
    model.to(orig_device)
    model.train(was_training)
    if device == "cuda":
        torch.cuda.empty_cache()
    return n_tok / dt


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "tests")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    try:
        from tests.test_causality import check_token_causality, check_shapes_token
        _CAUS_SRC = "tests.test_causality (import contractual)"
    except ModuleNotFoundError:
        # En este entorno puede haber un paquete 'tests' instalado que tape el
        # directorio tests/ del repo; se carga el archivo del repo por RUTA.
        import importlib.util

        _p = ROOT / "tests" / "test_causality.py"
        _spec = importlib.util.spec_from_file_location("_repo_test_causality", _p)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        check_token_causality = _mod.check_token_causality
        check_shapes_token = _mod.check_shapes_token
        _CAUS_SRC = str(_p)

    torch.manual_seed(1234)
    cfg = DeepLSTM.default_config()
    model = DeepLSTM(cfg)
    ok_all = True

    print("=" * 78)
    print(model.param_report())
    print("  d_model=%d H=%d n_layers=%d dropout=%.2f tie_weights=%s V=%d "
          "supports_state=%s learned_init=%s"
          % (model.d_model, model.hidden, model.n_layers, model.dropout_p,
             model.tie_weights, model.vocab_size, model.supports_state(),
             model.learned_init))
    parts = (("emb", model.emb), ("in_proj", model.in_proj), ("lstm", model.layers),
             ("norms", model.norms), ("norm_f", model.norm_f),
             ("out_proj", model.out_proj), ("norm_head", model.norm_head),
             ("head", model.head))
    for tag, mod in parts:
        n = sum(q.numel() for q in mod.parameters())
        print("    %-9s %10d  (%5.2f M)%s"
              % (tag, n, n / 1e6,
                 "  [atado al emb]" if tag == "head" and model.tie_weights else ""))
    n_init = sum(q.numel() for q in (model.h0, model.c0)) if model.learned_init else 0
    print("    %-9s %10d  (h0,c0 aprendidos) + 1 (skip_gate)" % ("init", n_init))
    ana = analytic_params(model.vocab_size, model.d_model, model.hidden,
                          model.n_layers, model.tie_weights, model.learned_init)
    same = ana == model.n_params()
    ok_all &= same
    print("  recuento analitico=%d  real=%d  coinciden=%s" % (ana, model.n_params(), same))
    lo, hi = PARAM_BUDGET
    inb = lo <= model.n_params() <= hi
    ok_all &= inb
    print("  presupuesto [%.0f, %.0f] M -> %s" % (lo / 1e6, hi / 1e6,
                                                  "DENTRO" if inb else "FUERA"))

    print("\n-- ajuste automatico de anchura con el Config() por defecto del laboratorio --")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m_def = DeepLSTM(Config())
        msgs = [str(x.message).split(".")[0] for x in w]
    print("  Config() (hidden=1024, n_layers=8) -> H=%d, %.2f M parametros"
          % (m_def.hidden, m_def.n_params() / 1e6))
    for m in msgs:
        print("  aviso: %s." % m)
    ok_all &= lo <= m_def.n_params() <= hi
    del m_def

    print("\n-- tests/test_causality.py --  (cargado de: %s)" % _CAUS_SRC)
    ok_shape = check_shapes_token(model, L=48)
    ok_caus = check_token_causality(model, L=48)
    ok_all &= ok_shape and ok_caus

    print("\n-- causalidad estricta (bit a bit + gradiente cero exacto) --")
    ok_strict = check_strict_causality(model, L=24)
    ok_all &= ok_strict

    print("\n-- equivalencia forward / step (muestreo O(1)) --")
    ok_eq = check_step_equivalence(model, batch=3, L=64)
    ok_all &= ok_eq

    print("\n-- formas L = 1, 17, 70, 1024 --")
    ok_sh = check_shapes(model, lengths=(1, 17, 70, 1024))
    ok_all &= ok_sh

    print("\n-- dropout variacional (misma mascara en todos los pasos) --")
    ok_ld = check_locked_dropout(model, batch=4, L=32, p=0.5)
    ok_all &= ok_ld

    print("\n-- la profundidad entrena gracias a los residuales (ablacion medida) --")
    ok_depth = check_depth_helps(n_layers=8, width=256, steps=60)
    ok_all &= ok_depth

    print("\n-- invariancia a la longitud del prefijo (el caso que se evalua) --")
    ok_pref = check_prefix_invariance(model, L_max=1024)
    ok_all &= ok_pref

    print("\n-- config alternativa del contrato: d_model=256 n_layers=4 n_heads=4 "
          "tie_weights=False --")
    # Esta rama NO estaba ejercitada: con tie_weights=False cambian el padding_idx
    # del embedding, la cabeza (deja de ser el mismo tensor que el embedding) y la
    # rama del recuento analitico. Es codigo distinto y hay que probarlo.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        alt = DeepLSTM(Config(d_model=256, n_layers=4, n_heads=4, tie_weights=False))
        alt_msgs = [str(a.message).split(".")[0] for a in w]
    print("  %s" % alt.param_report())
    for a in alt_msgs:
        print("  aviso: %s." % a)
    ana_alt = analytic_params(alt.vocab_size, alt.d_model, alt.hidden, alt.n_layers,
                              alt.tie_weights, alt.learned_init)
    untied = alt.head.weight.data_ptr() != alt.emb.weight.data_ptr()
    print("  recuento analitico=%d real=%d coinciden=%s  cabeza NO atada=%s  "
          "padding_idx=%s  fila PAD del embedding nula=%s"
          % (ana_alt, alt.n_params(), ana_alt == alt.n_params(), untied,
             alt.emb.padding_idx,
             bool(alt.emb.weight[PAD].abs().max().item() == 0.0)))
    ok_alt = (ana_alt == alt.n_params() and untied and alt.emb.padding_idx == PAD
              and lo <= alt.n_params() <= hi
              and check_shapes(alt, lengths=(1, 17, 70), verbose=True)
              and check_step_equivalence(alt, batch=2, L=48)
              and check_strict_causality(alt, L=20, verbose=True)
              and check_prefix_invariance(alt, L_max=256, lengths=(1, 17, 70)))
    ok_all &= ok_alt
    print("  -> %s" % ("OK, la config alternativa cumple el contrato" if ok_alt
                       else "FALLO"))
    del alt

    print("\n-- config minima (el modelo tiene que funcionar tambien pequeno) --")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        small = DeepLSTM(Config(d_model=128, hidden=192, n_layers=3, dropout=0.1,
                                seq_len=128))
    print("  %s" % small.param_report())
    ok_small = (check_shapes(small, lengths=(1, 5, 70), verbose=True)
                and check_step_equivalence(small, batch=2, L=32)
                and check_strict_causality(small, L=16, verbose=True))
    ok_all &= ok_small

    print("\n-- coste --")
    bench_train(model, batch=2, L=128, device="cpu", amp="off", iters=6, warmup=2)
    bench_step(model, batch=1, n_tok=100, device="cpu", prefix=70)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print("  [gpu] libre %.2f GiB de %.2f GiB" % (free / 2 ** 30, total / 2 ** 30))
        # 2.6 GiB y no 4: el pico REAL medido a batch 16 / L=1024 / bf16 es
        # 2.08 GiB. El umbral anterior hacia que el bench se saltara solo en una
        # GPU donde SI cabia, que es como se acabo publicando una cifra de
        # contencion en vez de una medida. Si aun asi no cabe, salta el except.
        if free / 2 ** 30 > 2.6:
            try:
                bench_train(model, batch=16, L=1024, device="cuda", amp="bf16",
                            iters=20, warmup=5)
                bench_step(model, batch=16, n_tok=200, device="cuda", prefix=70)
            except torch.cuda.OutOfMemoryError as e:
                print("  [gpu] sin memoria: %s" % str(e).splitlines()[0])
        else:
            print("  [gpu] OCUPADA por otros procesos: quedan %.2f GiB libres y el "
                  "modelo necesita 2.08 GiB de pico a batch 16 / L=1024 / bf16. "
                  "No se puede medir ahora; vale la cifra de CPU." % (free / 2 ** 30))

    print("\n" + "=" * 78)
    print("RESULTADO: %s" % ("TODO PASA" if ok_all else "HAY FALLOS"))
    sys.exit(0 if ok_all else 1)
