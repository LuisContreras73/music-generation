"""
Encoder BIDIRECCIONAL del prefijo + decoder CAUSAL de la continuacion.

Por que esta arquitectura y no otro decoder-only
------------------------------------------------
La prueba final del laboratorio no es "predice el siguiente token de una pieza
cualquiera": es "aqui tienes 5 s (100 pasos, ~70 tokens) de una melodia
MONOFONICA conocida, sigue tocando". En ese escenario el prefijo es FIJO y esta
COMPLETO antes de generar nada. Sobre el prefijo se puede mirar hacia ADELANTE
sin violar la causalidad, porque ese futuro ya existe y no se esta prediciendo:
condicionar en x[0:n] no es lo mismo que predecir x[0:n].

Un decoder causal puro se prohibe a si mismo esa informacion. Cuando llega a la
posicion 70 su representacion INTERNA del token 3 sigue siendo la que construyo
sin haber oido los tokens 4..69: sabe que notas hubo (la atencion causal en la
posicion 70 alcanza todo el prefijo) pero cada token del prefijo esta
representado de forma "ciega a su propio futuro". Un encoder bidireccional
produce, para cada posicion del prefijo, un vector que YA sabe como termina el
motivo: donde esta la cadencia, cual es la nota larga final, si la frase cierra
en la tonica o queda suspendida. La atencion cruzada del decoder consulta esos
vectores. Es exactamente la estructura del prompt acustico de VALL-E (Wang et
al., 2023) y de los seq2seq clasicos (Bahdanau et al., 2015): encoder
bidireccional sobre lo dado, decoder causal sobre lo que hay que producir.

LO QUE NO SE HACE: un BiLSTM sobre TODA la secuencia seria fuga causal pura.
La bidireccionalidad vive ENCERRADA en los primeros n_prefix tokens y en
ningun otro sitio.

El reto de compatibilidad con el contrato del laboratorio
---------------------------------------------------------
El contrato es forward(x: Long[B,L]) -> Float[B,L,V]: una sola secuencia, sin
sitio donde pasar "esto es prefijo y esto es continuacion". Convenio adoptado:

    x[:, :n_prefix]   -> prefijo      (lo lee el encoder bidireccional)
    x[:, n_prefix:]   -> continuacion (la genera el decoder causal)

La consecuencia que se suele dar por inevitable es que los logits DENTRO del
prefijo queden invalidos (dependerian del futuro del prefijo). AQUI NO PASA, y
es una decision de diseno deliberada: la atencion cruzada esta CERRADA para las
consultas con t < n_prefix. Las posiciones del prefijo se predicen con el
camino puramente causal del decoder (auto-atencion enmascarada, sin mirar el
encoder); solo las posiciones de la continuacion abren la atencion cruzada.
Con ese cierre:

    logits[:, t],  t <  n_prefix  depende solo de x[:, :t+1]            (causal)
    logits[:, t],  t >= n_prefix  depende de x[:, :t+1] por el camino causal
                                  Y de x[:, :n_prefix] bidireccionalmente,
                                  pero n_prefix <= t, asi que x[:, :n_prefix]
                                  esta CONTENIDO en x[:, :t+1].

Es decir: el modelo cumple la causalidad estricta del laboratorio EN TODAS las
posiciones, pasa tests/test_causality.py tal cual, y todos sus logits son
predicciones validas. La verosimilitud sobre la ventana completa SI es una
factorizacion autoregresiva legitima y por tanto comparable en bits/paso con
los demas modelos. Matiz honesto: en la region del prefijo el modelo esta
"amputado" (sin atencion cruzada), asi que ahi sera algo PEOR que un
decoder-only equivalente; su ventaja esta en la continuacion, que es lo que se
evalua. La comparacion informativa es bits/paso restringido a t >= n_prefix.

loss_mask: PARA EL INTEGRADOR
-----------------------------
    m = model.loss_mask(L)            -> Bool[L], True donde cuenta la perdida
    lg = logits[:, m]; tg = target[:, m]
    loss = F.cross_entropy(lg.reshape(-1, V), tg.reshape(-1), ignore_index=0)

Devuelve True solo en la continuacion (t >= n_prefix efectivo del ultimo
forward; ver model.last_n_prefix, que refleja el jitter de entrenamiento).
Recomendacion:

  * ENTRENAMIENTO: se PUEDE entrenar con TODAS las posiciones sin cometer
    ninguna incorreccion (ver arriba: no hay logits invalidos) y de hecho
    conviene, porque asi el tronco causal del decoder tambien aprende. El
    atributo de clase prefix_logits_valid = True documenta que no hay ninguna
    obligacion de enmascarar. Usar loss_mask en entrenamiento solo si se quiere
    concentrar toda la capacidad en el escenario de evaluacion.
  * EVALUACION comparativa contra los otros modelos: usar loss_mask para
    reportar bits/paso de la CONTINUACION. Ese es el numero comparable en
    igualdad de condiciones (los demas modelos tambien pueden evaluarse solo en
    t >= n_prefix). El bits/paso sobre la ventana COMPLETA de 1024 no es una
    comparacion justa para este modelo y no deberia usarse como veredicto.

Longitud de prefijo variable (el mismatch de contexto)
------------------------------------------------------
La sospecha (1) del laboratorio es que los modelos fallan porque se entrenan con
ventanas de 1024 tokens y en inferencia reciben 70. Aqui se ataca por tres vias:

  * JITTER DE PREFIJO: en train() n_prefix se sortea POR LOTE en
    [n_prefix_min, n_prefix_max] = [24, 192] por defecto. El modelo nunca
    aprende una frontera fija; aprende a continuar a partir de un resumen
    bidireccional de longitud variable, y en particular ve muchisimas veces la
    longitud exacta de la prueba (~70). En eval() no hay sorteo: se usa
    self.n_prefix, o el argumento explicito de forward.
  * POSICIONES RELATIVAS (RoPE) en toda la auto-atencion del decoder: no hay
    ninguna tabla de posiciones absolutas entrenada con indices 0..1023 que
    luego se use con 0..70. Lo unico absoluto que ve el decoder es un embedding
    de SEGMENTO de dos entradas (prefijo / continuacion), que es justo la senal
    que hace falta: "aqui acaba lo dado y empieza lo que tengo que inventar".
  * El encoder es RECURRENTE: no tiene ninguna nocion de longitud maxima; un
    prefijo de 70 y uno de 192 se procesan con el mismo mecanismo.

Por que BiLSTM como encoder y no un transformer bidireccional
--------------------------------------------------------------
Se eligio BiLSTM (2 capas, d_model/2 unidades por sentido, concatenadas a
d_model):
  1. El prefijo es CORTO (70-192 tokens) y de longitud VARIABLE. Una recurrencia
     no necesita codificacion posicional ninguna, asi que no hay nada que
     extrapolar entre 70 y 192; un transformer bidireccional necesitaria su
     propio esquema posicional y reintroduciria el problema que se intenta
     eliminar.
  2. El unico aporte real del encoder es la DIRECCION HACIA ATRAS (el decoder ya
     puede atender causalmente a los tokens del prefijo). El paso backward de un
     BiLSTM es la forma mas directa y barata de "haber oido el final del
     motivo": su estado en la posicion i resume exactamente x[i:n_prefix].
  3. Coste irrelevante: 70-192 pasos secuenciales de una LSTM de 256 unidades
     son ruido frente a 1024 posiciones de atencion del decoder.
  4. Evidencia del propio laboratorio: el mejor gen_score (80.8) lo tiene el
     LSTM, no el transformer. El sesgo inductivo recurrente le sienta bien a
     esta tokenizacion NOTE_ON/SHIFT, donde lo que manda es el orden local.
  5. Parametros: 3.15 M para las dos capas bidireccionales, que dejan sitio para
     un decoder profundo dentro del presupuesto de 20-30 M.

Diseno completo
---------------
    x [B,L] (Long)
      |-- x[:, :n_p] --> Embedding compartido --> BiLSTM(2 capas x 256 x 2 sent.)
      |                    --> LayerNorm --> Linear --> memoria [B,n_p,d]
      |                    --> mean-pool + max-pool --> vector global g [B,2d]
      |                    --> FiLM(g) = (gamma, beta) para la continuacion
      |
      '-- x --> Embedding compartido + embedding de SEGMENTO
                  --> FiLM aplicado SOLO en t >= n_p
                  --> n_dec x [ auto-atencion CAUSAL con RoPE
                                (+ atencion CRUZADA a la memoria, solo en las
                                   consultas t >= n_p, una capa de cada
                                   cross_every)
                                + FFN GELU ]
                  --> LayerNorm --> Linear atado al embedding --> logits [B,L,V]

La atencion cruzada (y el FiLM) se calculan SOLO sobre el trozo de consultas
t >= n_p y el resultado se concatena con ceros para el prefijo: no es una
mascara multiplicada por cero, es que esas posiciones literalmente no
participan en el calculo. Por eso la independencia es exacta BIT A BIT y no
"aproximada hasta 1e-7".

Hiperparametros propios (todos por getattr; src/config.py NO se toca)
    n_prefix       96      frontera por defecto en eval / inferencia
    n_prefix_min   24      limite inferior del jitter en train()
    n_prefix_max  192      limite superior del jitter en train()
    prefix_jitter  True    activa el sorteo de n_prefix durante entrenamiento
    enc_layers     2       capas del BiLSTM
    enc_hidden     d_model/2   unidades POR SENTIDO
    n_dec          n_layers-2  capas del decoder (6 con la config por defecto)
    cross_every    2       una capa con atencion cruzada de cada 2 (indices
                           impares: la ultima capa SIEMPRE lleva cruzada)

Coste medido (RTX 4070 SUPER 12.9 GB, config por defecto, 26.62 M parametros,
autocast bf16, fwd+bwd+clip+step, 10 pasos por repeticion tras 5-6 de
calentamiento)
    batch 16, L=1024 : 9.93 it/s (101 ms/it), pico 2.84 GiB  <-- mejor medida,
                       mediana 9.87 it/s (medicion limpia, sin dispersion)
    batch 24, L=1024 : pico 4.06 GiB
    CPU, batch 4, L=256 : 1.45 it/s (0.50 con la maquina cargada)
El PICO DE MEMORIA es el numero fiable y es reproducible: 2.84 GiB con batch 16
y seq_len=1024, menos de la mitad que los 5.21 GiB del Music Transformer. Cabe
de sobra en 12.9 GiB.
AVISO SOBRE LAS it/s: la GPU del laboratorio la comparten varios agentes y la
cifra es MUY inestable. Nueve repeticiones de la MISMA medida (batch 16,
L=1024) dieron 9.93, 5.86, 3.76, 2.88, 2.75, 2.38, 2.04, 1.67 y 1.02 it/s segun
lo que hubiera corriendo al lado; con 8-9 GiB ocupados por otros procesos
Windows/WDDM pagina y el tiempo se multiplica. El 9.93 (mediana 9.87 en esa
misma toma, es decir medida limpia y sin dispersion interna) es la unica tomada
con la tarjeta despejada de principio a fin, y corresponde a la version del
encoder con nn.LSTM fusionado, identica en calculo y en parametros a la actual
(ver PrefixBiEncoder). Interpretacion honesta: el modelo esta en el mismo orden
de magnitud que el Music Transformer (6.06 it/s con la GPU ociosa) y
probablemente por encima -- es coherente con su estructura, porque RoPE usa la
ruta rapida is_causal=True de SDPA en vez de un attn_mask float de 268 MiB por
capa y el decoder tiene 6 capas en vez de 8 -- pero con esta contencion NO se
puede afirmar un factor exacto. Lo que si es reproducible es el pico de memoria:
2.84-2.85 GiB en las nueve tomas.

USO EN GENERACION (importante para el integrador)
-------------------------------------------------
    model.set_prefix_len(len(tokens_prefijo))   # p.ej. 70, ANTES de generar
    model.eval()
    ... bucle de muestreo normal sobre forward(x) con x creciente ...
Si no se llama a set_prefix_len se usa n_prefix=96: sigue siendo causal y
correcto, pero los primeros 96-70 = 26 tokens generados saldrian del camino
causal sin atencion cruzada y ademas entrarian dentro del "prefijo" que lee el
encoder (lo cual sigue siendo causal, porque para t >= 96 esos tokens ya son
pasado). Funciona, pero se pierde parte de la ventaja del diseno.

LIMITACION DECLARADA
--------------------
Este es un modelo para el escenario PREFIJO + CONTINUACION. Es un LM valido de
proposito general (la factorizacion es correcta en todas las posiciones), pero
su capacidad esta deliberadamente sesgada hacia t >= n_prefix; en la region del
prefijo se comporta como un decoder-only mas pequeno y sin condicionar. No se
pretende que gane en bits/paso sobre la ventana completa de 1024: se pretende
que gane en la continuacion condicionada, que es lo que mide gen_score.

Contrato: TokenARModel, forward(x: Long[B,L]) -> Float[B,L,V] (logits de x[t+1]).
Causalidad estricta verificada: perturbar x[:, t] deja IDENTICOS BIT A BIT todos
los logits en posiciones < t, para todo t (dentro y fuera del prefijo).
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__" and __package__ in (None, ""):
    # Permite ejecutar la autoverificacion con:
    #     python src/models/prefix_encoder.py
    import sys as _sys_boot
    from pathlib import Path as _Path_boot

    _root_boot = _Path_boot(__file__).resolve().parents[2]
    for _p_boot in (str(_root_boot), str(_root_boot / "src")):
        if _p_boot not in _sys_boot.path:
            _sys_boot.path.insert(0, _p_boot)

# --- importaciones robustas: el modulo se usa como parte del paquete models
# pero tambien debe poder cargarse suelto teniendo src/ en sys.path ---
try:
    from .base import TokenARModel
except ImportError:                                      # pragma: no cover
    try:
        from models.base import TokenARModel
    except ImportError:
        from src.models.base import TokenARModel

try:
    from data.tokenizer import VOCAB_SIZE
except ImportError:                                      # pragma: no cover
    try:
        from src.data.tokenizer import VOCAB_SIZE
    except ImportError:
        VOCAB_SIZE = 155


# ---------------------------------------------------------------------------
# Posiciones rotatorias (RoPE)
# ---------------------------------------------------------------------------
def rope_cache(L: int, head_dim: int, device, base: float = 10000.0):
    """Devuelve (cos, sin) de forma [L, head_dim/2] en float32.

    RoPE (Su et al., 2021) rota cada par de canales de q y k un angulo
    proporcional a la posicion. El producto q_i . k_m resulta ser funcion de
    (i - m) y no de i y m por separado: la posicion entra de forma RELATIVA sin
    ninguna tabla entrenada y sin longitud maxima. Es justo la propiedad que
    interesa aqui, donde se entrena con L=1024 y se infiere con L~70.
    """
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(L, device=device, dtype=torch.float32)
    ang = torch.outer(t, inv)                     # [L, half]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B,H,L,hd] -> rotado. cos/sin: [L, hd/2] (convenio mitad-mitad)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos.to(x.dtype)[None, None]
    s = sin.to(x.dtype)[None, None]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def _heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """[B,L,D] -> [B,H,L,D/H]."""
    B, L, D = x.shape
    return x.view(B, L, n_heads, D // n_heads).transpose(1, 2)


def _unheads(x: torch.Tensor) -> torch.Tensor:
    """[B,H,L,hd] -> [B,L,H*hd]."""
    B, H, L, hd = x.shape
    return x.transpose(1, 2).reshape(B, L, H * hd)


# ---------------------------------------------------------------------------
# Encoder bidireccional del prefijo
# ---------------------------------------------------------------------------
class PrefixBiEncoder(nn.Module):
    """BiLSTM sobre los n_p primeros tokens -> memoria por posicion + vector global.

    Entrada:  e [B, n_p, d_model]  (ya embebido, el embedding se comparte con el
              decoder para no duplicar 79 k parametros y para que encoder y
              decoder hablen del mismo espacio de tokens).
    Salida:   mem [B, n_p, d_model]   estados por posicion (claves/valores de la
                                      atencion cruzada)
              g   [B, 2*d_model]      resumen global (mean-pool || max-pool)

    NO hay causalidad que respetar AQUI: este modulo solo ve el prefijo, que en
    el escenario de evaluacion esta dado por completo. La causalidad global la
    garantiza quien lo usa (ver PrefixEncoderDecoder.forward).

    POR QUE LA PILA NO ES UN UNICO nn.LSTM(num_layers=2, dropout=0.1)
    -----------------------------------------------------------------
    Son capas de 1 nivel con un nn.Dropout explicito entre ellas (misma
    regularizacion, mismos parametros: la capa 2 recibe 2*hidden = d_model, que
    es justo lo que recibiria dentro del modulo fusionado). El motivo es un BUG
    del entorno, no una preferencia: en esta build (torch 2.11.0+cu128, Python
    3.14, Windows) un nn.LSTM con num_layers >= 2 Y dropout > 0 sobre CUDA hace
    que el PROCESO REVIENTE AL SALIR con STATUS_STACK_BUFFER_OVERRUN
    (0xC0000409) en cuanto se ha hecho un backward; el estado de dropout de la
    RNN de cuDNN no se libera bien. Reproducido en 4 lineas sin nada de este
    laboratorio:

        m = nn.LSTM(512, 256, 2, bidirectional=True, dropout=0.1).cuda().train()
        y, _ = m(torch.randn(4, 96, 512, device="cuda")); y.sum().backward()
        # -> el interprete sale con codigo -1073740791

    Con dropout=0.0 el mismo codigo sale con 0. La pila explicita evita por
    completo esa ruta de cuDNN (es tambien lo que hace src/models/lstm_baseline.py,
    por otro motivo: poder inspeccionar el estado capa a capa).
    """

    def __init__(self, d_model: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList()
        d_in = d_model
        for _ in range(max(1, layers)):
            self.layers.append(nn.LSTM(input_size=d_in, hidden_size=hidden,
                                       num_layers=1, batch_first=True,
                                       bidirectional=True))
            d_in = 2 * hidden
        self.inter_drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, e: torch.Tensor):
        h = e
        for i, lstm in enumerate(self.layers):
            if h.is_cuda:
                # Evita el aviso de memoria no contigua de cuDNN cuando el
                # modulo se ha movido de dispositivo tras construirse.
                lstm.flatten_parameters()
            if i > 0:
                h = self.inter_drop(h)
            h, _ = lstm(h)                         # [B, n_p, 2*hidden]
        h = self.ln(h)
        mem = self.drop(self.proj(h))              # [B, n_p, d_model]
        g = torch.cat([mem.mean(dim=1), mem.amax(dim=1)], dim=-1)   # [B, 2*d]
        return mem, g


# ---------------------------------------------------------------------------
# Bloques del decoder
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Auto-atencion causal con RoPE. Estrictamente causal por is_causal=True."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, attn_dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        assert (d_model // n_heads) % 2 == 0, "RoPE necesita head_dim par"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.out._is_residual_out = True
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = apply_rope(_heads(q, self.n_heads), cos, sin)
        k = apply_rope(_heads(k, self.n_heads), cos, sin)
        v = _heads(v, self.n_heads)
        p = self.attn_dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        return self.drop(self.out(_unheads(y)))


class PrefixCrossAttention(nn.Module):
    """Atencion cruzada a la memoria del encoder, SOLO para las consultas de la
    continuacion.

    El troceado es la pieza critica del diseno: se reciben ya solo las consultas
    con t >= n_p (x_cont). Las posiciones del prefijo no entran en el calculo,
    de modo que su salida no es "cero multiplicado" sino AUSENCIA de camino.
    Sin RoPE aqui: la memoria ya lleva el orden dentro del propio estado
    recurrente, y una posicion relativa entre "token 500 de la continuacion" y
    "token 30 del prefijo" no significa nada estable cuando el prefijo cambia
    de longitud.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.out._is_residual_out = True
        self.ln_q = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = attn_dropout

    def forward(self, x_cont: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        q = _heads(self.q(self.ln_q(x_cont)), self.n_heads)
        k, v = self.kv(mem).chunk(2, dim=-1)
        k = _heads(k, self.n_heads)
        v = _heads(v, self.n_heads)
        p = self.attn_dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=False)
        return self.drop(self.out(_unheads(y)))


class DecoderBlock(nn.Module):
    """Pre-LN: auto-atencion causal -> (atencion cruzada opcional) -> FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 attn_dropout: float, cross: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, attn_dropout)
        self.cross = PrefixCrossAttention(d_model, n_heads, dropout, attn_dropout) if cross else None
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ff[2]._is_residual_out = True

    def forward(self, h: torch.Tensor, cos, sin, mem, n_p: int) -> torch.Tensor:
        h = h + self.attn(self.ln1(h), cos, sin)
        if self.cross is not None and mem is not None:
            # Solo las consultas de la continuacion. El prefijo se deja intacto
            # (ni siquiera se le suma cero: se concatena tal cual).
            h_pre, h_cont = h[:, :n_p], h[:, n_p:]
            h_cont = h_cont + self.cross(h_cont, mem)
            h = torch.cat([h_pre, h_cont], dim=1) if n_p > 0 else h_cont
        h = h + self.ff(self.ln2(h))
        return h


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class PrefixEncoderDecoder(TokenARModel):
    """Encoder bidireccional del prefijo + decoder causal con atencion cruzada."""

    name = "prefix_enc"

    #: Los logits del prefijo TAMBIEN son validos (la atencion cruzada esta
    #: cerrada para t < n_prefix), asi que enmascarar la perdida es OPCIONAL.
    #: Ver el docstring del modulo, seccion "loss_mask".
    prefix_logits_valid = True

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_model = int(cfg.d_model)
        n_heads = int(cfg.n_heads)
        d_ff = int(getattr(cfg, "d_ff", 4 * d_model))
        dropout = float(getattr(cfg, "dropout", 0.1))
        attn_dropout = float(getattr(cfg, "attn_dropout", 0.0))

        # --- hiperparametros propios: NO son campos de Config (no se toca
        # src/config.py). Se leen con getattr, asi que para cambiarlos hay que
        # fijarlos sobre la instancia de cfg antes de construir el modelo.
        self.n_prefix = int(getattr(cfg, "n_prefix", 96))
        self.n_prefix_min = int(getattr(cfg, "n_prefix_min", 24))
        self.n_prefix_max = int(getattr(cfg, "n_prefix_max", 192))
        self.prefix_jitter = bool(getattr(cfg, "prefix_jitter", True))
        enc_layers = int(getattr(cfg, "enc_layers", 2))
        enc_hidden = int(getattr(cfg, "enc_hidden", d_model // 2))
        n_dec = int(getattr(cfg, "n_dec", max(2, int(cfg.n_layers) - 2)))
        cross_every = int(getattr(cfg, "cross_every", 2))

        self.vocab_size = int(VOCAB_SIZE)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_dec = n_dec
        self.cross_every = cross_every
        self.tie_weights = bool(getattr(cfg, "tie_weights", True))
        #: n_prefix realmente usado en el ultimo forward (util con jitter).
        self.last_n_prefix = self.n_prefix

        # --- piezas compartidas ---
        self.tok_emb = nn.Embedding(self.vocab_size, d_model)
        # Embedding de SEGMENTO: 0 = prefijo, 1 = continuacion. Es la unica
        # senal posicional absoluta del modelo y marca la frontera.
        self.seg_emb = nn.Embedding(2, d_model)
        self.drop = nn.Dropout(dropout)

        # --- encoder bidireccional (solo prefijo) ---
        self.encoder = PrefixBiEncoder(d_model, enc_hidden, enc_layers, dropout)
        # FiLM: el vector global modula el embedding de entrada de la
        # continuacion. Inicializado a CERO -> al principio es la identidad y no
        # desestabiliza el arranque; el modelo decide cuanto usarlo.
        self.film = nn.Linear(2 * d_model, 2 * d_model)

        # --- decoder causal ---
        # La atencion cruzada va en una capa de cada cross_every, empezando por
        # el final: con n_dec=6 y cross_every=2 son las capas 1, 3 y 5, de forma
        # que la ULTIMA capa siempre condiciona (lo mas cerca posible del head).
        self.cross_layers = [i for i in range(n_dec) if (n_dec - 1 - i) % cross_every == 0]
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout, attn_dropout,
                         cross=(i in self.cross_layers))
            for i in range(n_dec)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = None if self.tie_weights else nn.Linear(d_model, self.vocab_size, bias=False)

        self._rope_cache = None       # (L, device, cos, sin)

        self.apply(self._init_weights)
        std_res = 0.02 / math.sqrt(2 * max(1, n_dec))
        for m in self.modules():
            if getattr(m, "_is_residual_out", False):
                nn.init.normal_(m.weight, mean=0.0, std=std_res)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.seg_emb.weight)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        # Deliberadamente NO toca nn.LSTM: la inicializacion uniforme por
        # defecto de PyTorch (U(-1/sqrt(h), 1/sqrt(h))) es la adecuada para las
        # puertas; un normal(0, 0.02) las dejaria casi muertas.
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # -- utilidades publicas ------------------------------------------------
    def set_prefix_len(self, n: int) -> None:
        """Fija la frontera prefijo/continuacion para eval e inferencia.

        LLAMAR ANTES DE GENERAR con la longitud REAL del prefijo en tokens
        (p.ej. 70 para los 100 pasos de la prueba). No afecta al entrenamiento
        si prefix_jitter esta activo.
        """
        self.n_prefix = max(0, int(n))
        self.last_n_prefix = self.n_prefix

    def resolve_n_prefix(self, L: int, n_prefix: int | None = None) -> int:
        """Frontera efectiva para una secuencia de longitud L.

        * n_prefix explicito -> se respeta (recortado a [0, L]).
        * train() con prefix_jitter -> sorteo en [n_prefix_min, n_prefix_max],
          recortado a L-1 para que quede al menos una posicion de continuacion.
        * eval() -> self.n_prefix, recortado a L.
        Si L <= n_prefix no hay continuacion: el modelo degenera con elegancia
        en un decoder causal sin condicionar (ver forward).
        """
        if n_prefix is not None:
            return max(0, min(int(n_prefix), L))
        if self.training and self.prefix_jitter and L > 2:
            lo = max(1, min(self.n_prefix_min, L - 1))
            hi = max(lo, min(self.n_prefix_max, L - 1))
            # torch.randint usa el RNG global -> reproducible con la semilla del
            # laboratorio y sin estado propio que guardar en el checkpoint.
            return int(torch.randint(lo, hi + 1, (1,)).item())
        return max(0, min(self.n_prefix, L))

    def loss_mask(self, L: int, n_prefix: int | None = None,
                  device=None) -> torch.Tensor:
        """Bool[L]: True en las posiciones que cuentan para la perdida.

        True solo en la CONTINUACION (t >= n_prefix efectivo). Si no se pasa
        n_prefix se usa self.last_n_prefix, es decir la frontera del ULTIMO
        forward (importante con el jitter de entrenamiento: llamar DESPUES del
        forward, no antes).

        Uso tipico en el bucle de entrenamiento/evaluacion:
            logits = model(x)                       # [B,L,V]
            m = model.loss_mask(x.shape[1], device=x.device)
            loss = F.cross_entropy(logits[:, m].reshape(-1, V),
                                   y[:, m].reshape(-1), ignore_index=0)

        Recordatorio: para este modelo enmascarar es OPCIONAL en entrenamiento
        (prefix_logits_valid=True); es OBLIGATORIO si se quiere un bits/paso
        comparable con los otros modelos en el escenario prefijo+continuacion.
        Si L <= n_prefix la mascara sale toda a False (no hay continuacion):
        el llamante debe saltarse ese lote o usar toda la ventana.
        """
        n_p = self.last_n_prefix if n_prefix is None else int(n_prefix)
        n_p = max(0, min(n_p, L))
        m = torch.zeros(L, dtype=torch.bool, device=device)
        m[n_p:] = True
        return m

    def _rope(self, L: int, device):
        c = self._rope_cache
        if c is None or c[0] < L or c[1] != device:
            cos, sin = rope_cache(max(L, 1024), self.d_model // self.n_heads, device)
            self._rope_cache = (cos.shape[0], device, cos, sin)
            c = self._rope_cache
        return c[2][:L], c[3][:L]

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor, n_prefix: int | None = None) -> torch.Tensor:
        """x: Long[B,L] -> logits Float[B,L,V] del token siguiente.

        El argumento n_prefix es OPCIONAL (el contrato del laboratorio solo
        exige forward(x)); sirve para forzar la frontera sin tocar el estado del
        modulo, p.ej. en los tests de causalidad.
        """
        assert x.dim() == 2, f"se esperaba [B,L], llego {tuple(x.shape)}"
        B, L = x.shape
        assert L >= 1, "secuencia vacia"
        n_p = self.resolve_n_prefix(L, n_prefix)
        self.last_n_prefix = n_p
        n_cont = L - n_p

        e = self.tok_emb(x)

        # --- encoder bidireccional sobre el prefijo -------------------------
        # Solo se ejecuta si hay continuacion que condicionar. Si n_cont == 0 la
        # memoria no la consumiria nadie y el encoder se queda sin gradiente,
        # que es exactamente lo correcto: en ese caso el modelo es un decoder
        # causal puro (caso limite documentado).
        mem = None
        if n_p >= 1 and n_cont >= 1:
            mem, g = self.encoder(e[:, :n_p])
            gamma, beta = self.film(g).chunk(2, dim=-1)          # [B,d] cada uno

        # --- entrada del decoder -------------------------------------------
        seg = torch.zeros(L, dtype=torch.long, device=x.device)
        seg[n_p:] = 1
        h = e + self.seg_emb(seg)[None]
        if mem is not None:
            # FiLM solo en la continuacion: mismo criterio que la atencion
            # cruzada, el prefijo no puede recibir informacion del encoder.
            h_pre, h_cont = h[:, :n_p], h[:, n_p:]
            h_cont = h_cont * (1.0 + gamma[:, None]) + beta[:, None]
            h = torch.cat([h_pre, h_cont], dim=1)
        h = self.drop(h)

        cos, sin = self._rope(L, x.device)
        for blk in self.blocks:
            h = blk(h, cos, sin, mem, n_p)
        h = self.ln_f(h)
        if self.tie_weights:
            return F.linear(h, self.tok_emb.weight)
        return self.head(h)

    # El decoder es atencional: no hay estado recurrente de tamano fijo que
    # exponer (el encoder si es recurrente, pero se re-ejecuta entero sobre el
    # prefijo, que es corto y FIJO durante toda la generacion).
    def supports_state(self) -> bool:
        return False

    @staticmethod
    def default_config():
        """Config del laboratorio tal cual: ya cae en 20-30 M (ver _selftest)."""
        try:
            from config import Config
        except ImportError:                                  # pragma: no cover
            from src.config import Config
        return Config(model="prefix_enc", family="token")


# ---------------------------------------------------------------------------
# Verificacion (ejecutar:  python src/models/prefix_encoder.py)
# ---------------------------------------------------------------------------
def _test_strict_causality(model, L: int = 96, n_prefix: int = 32, V: int = 155,
                           device: str = "cpu", verbose: bool = True) -> bool:
    """Causalidad estricta BIT A BIT, perturbando UNA sola posicion.

    PROPIEDAD COMPROBADA (la fuerte, la que de verdad importa):

        para todo t en [0, L):  cambiar SOLO x[:, t] deja logits[:, :t]
                                IDENTICOS BIT A BIT.

    Esto es mas exigente que el test del laboratorio en tres sentidos:
      * se perturba UNA posicion, no todo el sufijo (una fuga que se cancelase
        al perturbar en bloque quedaria al descubierto);
      * se exige igualdad EXACTA (torch.equal), no una tolerancia de 1e-4;
      * se recorre TODA t, incluyendo las posiciones del prefijo y la frontera.

    Incluye el caso delicado de esta arquitectura: t dentro del PREFIJO. Ahi la
    propiedad sigue siendo obligatoria, porque aunque el encoder mire hacia
    adelante DENTRO del prefijo, su salida solo la consumen posiciones
    t' >= n_prefix > t. Si la atencion cruzada estuviera abierta para las
    consultas del prefijo, este test fallaria exactamente ahi.

    Ademas se comprueba lo que SI debe ocurrir (que el modelo no ignore nada):
      * perturbar x[:, t] cambia algun logit en posiciones >= t;
      * perturbar dentro del prefijo cambia los logits de la continuacion
        (senal de que la atencion cruzada esta viva y transporta informacion).
    """
    model = model.to(device).eval()
    torch.manual_seed(0)
    x = torch.randint(3, V, (2, L), device=device)
    with torch.no_grad():
        base = model(x, n_prefix=n_prefix)
    ok = True
    worst_t, worst = -1, 0.0
    n_mudo = 0
    cross_alive = False
    for t in range(L):
        x2 = x.clone()
        # token distinto garantizado (dentro del bloque NOTE_ON/SHIFT)
        x2[:, t] = 3 + (x[:, t] - 3 + 37) % (V - 3)
        with torch.no_grad():
            pert = model(x2, n_prefix=n_prefix)
        if t > 0:
            same = torch.equal(base[:, :t], pert[:, :t])
            if not same:
                d = (base[:, :t] - pert[:, :t]).abs().max().item()
                if d > worst:
                    worst, worst_t = d, t
                ok = False
        d_fut = (base[:, t:] - pert[:, t:]).abs().max().item()
        if d_fut == 0.0:
            n_mudo += 1
        if t < n_prefix and n_prefix < L:
            if (base[:, n_prefix:] - pert[:, n_prefix:]).abs().max().item() > 0:
                cross_alive = True
    if verbose:
        print(f"  [estricto] L={L} n_prefix={n_prefix}: perturbar x[:,t] una a una, "
              f"t=0..{L - 1}")
        if ok:
            print(f"    logits[:, :t] IDENTICOS BIT A BIT en las {L - 1} pruebas -> OK")
        else:
            print(f"    FUGA CAUSAL en t={worst_t}: max|delta pasado|={worst:.3e} -> FALLO")
        print(f"    posiciones cuya perturbacion no cambio nada hacia adelante: "
              f"{n_mudo} (debe ser 0 o 1; la ultima posicion no tiene efecto "
              f"visible si L-1 es el ultimo logit)")
        if n_prefix < L:
            print(f"    perturbar el PREFIJO cambia los logits de la continuacion: "
                  f"{cross_alive} (debe ser True: el encoder transporta informacion)")
        else:
            print("    no hay continuacion (n_prefix >= L): el encoder no se "
                  "ejecuta, no se exige transporte de informacion")
    if n_prefix < L:
        ok &= cross_alive
    return ok


def _test_prefix_independence(model, L: int = 96, n_prefix: int = 32, V: int = 155,
                              device: str = "cpu", verbose: bool = True) -> bool:
    """Los logits del PREFIJO no dependen del resto del prefijo (ni del encoder).

    Comprueba la consecuencia concreta del cierre de la atencion cruzada: si se
    reescribe TODA la cola del prefijo (posiciones t+1..n_prefix-1) los logits
    en 0..t siguen siendo bit a bit los mismos. Es lo que hace que los logits
    del prefijo sean predicciones VALIDAS en este modelo y no basura que haya
    que enmascarar.
    """
    model = model.to(device).eval()
    torch.manual_seed(1)
    x = torch.randint(3, V, (2, L), device=device)
    t = max(0, n_prefix // 2)
    with torch.no_grad():
        base = model(x, n_prefix=n_prefix)
    x2 = x.clone()
    x2[:, t + 1:n_prefix] = torch.randint(3, V, (2, n_prefix - t - 1), device=device)
    with torch.no_grad():
        pert = model(x2, n_prefix=n_prefix)
    same = torch.equal(base[:, :t + 1], pert[:, :t + 1])
    d_cont = (base[:, n_prefix:] - pert[:, n_prefix:]).abs().max().item()
    if verbose:
        print(f"  [prefijo] reescribir x[:, {t + 1}:{n_prefix}] deja logits[:, :{t + 1}] "
              f"identicos: {same} -> {'OK' if same else 'FALLO'}")
        print(f"    y cambia los logits de la continuacion en {d_cont:.3e} "
              f"(debe ser > 0)")
    return bool(same) and d_cont > 0


def _test_shapes(model, V: int = 155, verbose: bool = True) -> bool:
    """Formas y casos limite, incluido L <= n_prefix (sin continuacion)."""
    ok = True
    n_p = model.n_prefix
    casos = [n_p + 1, 200, 1024, n_p, n_p - 1, 1, 2, 70]
    for L in casos:
        if L < 1:
            continue
        x = torch.randint(3, V, (2, L))
        with torch.no_grad():
            y = model(x)
        fin = bool(torch.isfinite(y).all().item())
        good = tuple(y.shape) == (2, L, V) and fin
        ok &= good
        n_eff = model.last_n_prefix
        m = model.loss_mask(L)
        nota = "sin continuacion (decoder causal puro)" if L <= n_p else ""
        if verbose:
            print(f"  L={L:5d} -> {tuple(y.shape)} finito={fin} n_prefix_ef={n_eff:4d} "
                  f"posiciones en la perdida={int(m.sum()):5d} "
                  f"{'OK' if good else 'FALLO'} {nota}")
    return ok


def _test_jitter(model, V: int = 155, verbose: bool = True) -> bool:
    """En train() la frontera se sortea; en eval() es determinista."""
    torch.manual_seed(0)
    x = torch.randint(3, V, (2, 512))
    model.train()
    vistos = set()
    for _ in range(12):
        with torch.no_grad():
            model(x)
        vistos.add(model.last_n_prefix)
    model.eval()
    fijos = set()
    for _ in range(3):
        with torch.no_grad():
            model(x)
        fijos.add(model.last_n_prefix)
    ok = len(vistos) > 1 and fijos == {model.n_prefix}
    dentro = all(model.n_prefix_min <= v <= model.n_prefix_max for v in vistos)
    ok &= dentro
    if verbose:
        print(f"  train(): {len(vistos)} fronteras distintas en 12 pasos, rango "
              f"[{min(vistos)}, {max(vistos)}] (limites [{model.n_prefix_min}, "
              f"{model.n_prefix_max}]) dentro={dentro}")
        print(f"  eval() : frontera fija {fijos} == n_prefix={model.n_prefix} "
              f"-> {'OK' if ok else 'FALLO'}")
    return ok


def _test_grad_flow(model, L: int = 200, n_prefix: int = 70, V: int = 155,
                    verbose: bool = True) -> bool:
    """Con la perdida restringida a la CONTINUACION, el encoder debe recibir
    gradiente.

    Es la comprobacion de que la atencion cruzada no es decorativa: si el
    encoder quedara desconectado (o el FiLM matara la senal) los gradientes de
    la BiLSTM serian exactamente cero y el modelo seria un decoder-only con
    3 M de parametros muertos.
    """
    model = model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(3)
    x = torch.randint(3, V, (2, L + 1))
    inp, tgt = x[:, :-1], x[:, 1:]
    logits = model(inp, n_prefix=n_prefix)
    m = model.loss_mask(L, n_prefix=n_prefix)
    loss = F.cross_entropy(logits[:, m].reshape(-1, V), tgt[:, m].reshape(-1))
    loss.backward()
    g_enc = max(p.grad.abs().max().item() for p in model.encoder.parameters()
                if p.grad is not None)
    n_enc_sin = sum(1 for p in model.encoder.parameters() if p.grad is None)
    g_film = model.film.weight.grad.abs().max().item()
    ok = g_enc > 0 and n_enc_sin == 0
    if verbose:
        print(f"  perdida solo en la continuacion ({int(m.sum())} de {L} posiciones), "
              f"loss={loss.item():.3f}")
        print(f"  max|grad| en el encoder={g_enc:.3e} (debe ser > 0), parametros "
              f"del encoder sin gradiente={n_enc_sin} (debe ser 0), "
              f"max|grad| FiLM={g_film:.3e} -> {'OK' if ok else 'FALLO'}")
    model.zero_grad(set_to_none=True)
    model.eval()
    return ok


def _has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:                                        # pragma: no cover
        return False


def _gpu_busy_mib() -> float:
    """VRAM ocupada por OTROS procesos (MiB). 0 si no se puede consultar."""
    if not _has_cuda():
        return 0.0
    try:
        free, total = torch.cuda.mem_get_info()
        mine = torch.cuda.memory_reserved()
        return max(0.0, (total - free - mine) / 2 ** 20)
    except Exception:                                        # pragma: no cover
        return 0.0


def _benchmark(batch: int = 16, L: int = 1024, steps: int = 10, warmup: int = 5,
               reps: int = 4, cfg=None, device: str = "cuda") -> float:
    """it/s reales (forward + backward + clip + step). bf16 en GPU, fp32 en CPU."""
    import time
    if device == "cuda" and not _has_cuda():
        print("  benchmark: sin CUDA, se omite")
        return 0.0
    if cfg is None:
        try:
            from config import Config
        except ImportError:                                  # pragma: no cover
            from src.config import Config
        cfg = Config()
    model = PrefixEncoderDecoder(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(3, model.vocab_size, (batch, L + 1), device=device)
    inp, tgt = x[:, :-1], x[:, 1:]
    use_amp = device == "cuda"

    def one_step():
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp)
        else:
            logits = model(inp)
        loss = F.cross_entropy(logits.float().reshape(-1, model.vocab_size),
                               tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss

    for _ in range(warmup):
        loss = one_step()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            loss = one_step()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / steps)
    best, med = min(times), sorted(times)[len(times) // 2]
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device == "cuda" else 0.0
    print(f"  benchmark {device}: batch={batch} L={L} -> MEJOR {1.0 / best:.2f} it/s "
          f"({best * 1e3:.0f} ms/it) | mediana {1.0 / med:.2f} it/s | "
          f"pico memoria {peak:.2f} GiB | loss={loss.item():.3f}"
          + (f" | VRAM de otros procesos: {_gpu_busy_mib():.0f} MiB" if device == "cuda" else ""))
    del model, opt, x
    if device == "cuda":
        torch.cuda.empty_cache()
    return 1.0 / best


def _test_overfit(steps: int = 150, verbose: bool = True) -> bool:
    """Sanity de optimizacion: memorizar un lote diminuto.

    No mide calidad, mide que la arquitectura no esta muerta: con el FiLM y el
    embedding de segmento inicializados a CERO y la atencion cruzada troceada,
    un error de cableado se manifestaria como una perdida plana. Se entrena SOLO
    sobre la continuacion (loss_mask), que es el modo en que se va a usar.
    """
    try:
        from config import Config
    except ImportError:                                      # pragma: no cover
        from src.config import Config
    torch.manual_seed(0)
    cfg = Config(d_model=128, n_layers=4, n_heads=4, d_ff=256, dropout=0.0)
    cfg.n_prefix = 32
    cfg.enc_hidden = 64
    cfg.prefix_jitter = False
    m = PrefixEncoderDecoder(cfg).train()
    x = torch.randint(3, m.vocab_size, (4, 129))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    hist = []
    for i in range(steps + 1):
        lg = m(x[:, :-1])
        mask = m.loss_mask(128)
        loss = F.cross_entropy(lg[:, mask].reshape(-1, m.vocab_size),
                               x[:, 1:][:, mask].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i % 50 == 0:
            hist.append((i, loss.item()))
    ok = hist[-1][1] < 2.0 and hist[-1][1] < hist[0][1]
    if verbose:
        print("  " + "  ".join(f"paso {i}: {v:.3f}" for i, v in hist)
              + f"  -> {'OK' if ok else 'FALLO'} (debe bajar de ~5.0 a < 2.0)")
    return ok


def _test_teardown_gpu(verbose: bool = True) -> bool:
    """El proceso debe salir con codigo 0 tras un paso de entrenamiento en GPU.

    Guardia de regresion para un fallo que NO se puede ver desde dentro del
    proceso: con nn.LSTM(num_layers>=2, dropout>0) sobre CUDA, esta build de
    torch revienta al FINALIZAR el interprete (0xC0000409) despues de un
    backward, con todo el calculo ya correcto y todas las metricas impresas. Un
    entrenamiento largo terminaria bien y el proceso moriria al salir, tirando
    lo que hubiera pendiente de escribir. Ver PrefixBiEncoder.
    """
    import subprocess
    import sys
    from pathlib import Path
    if not _has_cuda():
        print("  teardown GPU: sin CUDA, se omite")
        return True
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'src')\n"
        "import torch\n"
        "from config import Config\n"
        "from models.prefix_encoder import PrefixEncoderDecoder\n"
        "import torch.nn.functional as F\n"
        "m = PrefixEncoderDecoder(Config()).cuda().train()\n"
        "opt = torch.optim.AdamW(m.parameters(), lr=1e-4)\n"
        "x = torch.randint(3, 155, (4, 257), device='cuda')\n"
        "with torch.autocast('cuda', dtype=torch.bfloat16):\n"
        "    lg = m(x[:, :-1])\n"
        "loss = F.cross_entropy(lg.float().reshape(-1, 155), x[:, 1:].reshape(-1))\n"
        "loss.backward(); opt.step()\n"
        "print('paso ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(root),
                       capture_output=True, text=True)
    ok = r.returncode == 0
    if verbose:
        print(f"  subproceso 'paso de entrenamiento en GPU + salir': codigo "
              f"{r.returncode} (debe ser 0; -1073740791 = el bug de cuDNN) "
              f"-> {'OK' if ok else 'FALLO'}")
        if not ok:
            print(f"    stderr: {r.stderr.strip()[-300:]}")
    return ok


def _load_causality_checks(root):
    """Importa tests/test_causality.py (por nombre o por ruta)."""
    try:
        from tests.test_causality import check_shapes_token, check_token_causality
        return check_shapes_token, check_token_causality
    except ImportError:
        import importlib.util
        path = root / "tests" / "test_causality.py"
        spec = importlib.util.spec_from_file_location("_test_causality_local", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"  (test_causality cargado por ruta: {path})")
        return mod.check_shapes_token, mod.check_token_causality


def _selftest() -> bool:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for p in (str(root), str(root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from config import Config
    check_shapes_token, check_token_causality = _load_causality_checks(root)

    ok = True
    torch.manual_seed(0)

    # Modelo pequeno para los tests exhaustivos (L*forward por posicion).
    cfg_s = Config(d_model=64, n_layers=4, n_heads=4, d_ff=128, dropout=0.1)
    cfg_s.n_prefix = 16
    cfg_s.enc_hidden = 32
    ms = PrefixEncoderDecoder(cfg_s).eval()
    print(f"== modelo de prueba: {ms.param_report()} "
          f"(n_dec={ms.n_dec}, capas con cruzada={ms.cross_layers}, "
          f"n_prefix={ms.n_prefix}) ==")

    print("== test del laboratorio (tests/test_causality.py, sin tocarlo) ==")
    ok &= check_shapes_token(ms, L=32, V=ms.vocab_size)
    ok &= check_token_causality(ms, L=32, V=ms.vocab_size)
    ok &= check_token_causality(ms, L=64, V=ms.vocab_size, t=20)
    ok &= check_token_causality(ms, L=17, V=ms.vocab_size, t=3)   # L > n_prefix
    ok &= check_token_causality(ms, L=12, V=ms.vocab_size, t=5)   # L < n_prefix

    print("== causalidad estricta propia (bit a bit, una posicion cada vez) ==")
    ok &= _test_strict_causality(ms, L=48, n_prefix=16, V=ms.vocab_size)
    ok &= _test_strict_causality(ms, L=24, n_prefix=23, V=ms.vocab_size)
    ok &= _test_strict_causality(ms, L=20, n_prefix=20, V=ms.vocab_size)  # sin continuacion
    print("== independencia dentro del prefijo ==")
    ok &= _test_prefix_independence(ms, L=48, n_prefix=16, V=ms.vocab_size)
    print("== flujo de gradiente hacia el encoder (modelo de prueba) ==")
    ok &= _test_grad_flow(ms, L=64, n_prefix=16, V=ms.vocab_size)

    print("== formas y casos limite (modelo de prueba, n_prefix=16) ==")
    ok &= _test_shapes(ms, V=ms.vocab_size)
    print("== jitter de la frontera ==")
    ok &= _test_jitter(ms, V=ms.vocab_size)
    ms.eval()

    print("== mascara de perdida ==")
    m = ms.loss_mask(200, n_prefix=70)
    good = (not m[:70].any().item()) and m[70:].all().item() and m.shape == (200,)
    ok &= good
    print(f"  loss_mask(200, n_prefix=70): {int(m.sum())} posiciones activas, "
          f"primera activa={int(torch.nonzero(m)[0])} -> {'OK' if good else 'FALLO'}")
    m0 = ms.loss_mask(10, n_prefix=16)
    good0 = int(m0.sum()) == 0
    ok &= good0
    print(f"  loss_mask(10, n_prefix=16) (L <= n_prefix): {int(m0.sum())} activas "
          f"-> {'OK' if good0 else 'FALLO'}")

    print("== set_prefix_len y escenario de la prueba (prefijo de 70 tokens) ==")
    ms.set_prefix_len(70)
    x = torch.randint(3, ms.vocab_size, (1, 70))
    with torch.no_grad():
        y70 = ms(x)                      # solo prefijo: aun no hay continuacion
        xg = torch.cat([x, torch.randint(3, ms.vocab_size, (1, 30))], dim=1)
        y100 = ms(xg)
    good = tuple(y70.shape) == (1, 70, ms.vocab_size) and tuple(y100.shape) == (1, 100, ms.vocab_size)
    # El paso clave de la generacion real: al crecer x se activa el encoder,
    # pero los logits de las posiciones 0..69 NO pueden moverse.
    # OJO con el criterio: aqui se comparan dos forwards con L DISTINTO (70 y
    # 100), asi que SDPA elige otro teselado y la reasociacion en coma flotante
    # cambia los ultimos bits. Eso no es causalidad, es aritmetica: el mismo
    # experimento sobre src/models/music_transformer.py da exactamente el mismo
    # residuo, 2.384e-07 (medido). La igualdad BIT A BIT solo es exigible -- y
    # se exige, en _test_strict_causality -- entre forwards de la MISMA forma.
    d = (y70 - y100[:, :70]).abs().max().item()
    same = d < 1e-5
    ok &= good and same
    print(f"  max|logits(prefijo 70) - logits(100)[:, :70]| = {d:.3e} "
          f"(tolerancia 1e-5; el Music Transformer da 2.384e-07 en el mismo test) "
          f"-> {'OK' if good and same else 'FALLO'}")
    print("    (es la coherencia que necesita el muestreo incremental: activar el "
          "encoder no reescribe el pasado)")

    print("== configuracion por defecto del laboratorio ==")
    cfg = Config()
    m = PrefixEncoderDecoder(cfg)
    n_enc = sum(p.numel() for p in m.encoder.parameters()) + m.film.weight.numel() + m.film.bias.numel()
    n_dec_p = sum(p.numel() for p in m.blocks.parameters())
    print(f"  {m.param_report()} (d_model={cfg.d_model}, n_layers={cfg.n_layers} "
          f"-> n_dec={m.n_dec}, n_heads={cfg.n_heads}, d_ff={cfg.d_ff}, "
          f"tie={cfg.tie_weights}, n_prefix={m.n_prefix})")
    print(f"  desglose: encoder+FiLM {n_enc / 1e6:.2f} M | decoder {n_dec_p / 1e6:.2f} M "
          f"| embeddings {(m.tok_emb.weight.numel() + m.seg_emb.weight.numel()) / 1e6:.2f} M "
          f"| capas con atencion cruzada {m.cross_layers}")
    in_budget = 20e6 <= m.n_params() <= 30e6
    ok &= in_budget
    print(f"  presupuesto 20-30 M -> {'OK' if in_budget else 'FUERA DE PRESUPUESTO'}")
    with torch.no_grad():
        y = m.eval()(torch.randint(3, m.vocab_size, (2, 1024)))
    print(f"  forward L=1024 (config por defecto): {tuple(y.shape)} "
          f"finito={bool(torch.isfinite(y).all().item())}")
    del y
    print("== formas y casos limite (MODELO REAL, n_prefix=96) ==")
    ok &= _test_shapes(m, V=m.vocab_size)
    print("== causalidad estricta del MODELO REAL en el regimen de la prueba ==")
    # L=120 con n_prefix=96: prefijo corto tipo evaluacion + 24 tokens de
    # continuacion. 120 forwards del modelo de 26.6 M en CPU.
    ok &= _test_strict_causality(m, L=120, n_prefix=96, V=m.vocab_size)
    print("== flujo de gradiente hacia el encoder (MODELO REAL) ==")
    ok &= _test_grad_flow(m, L=200, n_prefix=70, V=m.vocab_size)

    print("== sanity de optimizacion (memorizar un lote, solo continuacion) ==")
    ok &= _test_overfit()

    print("== benchmark CPU (config por defecto, L=256, batch 4) ==")
    _benchmark(batch=4, L=256, steps=3, warmup=2, reps=2, device="cpu")

    if _has_cuda():
        print("== causalidad en GPU con autocast bf16 ==")
        mg = PrefixEncoderDecoder(Config()).cuda().eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            xg = torch.randint(3, mg.vocab_size, (2, 256), device="cuda")
            base = mg(xg, n_prefix=96)
            for t in (40, 96, 150):
                x2 = xg.clone()
                x2[:, t] = 3 + (xg[:, t] - 3 + 37) % (mg.vocab_size - 3)
                pert = mg(x2, n_prefix=96)
                same = torch.equal(base[:, :t], pert[:, :t])
                d_f = (base[:, t:] - pert[:, t:]).float().abs().max().item()
                ok &= bool(same)
                print(f"  GPU bf16 t={t:4d}: pasado identico bit a bit={same} "
                      f"futuro delta={d_f:.3e} -> {'OK' if same else 'FUGA CAUSAL'}")
            for L in (70, 1024):
                y = mg(torch.randint(3, mg.vocab_size, (2, L), device="cuda"))
                print(f"  GPU L={L}: {tuple(y.shape)} dtype={y.dtype} "
                      f"finito={bool(torch.isfinite(y).all().item())}")
        del mg, base, pert, y, xg, x2
        torch.cuda.empty_cache()
        busy = _gpu_busy_mib()
        print(f"== benchmark GPU (VRAM ocupada por otros: {busy:.0f} MiB) ==")
        try:
            _benchmark(batch=16, L=1024)
        except torch.cuda.OutOfMemoryError:
            print("  OOM: la GPU esta ocupada por otros procesos, se omite el "
                  "benchmark GPU (ver la cifra de CPU)")
            torch.cuda.empty_cache()
        print("== salida limpia del proceso tras entrenar en GPU ==")
        ok &= _test_teardown_gpu()
    else:
        print("== sin CUDA: benchmark GPU omitido ==")

    print(f"\nRESULTADO GLOBAL: {'TODO OK' if ok else 'HAY FALLOS'}")
    return ok


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(0 if _selftest() else 1)
