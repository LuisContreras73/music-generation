"""Modern Transformer: decoder-only con la receta estandar 2024-2025.

Contraste directo con src/models/music_transformer.py (Huang et al., 2018):
misma profundidad, mismo ancho y practicamente el mismo numero de parametros,
pero cambiando las cuatro piezas que la comunidad ha ido fijando desde
entonces (PaLM / LLaMA / Gemma / Qwen / OLMo comparten hoy exactamente estas):

  1. RoPE en vez de posicion absoluta o de sesgo relativo con skewing.
  2. RMSNorm en vez de LayerNorm, en configuracion pre-norma.
  3. SwiGLU en el feed-forward en vez de una MLP con GELU.
  4. QK-norm opcional (normalizar Q y K antes del producto escalar).
Ademas: sin sesgos en las proyecciones lineales, atencion por
F.scaled_dot_product_attention(is_causal=True), pesos atados entrada/salida e
inicializacion escalada 1/sqrt(2*n_layers) en las proyecciones residuales.

Por que cada pieza, y por que importan para ESTE problema
---------------------------------------------------------
El fallo que hay que atacar: en la prueba final el modelo recibe prefijos de
~70 tokens (100 pasos, 5 s) de melodias monofonicas y degenera en bucles
(gen_score 2.0-2.4 frente a 70-83 con prefijos del corpus de 200 pasos). Dos
sospechas: mismatch de LONGITUD de contexto (entrena con 1024, infiere con 70)
y mismatch de DOMINIO (textura rala monofonica).

(1) RoPE  (Su et al., 2021, arXiv:2104.09864).
    La posicion se inyecta ROTANDO q y k en planos 2D con un angulo
    proporcional a la posicion absoluta. Como una rotacion es ortogonal,

        <R(i) q, R(j) k> = q^T R(i)^T R(j) k = q^T R(j - i) k,

    el producto query-key resultante depende SOLO de la distancia j - i. Eso es
    justo lo que codifica la musica: un motivo es el mismo en el compas 2 y en
    el 40; lo que importa es "hace 12 tokens" y no "token numero 517".
    Consecuencias directas sobre el problema de los prefijos cortos:
      * El modelo entero queda EQUIVARIANTE A TRASLACION: no hay ni un solo
        parametro indexado por posicion absoluta (a diferencia de un embedding
        absoluto aprendido, y a diferencia de la tabla E_r[0..R-1] de Huang,
        que sigue siendo una tabla aprendida con un borde en R). La funcion que
        el modelo aplica a un contexto de 70 tokens es EXACTAMENTE la misma
        tanto si esos 70 tokens estan al principio de la pieza como si estan en
        el token 900 de una ventana de entrenamiento (se demuestra en
        _selftest: forward(x, pos_offset=0) vs forward(x, pos_offset=713),
        error 0 en float64). Es decir, del mismatch entrenar-1024/inferir-70
        RoPE elimina por construccion la mitad posicional; lo que queda es
        tener menos contexto, que es informacion que sencillamente no esta.
      * No hay longitud maxima cableada: el cache de cos/sin crece bajo demanda
        y el modelo corre con L=2048 (el doble de cfg.seq_len) sin tocar pesos
        ni producir NaN. La extrapolacion de RoPE no es perfecta, pero degrada
        de forma suave en vez de romperse; aqui se verifica que funciona y que
        los logits son finitos.
      * Memoria: el skewing de Huang construye un sesgo [B,H,L,L] por capa
        (268 MiB por capa con B=16, L=1024, bf16) y obliga a pasar un attn_mask
        float a SDPA, lo que descarta el kernel rapido. RoPE solo toca q y k
        ([B,H,L,dh]) y deja usar la ruta is_causal=True.

(2) RMSNorm  (Zhang & Sennrich, 2019, arXiv:1910.07467).
    LayerNorm resta la media y divide por la desviacion tipica; RMSNorm solo
    divide por la raiz del segundo momento y no tiene sesgo. Empiricamente la
    re-centracion no aporta nada en transformers (el residual arrastra su
    propia media) y quitarla ahorra una reduccion y un parametro por canal.
    Aqui la ruta MANUAL acumula la estadistica en float32 aunque el tensor
    venga en bf16 (es la unica parte del bloque donde una suma de 512 terminos
    en bf16 se nota); la ruta fusionada F.rms_norm hace su propia acumulacion
    interna, que NO esta bajo control de este fichero. Medidas contra una
    referencia en float64 del MISMO tensor bf16: fusionada 0.9-1.3 ulp de bf16,
    manual 0.4-0.8 ulp, para dim 64/512/1408/4096 (_test_rmsnorm). Las dos estan
    dentro del ruido de bf16; en float32 y float64 son identicas bit a bit. Pre-norma (norma dentro de la rama, nunca
    sobre el camino residual) para que el residual sea una identidad limpia y
    el modelo entrene a 8 capas sin warmup agresivo.

(3) SwiGLU  (Shazeer, 2020, arXiv:2002.05202).
        FFN(x) = W_down ( SiLU(W_gate x) * W_up x )
    La puerta multiplicativa hace la unidad bilineal: puede APAGAR canales en
    funcion de la propia entrada en vez de aplicar una no linealidad fija. Con
    tres matrices en vez de dos hay que corregir la dimension oculta a
    2/3 * 4 * d_model (aqui: 2/3 * cfg.d_ff, redondeado hacia arriba a multiplo
    de 64 -> 1408 con la config del laboratorio) para conservar el conteo de
    parametros y de FLOPs de la FFN clasica. W_gate y W_up van FUSIONADAS en un
    solo nn.Linear(d_model, 2*d_hidden): mismos parametros, un GEMM en vez de
    dos.

(4) QK-norm  (Henry et al., 2020; Dehghani et al., ViT-22B, 2023; hoy estandar
    en Gemma-2/3, Chameleon, OLMo-2, Qwen3).
    Se normaliza cada vector q y k POR CABEZA antes del producto escalar. El
    modo de fallo que evita es la divergencia lenta de ||q|| ||k||: los logits
    de atencion crecen, el softmax se satura, la entropia de atencion colapsa y
    en bf16 (7 bits de mantisa) aparecen picos y NaN. Con la norma el logit
    esta acotado por |g_q|_inf * |g_k|_inf * sqrt(d_head), donde g son las
    ganancias aprendibles de las dos RMSNorm. Con g en su valor inicial 1 eso es
    sqrt(d_head)=8; medido al init sobre una capa real: 4.1 con QK-norm frente a
    13.3 sin el. OJO: g aprende y no esta acotada, asi que si las ganancias
    suben a 4 la cota sube a 64; QK-norm frena el CRECIMIENTO de ||q|| ||k||
    (que es el modo de fallo), no impone una cota absoluta. El entrenamiento en
    bf16 aguanta lr mas altos. Para nuestro problema tiene un segundo efecto util: una atencion
    saturada es una atencion casi determinista, que es exactamente el regimen
    en el que un modelo autoregresivo se engancha a un bucle copiando su propio
    pasado reciente; mantener viva la entropia de atencion es una defensa
    barata contra la degeneracion (no es una garantia, pero ataca el mecanismo).
    Se aplica ANTES de RoPE, no despues: la rotacion preserva la norma, asi que
    el orden norma->rotacion deja intacta la propiedad relativa de RoPE (se
    verifica numericamente en _selftest con qk_norm activo).
    Es opcional: cfg.qk_norm (getattr, por defecto True). NO se anade ningun
    campo a src/config.py; se fija sobre la instancia (cfg.qk_norm = False).
    CUIDADO, y aqui la comparacion con cfg.attn_dropout que habia antes en este
    docstring era FALSA en los dos sentidos: attn_dropout SI es campo de Config
    (src/config.py) y ademas no cambia el conjunto de tensores; qk_norm ni es
    campo de Config ni es inocuo, porque ANADE O QUITA q_norm/k_norm. Un
    checkpoint entrenado con cfg.qk_norm=False y recargado por la via estandar
    del laboratorio (Config.save -> Config.load -> build_model ->
    load_state_dict(strict=True)) fallaba con
        Missing key(s) in state_dict: "blocks.0.attn.q_norm.weight", ...
    porque el flag se pierde en config.json y el modelo se reconstruye con
    QK-norm. Arreglado: ModernTransformer._load_from_state_dict deduce el flag
    de las CLAVES del checkpoint y se reconfigura solo antes de copiar pesos, en
    los dos sentidos. El checkpoint manda; config.json ya no puede mentir sobre
    esto. Verificado en _test_qk_norm_checkpoint.

(5) Atencion causal con F.scaled_dot_product_attention(is_causal=True): con
    is_causal el kernel ni siquiera visita los bloques enteramente futuros
    (37.5% menos de pares (i,m) a L=1024) y no materializa la matriz de
    probabilidades para el backward.

(6) Pesos atados si cfg.tie_weights (la matriz de embedding hace de capa de
    salida: 79 k parametros menos y un gradiente mas por token) e init
    normal(0, 0.02) con las proyecciones que ESCRIBEN en el residual
    (attn.proj y ff.w_down) escaladas por 1/sqrt(2*n_layers): con 2 escrituras
    por capa la varianza del stream residual crece ~linealmente con la
    profundidad si no se corrige.

Knobs extra leidos con getattr (ninguno es campo de Config)
    cfg.qk_norm      : bool  (True)    normalizar Q y K por cabeza
    cfg.rope_theta   : float (10000.0) base de las frecuencias de RoPE
    cfg.attn_dropout : float (0.0)     dropout sobre las probabilidades de atencion

Contrato: TokenARModel, forward(x: Long[B,L]) -> Float[B,L,V] (logits de x[t+1]).
Causalidad estricta: logits[:, t] depende solo de x[:, :t+1]. Verificado por
perturbacion (test generico) y por una comprobacion BIT A BIT posicion a
posicion en _selftest.

forward() acepta un argumento opcional pos_offset (por defecto 0) que desplaza
las posiciones de RoPE. No cambia el contrato -- model(x) sigue funcionando --
y sirve para muestreo con ventana deslizante y para el test de equivariancia.

MEDIDAS (todas reales, ninguna extrapolada; python src/models/modern_transformer.py)
------------------------------------------------------------------------------
Parametros con la config por defecto del laboratorio (d_model=512, n_layers=8,
n_heads=8, d_ff=2048 -> d_hidden SwiGLU=1408, tie_weights=True, qk_norm=True):

    25.78 M  (25 779 200)  =  embedding      79 360  (atado a la salida)
                            + atencion    8 388 608  (8 x 1 048 576 = qkv+proj)
                            + SwiGLU     17 301 504  (8 x 2 162 688)
                            + RMSNorm         9 728  (8 x (2x512 residuales +
                                                      2x64 de QK-norm) + 512)
    (el desglose anterior sumaba 25 780 224: contaba los 8 x 128 del QK-norm dos
     veces, en "atencion" y en "normas". La suma de arriba da 25 779 200 exacto,
     contrastado contra un recuento independiente en _test_param_count.)
    Dentro del presupuesto 20-30 M y a la altura del Music Transformer
    (25.6 M), que es con quien hay que comparar bpt.

Entrenamiento en la RTX 4070 SUPER (bf16 autocast, fwd+bwd+clip+step, MEJOR de
5 repeticiones de 10 pasos). La GPU es COMPARTIDA con otros agentes: los
numeros solo valen cuando nvidia-smi reporta la VRAM casi libre; con la GPU
llena Windows (WDDM) pagina y el mismo benchmark cae a 0.38 it/s sin avisar.
    batch 16, L=1024 : 5.96 it/s (168 ms/it), pico 5.09 GiB   <-- objetivo
    (referencia medida por el otro agente para music_transformer en la misma
     maquina: 6.06 it/s y 5.21 GiB; su ablacion sin atencion relativa: 11.25)
El requisito "batch 16, seq_len 1024, bf16, dentro de 12.9 GiB" se cumple con
5.09 GiB de pico, menos de la mitad del presupuesto.
HONESTIDAD sobre esa cifra: se midio con la ruta MANUAL de RMSNorm, que es la
que habia cuando la GPU estuvo libre (11.5 GiB); despues se cambio al kernel
fusionado F.rms_norm, que hace estrictamente menos trabajo y da la misma
salida, pero no se ha podido volver a medir porque desde entonces la GPU esta
ocupada al 100% por otros agentes (300-700 MiB libres). Asi que 5.96 it/s es
una COTA INFERIOR de lo que hace el fichero tal y como esta.

CPU (fp32, 8 hilos; util porque la GPU esta casi siempre ocupada):
    batch  4, L=256  : 1.37 it/s
    batch 16, L= 70  : 1.33 it/s
    forward B=1, L=70 (el prefijo exacto de la prueba final): 19.8 ms

AUDITORIA ADVERSARIAL (segundo agente, re-medido tras los arreglos de abajo)
    Las cifras de GPU NO se han podido reproducir: durante 200 s de sondeo con
    nvidia-smi el maximo de VRAM libre fue 1057 MiB (uso 100% por otros
    agentes) y el benchmark de batch 16 x 1024 necesita ~5.1 GiB. Los 5.96 it/s
    quedan como NO VERIFICADOS: ni confirmados ni desmentidos. Lo que si se
    verifico en GPU, con lo que cabia: el kernel CUDA de SDPA en bf16 da
    causalidad bit a bit exacta (max|delta logits en posiciones < t| = 0.0).
    CPU re-medida en esta auditoria (fp32, 8 hilos, mejor de 3, maquina
    compartida, con el kernel fusionado de RMSNorm ya activo):
        batch  4, L=256 (fwd+bwd+clip+step) : 1.49 it/s (669 ms/it)
        batch 16, L= 70 (fwd+bwd+clip+step) : 1.09 it/s (914 ms/it)
        forward B=1, L=70                   : 32.8 ms
    El requisito de memoria (batch 16, L=1024, bf16, < 12.9 GiB) no se ha
    podido re-medir por lo mismo; el pico de 5.09 GiB reportado es plausible
    para 25.78 M parametros pero queda SIN CONFIRMAR en esta auditoria.

Sanidad de entrenamiento (_test_train_sanity, CPU, 3 capas, d_model=128):
    los 26 tensores de parametros reciben gradiente no nulo; la perdida sobre
    una secuencia fija de 128 tokens cae de 5.102 a 0.0013 en 200 pasos
    (ln 155 = 5.043 es el valor de un modelo uniforme: el init esta calibrado)
    y el prefijo de 70 tokens evaluado solo coincide con el mismo prefijo
    dentro de la secuencia larga (2.9e-6, redondeo de SDPA).

COMO USARLO PARA EL PROBLEMA DE LOS PREFIJOS CORTOS
---------------------------------------------------
La arquitectura elimina la parte POSICIONAL del mismatch (equivariancia a
traslacion demostrada abajo) y quita el freno de memoria del skewing, pero no
resuelve sola el mismatch de DOMINIO (nunca vio texturas monofonicas ralas) ni
garantiza que no haya bucles. Lo que toca combinar, y que vive fuera de este
fichero: src/augment.py::thin_voices para que el entrenamiento vea texturas
casi monofonicas, y src/sampling.py (repetition_penalty / no_repeat_ngram /
min_p) en el muestreo.
Riesgos conocidos de esta arquitectura, por orden de importancia:
  * RoPE con theta=10000 sobre una tokenizacion NOTE_ON/SHIFT no tiene por que
    ser el theta optimo: la frecuencia mas alta da una vuelta completa cada
    2*pi = 6.3 tokens (dos o tres eventos) y un motivo musical vive a 10-40
    tokens. Si el modelo sale flojo en estructura, cfg.rope_theta = 500 o 2000
    es la primera ablacion a probar (un solo numero, sin tocar el codigo).
  * El decaimiento implicito de RoPE con la distancia puede penalizar la
    dependencia larga frente al sesgo aprendido de Huang; es exactamente la
    comparacion que este modelo existe para hacer.
  * Con weight decay: conviene EXCLUIR del decay las ganancias de RMSNorm (33
    tensores de los 66 con la config por defecto: 4 normas por bloque mas la
    final; se reconocen porque son los unicos parametros de dimension 1). Si el entrenador del laboratorio aplica
    AdamW a todo por igual el efecto es pequeno pero real; es cosa del trainer,
    no de este fichero.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__" and __package__ in (None, ""):
    # Permite ejecutar la autoverificacion con:  python src/models/modern_transformer.py
    import sys as _sys_boot
    from pathlib import Path as _Path_boot

    _root_boot = _Path_boot(__file__).resolve().parents[2]
    for _p_boot in (str(_root_boot), str(_root_boot / "src")):
        if _p_boot not in _sys_boot.path:
            _sys_boot.path.insert(0, _p_boot)

# --- importaciones robustas: el modulo se usa como parte del paquete models
# (models/__init__.py hace build_model) pero tambien debe poder cargarse suelto
# teniendo src/ en sys.path ---
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
# RMSNorm
# ---------------------------------------------------------------------------
_HAS_FUSED_RMS = hasattr(F, "rms_norm")


class RMSNorm(nn.Module):
    """y = x / rms(x) * g   con rms(x) = sqrt(mean(x^2) + eps).

    Sin resta de media y sin sesgo (Zhang & Sennrich, 2019). La reduccion se
    acumula en float32 aunque la entrada sea bf16 y se devuelve en el dtype de
    entrada. Opera sobre la ULTIMA dimension, asi que sirve tanto para el
    stream residual [B,L,d_model] como para los vectores por cabeza
    [B,H,L,d_head] del QK-norm; en ningun caso mezcla informacion entre
    posiciones (irrelevante para la causalidad).

    Se usa F.rms_norm (fusionado, torch >= 2.4) cuando existe; si no, la ruta
    manual equivalente. Importa: la version manual encadena 6 operaciones sobre
    tensores de [B,L,d_model] en float32 (33 MiB cada temporal con B=16,
    L=1024) y, con 2 normas por bloque mas las 2 del QK-norm, son ~2.6 GiB de
    trafico de memoria por paso que el kernel fusionado se ahorra.
    La ganancia se castea al dtype de la entrada porque con dtypes mezclados
    torch NO despacha al kernel fusionado (avisa por consola y cae a la ruta
    lenta); es exactamente lo que autocast ya hace con los pesos de todos los
    Linear. En float32 y float64 la salida es IDENTICA bit a bit a la manual y
    en bf16 la diferencia esta en el ulp (verificado en _test_rmsnorm).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    @staticmethod
    def manual(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        dt = x.dtype
        # Se acumula en float32 si la entrada es de menor precision (bf16/fp16);
        # si ya es float64 NO se degrada a float32 (los tests de la propiedad
        # relativa de RoPE corren en float64 y .float() los truncaria a 1e-7).
        acc = x if dt == torch.float64 else x.float()
        acc = acc * torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + eps)
        return (acc * weight.to(acc.dtype)).to(dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_FUSED_RMS:
            return F.rms_norm(x, self.weight.shape, self.weight.to(x.dtype), self.eps)
        return self.manual(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
def rope_angles(L: int, d_head: int, theta: float, device, dtype,
                offset: int = 0):
    """cos/sin de RoPE para las posiciones [offset, offset+L): dos [L, d_head].

    Frecuencias  w_k = theta^(-2k/d_head),  k = 0 .. d_head/2 - 1, es decir
    periodos de 2*pi (k=0) hasta 2*pi*theta (k = d/2 - 1). Cada componente j se
    rota contra la j + d_head/2 (convenio "rotate_half" de GPT-NeoX/LLaMA), de
    ahi el cat() que duplica el angulo. El convenio de emparejamiento es
    irrelevante para la propiedad relativa mientras sea el mismo en q y en k.
    """
    assert d_head % 2 == 0, "d_head debe ser par para RoPE"
    half = torch.arange(0, d_head, 2, device=device, dtype=dtype)
    inv = torch.as_tensor(float(theta), device=device, dtype=dtype) ** (-half / d_head)
    pos = torch.arange(offset, offset + L, device=device, dtype=dtype)
    ang = torch.outer(pos, inv)                                  # [L, d_head/2]
    return torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1): el companero ortogonal de la rotacion 2D."""
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rota x [..., L, dh] con cos/sin [L, dh]. Preserva la norma de cada vector.

    cos/sin se convierten al dtype de x para no promocionar silenciosamente la
    atencion a float32 cuando el cache esta en float32 y x en bf16.
    """
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    return x * cos + rotate_half(x) * sin


class Rotary(nn.Module):
    """Cache de cos/sin compartido por todas las capas.

    * Se construye bajo demanda y crece a la siguiente potencia de dos, para no
      reasignar en cada llamada: en muestreo con ventana deslizante L cambia en
      cada paso.
    * Los buffers son NO persistentes: no entran en state_dict(), asi que un
      checkpoint no queda atado a la longitud con la que se entreno (es lo que
      permite cargar el modelo y correrlo con L mayor).
    * Si alguien pide float64 (los tests de la propiedad relativa) el cache se
      reconstruye en float64; para el resto se mantiene en float32 y se castea
      al dtype del tensor dentro de apply_rope.
    """

    def __init__(self, d_head: int, theta: float = 10000.0, init_len: int = 0):
        super().__init__()
        self.d_head = int(d_head)
        self.theta = float(theta)
        self.register_buffer("cos_cache", torch.zeros(0, self.d_head), persistent=False)
        self.register_buffer("sin_cache", torch.zeros(0, self.d_head), persistent=False)
        # Precision con la que se CALCULARON los senos: no se puede deducir del
        # dtype del buffer, porque model.double() lo convierte a float64 sin
        # recuperar los digitos que se perdieron al calcularlo en float32.
        self._built_dtype = None
        if init_len > 0:
            self._build(int(init_len), torch.device("cpu"), torch.float32)

    def _build(self, need: int, device, dtype) -> None:
        n = 1 << max(6, int(max(1, need) - 1).bit_length())      # potencia de dos >= need
        cos, sin = rope_angles(n, self.d_head, self.theta, device, dtype)
        self.cos_cache, self.sin_cache = cos, sin
        self._built_dtype = dtype

    def forward(self, L: int, offset: int = 0, device=None, dtype=torch.float32):
        # device se NORMALIZA antes de comparar. Sin esto, llamar al modulo con
        # el valor por defecto de su propia firma (device=None) o con una cadena
        # ("cuda") hacia que c.device != device fuese SIEMPRE cierto: el cache se
        # reconstruia entero en cada llamada (medido: el data_ptr cambiaba en
        # cada invocacion) y, peor, con device=None sobre un modelo en GPU se
        # reconstruia en CPU y apply_rope reventaba por mezcla de dispositivos.
        # forward() del modelo siempre pasa h.device, asi que esto solo afectaba
        # a llamadas externas (muestreo con ventana deslizante, diagnostico).
        device = self.cos_cache.device if device is None else torch.device(device)
        need = int(L) + int(offset)
        want = torch.float64 if dtype == torch.float64 else torch.float32
        c = self.cos_cache
        if c.shape[0] < need or c.device != device or self._built_dtype != want:
            self._build(need, device, want)
        return self.cos_cache[offset:offset + L], self.sin_cache[offset:offset + L]

    def extra_repr(self) -> str:
        return f"d_head={self.d_head}, theta={self.theta}, cache={self.cos_cache.shape[0]}"


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------
def swiglu_hidden(d_ff: int, multiple_of: int = 64) -> int:
    """2/3 * d_ff redondeado hacia ARRIBA a multiplo de multiple_of.

    d_ff en Config es 4*d_model, asi que esto implementa la correccion clasica
    2/3 * 4 * d_model de Shazeer 2020 / LLaMA: SwiGLU tiene tres matrices en
    vez de dos, y con el factor 2/3 los parametros y los FLOPs de la FFN
    coinciden con los de la MLP con GELU a la que sustituye.
    Con d_ff=2048: 2/3*2048 = 1365.3 -> 1408 (22*64).
    """
    h = int(2 * int(d_ff) / 3)
    m = max(1, int(multiple_of))
    return m * ((h + m - 1) // m)


class SwiGLU(nn.Module):
    """FFN con puerta: W_down( SiLU(W_gate x) * W_up x ). Sin sesgos."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.d_hidden = int(d_hidden)
        # W_gate y W_up fusionadas: mismos parametros, un solo GEMM.
        self.w_in = nn.Linear(d_model, 2 * self.d_hidden, bias=False)
        self.w_down = nn.Linear(self.d_hidden, d_model, bias=False)
        self.w_down._is_residual_out = True
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_in(x).chunk(2, dim=-1)
        return self.drop(self.w_down(F.silu(gate) * up))


# ---------------------------------------------------------------------------
# Atencion
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Auto-atencion causal multi-cabeza con RoPE y QK-norm opcional.

    Orden de operaciones (importa):
        qkv -> separar cabezas -> [QK-norm] -> RoPE(q), RoPE(k) -> SDPA causal
    La norma va antes de la rotacion porque la rotacion preserva la norma: asi
    QK-norm no rompe la equivalencia <R(i)q, R(j)k> = q^T R(j-i) k.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 qk_norm: bool = True, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser multiplo de n_heads"
        self.n_heads = int(n_heads)
        self.d_head = d_model // self.n_heads
        self.p_drop = float(attn_dropout)      # dropout sobre las probabilidades
        self.qk_norm = bool(qk_norm)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj._is_residual_out = True                 # marca para el init escalado
        self.resid_drop = nn.Dropout(dropout)
        if self.qk_norm:
            # Una ganancia por canal de cabeza, compartida por las cabezas de la
            # capa (2*d_head parametros por capa: 128 con d_head=64).
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

    def set_qk_norm(self, enabled: bool) -> None:
        """Activa o desactiva QK-norm en caliente (anade o quita q_norm/k_norm).

        Lo usa ModernTransformer._load_from_state_dict para adaptarse a lo que
        diga el CHECKPOINT, porque cfg.qk_norm no sobrevive a Config.save().
        """
        enabled = bool(enabled)
        if enabled == self.qk_norm:
            return
        if enabled:
            ref = self.qkv.weight
            self.q_norm = RMSNorm(self.d_head).to(device=ref.device, dtype=ref.dtype)
            self.k_norm = RMSNorm(self.d_head).to(device=ref.device, dtype=ref.dtype)
        else:
            for nm in ("q_norm", "k_norm"):
                self._modules.pop(nm, None)
        self.qk_norm = enabled

    def _qk(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Devuelve (q, k, v) [B,H,L,dh] ya normalizados y rotados."""
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self._qk(x, cos, sin)
        p = self.p_drop if self.training else 0.0
        # is_causal=True: la mascara triangular la aplica el kernel, que ademas
        # se salta los bloques enteramente futuros. scale por defecto = 1/sqrt(dh).
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        y = y.transpose(1, 2).reshape(B, L, D)
        return self.resid_drop(self.proj(y))

    @torch.no_grad()
    def qk_scores(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Productos q_i . k_m INTERNOS (sin escala ni mascara): [B,H,L,L].

        Solo para diagnostico y para el test de que RoPE es realmente relativo;
        no se usa en forward().
        """
        q, k, _ = self._qk(x, cos, sin)
        return torch.matmul(q, k.transpose(-2, -1))


# ---------------------------------------------------------------------------
# Bloque
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Bloque pre-norma:  x = x + Attn(RMS(x));  x = x + SwiGLU(RMS(x))."""

    def __init__(self, d_model: int, n_heads: int, d_hidden: int, dropout: float,
                 qk_norm: bool = True, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, qk_norm, attn_dropout)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_hidden, dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class ModernTransformer(TokenARModel):
    """Decoder-only 2024-2025: RoPE + RMSNorm + SwiGLU + QK-norm.

    Sin ningun parametro indexado por posicion absoluta: funciona igual con
    L=70 que con L=1024 y corre (sin reentrenar) con L > cfg.seq_len.
    """

    name = "modern"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_model = int(cfg.d_model)
        n_layers = int(cfg.n_layers)
        n_heads = int(cfg.n_heads)
        d_ff = int(getattr(cfg, "d_ff", 4 * d_model))
        dropout = float(getattr(cfg, "dropout", 0.1))
        # Knobs fuera de Config (getattr, sin tocar src/config.py):
        attn_dropout = float(getattr(cfg, "attn_dropout", 0.0))
        self.qk_norm = bool(getattr(cfg, "qk_norm", True))
        self.rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
        self.attn_dropout = attn_dropout
        self.tie_weights = bool(getattr(cfg, "tie_weights", True))
        self.max_seq_len = int(getattr(cfg, "seq_len", 1024))
        self.vocab_size = int(VOCAB_SIZE)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_hidden = swiglu_hidden(d_ff)

        self.tok_emb = nn.Embedding(self.vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        # Un solo cache de cos/sin para las 8 capas (se calcula una vez por forward).
        self.rope = Rotary(self.d_head, self.rope_theta, init_len=self.max_seq_len)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, self.d_hidden, dropout, self.qk_norm, attn_dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.head = None if self.tie_weights else nn.Linear(d_model, self.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Escala 1/sqrt(2*n_layers) en las proyecciones que escriben en el
        # residual: mantiene acotada la varianza acumulada del stream.
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
        # RMSNorm.weight se queda en 1 (init por defecto).

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """x: Long[B,L] -> logits Float[B,L,V] del token siguiente.

        pos_offset desplaza las posiciones de RoPE (por defecto 0). Como el
        modelo es equivariante a traslacion, cambiarlo NO altera la salida
        salvo por redondeo; existe para muestreo con ventana deslizante con
        cache de KV y para el test de equivariancia.
        """
        assert x.dim() == 2, f"se esperaba [B,L], llego {tuple(x.shape)}"
        B, L = x.shape
        assert L >= 1, "secuencia vacia"
        h = self.drop(self.tok_emb(x))
        cos, sin = self.rope(L, int(pos_offset), device=h.device, dtype=h.dtype)
        for blk in self.blocks:
            h = blk(h, cos, sin)
        h = self.norm_f(h)
        if self.tie_weights:
            return F.linear(h, self.tok_emb.weight)
        return self.head(h)

    def set_qk_norm(self, enabled: bool) -> None:
        """Activa/desactiva QK-norm en todas las capas (cambia el state_dict)."""
        for blk in self.blocks:
            blk.attn.set_qk_norm(enabled)
        self.qk_norm = bool(enabled)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """El CHECKPOINT manda sobre cfg en lo que toca a qk_norm.

        qk_norm no es campo de Config, asi que se pierde en config.json, pero SI
        cambia el conjunto de tensores: sin esto, entrenar con qk_norm=False y
        recargar por la via estandar del laboratorio fallaba con "Missing key(s)
        ... q_norm.weight" (y al reves, con "Unexpected key(s)"). Se deduce de
        las claves y el modelo se reconfigura ANTES de copiar los pesos.
        nn.Module.load_state_dict invoca el _load_from_state_dict del padre antes
        de recorrer los hijos, por eso llega a tiempo.
        """
        if any(k.startswith(prefix + "blocks.") for k in state_dict):
            want = (prefix + "blocks.0.attn.q_norm.weight") in state_dict
            if want != self.qk_norm:
                self.set_qk_norm(want)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    # El modelo no es recurrente: no hay estado de tamano fijo. El muestreo usa
    # ventana deslizante sobre forward() (opcionalmente con pos_offset).
    def supports_state(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Verificacion (ejecutar:  python src/models/modern_transformer.py)
# ---------------------------------------------------------------------------
def _test_rmsnorm(verbose: bool = True) -> bool:
    """RMSNorm: formula correcta, fusionada vs manual y error REAL en bf16.

    (a) float32/float64: igualdad EXACTA fusionada == manual, y no solo a
        dim=64: se prueban las dimensiones que el modelo usa de verdad
        (d_head=64, d_model=512, d_hidden=1408, y 4096 de margen).
    (b) bf16: cada ruta contra una referencia en float64 del MISMO tensor bf16.
        La version anterior de este test comparaba fusionada contra manual con
        tolerancia 2^-7 relativa y solo a dim=64; eso no mide el error de la
        norma (enfrenta las dos rutas entre si, sin verdad de referencia) y a
        dim=4096 rozaba la tolerancia, es decir era un test a punto de volverse
        intermitente. Aqui cada ruta se mide contra la verdad y se exige <= 2 ulp
        de bf16.
    (c) es RMSNorm y NO LayerNorm: no resta la media (x y x+c dan salidas
        distintas y la media de la salida no es cero).
    """
    torch.manual_seed(0)
    ok = True
    for dim in (64, 512, 1408, 4096):
        for dt in (torch.float32, torch.float64):
            nm = RMSNorm(dim).to(dt)
            with torch.no_grad():
                nm.weight.normal_(1.0, 0.1)
            x = torch.randn(4, dim, dtype=dt)
            err = (nm(x) - RMSNorm.manual(x, nm.weight, nm.eps)).abs().max().item()
            good = err == 0.0
            ok = ok and good
            if verbose:
                print(f"  dim={dim:5d} {str(dt):>16}: max|fusionada - manual| = "
                      f"{err!r} (exige 0.0 exacto) -> {'OK' if good else 'FALLO'}")
    for dim in (64, 512, 1408, 4096):
        nm = RMSNorm(dim)
        with torch.no_grad():
            nm.weight.normal_(1.0, 0.1)
        xb = torch.randn(4, dim).bfloat16()
        ref = RMSNorm.manual(xb.double(), nm.weight.double(), nm.eps)
        ulp = ref.abs().max().item() * 2 ** -8          # 1 ulp de bf16 a esa escala
        e_fus = (nm(xb).double() - ref).abs().max().item()
        e_man = (RMSNorm.manual(xb, nm.weight, nm.eps).double() - ref).abs().max().item()
        good = e_fus <= 2 * ulp and e_man <= 2 * ulp
        ok = ok and good
        if verbose:
            print(f"  dim={dim:5d} bf16 vs referencia float64 del MISMO tensor: "
                  f"fusionada {e_fus / ulp:4.2f} ulp, manual {e_man / ulp:4.2f} ulp "
                  f"(tol 2 ulp) -> {'OK' if good else 'FALLO'}")
    nm = RMSNorm(16).double()
    with torch.no_grad():
        nm.weight.normal_(1.0, 0.2)
    v = torch.randn(4, 16, dtype=torch.float64)
    shift = (nm(v) - nm(v + 3.0)).abs().max().item()
    media = nm(v).mean(-1).abs().max().item()
    good = shift > 1e-6 and media > 1e-6
    ok = ok and good
    if verbose:
        print(f"  no es LayerNorm: |RMS(x)-RMS(x+3)| = {shift:.3e} (>0) y la media de "
              f"la salida es {media:.3e} (!=0) -> {'OK' if good else 'FALLO'}")
        print(f"  (F.rms_norm disponible: {_HAS_FUSED_RMS})")
    return ok


def _test_rope_math(verbose: bool = True) -> bool:
    """Propiedades algebraicas de la rotacion, en float64.

    (a) La rotacion preserva la norma de cada vector: ||R(i) q|| == ||q||.
    (b) <R(i)q, R(j)k> depende SOLO de j - i: se compara el mismo par (q,k) a
        distancia d colocado en muchas posiciones absolutas distintas.
    (c) Contra una referencia compleja explicita:
        <R(i)q, R(j)k> = Re( sum_p  z_q[p]* . z_k[p] . e^{i w_p (j - i)} ).
    """
    torch.manual_seed(0)
    dh, theta = 64, 10000.0
    q = torch.randn(dh, dtype=torch.float64)
    k = torch.randn(dh, dtype=torch.float64)
    dev = torch.device("cpu")

    def dot(i: int, j: int) -> float:
        ci, si = rope_angles(1, dh, theta, dev, torch.float64, offset=i)
        cj, sj = rope_angles(1, dh, theta, dev, torch.float64, offset=j)
        qi = apply_rope(q.view(1, 1, 1, dh), ci, si)
        kj = apply_rope(k.view(1, 1, 1, dh), cj, sj)
        return float((qi * kj).sum())

    c0, s0 = rope_angles(5, dh, theta, dev, torch.float64, offset=997)
    nrm = (apply_rope(q.view(1, 1, 1, dh).expand(1, 1, 5, dh).contiguous(), c0, s0)
           .norm(dim=-1) - q.norm()).abs().max().item()

    worst_rel = 0.0
    for d in (0, 1, 3, 12, 70, 511):
        vals = [dot(i, i + d) for i in (0, 1, 7, 64, 333, 1024, 4096)]
        worst_rel = max(worst_rel, max(abs(v - vals[0]) for v in vals))
    # referencia compleja
    half = torch.arange(0, dh, 2, dtype=torch.float64)
    w = theta ** (-half / dh)
    zq = torch.complex(q[:dh // 2], q[dh // 2:])
    zk = torch.complex(k[:dh // 2], k[dh // 2:])
    worst_ref = 0.0
    for (i, j) in ((0, 0), (3, 17), (500, 87), (1024, 2048)):
        ref = float((zq.conj() * zk * torch.exp(1j * w * (j - i))).sum().real)
        worst_ref = max(worst_ref, abs(dot(i, j) - ref))

    ok = nrm < 1e-12 and worst_rel < 1e-10 and worst_ref < 1e-10
    if verbose:
        print(f"  (a) ||R(i)q|| - ||q||                       = {nrm:.3e} (tol 1e-12)")
        print(f"  (b) max |<R(i)q,R(i+d)k> - <R(0)q,R(d)k)>|  = {worst_rel:.3e} (tol 1e-10)")
        print(f"  (c) max |dot - referencia compleja|         = {worst_ref:.3e} (tol 1e-10)")
        print(f"  -> RoPE es relativo: {'OK' if ok else 'FALLO'}")
    return ok


def _test_rope_relative_layer(verbose: bool = True) -> bool:
    """RoPE relativo en la CAPA real de atencion (con QK-norm activo).

    Se toma la MISMA subsecuencia de estados ocultos y se coloca en posiciones
    absolutas distintas (offset 0, 1, 97, 1024, 4096). Los productos internos
    q_i . k_m de la capa deben coincidir para todo par (i,m), porque solo
    dependen de i - m.
    """
    torch.manual_seed(0)
    d_model, n_heads, L = 64, 4, 12
    att = CausalSelfAttention(d_model, n_heads, dropout=0.0, qk_norm=True).double().eval()
    rope = Rotary(d_model // n_heads, 10000.0)
    x = torch.randn(2, L, d_model, dtype=torch.float64)
    cos0, sin0 = rope(L, 0, device=x.device, dtype=torch.float64)
    ref = att.qk_scores(x, cos0, sin0)
    ok = True
    for off in (1, 97, 1024, 4096):
        cos, sin = rope(L, off, device=x.device, dtype=torch.float64)
        err = (att.qk_scores(x, cos, sin) - ref).abs().max().item()
        good = err < 1e-9
        ok = ok and good
        if verbose:
            print(f"  subsecuencia desplazada a la posicion {off:5d}: "
                  f"max|Q.K - Q.K(offset 0)| = {err:.3e} (tol 1e-9) "
                  f"-> {'OK' if good else 'FALLO'}")
    # Control negativo: si la posicion NO fuese relativa, comparar pares con
    # distinta distancia deberia dar una diferencia grande.
    diff = (ref[:, :, 1:, :-1] - ref[:, :, :-1, :-1]).abs().max().item()
    if verbose:
        print(f"  (control: pares a distinta distancia difieren en {diff:.3e}, "
              f"la invariancia de arriba no es trivial)")
    return ok and diff > 1e-3


def _test_translation_equivariance(model, verbose: bool = True) -> bool:
    """El MODELO COMPLETO es invariante al desplazamiento absoluto.

    Es la consecuencia practica de RoPE para nuestro problema: un prefijo de 70
    tokens produce exactamente los mismos logits este donde este en la ventana,
    asi que el mismatch entrenar-con-1024 / inferir-con-70 no tiene componente
    posicional.
    """
    torch.manual_seed(0)
    m = model.double().eval()
    x = torch.randint(3, m.vocab_size, (2, 70))
    with torch.no_grad():
        a = m(x, pos_offset=0)
        errs = []
        for off in (1, 713, 4096):
            errs.append((m(x, pos_offset=off) - a).abs().max().item())
    scale = a.abs().max().item()
    ok = max(errs) < 1e-9
    if verbose:
        for off, e in zip((1, 713, 4096), errs):
            print(f"  logits(L=70) con pos_offset={off:5d} vs 0: max|diff| = {e:.3e} "
                  f"(escala de los logits: {scale:.2f}, tol 1e-9)")
        print(f"  -> equivariancia a traslacion: {'OK' if ok else 'FALLO'}")
    m.float()
    return ok


def _test_strict_causality(model, L: int = 48, verbose: bool = True) -> bool:
    """Comprobacion BIT A BIT, posicion a posicion.

    Para cada t se perturba UNICAMENTE x[:, t] y se exige que los logits en
    TODAS las posiciones < t sean identicos bit a bit (diferencia exactamente
    0.0, no "pequena"), y que la posicion t SI cambie (si no cambiase, el
    modelo estaria ignorando su entrada).
    """
    m = model.eval()
    torch.manual_seed(0)
    x = torch.randint(3, m.vocab_size, (2, L))
    with torch.no_grad():
        base = m(x)
    worst_past, min_self = 0.0, float("inf")
    for t in range(L):
        x2 = x.clone()
        # token distinto garantizado
        x2[:, t] = 3 + (x[:, t] - 3 + 37) % (m.vocab_size - 3)
        with torch.no_grad():
            pert = m(x2)
        if t > 0:
            worst_past = max(worst_past, (base[:, :t] - pert[:, :t]).abs().max().item())
        min_self = min(min_self, (base[:, t] - pert[:, t]).abs().max().item())
    ok = worst_past == 0.0 and min_self > 0.0
    if verbose:
        print(f"  perturbando solo x[:,t] para los {L} valores de t: "
              f"max|delta logits en posiciones < t| = {worst_past:.1e} "
              f"(exige == 0.0 exacto)")
        print(f"  minimo sobre t de max|delta logits en la posicion t| = {min_self:.3e} "
              f"(debe ser > 0)")
        print(f"  -> causalidad estricta bit a bit: {'OK' if ok else 'FUGA CAUSAL'}")
    return ok


def _test_param_count(verbose: bool = True) -> bool:
    """Conteo de parametros contra una formula independiente, no contra si mismo."""
    from config import Config
    ok = True
    for kw, qk in ((dict(), True), (dict(), False),
                   (dict(d_model=256, n_layers=4, n_heads=4, tie_weights=False), True)):
        cfg = Config(**kw)
        cfg.qk_norm = qk
        m = ModernTransformer(cfg)
        V, D, Ly, dh, hid = m.vocab_size, m.d_model, m.n_layers, m.d_head, m.d_hidden
        exp = (V * D                          # embedding
               + Ly * 4 * D * D               # qkv (3) + proj (1)
               + (Ly * 2 * dh if qk else 0)   # ganancias de QK-norm
               + Ly * 3 * D * hid             # SwiGLU: w_in (2*hid) + w_down
               + Ly * 2 * D + D               # norm1, norm2 por bloque + norm_f
               + (0 if m.tie_weights else V * D))   # cabeza no atada
        got = m.n_params()
        good = exp == got
        ok = ok and good
        if verbose:
            print(f"  d_model={D} n_layers={Ly} qk_norm={qk} tie={m.tie_weights}: "
                  f"esperado {exp} ({exp/1e6:.4f} M) real {got} ({got/1e6:.4f} M) "
                  f"-> {'OK' if good else 'FALLO'}")
        del m
    return ok


def _test_qk_norm_checkpoint(verbose: bool = True) -> bool:
    """Un checkpoint se recarga aunque config.json haya perdido cfg.qk_norm.

    Es el caso real del laboratorio: Config.save() no guarda qk_norm (no es campo
    del dataclass), asi que Config.load() devuelve el valor por defecto True. Si
    el modelo se entreno con qk_norm=False, load_state_dict(strict=True) fallaba
    con "Missing key(s) ... q_norm.weight" (y al reves con "Unexpected key(s)").
    """
    from config import Config

    def build(qk):
        cfg = Config(d_model=128, n_layers=2, n_heads=4, d_ff=512, dropout=0.0)
        cfg.qk_norm = qk
        return ModernTransformer(cfg).eval()

    ok = True
    for entrenado, recargado in ((False, True), (True, False), (True, True), (False, False)):
        a = build(entrenado)
        b = build(recargado)          # el flag que dice el config.json recargado
        x = torch.randint(3, a.vocab_size, (2, 33))
        try:
            b.load_state_dict(a.state_dict(), strict=True)
            with torch.no_grad():
                d = (a(x) - b(x)).abs().max().item()
            good = d == 0.0 and b.qk_norm == entrenado
            msg = f"logits identicos bit a bit ({d!r}), b.qk_norm={b.qk_norm}"
        except Exception as e:                       # noqa: BLE001
            good, msg = False, f"EXCEPCION {type(e).__name__}: {str(e).splitlines()[0][:90]}"
        ok = ok and good
        if verbose:
            print(f"  entrenado con qk_norm={str(entrenado):5s} -> recargado en un modelo "
                  f"construido con qk_norm={str(recargado):5s}: {msg} "
                  f"-> {'OK' if good else 'FALLO'}")
    return ok


def _test_rope_cache(verbose: bool = True) -> bool:
    """El cache de cos/sin no se reconstruye en cada llamada (y crece a demanda)."""
    r = Rotary(16, init_len=64)
    r(8, 0, device=torch.device("cpu"), dtype=torch.float32)
    p0 = r.cos_cache.data_ptr()
    r(8, 0)                                  # device=None: el defecto de la firma
    p1 = r.cos_cache.data_ptr()
    r(8, 0, device="cpu")                    # device como cadena
    p2 = r.cos_cache.data_ptr()
    estable = (p0 == p1 == p2)
    n0 = r.cos_cache.shape[0]
    cos, sin = r(70, 4096, device=torch.device("cpu"), dtype=torch.float32)
    crecio = r.cos_cache.shape[0] >= 4166
    formas = tuple(cos.shape) == (70, 16) and tuple(sin.shape) == (70, 16)
    ok = estable and crecio and formas
    if verbose:
        print(f"  llamadas con device=None / 'cpu' / torch.device('cpu') reutilizan el "
              f"cache (mismo data_ptr): {estable}")
        print(f"  crece a demanda: {n0} -> {r.cos_cache.shape[0]} filas para L=70 "
              f"offset=4096 (hacen falta 4166); formas de la rodaja correctas: {formas}")
        print(f"  -> cache de RoPE: {'OK' if ok else 'FALLO'}")
    return ok


def _test_train_sanity(verbose: bool = True) -> bool:
    """El modelo ENTRENA de verdad (no solo pasa los tests de forma).

    (a) Todos los tensores de parametros reciben gradiente no nulo (si alguno
        quedara desconectado -- tipico al atar pesos o al marcar mal el init
        residual -- se veria aqui).
    (b) Memoriza una secuencia fija: la perdida debe caer de ln(155)=5.04 a ~0.
    (c) Un prefijo de 70 tokens evaluado SOLO da los mismos logits que dentro
        de la secuencia larga. La causalidad lo exige matematicamente; la
        diferencia residual es de redondeo (SDPA divide el trabajo en bloques
        distintos segun L, asi que aqui no hay igualdad bit a bit, solo ~1e-6
        en float32; la igualdad exacta se comprueba a L fijo en
        _test_strict_causality).
    """
    from config import Config
    torch.manual_seed(0)
    cfg = Config(d_model=128, n_layers=3, n_heads=4, d_ff=512, dropout=0.0)
    m = ModernTransformer(cfg).train()
    x = torch.randint(3, m.vocab_size, (2, 64))
    F.cross_entropy(m(x[:, :-1]).reshape(-1, m.vocab_size),
                    x[:, 1:].reshape(-1)).backward()
    huerfanos = [n for n, p in m.named_parameters()
                 if p.grad is None or p.grad.abs().max().item() == 0.0]
    m.zero_grad(set_to_none=True)

    seq = torch.randint(3, m.vocab_size, (1, 128))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    first = last = float("nan")
    for step in range(201):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(seq[:, :-1]).reshape(-1, m.vocab_size),
                               seq[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    m.eval()
    with torch.no_grad():
        acc = (m(seq[:, :-1]).argmax(-1) == seq[:, 1:]).float().mean().item()
        d70 = (m(seq[:, :70])[:, :70] - m(seq[:, :-1])[:, :70]).abs().max().item()
    ok = not huerfanos and last < 0.05 and acc > 0.99 and d70 < 1e-4
    if verbose:
        print(f"  parametros sin gradiente: {huerfanos if huerfanos else 'ninguno'} "
              f"({sum(1 for _ in m.parameters())} tensores)")
        print(f"  overfit de 128 tokens: loss {first:.3f} -> {last:.4f} en 200 pasos "
              f"(ln 155 = 5.043), aciertos {acc:.3f}")
        print(f"  logits del prefijo de 70 solo vs dentro de la secuencia larga: "
              f"max|diff| = {d70:.3e} (tol 1e-4, redondeo de SDPA)")
        print(f"  -> el modelo entrena: {'OK' if ok else 'FALLO'}")
    return ok


def _has_cuda() -> bool:
    """CUDA realmente usable (is_available() puede ser True con 0 dispositivos)."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _gpu_free_mib() -> float:
    """MiB libres segun nvidia-smi (la GPU del laboratorio es compartida)."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return -1.0


def _benchmark(batch: int = 16, L: int = 1024, steps: int = 10, warmup: int = 5,
               reps: int = 5, device: str = "cuda", cfg=None, amp: bool = True) -> float:
    """it/s reales (forward + backward + clip + step). MEJOR de varias repeticiones.

    La mejor repeticion es el estimador robusto en una GPU compartida: la
    contencion solo puede hacer que una repeticion salga mas lenta.
    """
    import time
    if cfg is None:
        from config import Config
        cfg = Config()
    model = ModernTransformer(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(3, model.vocab_size, (batch, L + 1), device=device)
    inp, tgt = x[:, :-1], x[:, 1:]
    use_amp = amp and device == "cuda"

    def one_step():
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(inp)
        else:
            logits = model(inp)
        loss = F.cross_entropy(logits.float().reshape(-1, model.vocab_size), tgt.reshape(-1))
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
    best = min(times)
    med = sorted(times)[len(times) // 2]
    extra = ""
    if device == "cuda":
        extra = (f" | pico memoria {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB"
                 f" | libre en la GPU {_gpu_free_mib():.0f} MiB")
    print(f"  benchmark {device}: batch={batch} L={L} {'bf16' if use_amp else 'fp32'} -> "
          f"MEJOR {1.0 / best:.2f} it/s ({best * 1e3:.0f} ms/it) | "
          f"mediana {1.0 / med:.2f} it/s | loss={loss.item():.3f}{extra}")
    del model, opt, x
    if device == "cuda":
        torch.cuda.empty_cache()
    return 1.0 / best


def _load_causality_checks(root):
    """Importa tests/test_causality.py (por ruta si el import normal falla).

    Motivo: tests/ no tiene __init__.py y en este entorno hay un paquete
    'tests' en site-packages que lo ensombrece.
    """
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


def _selftest(bench: bool = True) -> bool:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for p in (str(root), str(root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from config import Config
    check_shapes_token, check_token_causality = _load_causality_checks(root)

    ok = True
    print("== RMSNorm fusionada vs manual ==")
    ok &= _test_rmsnorm()
    print("== RoPE: propiedad relativa (algebra, float64) ==")
    ok &= _test_rope_math()
    print("== RoPE: propiedad relativa en la capa de atencion real (QK-norm activo) ==")
    ok &= _test_rope_relative_layer()

    print("== equivariancia a traslacion del modelo completo (float64) ==")
    ok &= _test_translation_equivariance(
        ModernTransformer(Config(n_layers=2, d_model=64, n_heads=4, d_ff=256)))

    for qk in (True, False):
        cfg = Config(n_layers=2, d_model=128, n_heads=4, d_ff=512)
        cfg.qk_norm = qk
        m = ModernTransformer(cfg).eval()
        print(f"== contrato (qk_norm={qk}, {m.param_report()}, d_hidden={m.d_hidden}) ==")
        ok &= check_shapes_token(m, L=32, V=m.vocab_size)
        ok &= check_token_causality(m, L=32, V=m.vocab_size)
        ok &= check_token_causality(m, L=17, V=m.vocab_size, t=3)
        ok &= _test_strict_causality(m, L=48)
        for L in (1, 17, 70, 1024, 2048):
            with torch.no_grad():
                y = m(torch.randint(3, m.vocab_size, (1, L)))
            fin = bool(torch.isfinite(y).all().item())
            good = tuple(y.shape) == (1, L, m.vocab_size) and fin
            ok &= good
            print(f"  L={L:5d} -> {tuple(y.shape)} finito={fin} "
                  f"|logits|max={y.abs().max().item():6.3f} {'OK' if good else 'FALLO'}")

    print("== conteo de parametros contra formula independiente ==")
    ok &= _test_param_count()
    print("== cache de RoPE (no se reconstruye en cada llamada) ==")
    ok &= _test_rope_cache()
    print("== checkpoint: qk_norm se deduce del state_dict, no de config.json ==")
    ok &= _test_qk_norm_checkpoint()

    print("== sanidad de entrenamiento (CPU, 200 pasos) ==")
    ok &= _test_train_sanity()

    cfg = Config()
    m = ModernTransformer(cfg)
    print("== configuracion por defecto del laboratorio ==")
    print(f"  {m.param_report()} (d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_heads={cfg.n_heads}, d_ff={cfg.d_ff} -> d_hidden SwiGLU={m.d_hidden}, "
          f"qk_norm={m.qk_norm}, theta={m.rope_theta:.0f}, tie={m.tie_weights})")
    emb = m.tok_emb.weight.numel()
    per_layer = sum(p.numel() for p in m.blocks[0].parameters())
    print(f"  desglose: embedding {emb/1e6:.3f} M (atado a la salida) + "
          f"{cfg.n_layers} x {per_layer/1e6:.3f} M por bloque + "
          f"norma final {m.norm_f.weight.numel()}")
    print("  causalidad estricta con la config por defecto (L=48):")
    ok &= _test_strict_causality(m.eval(), L=48)
    del m

    if bench and _has_cuda():
        free = _gpu_free_mib()
        print(f"== GPU (libre: {free:.0f} MiB) ==")
        # Guarda de cordura: con la GPU del laboratorio llena (otros agentes)
        # Windows no lanza OOM, PAGINA: un forward de L=1024 puede tardar
        # minutos y el selftest parece colgado. Por debajo de 2.5 GiB libres no
        # se toca la GPU y se dice por que.
        if 0 <= free < 2500:
            print("  saltada: hacen falta ~5.1 GiB para el benchmark de batch 16 "
                  "y con la VRAM llena WDDM pagina en vez de fallar (el forward "
                  "de L=1024 tarda minutos). Vuelve a ejecutarlo con la GPU libre.")
            print("== resultado global ==")
            print("  TODO OK (sin GPU)" if ok else "  HAY FALLOS")
            return ok
        try:
            mg = ModernTransformer(Config()).cuda().eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for L in (70, 1024, 2048):
                    y = mg(torch.randint(3, mg.vocab_size, (2, L), device="cuda"))
                    print(f"  GPU L={L}: {tuple(y.shape)} dtype={y.dtype} "
                          f"finito={bool(torch.isfinite(y).all().item())}")
                x = torch.randint(3, mg.vocab_size, (2, 128), device="cuda")
                t = 64
                base = mg(x)
                x2 = x.clone()
                x2[:, t] = 3 + (x[:, t] - 3 + 37) % (mg.vocab_size - 3)
                pert = mg(x2)
            d_past = (base[:, :t] - pert[:, :t]).abs().max().item()
            d_self = (base[:, t] - pert[:, t]).abs().max().item()
            gpu_ok = d_past == 0.0
            ok &= gpu_ok
            print(f"  GPU causalidad bf16 (perturbando solo x[:,{t}]): "
                  f"max|delta pasado|={d_past:.1e} propio={d_self:.3e} -> "
                  f"{'OK' if gpu_ok else 'FUGA CAUSAL'}")
            del mg, base, pert, y, x, x2
            torch.cuda.empty_cache()
            _benchmark(batch=16, L=1024, device="cuda")
        except RuntimeError as e:
            # OutOfMemoryError es subclase de RuntimeError; cualquier otro
            # RuntimeError es un fallo de verdad y se vuelve a lanzar.
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            print(f"  SIN MEMORIA en la GPU (compartida con otros agentes): "
                  f"{str(e).splitlines()[0]}")
            print("  -> se reportan las cifras de CPU y la medida previa con la GPU libre")
    elif bench:
        print("== GPU no disponible ==")

    print("== resultado global ==")
    print("  TODO OK" if ok else "  HAY FALLOS")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
