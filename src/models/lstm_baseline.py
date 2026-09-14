"""
Linea base recurrente del laboratorio: LSTM autoregresivo a nivel de token.

Arquitectura (token-level, V = 155, ver src/data/tokenizer.py)

    x [B,L] (Long)
      -> Embedding(V, d_model)          + dropout
      -> cfg.n_layers x [nn.LSTM de 1 capa (d_model o hidden) -> hidden]
         con dropout entre capas        (ver "POR QUE LA PILA NO ES UN UNICO
                                         nn.LSTM" mas abajo)
      -> dropout
      -> Linear(hidden -> d_model)      (solo si hidden != d_model)
      -> LayerNorm(d_model)
      -> Linear(d_model -> V)           (peso atado al embedding si cfg.tie_weights)
    logits [B,L,V]  del token x[t+1]

Por que un LSTM como linea base
-------------------------------
Es la referencia clasica de este problema: Performance-RNN usa una pila
recurrente sobre un vocabulario NOTE_ON/TIME_SHIFT casi identico al nuestro.
Su virtud aqui no es la calidad sino el contraste: memoria de tamano FIJO
(el estado h,c) frente a la atencion del Music Transformer, que puede mirar
cualquier token del pasado. La diferencia de perdida entre ambos mide cuanto
vale el acceso explicito al pasado en piano transcrito.

Causalidad
----------
Un LSTM es causal por construccion: el estado del paso t se calcula a partir de
x[:t+1] y de nada mas, y la cabeza es puntual en el tiempo, asi que no existe
camino desde x[t' > t] hacia logits[t]. Aun asi se ejecuta el test del
laboratorio (ver bloque __main__).

Estado recurrente (muestreo O(1))
---------------------------------
supports_state() -> True y step(x_t, state) generan un token con coste
constante en vez de re-procesar el prefijo completo. forward() y step()
comparten exactamente las mismas piezas (_embed, _run_rnn, _head_from_rnn),
por lo que dan los mismos logits; se verifica con check_step_equivalence
(tolerancia 1e-4, float32, modelo en eval()).

Presupuesto de parametros
-------------------------
Los valores por defecto de Config (n_layers=8, hidden=1024) estan pensados para
el transformer: apilar 8 capas LSTM de 1024 da 65.68 M parametros, fuera del
presupuesto de 15-45 M, y ademas casi triplica el tiempo por paso (en un RNN la
profundidad es coste puramente secuencial). La configuracion de la linea base
es LSTMBaseline.default_config() -> n_layers=3, hidden=1024, d_model=512,
23.70 M parametros. El constructor NO modifica el cfg que recibe (respeta lo que
le pasan) pero avisa por warnings si el modelo cae fuera del presupuesto.

Coste medido (RTX 4070 SUPER 12.9 GB, default_config, L=1024, autocast bf16)
----------------------------------------------------------------------------
Cifras remedidas en auditoria con la GPU OCIOSA (184 MiB en uso, 5% de
utilizacion), 30 iteraciones cronometradas tras 8 de calentamiento:

    entrenamiento  batch 16 -> mediana 7.24 it/s (138 ms/it), mejor 7.90,
                               p90 6.59, pico 1.32 GiB
    entrenamiento  batch 24 -> mediana 6.86 it/s (146 ms/it), mejor 7.31,
                               p90 6.64, pico 1.80 GiB
    muestreo con step()     -> ~500 tokens/s por secuencia (2.0 ms/token),
                               coste O(1) por token
Reproducido dos veces a batch 16 (7.24 y 7.29 it/s, 0.7% de diferencia).

La GPU de esta maquina se comparte con otros agentes del laboratorio, asi que la
mediana solo vale si la GPU esta libre: con otro proceso al 100% se han medido
1400-2800 ms/it (0.35-0.7 it/s) para el mismo modelo y el mismo codigo. Por eso
bench_gpu informa mediana, mejor Y p90 mas el cociente mejor/mediana: si ese
cociente se aleja de 1, la tirada mide la contencion y no el modelo. Memoria:
sobra margen para batch 24 en 12.9 GB (1.80 de 12.9 GiB).

POR QUE LA PILA NO ES UN UNICO nn.LSTM(num_layers=n, dropout=p)
---------------------------------------------------------------
En esta maquina (Windows 11, torch 2.11.0+cu128, cuDNN 9.19) un nn.LSTM con
num_layers > 1 Y dropout > 0 ejecutado en train() sobre CUDA mata el proceso al
cerrarse, con 0xC0000409 / STATUS_STACK_BUFFER_OVERRUN (LASTEXITCODE =
-1073740791). El fallo ocurre al descargar la libreria, cuando el trabajo ya ha
terminado: resultados, gradientes y checkpoints son correctos; lo unico corrupto
es el codigo de salida del proceso.

Aislado con un nn.LSTM pelado, sin nada de este modulo:
    num_layers=3, dropout=0.1, train()   -> 0xC0000409
    num_layers=3, dropout=0.0, train()   -> salida 0
    num_layers=3, dropout=0.1, eval()    -> salida 0   (en eval cuDNN no crea
                                                        el estado de dropout)
    num_layers=1, dropout=0.0, train()   -> salida 0
    MLP equivalente en CUDA (control)    -> salida 0
Lo que revienta es el estado de dropout de cuDNN. No se puede evitar desde
dentro del proceso: se probo terminar con os._exit(0), saltandose el cierre del
interprete, y el proceso muere igual con 0xC0000409 (ExitProcess sigue
descargando las DLL).

De ahi el diseno: la pila son nn.LSTM de UNA capa (que nunca activan el dropout
de cuDNN) con nn.Dropout propio entre ellas.

Coste de no fusionar, remedido en auditoria alternando las dos variantes en el
mismo proceso (para cancelar la contencion) y con EXACTAMENTE los mismos
parametros en ambas (23.70 M, sin capas muertas que inflaran el optimizador):

    apilada (n x nn.LSTM de 1 capa)  mediana 133.8 ms/it  (7.47 it/s)
    fusionada (nn.LSTM num_layers=3) mediana 107.6 ms/it  (9.30 it/s)
    sobrecoste: +24% en mediana, +20% en el mejor tiempo

Son ~26 ms por iteracion, unos 5 min mas en una tirada de 12000 pasos. A cambio:
codigo de salida fiable (verificado, salida 0), dropout bajo nuestro control y
estado accesible capa a capa. Si algun dia se arregla en torch/cuDNN, volver al
kernel fusionado es cambiar _stack por una sola llamada.

Nota sobre precision mixta (medido aqui: torch 2.11+cu128, cuDNN 9.19)
----------------------------------------------------------------------
PyTorch registra _cudnn_rnn con politica de autocast fp16 FIJA, asi que dentro
de torch.autocast("cuda", dtype=torch.bfloat16) la pila recurrente se ejecuta
igualmente en float16: la salida de las capas LSTM sale float16 (comprobado),
aunque los logits finales salgan bfloat16 porque la cabeza si respeta el dtype
pedido. No es un defecto de este modulo, es la politica del framework.
Consecuencias practicas:
  * bf16 y fp16 rinden lo mismo en este modelo.
  * el paso hacia atras corre en fp16 sin GradScaler, con riesgo teorico de
    desbordamiento. En la practica el estado de celda esta acotado (|c| crece
    como mucho ~1 por paso) y los gradientes medidos salen finitos (|grad| del
    orden de 0.6 a 5.6 en las mediciones), pero si el entrenamiento diverge la
    valvula de escape es fp32_rnn=True: mantiene la pila recurrente en float32
    aunque el resto corra bajo autocast (cuesta ~2-3x en velocidad y no altera
    la equivalencia forward/step, verificado).
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
except ImportError:                      # ejecucion directa del archivo: src/ no esta en sys.path
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    for _p in (str(_ROOT), str(_ROOT / "src")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from config import Config
    from models.base import TokenARModel
    from data.tokenizer import VOCAB_SIZE, PAD


PARAM_BUDGET = (15e6, 45e6)              # limites del laboratorio, en parametros


def _cast_state(state, dtype: torch.dtype):
    """Devuelve (h, c) en el dtype pedido; None se propaga tal cual."""
    if state is None:
        return None
    h, c = state
    return (h.to(dtype), c.to(dtype))


class LSTMBaseline(TokenARModel):
    """LSTM apilado con cabeza atada opcional. Contrato: forward([B,L]) -> [B,L,V]."""

    name = "lstm"

    # Si True, la pila recurrente corre en float32 incluso bajo autocast.
    # Ver la nota sobre precision mixta en el docstring del modulo.
    fp32_rnn: bool = False

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        V = int(VOCAB_SIZE)
        d = int(cfg.d_model)
        h = int(cfg.hidden)
        nl = int(cfg.n_layers)
        p = float(cfg.dropout)
        assert nl >= 1 and d >= 1 and h >= 1, "n_layers/d_model/hidden deben ser >= 1"

        self.vocab_size, self.d_model, self.hidden, self.n_layers = V, d, h, nl
        self.tie_weights = bool(cfg.tie_weights)

        # --- entrada ---
        # padding_idx solo cuando NO se atan pesos: con pesos atados la cabeza
        # inyectaria gradiente en la misma fila de PAD que el embedding mantiene
        # a cero, dejando esa "fila congelada" a medias. La perdida ignora PAD
        # con ignore_index, asi que no se pierde nada.
        self.emb = nn.Embedding(V, d, padding_idx=None if self.tie_weights else PAD)
        self.emb_drop = nn.Dropout(p)

        # --- pila recurrente ---
        # Pila de LSTM de UNA capa con dropout propio entre ellas, en lugar de
        # un unico nn.LSTM(num_layers=nl, dropout=p). Motivo: el dropout interno
        # de cuDNN corrompe la salida del proceso en esta maquina (ver el aviso
        # de entorno en el docstring del modulo) y no hay forma de arreglarlo
        # desde dentro del proceso. Cuesta +24% de tiempo por iteracion (medido:
        # 133.8 ms frente a 107.6 ms) porque cuDNN ya no puede fundir las capas
        # en una sola llamada, que a 12000 pasos son ~5 min mas de reloj: barato
        # a cambio de un codigo de salida fiable.
        # La semantica del dropout es la misma que la de nn.LSTM
        # (elemento a elemento, entre capas, remuestreado en cada paso; no es
        # dropout variacional a la AWD-LSTM), y ademas queda bajo nuestro control.
        self.layers = nn.ModuleList([
            nn.LSTM(input_size=d if i == 0 else h, hidden_size=h,
                    num_layers=1, batch_first=True) for i in range(nl)])
        self.layer_drop = nn.Dropout(p)     # entre capas, no tras la ultima
        self.out_drop = nn.Dropout(p)       # tras la pila, antes de la cabeza

        # --- cabeza ---
        # Proyeccion a d_model para poder atar pesos con el embedding. Si
        # hidden == d_model no hace falta y se ahorra la matriz.
        self.proj = nn.Linear(h, d, bias=False) if h != d else nn.Identity()
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        if self.tie_weights:
            self.head.weight = self.emb.weight     # comparte el tensor, no una copia

        self._init_weights()

        n = self.n_params()
        lo, hi = PARAM_BUDGET
        if not (lo <= n <= hi):
            warnings.warn(
                "LSTMBaseline: %.2f M parametros, fuera del presupuesto "
                "[%.0f, %.0f] M (n_layers=%d, hidden=%d, d_model=%d). Para la "
                "linea base usa LSTMBaseline.default_config()."
                % (n / 1e6, lo / 1e6, hi / 1e6, nl, h, d),
                RuntimeWarning, stacklevel=2)

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
                    # Ortogonal por bloque de compuerta (i, f, g, o): deja el radio
                    # espectral en 1 y retrasa explosion/desvanecimiento del gradiente.
                    with torch.no_grad():
                        for k in range(4):
                            nn.init.orthogonal_(prm.data[k * self.hidden:(k + 1) * self.hidden])
                elif pname.startswith("bias_"):
                    nn.init.zeros_(prm)
            # Sesgo de la compuerta de olvido a 1: el estado arranca "recordando"
            # (Gers et al. 2000). Orden de compuertas en nn.LSTM: i, f, g, o.
            with torch.no_grad():
                layer.bias_ih_l0[self.hidden:2 * self.hidden].fill_(1.0)

        if isinstance(self.proj, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        if not self.tie_weights:
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    # ------------------------------------- piezas compartidas forward / step
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.long, torch.int):
            x = x.long()
        return self.emb_drop(self.emb(x))

    def _check_state(self, state, x: torch.Tensor):
        """Valida la forma de (h, c) y lo lleva al dtype/device de la entrada.

        Cubre dos fallos reales detectados en auditoria:

        * Un state con MAS de n_layers filas se aceptaba en silencio: _stack solo
          consume h0[i:i+1] para i < n_layers, asi que las filas sobrantes se
          descartaban sin avisar. Arrastrar el estado de un modelo mas profundo
          (otro checkpoint, otra config) seguia generando desde un estado
          equivocado en vez de fallar. Ahora es un AssertionError con la forma.
        * Cebar el prefijo bajo autocast devuelve (h, c) en float16 (politica de
          autocast de _cudnn_rnn, ver la nota del docstring del modulo) y el
          siguiente step() fuera de autocast reventaba con "Input and hidden
          tensors are not the same dtype". Ahora el estado se castea al dtype de
          la entrada, asi que cebar y muestrear pueden estar en contextos de
          precision distintos.
        """
        h, c = state
        want = (self.n_layers, int(x.size(0)), self.hidden)
        for tag, t in (("h", h), ("c", c)):
            assert tuple(t.shape) == want, (
                "state[%s] tiene forma %s, se esperaba [n_layers,B,hidden]=%s"
                % (tag, tuple(t.shape), list(want)))
        if h.dtype != x.dtype or h.device != x.device:
            h = h.to(device=x.device, dtype=x.dtype)
        if c.dtype != x.dtype or c.device != x.device:
            c = c.to(device=x.device, dtype=x.dtype)
        return (h, c)

    def _stack(self, x: torch.Tensor, state):
        """Recorre las capas. state = (h, c) con h,c de forma [n_layers,B,hidden],
        el mismo convenio que nn.LSTM, para que sea intercambiable."""
        if state is not None:
            state = self._check_state(state, x)
        h0, c0 = (None, None) if state is None else state
        hs, cs = [], []
        for i, layer in enumerate(self.layers):
            st_i = None if state is None else (h0[i:i + 1].contiguous(),
                                               c0[i:i + 1].contiguous())
            x, (h_i, c_i) = layer(x, st_i)
            hs.append(h_i)
            cs.append(c_i)
            if i < self.n_layers - 1:
                x = self.layer_drop(x)
        return x, (torch.cat(hs, dim=0), torch.cat(cs, dim=0))

    def _run_rnn(self, e: torch.Tensor, state):
        """Pila recurrente. Devuelve (salida [B,L,hidden], nuevo state)."""
        if self.fp32_rnn:
            dev = "cuda" if e.is_cuda else "cpu"
            with torch.autocast(device_type=dev, enabled=False):
                return self._stack(e.float(), _cast_state(state, torch.float32))
        return self._stack(e, state)

    def _head_from_rnn(self, y: torch.Tensor) -> torch.Tensor:
        """[B,L,hidden] -> logits [B,L,V]. Es puntual en el tiempo, asi que no
        introduce ninguna dependencia con el futuro."""
        z = self.out_drop(y)
        z = self.proj(z)
        z = self.norm(z)
        return self.head(z)

    # ----------------------------------------------------------- API publica
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x Long[B,L] -> logits Float[B,L,V] del token x[t+1]."""
        assert x.dim() == 2, "se esperaba x [B,L], llego %s" % (tuple(x.shape),)
        y, _ = self._run_rnn(self._embed(x), None)
        return self._head_from_rnn(y)

    def forward_state(self, x: torch.Tensor, state=None):
        """Como forward pero arrastrando estado: devuelve (logits, state).

        Sirve para BPTT truncado (encadenar ventanas contiguas) y para cebar el
        muestreo con un prefijo en una sola llamada.
        """
        assert x.dim() == 2, "se esperaba x [B,L], llego %s" % (tuple(x.shape),)
        y, st = self._run_rnn(self._embed(x), state)
        return self._head_from_rnn(y), st

    def supports_state(self) -> bool:
        return True

    def step(self, x_t: torch.Tensor, state=None):
        """Un token: x_t Long[B,1] (o [B]) -> (logits [B,1,V], state).

        Coste O(1) por token, independiente de la longitud ya generada.
        """
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(1)
        assert x_t.dim() == 2 and x_t.size(1) == 1, \
            "step espera x_t [B,1], llego %s" % (tuple(x_t.shape),)
        y, st = self._run_rnn(self._embed(x_t), state)
        return self._head_from_rnn(y), st

    def init_state(self, batch_size: int, device=None, dtype=torch.float32):
        """Estado (h, c) explicito a ceros. Equivale a pasar state=None."""
        dev = device if device is not None else self.emb.weight.device
        z = torch.zeros((self.n_layers, int(batch_size), self.hidden),
                        device=dev, dtype=dtype)
        return (z, z.clone())

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
        """Config de la linea base: 23.70 M parametros, dentro del presupuesto.

        n_layers=3 y no los 8 del transformer, por dos razones. La primera es el
        presupuesto: 8 capas de 1024 son 65.68 M parametros, muy por encima de
        los 45 M del laboratorio. La segunda es que la profundidad no compra lo
        que a este modelo le falta -- el cuello de botella de un LSTM es el
        tamano del estado, no el numero de capas -- mientras que si multiplica
        un coste que en un RNN es estrictamente secuencial. Tres capas de 1024
        es tambien la forma clasica de esta linea base (Performance-RNN).
        """
        base = dict(name="lstm", model="lstm", family="token",
                    d_model=512, hidden=1024, n_layers=3, dropout=0.1,
                    tie_weights=True, seq_len=1024, batch_size=16)
        base.update(over)
        return Config(**base)


# ---------------------------------------------------------------------------
# verificaciones propias del modulo
# ---------------------------------------------------------------------------
def check_step_equivalence(model: LSTMBaseline, batch: int = 3, L: int = 64,
                           device: str = "cpu", tol: float = 1e-4,
                           verbose: bool = True) -> bool:
    """forward(x) debe coincidir con recorrer step() token a token.

    Es la prueba que legitima el muestreo O(1): si no coincidieran, generar con
    step() estaria muestreando de un modelo distinto del que se entreno con
    forward(). Se comprueba tambien el troceado irregular con forward_state,
    que es lo que hace un muestreador que ceba con un prefijo.

    En CUDA se desactiva TF32 durante la comprobacion. Con TF32 activado (el
    valor por defecto de torch.backends.cudnn.allow_tf32) las GEMM internas de
    cuDNN usan 10 bits de mantisa y cuDNN elige algoritmos distintos segun la
    longitud de secuencia (la pasada completa frente a los pasos de longitud 1),
    de modo que la diferencia sube a 5e-4 - 8e-4: es ruido de precision
    reducida, no una discrepancia de modelo (con float32 real la diferencia
    baja a ~6e-7). Se informa igualmente el valor con TF32 activado, porque es
    la diferencia que vera de verdad quien muestree con los ajustes por defecto
    (irrelevante tras el softmax sobre 155 clases).
    """
    # Se guardan modo y device para restaurarlos: esta funcion se puede llamar
    # como sanity-check DENTRO de un bucle de entrenamiento, y dejar el modelo en
    # eval() (o bajado a CPU) apagaria el dropout el resto del entrenamiento sin
    # que nada avisase.
    was_training = model.training
    orig_device = next(model.parameters()).device
    model = model.to(device).eval()
    torch.manual_seed(0)
    x = torch.randint(3, model.vocab_size, (batch, L), device=device)

    def _sweep():
        with torch.no_grad():
            full = model(x)                                # [B,L,V]

            st, outs = None, []                            # token a token
            for t in range(L):
                lg, st = model.step(x[:, t:t + 1], st)
                outs.append(lg)
            seq = torch.cat(outs, dim=1)

            st, outs, i = None, [], 0                      # por trozos irregulares
            for n in (7, 1, 20, 3, L):
                n = min(n, L - i)
                if n <= 0:
                    break
                lg, st = model.forward_state(x[:, i:i + n], st)
                outs.append(lg)
                i += n
            chunk = torch.cat(outs, dim=1)
        return (full, (full - seq).abs().max().item(),
                (full - chunk).abs().max().item())

    tf32_cudnn = torch.backends.cudnn.allow_tf32
    tf32_mm = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        full, d_step, d_chunk = _sweep()
        if device != "cpu" and (tf32_cudnn or tf32_mm):
            torch.backends.cudnn.allow_tf32 = tf32_cudnn
            torch.backends.cuda.matmul.allow_tf32 = tf32_mm
            _, d_step32, d_chunk32 = _sweep()
            if verbose:
                print("  [equivalencia] (informativo, TF32 activado) "
                      "max|forward-step|=%.3e  max|forward-trozos|=%.3e"
                      % (d_step32, d_chunk32))
    finally:
        torch.backends.cudnn.allow_tf32 = tf32_cudnn
        torch.backends.cuda.matmul.allow_tf32 = tf32_mm
        model.to(orig_device)
        model.train(was_training)
    ok = d_step < tol and d_chunk < tol and bool(torch.isfinite(full).all().item())
    if verbose:
        print("  [equivalencia] device=%-4s max|forward-step|=%.3e  "
              "max|forward-trozos|=%.3e  tol=%.0e -> %s"
              % (device, d_step, d_chunk, tol, "OK" if ok else "FALLO"))
    return ok


def check_strict_causality(model: LSTMBaseline, batch: int = 2, L: int = 24,
                           device: str = "cpu", verbose: bool = True) -> bool:
    """Causalidad mas dura que check_token_causality, por dos vias.

    1) POSICION A POSICION. El test del laboratorio reescribe TODO el sufijo
       x[t+1:] de golpe y admite una diferencia < 1e-4 en el pasado. Aqui se
       perturba UN SOLO token x[:, t] y se exige que los logits de todas las
       posiciones < t sean identicos BIT A BIT (torch.equal), no aproximados; se
       comprueba ademas explicitamente la posicion t-1, la primera que caeria si
       hubiera un desplazamiento de una casilla (el clasico off-by-one de una
       mascara con diagonal=0 en vez de diagonal=1). Y se exige que el logit EN t
       si cambie, para descartar el falso positivo de un modelo que ignora su
       entrada.
    2) POR GRADIENTE, exigiendo CERO EXACTO. Se engancha un hook al embedding
       para capturar su salida como hoja derivable y se retropropaga desde
       logits[:, t]: d logits[:, t] / d emb[:, t'] debe ser 0.0 exacto para todo
       t' > t. El gradiente ve todos los caminos del grafo, incluidos los que una
       perturbacion discreta podria no despertar.
    """
    assert L >= 4, "check_strict_causality necesita L >= 4 para tener posiciones que sondear"
    was_training = model.training
    orig_device = next(model.parameters()).device
    model = model.to(device).eval()
    torch.manual_seed(0)
    V = model.vocab_size
    x = torch.randint(3, V, (batch, L), device=device)
    try:
        # --- 1) posicion a posicion, igualdad bit a bit ---
        with torch.no_grad():
            base = model(x)
        probes = sorted({t for t in (1, 2, L // 2, L - 2, L - 1) if 1 <= t < L})
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

        # --- 2) gradiente exacto a traves de forward() real ---
        holder = {}

        def _hook(_mod, _inp, out):
            e = out.detach().clone().requires_grad_(True)
            holder["e"] = e
            return e

        hk = model.emb.register_forward_hook(_hook)
        try:
            grads = []
            for t in sorted({t for t in (0, 1, L // 2, L - 1) if 0 <= t < L}):
                model.zero_grad(set_to_none=True)
                holder.clear()
                model(x)[:, t].sum().backward()
                g = holder["e"].grad.abs().sum(-1)          # [B,L]
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
        print("  [estricto] posicion a posicion t=%s: logits<t identicos BIT A BIT "
              "en %d/%d pruebas" % (probes, len(probes) * 3 - viol, len(probes) * 3))
        print("             max|delta| en t-1=%.3e (exige 0 exacto)  "
              "min|delta| en t=%.3e (exige >0)" % (worst_prev, min_at_t))
        print("             gradiente: max|dlogits[t]/demb[t'>t]| = %s  (exige 0.0 exacto)"
              % ", ".join("t=%d:%.1e" % (t, gf) for t, gf, _ in grads))
        print("             -> %s" % ("OK, causalidad estricta" if ok else "FUGA CAUSAL"))
    return ok


def bench_gpu(model: LSTMBaseline, batch: int = 16, L: int = 1024, amp: str = "bf16",
              iters: int = 30, warmup: int = 8) -> float:
    """Paso de entrenamiento completo (fwd + bwd + clip + optimizador) en GPU.

    Se sincroniza CUDA antes y despues de cada iteracion (sin eso se mediria solo
    el tiempo de encolar los kernels) y se cronometra con perf_counter, no con
    time.time: en Windows time.time tiene una resolucion de ~15.6 ms, un 11% de
    una iteracion de 138 ms.

    Se informa MEDIANA, MEJOR y p90, no la media: esta maquina comparte la GPU con
    otros procesos y la media se contamina con los picos de contencion. Con la GPU
    ociosa la mediana y el mejor tiempo caen a menos de un 10% uno del otro
    (7.24 y 7.90 it/s medidos); con otro proceso al 100% la mediana se hunde a
    0.3-1.7 it/s, que mide la contencion y no el modelo. Si mediana y mejor se
    separan mucho, la tirada no es utilizable: repitela con la GPU libre.
    """
    if not torch.cuda.is_available():
        print("  [bench] sin GPU, omitido")
        return 0.0
    dev = torch.device("cuda")
    was_training = model.training
    orig_device = next(model.parameters()).device
    model = model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(3, model.vocab_size, (batch, L + 1), device=dev)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD)
    torch.cuda.reset_peak_memory_stats()
    obs_dtype, gnorm, loss_val, times = None, float("nan"), float("nan"), []
    for i in range(warmup + iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            logits = model(x[:, :-1])
            obs_dtype = logits.dtype
        loss = lossf(logits.reshape(-1, model.vocab_size).float(), x[:, 1:].reshape(-1))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
            gnorm, loss_val = float(gn.detach()), float(loss.detach())
    times.sort()
    med, best = times[len(times) // 2], times[0]
    p90 = times[min(len(times) - 1, int(0.9 * len(times)))]
    print("  [bench] batch=%d L=%d amp=%s n=%d -> mediana %.3f it/s (%.1f ms/it)  "
          "mejor %.3f it/s (%.1f ms/it)  p90 %.3f it/s  pico VRAM=%.2f GiB"
          % (batch, L, amp, len(times), 1.0 / med, med * 1e3, 1.0 / best,
             best * 1e3, 1.0 / p90, torch.cuda.max_memory_allocated() / 2 ** 30))
    print("          dtype de logits observado=%s  loss=%.4f (ln(155)=%.4f)  "
          "|grad|=%.3f finito=%s  contencion(mejor/mediana)=%.2fx"
          % (obs_dtype, loss_val, float(torch.log(torch.tensor(155.0))),
             gnorm, gnorm == gnorm, med / best))
    model.to(orig_device)
    model.train(was_training)
    del opt, x
    torch.cuda.empty_cache()
    return 1.0 / med


def bench_step_gpu(model: LSTMBaseline, batch: int = 16, n_tok: int = 256) -> float:
    """Muestreo autoregresivo con step(): tokens/s por secuencia (coste O(1))."""
    if not torch.cuda.is_available():
        return 0.0
    dev = torch.device("cuda")
    was_training = model.training
    orig_device = next(model.parameters()).device
    model = model.to(dev).eval()
    with torch.no_grad():
        x = torch.randint(3, model.vocab_size, (batch, 1), device=dev)
        st = None
        for _ in range(8):                                  # calentamiento
            lg, st = model.step(x, st)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        st = None
        for _ in range(n_tok):
            lg, st = model.step(x, st)
            x = lg[:, -1].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print("  [muestreo] step() batch=%d -> %.0f tokens/s por secuencia (%.2f ms/token)"
          % (batch, n_tok / dt, dt / n_tok * 1e3))
    model.to(orig_device)
    model.train(was_training)
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
        # En este entorno hay un paquete 'tests' instalado en site-packages que
        # tapa el directorio tests/ del repo (un paquete regular gana a un
        # namespace package, y tests/ no tiene __init__.py). Se carga el archivo
        # del repo por RUTA EXPLICITA en vez de fiarse del orden de sys.path:
        # asi la comprobacion viene con certeza de tests/test_causality.py de
        # este repo y no de cualquier otro modulo que se llame igual.
        import importlib.util

        _p = ROOT / "tests" / "test_causality.py"
        _spec = importlib.util.spec_from_file_location("_repo_test_causality", _p)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        check_token_causality = _mod.check_token_causality
        check_shapes_token = _mod.check_shapes_token
        _CAUS_SRC = str(_p)

    torch.manual_seed(1234)
    cfg = LSTMBaseline.default_config()
    model = LSTMBaseline(cfg)

    print(model.param_report())
    print("  d_model=%d hidden=%d n_layers=%d dropout=%.2f tie_weights=%s V=%d "
          "supports_state=%s"
          % (cfg.d_model, cfg.hidden, cfg.n_layers, cfg.dropout, cfg.tie_weights,
             model.vocab_size, model.supports_state()))
    for tag, mod in (("emb", model.emb), ("lstm", model.layers), ("proj", model.proj),
                     ("norm", model.norm), ("head", model.head)):
        n = sum(q.numel() for q in mod.parameters())
        print("    %-5s %10d  (%5.2f M)%s"
              % (tag, n, n / 1e6, "  [atado al emb]" if tag == "head" and cfg.tie_weights else ""))

    print("\n-- tests/test_causality.py --  (cargado de: %s)" % _CAUS_SRC)
    ok_shape = check_shapes_token(model)
    ok_caus = check_token_causality(model)

    print("\n-- causalidad estricta (bit a bit + gradiente cero exacto) --")
    ok_strict = check_strict_causality(model)

    print("\n-- equivalencia forward / step --")
    ok_eq = check_step_equivalence(model, device="cpu")
    if torch.cuda.is_available():
        ok_eq = check_step_equivalence(model, device="cuda") and ok_eq
    model.to("cpu")

    print("\n-- velocidad --")
    it_s = bench_gpu(LSTMBaseline(cfg), batch=16, L=1024, amp="bf16")
    bench_step_gpu(LSTMBaseline(cfg), batch=16, n_tok=256)

    print("\n-- referencia: Config() crudo, n_layers=8 (fuera de presupuesto) --")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("  " + LSTMBaseline(Config()).param_report())

    allok = ok_shape and ok_caus and ok_strict and ok_eq
    print("\nRESULTADO: %s" % ("TODO OK" if allok else "HAY FALLOS"))
    # Salida normal: la pila de capas de una en una no crea estado de dropout
    # en cuDNN, asi que el proceso cierra limpio y el codigo de salida sirve.
    sys.exit(0 if allok else 1)
