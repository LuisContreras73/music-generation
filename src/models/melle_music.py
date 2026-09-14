"""MELLE adaptado a piano-roll binario de onsets  ->  MELLEMusic.

Referencia
----------
Meng et al. 2024, "Autoregressive Speech Synthesis without Vector Quantization"
(MELLE).  MELLE es un LM autoregresivo de valores CONTINUOS: en lugar de
predecir tokens de un codebook, en cada paso predice el siguiente frame de
mel-espectrograma. Sus cuatro ingredientes distintivos, todos portados aqui:

  1. Latent sampling module (variacional).  El estado del decoder h_t predice
     media mu_t y log-varianza logvar_t de una gaussiana diagonal; se muestrea
     z_t = mu_t + sigma_t * eps con el truco de reparametrizacion y se anade un
     termino KL al objetivo.  Esto sustituye al muestreo categorico y da
     diversidad SIN cuantizacion vectorial.
  2. Perdida de regresion L1 + L2 sobre el frame reconstruido desde z_t.
  3. Spectrogram flux loss: premia que el frame predicho DIFIERA del frame
     anterior; evita el colapso a una secuencia de frames constantes.
  4. Stop prediction: cabeza binaria de fin de secuencia, con peso pequeno.

Adaptacion al dominio (decisiones de ingenieria y desviaciones del paper)
------------------------------------------------------------------------
* La entrada no es un mel continuo sino un frame BINARIO de 88 dimensiones
  (onsets, 20 Hz).  Se proyecta con un "pre-net" MLP con dropout (como el
  pre-net de Tacotron / MELLE), 88 -> cfg.hidden -> cfg.d_model, cerrado con un
  LayerNorm.  El LayerNorm no es decorativo: con la densidad real del corpus
  (0.0051, ~1.86 pitches activos por paso activo) e init normal(0, 0.02) la
  salida cruda del pre-net tiene std ~0.004 frente a rms(PE) = 0.707, o sea que
  la senal de CONTENIDO seria el 0.004% de la varianza que entra al transformer
  y el modelo arrancaria viendo casi solo posiciones.  El LayerNorm pone el
  contenido a escala unidad y acelera de forma medible el arranque.
* Backbone: decoder TRANSFORMER causal sobre frames, pre-LN, atencion con
  F.scaled_dot_product_attention(is_causal=True).  Posiciones sinusoidales
  absolutas (no hay bias relativo porque SDPA con is_causal no lo admite sin
  materializar la matriz [T,T]; cfg.rel_attn se ignora en este modelo).
* El latente es ABSTRACTO de dimension cfg.latent_dim (en el paper z_t tiene la
  dimension del mel y ES el frame predicho).  Un gaussiano diagonal directamente
  sobre 88 dimensiones binarias no tiene sentido, asi que z_t se decodifica a la
  grilla de 88 pitches con un cabezal + post-net causal.  El latente actua como
  cuello de botella: TODA la salida pasa por el.
* CAMINO UNICO desde z.  logits, recon y stop se derivan exclusivamente de z
  (no hay rama paralela que ignore el muestreo latente).  En train() z se
  muestrea; en eval() z = mu, luego el forward es DETERMINISTA.  Para generar
  con diversidad latente sin tocar src/generate.py (que llama a model(x) en
  eval y no a sample_next) basta poner model.sample_latent_in_eval = True.
* La "reconstruccion continua" es recon = sigmoid(logits) en (0,1), es decir la
  media Bernoulli del frame.  Consecuencias buenas de esta eleccion:
    - la verosimilitud discreta ("logits", obligatoria para reportar bits/paso
      comparables con los modelos token-level) y la regresion L1/L2 son
      consistentes entre si, no dos cabezales que pueden contradecirse;
    - abs(recon - x_prev) <= 1, luego la flux loss (que es NEGATIVA por
      construccion: se minimiza maximizando la variacion) esta ACOTADA en
      [-1, 0] y no puede divergir a -inf.  En el paper hay que confiar en un
      peso pequeno; aqui el acotamiento es estructural.
* Post-net: en MELLE es una pila de conv1d que refina el frame.  Aqui son
  convoluciones 1D CAUSALES (padding solo por la izquierda) con residual y
  salida inicializada a cero, asi que al inicio logits == logits_pre.
* KL: por defecto la forma de MELLE, D_KL(N(mu, sigma^2) || N(mu, I)) =
  0.5 * (sigma^2 - logvar - 1), que regulariza SOLO la varianza (penaliza el
  colapso sigma -> 0 y la explosion sigma -> inf) y deja libre la media, que es
  la que transporta el contenido musical.  kl_mode="prior" da la KL de VAE
  clasica contra N(0, I).  DESVIACION DEL PAPER: la KL se reduce como MEDIA
  sobre las cfg.latent_dim dimensiones, no como SUMA; asi cfg.kl_weight
  significa lo mismo independientemente de latent_dim, pero el peso efectivo
  respecto al paper es cfg.kl_weight / latent_dim (0.001 / 64 = 1.6e-5).
* Stop target: si el bucle de entrenamiento no pasa stop_target, se deduce del
  mask y SOLO se marca stop=1 cuando la ventana lleva padding (ahi la pieza
  termina de verdad); en una ventana interior el objetivo es todo ceros.  Marcar
  el borde de toda ventana como "fin de pieza" seria una etiqueta falsa que
  inyecta gradiente espurio en el tronco compartido (stop_head lee la misma f
  que roll_head).
* forward tambien devuelve "aux" con los mismos terminos ya reducidos, por
  contrato de base.FrameARModel.  src/train.py usa loss_terms, luego ese aux es
  redundante: construye un segundo grafo de perdida que se descarta.  Se puede
  desactivar con model.aux_in_forward = False (los terminos salen a cero).

Colapso a todo-cero: como se mitiga
-----------------------------------
El 75.6% de los frames del corpus estan vacios y la densidad de onsets es
0.0051 por celda (t, pitch).  El predictor constante ya consigue ~0.046
bits/celda, asi que el gradiente hacia "todo apagado" es enorme y es el modo de
fallo real.  Mitigaciones implementadas:
  * pos_weight (argumento del constructor, override por cfg.pos_weight):
    reponderacion de la clase positiva en la BCE.  El inverso completo de la
    frecuencia (1/0.0051 ~ 196) descalibra las probabilidades y arruina el
    bits/paso; el valor por defecto es moderado (4.0).  El termino
    "bits_per_step" que devuelve loss_terms se calcula SIEMPRE sin reponderar,
    de modo que la metrica reportada sigue siendo la verosimilitud honesta.
  * La MISMA reponderacion se aplica a la regresion L1/L2, que es donde el
    colapso es mas peligroso: sin ella el minimo de L1 con 99.5% de ceros es
    literalmente recon = 0.
  * Flux loss (cfg.flux_weight): castiga que el frame predicho se parezca al
    anterior; una secuencia constante (por ejemplo todo cero) es exactamente el
    maximo de esta perdida.
  * El bias del cabezal de salida se inicializa en logit(0.0051) = -5.27, es
    decir el modelo ARRANCA en la marginal del corpus en vez de tener que
    descender hasta ella (que es la trayectoria que induce el colapso).

Contrato
--------
forward(x: Float[B,T,88], mask=None) -> dict con
    "logits"      Float[B,T,88]  logits Bernoulli del frame t+1  (OBLIGATORIO)
    "recon"       Float[B,T,88]  reconstruccion continua = sigmoid(logits)
    "aux"         dict de escalares YA reducidos y SIN pesos:
                  "kl", "reg_l1", "reg_l2", "flux", "stop"
    ademas: "logits_pre", "recon_pre", "mu", "logvar", "stop_logits", "prev"
loss_terms(out, target, mask=None, stop_target=None) -> dict con cada termino y
    "loss" total, aplicando cfg.kl_weight / cfg.flux_weight y los pesos del
    constructor.  mask [B,T] con 1 = posicion valida, alineada con target
    (igual que lo entrega data.datasets.FrameWindows: x, target, mask).

Causalidad: logits[:, t] depende solo de x[:, :t+1] (atencion causal + conv
causal en el post-net).  Verificado por gradiente con
tests/test_causality.check_frame_causality.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .base import FrameARModel
except ImportError:                                  # cargado con sys.path=src
    from models.base import FrameARModel

N_PITCH = 88
ONSET_DENSITY = 0.0051        # densidad media de onsets por celda (t, pitch)
LOGVAR_CLAMP = 8.0            # rango seguro de log-varianza


# --------------------------------------------------------------------------- #
#  Piezas del backbone
# --------------------------------------------------------------------------- #
def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """Codificacion posicional sinusoidal clasica -> [max_len, d_model]."""
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    idx = torch.arange(0, d_model, 2, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * idx / d_model)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    n_cos = pe[:, 1::2].shape[1]
    pe[:, 1::2] = torch.cos(pos * div)[:, :n_cos]
    return pe


class _CausalSelfAttention(nn.Module):
    """Atencion multi-cabeza causal via scaled_dot_product_attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model debe ser divisible por n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.p = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (B, T, self.n_heads, self.d_head)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.p if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.drop(self.proj(y))


class _Block(nn.Module):
    """Bloque pre-LN: atencion causal + FFN, ambos residuales."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class _CausalPostNet(nn.Module):
    """Post-net de MELLE con convoluciones 1D CAUSALES sobre el eje temporal.

    Refina el frame predicho mirando solo frames predichos anteriores, luego no
    rompe la causalidad.  La ultima conv se inicializa a cero: al comienzo del
    entrenamiento el post-net es la identidad (residual nulo).
    """

    def __init__(self, n_pitch: int, hidden: int, kernel: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.kernel = int(kernel)
        chans = [n_pitch] + [hidden] * max(int(n_layers) - 1, 0) + [n_pitch]
        self.convs = nn.ModuleList(
            nn.Conv1d(chans[i], chans[i + 1], self.kernel)
            for i in range(len(chans) - 1)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        h = y.transpose(1, 2)                       # [B,88,T]
        last = len(self.convs) - 1
        for i, conv in enumerate(self.convs):
            h = F.pad(h, (self.kernel - 1, 0))      # padding SOLO por la izquierda
            h = conv(h)
            if i != last:
                h = self.drop(F.gelu(h))
        return h.transpose(1, 2)                    # [B,T,88]


# --------------------------------------------------------------------------- #
#  Modelo
# --------------------------------------------------------------------------- #
class MELLEMusic(FrameARModel):
    """MELLE sobre piano-roll: transformer causal + latente gaussiano por paso.

    Parameters
    ----------
    cfg : src.config.Config
        Usa d_model, n_layers, n_heads, d_ff, dropout, hidden, latent_dim,
        kl_weight, flux_weight, seq_len.  Cualquier atributo extra del cfg que
        se llame igual que uno de los argumentos de abajo tiene prioridad.
    pos_weight : float
        Peso de la clase positiva (onset) en la BCE y en la regresion.  Mitiga
        el colapso a todo-cero; ver el docstring del modulo.
    stop_weight : float
        Peso de la cabeza de fin de secuencia.  Pequeno a proposito: en ventanas
        aleatorias de un corpus de 51 M pasos el "final" que ve el modelo es
        casi siempre el borde de la ventana, no el final real de la pieza, salvo
        que el bucle de entrenamiento pase stop_target explicito.
    reg_weight, bce_weight : float
        Pesos de la regresion (L1+L2) y de la verosimilitud discreta.
    stop_pos_weight : float
        Reponderacion de la clase positiva de la cabeza de stop (hay como maximo
        un 1 por fila frente a T-1 ceros).
    kl_mode : {"melle", "prior"}
        Referencia de la KL: N(mu, I) (paper) o N(0, I) (VAE clasico).
    aux_in_forward : bool
        Si False, forward devuelve aux con ceros (ahorra el grafo extra cuando
        el bucle de entrenamiento usa loss_terms, que es la via recomendada;
        src/train.py esta en ese caso).

    Atributos publicos ajustables tras construir
    --------------------------------------------
    sample_latent_in_eval : bool (defecto False)
        Si True, forward muestrea z tambien en eval().  Sirve para que
        src/generate.py::sample_frames (que llama a model(x) en eval) recupere
        la diversidad latente de MELLE.  Con False, forward es determinista en
        eval, que es lo que exige el contrato y lo que verifica el test.
    aux_in_forward : bool
        Igual que el argumento del constructor, mutable en caliente.
    """

    name = "melle"
    family = "frame"

    def __init__(self, cfg, pos_weight: float = 4.0, stop_weight: float = 0.05,
                 reg_weight: float = 1.0, bce_weight: float = 1.0,
                 stop_pos_weight: float = 20.0, kl_mode: str = "melle",
                 postnet_hidden: int = 192, postnet_kernel: int = 5,
                 postnet_layers: int = 2, aux_in_forward: bool = True,
                 onset_density: float = ONSET_DENSITY):
        super().__init__()
        self.cfg = cfg
        d_model = int(cfg.d_model)
        d_ff = int(cfg.d_ff)
        n_heads = int(cfg.n_heads)
        n_layers = int(cfg.n_layers)
        dropout = float(cfg.dropout)
        hidden = int(getattr(cfg, "hidden", 4 * d_model))
        latent = int(getattr(cfg, "latent_dim", 64))

        self.d_model, self.n_layers, self.latent_dim = d_model, n_layers, latent
        self.n_pitch = N_PITCH

        # --- pesos de la perdida (cfg tiene prioridad sobre los argumentos) ---
        def g(key, default):
            return float(getattr(cfg, key, default))

        self.pos_weight = g("pos_weight", pos_weight)
        self.stop_weight = g("stop_weight", stop_weight)
        self.reg_weight = g("reg_weight", reg_weight)
        self.bce_weight = g("bce_weight", bce_weight)
        self.stop_pos_weight = g("stop_pos_weight", stop_pos_weight)
        self.kl_weight = g("kl_weight", 0.001)
        self.flux_weight = g("flux_weight", 0.1)
        self.kl_mode = str(getattr(cfg, "kl_mode", kl_mode))
        if self.kl_mode not in ("melle", "prior"):
            raise ValueError("kl_mode debe ser 'melle' o 'prior'")
        self.aux_in_forward = bool(aux_in_forward)
        # Gancho de generacion: src/generate.py::sample_frames llama a
        # model(x) en eval(), no a sample_next, luego por defecto NO habria
        # muestreo latente al generar (z = mu) y se perderia la diversidad que
        # es el punto de MELLE.  Poniendo model.sample_latent_in_eval = True se
        # reactiva el muestreo sin tocar generate.py.  Por defecto False para
        # que forward sea determinista en eval (requisito del contrato).
        self.sample_latent_in_eval = False

        # --- pre-net: frame binario 88-d -> d_model (MLP con dropout) ---
        self.prenet = nn.Sequential(
            nn.Linear(N_PITCH, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # --- posiciones sinusoidales (buffer no persistente) ---
        max_len = max(int(getattr(cfg, "seq_len", 1024)) + 1, 2048)
        self.register_buffer("pe", _sinusoidal_pe(max_len, d_model),
                             persistent=False)
        self.emb_drop = nn.Dropout(dropout)

        # --- backbone transformer causal ---
        self.blocks = nn.ModuleList(
            _Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)

        # --- latent sampling module (el corazon de MELLE) ---
        self.to_mu = nn.Linear(d_model, latent)
        self.to_logvar = nn.Linear(d_model, latent)

        # --- decoder del latente: TODA la salida pasa por z ---
        self.z_proj = nn.Sequential(
            nn.Linear(latent, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.GELU(),
        )
        self.roll_head = nn.Linear(d_model, N_PITCH)
        self.postnet = _CausalPostNet(N_PITCH, postnet_hidden, postnet_kernel,
                                      postnet_layers, dropout)
        self.stop_head = nn.Linear(d_model, 1)

        self.apply(self._init_weights)
        self._init_special(onset_density)

    # ------------------------------------------------------------------ init
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _init_special(self, onset_density: float) -> None:
        # proyecciones residuales escaladas (estabilidad con muchas capas)
        std = 0.02 / math.sqrt(2.0 * max(self.n_layers, 1))
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, 0.0, std)
            nn.init.normal_(blk.ff[2].weight, 0.0, std)
        # arrancar en la marginal del corpus, no en p=0.5 (anti-colapso)
        p = min(max(float(onset_density), 1e-6), 1.0 - 1e-6)
        nn.init.constant_(self.roll_head.bias, math.log(p / (1.0 - p)))
        # sigma inicial ~ 0.37: muestreo activo pero no dominante
        nn.init.normal_(self.to_logvar.weight, 0.0, 1e-3)
        nn.init.constant_(self.to_logvar.bias, -2.0)
        # post-net = identidad al inicio
        nn.init.zeros_(self.postnet.convs[-1].weight)
        # la pieza casi nunca termina en un paso dado -> stop logit muy negativo
        nn.init.constant_(self.stop_head.bias, -5.0)

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> dict:
        """x [B,T,88] binario -> dict (ver el docstring del modulo).

        `mask` (opcional) marca frames de ENTRADA validos [B,T] y solo se usa
        para los escalares de "aux"; los logits no dependen de ella.
        """
        if x.dim() != 3 or x.shape[-1] != N_PITCH:
            raise ValueError(f"se esperaba x [B,T,{N_PITCH}], llego {tuple(x.shape)}")
        B, T, _ = x.shape
        if T > self.pe.shape[0]:
            raise ValueError(f"T={T} supera el buffer posicional ({self.pe.shape[0]})")

        h = self.prenet(x)
        h = self.emb_drop(h + self.pe[:T].unsqueeze(0).to(h.dtype))
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)                                    # [B,T,d_model]

        # --- latent sampling module: muestreo en train, media en eval ---
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)
        if self.training or self.sample_latent_in_eval:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu                                          # eval determinista

        # --- camino unico desde z: nada esquiva el latente ---
        f = self.z_proj(z)
        logits_pre = self.roll_head(f)
        logits = logits_pre + self.postnet(logits_pre)
        out = {
            "logits": logits,                               # Bernoulli del frame t+1
            "recon": torch.sigmoid(logits),                 # frame continuo en (0,1)
            "logits_pre": logits_pre,
            "recon_pre": torch.sigmoid(logits_pre),
            "mu": mu,
            "logvar": logvar,
            "stop_logits": self.stop_head(f).squeeze(-1),   # [B,T]
            "prev": x,                                      # frame anterior real
        }
        out["aux"] = self._aux_in_window(out, x, mask)
        return out

    # ------------------------------------------------------------- utilidades
    @staticmethod
    def _masked_mean(t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Media de t sobre posiciones validas.  t [B,T] o [B,T,P], m [B,T]."""
        if t.dim() == 3:
            m = m.unsqueeze(-1)
        num = (t * m).sum()
        den = m.expand_as(t).sum().clamp(min=1.0)
        return num / den

    def _aux_in_window(self, out: dict, x: torch.Tensor,
                       mask: Optional[torch.Tensor]) -> dict:
        """Escalares de aux con el desplazamiento interno de la ventana.

        Dentro de la ventana el objetivo de logits[:, t] es x[:, t+1], luego
        solo las posiciones 0..T-2 tienen objetivo.  Son los MISMOS terminos que
        devuelve loss_terms, ya reducidos y SIN sus pesos.
        """
        keys = ("kl", "reg_l1", "reg_l2", "flux", "stop")
        T = x.shape[1]
        if (not self.aux_in_forward) or T < 2:
            zero = x.new_zeros(())
            return {k: zero.clone() for k in keys}
        sub_keys = ("logits", "recon", "recon_pre", "mu", "logvar", "stop_logits")
        sub = {k: out[k][:, :-1] for k in sub_keys}
        sub["prev"] = x[:, :-1]
        if mask is None:
            m = x.new_ones(x.shape[0], T - 1)
        else:
            m = mask[:, :-1] * mask[:, 1:]
        terms = self._terms(sub, x[:, 1:], m, stop_target=None)
        return {k: terms[k] for k in keys}

    def _terms(self, pred: dict, target: torch.Tensor, mask: torch.Tensor,
               stop_target: Optional[torch.Tensor] = None,
               flux_mask: Optional[torch.Tensor] = None) -> dict:
        """Todos los terminos de MELLE, reducidos a escalar y SIN pesos.

        Todo en float32 aunque el forward corra en bf16 (BCE y exp son
        sensibles a la precision reducida).
        """
        logits = pred["logits"].float()
        recon = pred["recon"].float()
        recon_pre = pred.get("recon_pre", pred["recon"]).float()
        mu = pred["mu"].float()
        logvar = pred["logvar"].float()
        prev = pred["prev"].float()
        y = target.float()
        mask = mask.float()

        # 1) verosimilitud discreta (Bernoulli) con reponderacion de onsets
        w = 1.0 + (self.pos_weight - 1.0) * y
        bce_raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        bce = self._masked_mean(bce_raw * w, mask)

        # metrica honesta y comparable: bits por paso (sin reponderar, 88 pitches)
        with torch.no_grad():
            bits = self._masked_mean(bce_raw, mask) * (N_PITCH / math.log(2.0))

        # 2) regresion L1 + L2 del frame continuo, antes y despues del post-net
        #    (MELLE aplica L_reg a las dos salidas)
        d1 = (recon - y).abs() + (recon_pre - y).abs()
        d2 = (recon - y).pow(2) + (recon_pre - y).pow(2)
        reg_l1 = self._masked_mean(d1 * w, mask)
        reg_l2 = self._masked_mean(d2 * w, mask)

        # 3) KL del latente.  "melle": D_KL(N(mu,sigma^2) || N(mu,I)) regulariza
        #    solo la varianza; "prior": VAE clasico contra N(0,I).
        var = logvar.exp()
        kl_el = 0.5 * (var - logvar - 1.0)
        if self.kl_mode == "prior":
            kl_el = kl_el + 0.5 * mu.pow(2)
        kl = self._masked_mean(kl_el, mask)

        # 4) flux loss: NEGATIVA, se minimiza maximizando |frame_t - frame_{t-1}|.
        #    Acotada en [-1,0] porque recon es una sigmoide y prev es binario.
        fm = mask if flux_mask is None else flux_mask.float()
        flux = -self._masked_mean((recon - prev).abs(), fm)

        # 5) stop prediction: stop_target real si lo da el bucle; si no, 1 en la
        #    ultima posicion valida SOLO en ventanas con padding (ver abajo)
        stop_logits = pred["stop_logits"].float()
        if stop_target is None:
            # Solo hay evidencia de "fin de pieza" si la ventana lleva PADDING:
            # data.datasets.FrameWindows rellena con ceros unicamente cuando el
            # bloque de la pieza se agota, luego ahi la ultima posicion valida
            # ES el final real.  En una ventana INTERIOR (mask todo 1) marcar el
            # ultimo paso como final es una etiqueta FALSA -> objetivo todo 0.
            n_steps = mask.shape[1]
            lengths = mask.sum(dim=1).long()
            ends = (lengths > 0) & (lengths < n_steps)
            stop_target = torch.zeros_like(mask)
            rows = torch.arange(mask.shape[0], device=mask.device)
            stop_target[rows, (lengths - 1).clamp(min=0)] = ends.float()
        else:
            stop_target = stop_target.float()
        sw = 1.0 + (self.stop_pos_weight - 1.0) * stop_target
        stop_raw = F.binary_cross_entropy_with_logits(stop_logits, stop_target,
                                                      reduction="none")
        stop = self._masked_mean(stop_raw * sw, mask)

        return {"bce": bce, "reg_l1": reg_l1, "reg_l2": reg_l2, "kl": kl,
                "flux": flux, "stop": stop, "bits_per_step": bits,
                "n_valid": mask.sum().detach()}

    # ---------------------------------------------------------------- perdida
    def loss_terms(self, out: dict, target: torch.Tensor,
                   mask: Optional[torch.Tensor] = None,
                   stop_target: Optional[torch.Tensor] = None) -> dict:
        """Perdida completa de MELLE.

        Parameters
        ----------
        out : dict devuelto por forward.
        target : Float[B,T,88] frame objetivo alineado con out["logits"]
            (target[:, t] es el frame que predice logits[:, t]).
        mask : Float[B,T] con 1 = posicion valida.  Las ventanas del final de
            una pieza llevan padding, asi que hay que enmascarar.  None = todo
            valido.
        stop_target : Float[B,T] opcional, 1 donde la pieza termina de verdad.
            Si es None se deduce del mask: stop=1 en la ultima posicion valida
            SOLO si la ventana lleva padding (mask.sum < T), porque eso es lo
            unico que indica que la pieza se acabo dentro de la ventana.  Una
            ventana interior (mask todo 1) recibe objetivo todo ceros.

        Returns
        -------
        dict con "bce", "reg_l1", "reg_l2", "kl", "flux", "stop",
        "bits_per_step" (diagnostico sin reponderar), "n_valid" y "loss" total.
        Los pesos aplicados son cfg.kl_weight, cfg.flux_weight y los del
        constructor (bce_weight, reg_weight, stop_weight).
        """
        logits = out["logits"]
        if target.shape != logits.shape:
            raise ValueError(f"target {tuple(target.shape)} no encaja con "
                             f"logits {tuple(logits.shape)}")
        if mask is None:
            mask = logits.new_ones(logits.shape[:2])

        pred = dict(out)
        prev = out.get("prev")
        flux_mask = None
        if prev is None or prev.shape != target.shape:
            # sin el frame de entrada: usar target desplazado y excluir t=0 de
            # la flux loss (ahi no existe "frame anterior" conocido)
            prev = torch.cat([target[:, :1], target[:, :-1]], dim=1)
            flux_mask = mask.clone()
            flux_mask[:, 0] = 0.0
        pred["prev"] = prev

        t = self._terms(pred, target, mask, stop_target=stop_target,
                        flux_mask=flux_mask)
        t["loss"] = (self.bce_weight * t["bce"]
                     + self.reg_weight * (t["reg_l1"] + t["reg_l2"])
                     + self.kl_weight * t["kl"]
                     + self.flux_weight * t["flux"]
                     + self.stop_weight * t["stop"])
        return t

    # --------------------------------------------------------------- muestreo
    @torch.no_grad()
    def sample_next(self, x: torch.Tensor, temperature: float = 1.0,
                    sample_latent: bool = True,
                    threshold: Optional[float] = None) -> torch.Tensor:
        """Muestrea el siguiente frame binario [B,88] dado el contexto x [B,T,88].

        Muestreo O(T^2): el backbone no es recurrente, luego supports_state()
        sigue siendo False.  `sample_latent=True` reactiva el latente gaussiano
        aunque el modulo este en eval (es la fuente de diversidad de MELLE);
        `threshold` fuerza una decision determinista en vez de muestrear la
        Bernoulli.
        """
        was_training = self.training
        self.eval()
        try:
            if sample_latent:
                # forward manual para muestrear z tambien en eval.  El pasado usa
                # z = mu (determinista) y solo el ultimo paso se muestrea; asi el
                # post-net causal recibe la secuencia completa de logits_pre, como
                # en entrenamiento, y no un contexto de ceros.
                T = x.shape[1]
                h = self.prenet(x)
                h = h + self.pe[:T].unsqueeze(0).to(h.dtype)
                for blk in self.blocks:
                    h = blk(h)
                h = self.ln_f(h)
                mu = self.to_mu(h)
                logvar = self.to_logvar(h).clamp(-LOGVAR_CLAMP, LOGVAR_CLAMP)
                z = mu.clone()
                sigma = torch.exp(0.5 * logvar[:, -1])
                z[:, -1] = mu[:, -1] + sigma * torch.randn_like(sigma)
                f = self.z_proj(z)
                lp = self.roll_head(f)
                logits = (lp + self.postnet(lp))[:, -1]
            else:
                logits = self.forward(x)["logits"][:, -1]
        finally:
            if was_training:
                self.train()
        p = torch.sigmoid(logits.float() / max(float(temperature), 1e-6))
        if threshold is not None:
            return (p > float(threshold)).to(x.dtype)
        return torch.bernoulli(p).to(x.dtype)
