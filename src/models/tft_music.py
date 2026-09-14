"""Temporal Fusion Transformer (Lim, Arik, Loeff & Pfister, 2021) portado a un
modelo de lenguaje AUTOREGRESIVO CAUSAL sobre tokens musicales.

Que se conserva del paper
-------------------------
1. GatedResidualNetwork (GRN)      : Dense -> ELU -> Dense -> GLU -> Add & Norm,
                                     con contexto estatico opcional. Ladrillo basico.
2. VariableSelectionNetwork (VSN)  : pesos softmax por variable producidos por una
                                     GRN sobre la concatenacion de todas las
                                     variables; cada variable pasa por su propia GRN.
3. Encoder LSTM unidireccional     : "locality enhancement" (el paper usa un
                                     seq2seq LSTM; aqui solo existe el pasado, asi
                                     que queda un unico LSTM causal) + gate residual.
4. Static covariate encoder        : 4 GRNs sobre un embedding estatico que
                                     inicializa el LSTM y enriquece la atencion.
5. Interpretable Multi-Head Attn   : cabezas con matriz de VALORES COMPARTIDA y
                                     promedio de cabezas, mas mascara causal.
6. Position-wise feed-forward GRN  + gate + Add & Norm, y cabeza lineal a V.

Que se elimina (no aplica a un LM causal)
-----------------------------------------
* Decoder de horizonte futuro y covariables conocidas a futuro: en generacion
  autoregresiva no existe nada conocido en t+1. Solo queda la rama "encoder".
* Salida por cuantiles / quantile loss: la salida es una distribucion
  CATEGORICA sobre el vocabulario (V=155) con cabeza lineal.
* Metadatos estaticos reales: el dataset no los tiene. El vector estatico se
  deriva del PROMEDIO ACUMULADO causal de los embeddings seleccionados.

Variables de entrada de la VSN (las "input covariates" del TFT)
---------------------------------------------------------------
Todas son funcion PUNTUAL del token en t (o de su indice), por lo que son
causales por construccion:
    token_id      embedding del identificador de token          (V=155)
    token_type    embedding del tipo: especial / nota / shift    (3)
    pitch         embedding del pitch, o id 88 "sin pitch"       (89)
    shift_dur     embedding de la duracion del shift, o 64       (65)
    pitch_class   embedding de pitch % 12, o 12 "sin pitch"      (13)
    log_pos       feature continua log(1 + posicion) normalizada (1 -> d_model)

Causalidad
----------
logits[:, t] depende solo de x[:, :t+1]. Los tres puntos delicados son:
  * el promedio acumulado del contexto estatico -> cumsum(., dim=1)/(t+1),
    estrictamente causal (nunca mira t+1);
  * el LSTM -> unidireccional, y su estado inicial se toma del contexto
    estatico en t=0 (que depende solo de x[:, 0]);
  * la atencion -> mascara triangular superior estricta (diagonal=1).
Verificado con tests/test_causality.py y con una perturbacion del ULTIMO token.

Notas de configuracion
----------------------
* cfg.n_layers se mapea a  n_blocks = max(1, n_layers // 2)  bloques de fusion
  temporal (el TFT original tiene UNO solo). Cada bloque pesa ~1.3x una capa
  transformer estandar por sus dos GRNs de ancho completo, y esta division
  deja el modelo dentro del presupuesto de 15-45 M parametros.
* El ancho oculto de todas las GRNs es d_model, como en el paper; por eso
  cfg.d_ff, cfg.hidden, cfg.rel_attn y cfg.max_rel_dist NO se usan (el TFT no
  tiene FF de ancho propio ni atencion relativa: la posicion la aporta el LSTM).

Rendimiento (RTX 4070 SUPER 12.9 GB, torch 2.11+cu128, bf16, L=1024, fwd+bwd+step
con torch.cuda.synchronize alrededor de cada iteracion)
---------------------------------------------------------------------------
  31.20 M parametros con la Config por defecto (d_model=512, n_layers=8 -> 4 bloques)
  batch 16 -> 5.37 it/s (mediana) / 5.56 (mejor),  88.0 k tokens/s, pico 5.69 GiB
  batch 24 -> 3.81 it/s (mediana) / 3.88 (mejor),  93.7 k tokens/s, pico 8.15 GiB
Ambos lotes caben con holgura en 12.9 GB. OJO: la GPU de este laboratorio se
comparte entre agentes; con la GPU al 100% por otro proceso la misma
configuracion cae a 2.5 it/s, asi que cualquier it/s debe reportarse junto al
estado de la GPU.
La atencion usa scaled_dot_product_attention en el camino de entrenamiento (sin
materializar L x L) y el camino explicito con mascara solo en interpret();
ambos coinciden a 3.6e-7 relativo en fp32 y a 1.1e-16 en fp64.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TokenARModel, causal_mask

try:                                    # sys.path incluye 'src' (ver tests/)
    from data.tokenizer import (VOCAB_SIZE, NOTE_OFF_ID, SHIFT_OFF_ID,
                                N_PITCH, MAX_SHIFT)
except Exception:                       # respaldo: constantes documentadas
    VOCAB_SIZE, NOTE_OFF_ID, SHIFT_OFF_ID, N_PITCH, MAX_SHIFT = 155, 3, 91, 88, 64

VAR_NAMES = ("token_id", "token_type", "pitch", "shift_dur", "pitch_class", "log_pos")
N_VARS = len(VAR_NAMES)

# ids de los embeddings "ausente"
NO_PITCH = N_PITCH          # 88
NO_SHIFT = MAX_SHIFT        # 64
NO_PC = 12
TYPE_SPECIAL, TYPE_NOTE, TYPE_SHIFT = 0, 1, 2


# ---------------------------------------------------------------------------
# Bloques del TFT
# ---------------------------------------------------------------------------
class GatedLinearUnit(nn.Module):
    """GLU(gamma) = sigmoid(W4 gamma + b4) * (W5 gamma + b5).

    Es el mecanismo con el que el TFT decide cuanta capacidad usar en cada
    componente: si la compuerta se satura a 0 el bloque queda anulado.
    """

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_in, 2 * d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.fc(self.drop(x)).chunk(2, dim=-1)
        return torch.sigmoid(gate) * value


class GateAddNorm(nn.Module):
    """GLU + conexion residual + LayerNorm (el 'Add & Norm' con compuerta)."""

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.glu = GatedLinearUnit(d_model, d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(self.glu(x) + skip)


class GatedResidualNetwork(nn.Module):
    """GRN del paper (ecuaciones 2-4):

        eta2 = ELU(W2 a + W3 c + b2)
        eta1 = W1 eta2 + b1
        GRN  = LayerNorm(a + GLU(eta1))

    `c` es el contexto estatico (opcional, sin bias como en el paper). Si las
    dimensiones de entrada y salida difieren se proyecta el residuo.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int,
                 dropout: float = 0.0, d_context: int | None = None):
        super().__init__()
        self.skip = None if d_in == d_out else nn.Linear(d_in, d_out, bias=False)
        self.fc_in = nn.Linear(d_in, d_hidden)
        self.fc_ctx = None if d_context is None else nn.Linear(d_context, d_hidden, bias=False)
        self.fc_hid = nn.Linear(d_hidden, d_hidden)
        self.glu = GatedLinearUnit(d_hidden, d_out, dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, a: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        res = a if self.skip is None else self.skip(a)
        h = self.fc_in(a)
        if self.fc_ctx is not None and c is not None:
            h = h + self.fc_ctx(c)
        h = self.fc_hid(F.elu(h))
        return self.norm(res + self.glu(h))


class VariableSelectionNetwork(nn.Module):
    """Seleccion instantanea de variables (seccion 4.2 del paper).

    Entrada  v : [B, L, n_vars, d_model]  (cada variable ya proyectada a d_model)
    Salida   (combinado [B, L, d_model], pesos [B, L, n_vars])

    Los pesos salen de una GRN sobre la CONCATENACION de todas las variables
    seguida de softmax; cada variable se transforma con su PROPIA GRN (pesos
    compartidos a lo largo del tiempo). Todo es puntual en t -> causal.
    """

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.0,
                 d_context: int | None = None):
        super().__init__()
        self.n_vars = n_vars
        self.weight_grn = GatedResidualNetwork(
            n_vars * d_model, d_model, n_vars, dropout, d_context)
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(d_model, d_model, d_model, dropout)
            for _ in range(n_vars)])

    def forward(self, v: torch.Tensor, c: torch.Tensor | None = None):
        flat = v.flatten(start_dim=-2)                        # [B,L,n_vars*d]
        w = torch.softmax(self.weight_grn(flat, c), dim=-1)   # [B,L,n_vars]
        proc = torch.stack([grn(v[..., i, :]) for i, grn in enumerate(self.var_grns)],
                           dim=-2)                            # [B,L,n_vars,d]
        return (w.unsqueeze(-1) * proc).sum(dim=-2), w


class InterpretableMultiHeadAttention(nn.Module):
    """Atencion interpretable del TFT (seccion 4.4) con mascara causal.

    Todas las cabezas COMPARTEN la matriz de valores W_V y sus salidas se
    PROMEDIAN (en vez de concatenarse). Asi cada cabeza opera sobre el mismo
    subespacio de valores y el promedio de los pesos de atencion sigue siendo
    una distribucion valida e interpretable sobre el pasado.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser divisible por n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = float(dropout)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, self.d_head, bias=False)   # COMPARTIDA
        self.w_out = nn.Linear(self.d_head, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                need_weights: bool = False):
        B, L, _ = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.w_q(x).view(B, L, H, dh).transpose(1, 2)        # [B,H,L,dh]
        k = self.w_k(x).view(B, L, H, dh).transpose(1, 2)        # [B,H,L,dh]
        v = self.w_v(x).unsqueeze(1)                             # [B,1,L,dh] compartida

        if need_weights:
            # camino explicito: materializa la matriz [B,H,L,L] para inspeccion
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)
            if mask is None:
                mask = torch.triu(torch.ones(L, L, dtype=torch.bool,
                                             device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            # la diagonal nunca se enmascara -> ninguna fila queda entera en -inf
            ctx = torch.matmul(self.attn_drop(attn), v)          # [B,H,L,dh]
        else:
            # camino rapido: SDPA (flash / memory-efficient) con is_causal.
            # Identico al anterior: mean_h softmax(Q K^T / sqrt(dh)) V, con V
            # compartida (se replica por cabeza con expand, coste despreciable).
            # Evita materializar L x L -> ~2 GB menos a B=16, L=1024.
            ctx = F.scaled_dot_product_attention(
                q, k, v.expand(B, H, L, dh).contiguous(),
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True)
            attn = None
        ctx = ctx.mean(dim=1)                                    # promedio de cabezas
        out = self.out_drop(self.w_out(ctx))
        return out, attn


class TemporalFusionBlock(nn.Module):
    """Enriquecimiento estatico -> atencion causal -> GRN position-wise.

    Reproduce las capas 'static enrichment', 'temporal self-attention' y
    'position-wise feed-forward' del TFT, cada una con su GateAddNorm.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.enrich = GatedResidualNetwork(d_model, d_model, d_model, dropout,
                                          d_context=d_model)
        self.attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.gate_attn = GateAddNorm(d_model, dropout)
        self.ff = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.gate_ff = GateAddNorm(d_model, dropout)

    def forward(self, phi: torch.Tensor, c_enrich: torch.Tensor,
                mask: torch.Tensor | None = None, need_weights: bool = False):
        theta = self.enrich(phi, c_enrich)                       # theta(t,n)
        attn_out, w = self.attn(theta, mask, need_weights)
        delta = self.gate_attn(attn_out, theta)                  # delta(t,n)
        psi = self.ff(delta)                                     # psi(t,n)
        return self.gate_ff(psi, phi), w                         # skip a phi (paper)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class TFTMusic(TokenARModel):
    """TFT causal para tokens musicales.  forward: [B,L] long -> [B,L,155]."""

    name = "tft"

    def __init__(self, cfg):
        super().__init__()
        d = int(cfg.d_model)
        self.d_model = d
        self.dropout = float(getattr(cfg, "dropout", 0.1))
        self.n_blocks = max(1, int(cfg.n_layers) // 2)
        self.vocab_size = VOCAB_SIZE
        self.max_len = int(getattr(cfg, "seq_len", 1024))

        # --- tablas de features derivadas del token (constantes, sin gradiente) ---
        ids = torch.arange(VOCAB_SIZE)
        is_note = (ids >= NOTE_OFF_ID) & (ids < SHIFT_OFF_ID)
        is_shift = ids >= SHIFT_OFF_ID
        type_of = torch.full((VOCAB_SIZE,), TYPE_SPECIAL, dtype=torch.long)
        type_of[is_note] = TYPE_NOTE
        type_of[is_shift] = TYPE_SHIFT
        pitch_of = torch.full((VOCAB_SIZE,), NO_PITCH, dtype=torch.long)
        pitch_of[is_note] = ids[is_note] - NOTE_OFF_ID
        shift_of = torch.full((VOCAB_SIZE,), NO_SHIFT, dtype=torch.long)
        shift_of[is_shift] = ids[is_shift] - SHIFT_OFF_ID
        pc_of = torch.full((VOCAB_SIZE,), NO_PC, dtype=torch.long)
        pc_of[is_note] = (ids[is_note] - NOTE_OFF_ID) % 12
        self.register_buffer("type_of", type_of, persistent=False)
        self.register_buffer("pitch_of", pitch_of, persistent=False)
        self.register_buffer("shift_of", shift_of, persistent=False)
        self.register_buffer("pc_of", pc_of, persistent=False)

        # feature continua log(1+pos) normalizada a ~[0,1]
        self.register_buffer(
            "pos_feat", self._make_pos_feat(self.max_len, self.max_len),
            persistent=False)

        # --- embeddings de las variables (cada una a d_model) ---
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d)
        self.type_emb = nn.Embedding(3, d)
        self.pitch_emb = nn.Embedding(N_PITCH + 1, d)
        self.shift_emb = nn.Embedding(MAX_SHIFT + 1, d)
        self.pc_emb = nn.Embedding(13, d)
        self.pos_proj = nn.Linear(1, d)

        # --- seleccion de variables ---
        # Nota de diseno: la VSN del paper recibe el contexto estatico c_s. Aqui
        # el vector estatico se DERIVA de la salida de la VSN (no hay metadatos
        # estaticos), asi que realimentarlo seria circular; la VSN va sin
        # contexto (el paper declara ese termino opcional en la GRN).
        self.vsn = VariableSelectionNetwork(N_VARS, d, self.dropout)

        # --- locality enhancement: LSTM unidireccional (causal) + gate ---
        self.lstm = nn.LSTM(d, d, num_layers=1, batch_first=True)
        self.gate_lstm = GateAddNorm(d, self.dropout)

        # --- static covariate encoder (4 GRNs como en el paper) ---
        self.static_grn = GatedResidualNetwork(d, d, d, self.dropout)   # embedding estatico
        self.static_h = GatedResidualNetwork(d, d, d, self.dropout)     # c_h
        self.static_c = GatedResidualNetwork(d, d, d, self.dropout)     # c_c
        self.static_e = GatedResidualNetwork(d, d, d, self.dropout)     # c_e

        # --- bloques de fusion temporal ---
        self.blocks = nn.ModuleList([
            TemporalFusionBlock(d, int(cfg.n_heads), self.dropout)
            for _ in range(self.n_blocks)])

        # --- cabeza categorica (NO cuantiles) ---
        self.head = nn.Linear(d, VOCAB_SIZE, bias=False)

        self._init_weights()
        # el atado se hace DESPUES de inicializar para que la matriz compartida
        # conserve la escala de embedding (std 0.02); con la init por defecto de
        # nn.Embedding (N(0,1)) los logits salian con std ~sqrt(d)=22 y la
        # perdida inicial era ~128 en vez de ln(155)=5.04.
        if bool(getattr(cfg, "tie_weights", True)):
            self.head.weight = self.tok_emb.weight

        # ultimos pesos de VSN promediados en el tiempo, [B, n_vars] (detached)
        self.last_vsn_weights: torch.Tensor | None = None

    # -- inicializacion -----------------------------------------------------
    def _init_weights(self) -> None:
        """Xavier en las capas densas (como la implementacion de referencia del
        TFT), ortogonal en las recurrencias del LSTM y embeddings pequenos."""
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.LayerNorm):
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.LSTM):
                for pname, p in mod.named_parameters():
                    if "weight_ih" in pname:
                        nn.init.xavier_uniform_(p)
                    elif "weight_hh" in pname:
                        nn.init.orthogonal_(p)
                    elif "bias" in pname:
                        nn.init.zeros_(p)
        for emb in (self.tok_emb, self.type_emb, self.pitch_emb,
                    self.shift_emb, self.pc_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    # -- utilidades ---------------------------------------------------------
    @staticmethod
    def _make_pos_feat(length: int, scale: int, device=None) -> torch.Tensor:
        """log(1+pos) / log(1+scale-1), normalizada a ~[0,1] para pos < scale.

        Se construye directamente en `device` para no pagar una copia
        host->device en cada forward cuando L > cfg.seq_len.
        """
        pos = torch.arange(length, dtype=torch.float32, device=device)
        return (torch.log1p(pos) / math.log1p(max(scale - 1, 1))).view(1, length, 1)

    def _variables(self, x: torch.Tensor) -> torch.Tensor:
        """[B,L] long -> [B,L,n_vars,d_model]. Todo puntual en t."""
        B, L = x.shape
        if L <= self.pos_feat.shape[1]:
            pf = self.pos_feat[:, :L]
        else:                                   # ventana mas larga que cfg.seq_len
            pf = self._make_pos_feat(L, self.max_len, self.pos_feat.device)
        pf = pf.to(self.tok_emb.weight.dtype).expand(B, L, 1)
        return torch.stack([
            self.tok_emb(x),
            self.type_emb(self.type_of[x]),
            self.pitch_emb(self.pitch_of[x]),
            self.shift_emb(self.shift_of[x]),
            self.pc_emb(self.pc_of[x]),
            self.pos_proj(pf),
        ], dim=-2)

    @staticmethod
    def _causal_cummean(z: torch.Tensor) -> torch.Tensor:
        """Media acumulada ESTRICTAMENTE causal: out[t] = mean(z[0..t]).

        Se acumula en AL MENOS float32 aunque el modelo corra en bf16/fp16
        (sumar 1024 terminos en bf16 pierde precision apreciable), pero sin
        degradar nunca un modelo en float64: `.float()` a secas rebajaba fp64
        a fp32 y metia un error de 1e-7 en las auditorias de precision.
        """
        L = z.shape[1]
        acc = z.dtype if z.dtype in (torch.float32, torch.float64) else torch.float32
        denom = torch.arange(1, L + 1, device=z.device, dtype=acc).view(1, L, 1)
        return (z.to(acc).cumsum(dim=1) / denom).to(z.dtype)

    # -- nucleo -------------------------------------------------------------
    def _run(self, x: torch.Tensor, collect: bool = False):
        assert x.dim() == 2, "se espera x [B,L] de tipo entero"
        # el dataset entrega int64, pero tokens.bin es uint8: nn.Embedding solo
        # acepta long/int, asi que normalizamos (no copia si ya es long).
        if x.dtype != torch.long:
            x = x.long()
        L = x.shape[1]

        # 1) variables + seleccion instantanea
        v = self._variables(x)
        sel, vsn_w = self.vsn(v)                       # [B,L,d], [B,L,n_vars]
        self.last_vsn_weights = vsn_w.detach().mean(dim=1)

        # 2) static covariate encoder sobre la media acumulada causal
        static = self.static_grn(self._causal_cummean(sel))       # [B,L,d]
        c_e = self.static_e(static)                               # enriquecimiento
        # el estado inicial del LSTM es un vector ESTATICO: se toma en t=0, que
        # depende unicamente de x[:, 0] (causal para todo t >= 0)
        h0 = self.static_h(static[:, :1]).transpose(0, 1).contiguous()   # [1,B,d]
        c0 = self.static_c(static[:, :1]).transpose(0, 1).contiguous()

        # 3) locality enhancement (LSTM unidireccional) + gate residual.
        # cuDNN NO tiene kernel de RNN en bfloat16: dentro de una region
        # autocast("cuda", bfloat16) nn.LSTM devuelve FLOAT16 (medido). El
        # forward no desborda (h esta acotado por tanh, |h| <= 1), pero el
        # backward de 1024 pasos recurrentes se calcularia en fp16 SIN
        # GradScaler (el laboratorio usa bf16 justo para no necesitarlo), con
        # riesgo de underflow en las dependencias largas. Se ejecuta fuera de
        # autocast en el dtype de sus propios pesos: coste medido x0.97 en
        # it/s (dentro del ruido) y adios al fp16 silencioso. Ademas arregla
        # autocast('cpu', bfloat16), que antes petaba en oneDNN ("could not
        # create a primitive descriptor for the LSTM forward propagation").
        w_dtype = self.lstm.weight_ih_l0.dtype
        with torch.autocast(device_type=sel.device.type, enabled=False):
            lstm_out, _ = self.lstm(sel.to(w_dtype),
                                    (h0.to(w_dtype), c0.to(w_dtype)))
        phi = self.gate_lstm(lstm_out.to(sel.dtype), sel)         # [B,L,d]

        # 4) bloques de fusion temporal con atencion interpretable enmascarada.
        # La mascara explicita solo se construye si hay que devolver los pesos;
        # en el camino rapido la causalidad la impone SDPA con is_causal=True.
        mask = causal_mask(L, x.device).view(1, 1, L, L) if collect else None
        attn_all = []
        for blk in self.blocks:
            phi, w = blk(phi, c_e, mask, need_weights=collect)
            if collect:
                attn_all.append(w)

        logits = self.head(phi)                                   # [B,L,V]
        if not collect:
            return logits, None
        return logits, {"vsn_weights_t": vsn_w, "attn_heads": attn_all}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x, collect=False)[0]

    # -- interpretabilidad --------------------------------------------------
    @torch.no_grad()
    def interpret(self, x: torch.Tensor, heads: bool = False) -> dict:
        """Pesos de seleccion de variables y de atencion.

        Devuelve
            variable_names  : tuple[str]              nombres de las n_vars variables
            vsn_weights     : [B, n_vars]             promedio temporal (que mira el modelo)
            vsn_weights_t   : [B, L, n_vars]          traza temporal
            attention       : [B, n_blocks, L, L]     atencion promediada por cabeza
            attention_heads : [B, n_blocks, H, L, L]  solo si heads=True (pesa mucho)
            logits          : [B, L, V]

        OJO con la memoria: attention crece como B*n_blocks*L^2; usa lotes
        pequenos (B <= 4) para L = 1024.
        """
        was_training = self.training
        self.eval()
        logits, info = self._run(x, collect=True)
        vsn_w = info["vsn_weights_t"]
        att = torch.stack([a.mean(dim=1) for a in info["attn_heads"]], dim=1)
        out = {
            "variable_names": VAR_NAMES,
            "vsn_weights": vsn_w.mean(dim=1).float().cpu(),
            "vsn_weights_t": vsn_w.float().cpu(),
            "attention": att.float().cpu(),
            "logits": logits.float().cpu(),
        }
        if heads:
            out["attention_heads"] = torch.stack(info["attn_heads"], dim=1).float().cpu()
        if was_training:
            self.train()
        return out

    # -- muestreo -----------------------------------------------------------
    def supports_state(self) -> bool:
        """False: hay atencion sobre todo el pasado, no es un recurrente puro;
        el muestreo O(1) exigiria cache KV ademas del estado del LSTM."""
        return False

    def param_report(self) -> str:
        return (f"{self.name} [{self.family}] {self.n_params()/1e6:.2f} M parametros "
                f"(d_model={self.d_model}, bloques={self.n_blocks}, n_vars={N_VARS})")
