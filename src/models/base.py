"""Contrato comun de los modelos autoregresivos del laboratorio.

Dos familias
------------
TokenARModel  (family="token")
    forward(x: Long[B,L])                 -> Float[B,L,V]   logits del token t+1
    Vocabulario V=155 (ver data/tokenizer.py). PAD=0 se ignora en la perdida.

FrameARModel  (family="frame")
    forward(x: Float[B,T,88])             -> dict con:
        "logits" : Float[B,T,88]  logits Bernoulli del frame t+1  (OBLIGATORIO:
                   da una verosimilitud discreta comparable en bits/paso con
                   los modelos token-level)
        y opcionalmente "aux" : dict[str, Tensor] con terminos de perdida extra
                   (p.ej. regresion latente, KL, flux) ya reducidos a escalar.

REGLA DURA: causalidad estricta. logits[:, t] solo puede depender de x[:, :t+1].
Se verifica por gradiente en tests/test_causality.py; un modelo que no pase
ese test NO se entrena.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ARModel(nn.Module):
    family: str = "token"
    name: str = "base"

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_report(self) -> str:
        tot = self.n_params()
        return f"{self.name} [{self.family}] {tot/1e6:.2f} M parametros"


class TokenARModel(ARModel):
    family = "token"

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,L] -> [B,L,V]
        raise NotImplementedError

    # --- interfaz opcional de estado recurrente (LSTM la usa para muestreo O(1)) ---
    def supports_state(self) -> bool:
        return False

    def step(self, x_t: torch.Tensor, state):
        """Un paso: x_t [B,1] -> (logits [B,1,V], nuevo state)."""
        raise NotImplementedError


class FrameARModel(ARModel):
    family = "frame"

    def forward(self, x: torch.Tensor) -> dict:          # [B,T,88] -> dict
        raise NotImplementedError

    def supports_state(self) -> bool:
        return False

    def step(self, x_t: torch.Tensor, state):
        raise NotImplementedError


def causal_mask(L: int, device=None) -> torch.Tensor:
    """True donde se debe ENMASCARAR (posiciones futuras)."""
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
