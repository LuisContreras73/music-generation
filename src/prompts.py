"""Construccion de prefijos (prompts) reales del corpus para condicionar la generacion.

Detalle importante: los prefijos token-level tienen todos EXACTAMENTE el mismo
numero de tokens, tomados literalmente del corpus. Asi el batch de muestreo no
necesita padding; meter PAD a la izquierda contaminaria el contexto causal,
porque en entrenamiento el PAD solo aparece a la derecha y el modelo nunca
aprendio a ignorarlo.
"""
from __future__ import annotations
import numpy as np
import torch

from data.datasets import get_roll, get_tokens, load_meta
from data.tokenizer import BOS, is_shift, tok_shift

TOKENS_PER_STEP = 0.697          # medido en el corpus (35.87 M tokens / 51.46 M pasos)


def steps_in(tokens) -> int:
    return int(sum(tok_shift(int(t)) for t in np.asarray(tokens).ravel() if is_shift(int(t))))


def make_token_prompts(n: int, prime_steps: int = 200, split: str = "val",
                       seed: int = 0, max_len: int = 1024):
    """n prefijos de LONGITUD FIJA en tokens (~prime_steps pasos), de piezas distintas.

    Devuelve (tensor [n, L] sin padding, pasos cubiertos por prefijo, indices de pieza).
    """
    n_tok = int(max(8, min(max_len - 1, round(prime_steps * TOKENS_PER_STEP))))
    _, splits = load_meta()
    idx = np.asarray(splits[split])
    rng = np.random.default_rng(seed)
    prefs, steps, srcs = [], [], []
    for si in rng.permutation(idx):
        tk = get_tokens(int(si))
        if len(tk) < n_tok + 40:
            continue
        # arranca en un punto interior aleatorio y alineado a un token de tiempo,
        # para no empezar siempre en el ataque inicial de la pieza
        lo = 1; hi = max(2, len(tk) - n_tok - 1)
        s = int(rng.integers(lo, hi))
        while s < len(tk) - n_tok - 1 and not is_shift(int(tk[s - 1])):
            s += 1
        p = np.concatenate([[BOS], tk[s:s + n_tok - 1]]).astype(np.int64)
        prefs.append(p); steps.append(steps_in(p)); srcs.append(int(si))
        if len(prefs) >= n:
            break
    return torch.from_numpy(np.stack(prefs)), np.asarray(steps), srcs


def make_frame_prompts(n: int, prime_steps: int = 200, split: str = "val", seed: int = 0):
    """n prefijos de piano-roll [n, prime_steps, 88] del split indicado."""
    _, splits = load_meta()
    idx = np.asarray(splits[split])
    rng = np.random.default_rng(seed)
    out, srcs = [], []
    for si in rng.permutation(idx):
        r = get_roll(int(si))
        if len(r) <= prime_steps + 10:
            continue
        s = int(rng.integers(0, len(r) - prime_steps - 1))
        out.append(r[s:s + prime_steps]); srcs.append(int(si))
        if len(out) >= n:
            break
    return torch.from_numpy(np.stack(out).astype(np.float32)), srcs
