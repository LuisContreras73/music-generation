"""Muestreo autoregresivo y decodificacion a piano-roll.

Familia token
-------------
Muestreo con ventana deslizante (sin KV-cache: el contexto se recorta a las
ultimas cfg.seq_len posiciones). Se genera hasta acumular `n_steps` pasos de
50 ms, no hasta un numero fijo de tokens, para que todas las muestras tengan
la MISMA duracion musical y sean comparables.

Se registran DOS medidas de aprendizaje de la gramatica de la representacion.
En el corpus real es imposible que una nota se repita dentro del mismo paso
temporal (la tokenizacion emite un conjunto ordenado por paso), asi que la
probabilidad que el modelo asigne a esa repeticion mide directamente cuanto ha
aprendido de la estructura:

  grammar_prob_mass  masa de probabilidad media que el modelo pone sobre notas
                     ya emitidas en el paso actual, medida ANTES de enmascarar.
                     Es la metrica valida cuando enforce_grammar=True, porque en
                     ese caso el token prohibido nunca llega a muestrearse.
  grammar_violation_rate  proporcion de tokens de nota realmente muestreados que
                     repiten una nota del paso actual. Solo es informativa con
                     enforce_grammar=False; con el enmascarado activo vale 0 por
                     construccion.

Familia frame
-------------
Muestreo Bernoulli independiente por nota dentro de cada frame. Es la
limitacion intrinseca de la familia (no modela la correlacion intra-frame,
es decir los acordes); se documenta y se mide.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from data.tokenizer import (PAD, BOS, EOS, NOTE_OFF_ID, SHIFT_OFF_ID, VOCAB_SIZE,
                            MAX_SHIFT, N_PITCH, decode_tokens, is_shift, tok_shift)


def _filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """Aplica top-k y/o nucleus (top-p) sobre la ultima dimension."""
    if top_k and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        kth = logits.topk(k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        srt, idx = torch.sort(logits, descending=True, dim=-1)
        probs = srt.softmax(-1).cumsum(-1)
        drop = probs - srt.softmax(-1) >= top_p          # deja siempre >=1 token
        srt = srt.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, srt)
    return logits


@torch.no_grad()
def sample_tokens(model, prefix: torch.Tensor, n_steps: int, *, seq_len: int = 1024,
                  temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0,
                  enforce_grammar: bool = True, allow_eos: bool = False,
                  max_tokens: int | None = None, device: str = "cuda",
                  generator: torch.Generator | None = None) -> dict:
    """Continua `prefix` [B, L0] hasta acumular n_steps pasos de 50 ms por muestra.

    Devuelve dict con:
      tokens   list[np.ndarray]  secuencia COMPLETA (prefijo + continuacion) por muestra
      new      list[np.ndarray]  solo los tokens generados
      steps    np.ndarray        pasos temporales generados por muestra
      grammar_violation_rate float
      n_tokens_generated int
    """
    model.eval()
    x = prefix.to(device).long()
    B = x.shape[0]
    max_tokens = max_tokens or int(n_steps * 3 + 64)

    steps = torch.zeros(B, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    emitted = [set() for _ in range(B)]          # notas ya puestas en el paso actual
    new_tokens: list[list[int]] = [[] for _ in range(B)]
    n_note_tok = 0
    n_violations = 0
    gram_mass_sum = 0.0
    gram_mass_n = 0

    ban = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    ban[PAD] = True; ban[BOS] = True
    if not allow_eos:
        ban[EOS] = True

    for _ in range(max_tokens):
        if bool(done.all()):
            break
        ctx = x[:, -seq_len:]
        logits = model(ctx)[:, -1, :].float()
        logits = logits.masked_fill(ban, float("-inf"))

        # masa de probabilidad sobre notas imposibles, ANTES de enmascarar:
        # es la medida valida de aprendizaje gramatical cuando se enmascara
        probs_raw = logits.softmax(-1)
        for b in range(B):
            if emitted[b] and not bool(done[b]):
                idxs = [NOTE_OFF_ID + p for p in emitted[b]]
                gram_mass_sum += float(probs_raw[b, idxs].sum())
                gram_mass_n += 1

        # no pasarse del objetivo de duracion: prohibe shifts que excedan n_steps
        remain = (n_steps - steps).clamp(min=0)
        for b in range(B):
            r = int(remain[b].item())
            if r < MAX_SHIFT:
                lo = SHIFT_OFF_ID + max(r, 0)     # SHIFT(d) con d > r
                if lo < VOCAB_SIZE:
                    logits[b, lo:] = float("-inf")
            if enforce_grammar and emitted[b]:
                idxs = [NOTE_OFF_ID + p for p in emitted[b]]
                logits[b, idxs] = float("-inf")

        if temperature != 1.0:
            logits = logits / max(temperature, 1e-5)
        logits = _filter_logits(logits, top_k, top_p)
        # si una fila quedo totalmente enmascarada, fuerza un SHIFT(1)
        dead = ~torch.isfinite(logits).any(-1)
        if bool(dead.any()):
            logits[dead] = float("-inf")
            logits[dead, SHIFT_OFF_ID] = 0.0
        nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)   # [B,1]

        for b in range(B):
            if bool(done[b]):
                continue
            t = int(nxt[b, 0].item())
            new_tokens[b].append(t)
            if t >= SHIFT_OFF_ID:                             # SHIFT
                steps[b] += t - SHIFT_OFF_ID + 1
                emitted[b] = set()
                if int(steps[b].item()) >= n_steps:
                    done[b] = True
            elif t >= NOTE_OFF_ID:                            # NOTE_ON
                p = t - NOTE_OFF_ID
                n_note_tok += 1
                if p in emitted[b]:
                    n_violations += 1
                emitted[b].add(p)
            else:                                             # EOS
                done[b] = True
        x = torch.cat([x, nxt], dim=1)

    pref = prefix.cpu().numpy()
    return dict(
        tokens=[np.concatenate([pref[b], np.asarray(new_tokens[b], np.int64)]) for b in range(B)],
        new=[np.asarray(new_tokens[b], np.int64) for b in range(B)],
        steps=steps.cpu().numpy(),
        grammar_violation_rate=float(n_violations / max(n_note_tok, 1)),
        grammar_prob_mass=float(gram_mass_sum / max(gram_mass_n, 1)),
        n_tokens_generated=int(sum(len(t) for t in new_tokens)),
    )


@torch.no_grad()
def sample_frames(model, prefix: torch.Tensor, n_new: int, *, seq_len: int = 1024,
                  temperature: float = 1.0, threshold: float | None = None,
                  device: str = "cuda", generator: torch.Generator | None = None) -> dict:
    """Continua un piano-roll [B, T0, 88] con n_new frames (familia frame)."""
    model.eval()
    x = prefix.to(device).float()
    for _ in range(n_new):
        out = model(x[:, -seq_len:])
        logits = (out["logits"] if isinstance(out, dict) else out)[:, -1, :].float()
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-5)
        p = logits.sigmoid()
        if threshold is None:
            nxt = (torch.rand(p.shape, device=device, generator=generator) < p).float()
        else:
            nxt = (p > threshold).float()
        x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
    rolls = x.cpu().numpy().astype(np.uint8)
    return dict(rolls=[r for r in rolls], new=[r[-n_new:] for r in rolls])


def tokens_to_roll(tokens, max_steps: int | None = None) -> np.ndarray:
    return decode_tokens(tokens, n_positions=N_PITCH, max_steps=max_steps)


def split_prime_continuation(tokens, prime_tokens: int, max_steps: int | None = None):
    """Roll del prefijo y roll completo, para dibujar la costura de la continuacion."""
    full = decode_tokens(tokens, max_steps=max_steps)
    n_prime_steps = 0
    for t in np.asarray(tokens[:prime_tokens]).ravel():
        if is_shift(int(t)):
            n_prime_steps += tok_shift(int(t))
    return full, n_prime_steps
