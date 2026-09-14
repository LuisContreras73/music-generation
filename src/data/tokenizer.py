"""
Tokenizador de eventos para piano-roll de ONSETS [T,88] @ 20 Hz.

La grilla del dataset es plana (no hay compas): el histograma de onsets sobre
una grilla de 16 pasos es uniforme (0.0625 +/- 0.001). Por eso NO se usa la
tokenizacion REMI (Bar/Position) sino un esquema estilo Performance-RNN /
Music Transformer: NOTE_ON + TIME_SHIFT.

Vocabulario (V = 155):
    0            PAD
    1            BOS
    2            EOS
    3 .. 90      NOTE_ON(p),  p in [0, 88)      -> 88 tokens
    91 .. 154    SHIFT(d),    d in [1, 64]      -> 64 tokens

Invariantes:
  * Dentro de un paso temporal las notas se emiten en orden ascendente de
    pitch  -> orden canonico, biyectivo.
  * Un gap g > 64 se codifica como floor(g/64) x SHIFT(64) + SHIFT(g % 64).
    Cubre el 100% de los casos (gap max observado = 1200) y solo el 0.4% de
    los gaps supera 64, asi que el sobrecoste es despreciable.
  * encode -> decode es EXACTAMENTE la identidad (verificado en tests).
"""
from __future__ import annotations
import numpy as np

PAD, BOS, EOS = 0, 1, 2
NOTE_OFF_ID = 3                 # inicio del bloque NOTE_ON
N_PITCH = 88
SHIFT_OFF_ID = NOTE_OFF_ID + N_PITCH        # 91
MAX_SHIFT = 64
VOCAB_SIZE = SHIFT_OFF_ID + MAX_SHIFT       # 155

NOTE_MIN_MIDI = 21              # A0  (pitch 0 del roll)


def note_token(pitch: int) -> int:
    return NOTE_OFF_ID + int(pitch)


def shift_token(d: int) -> int:
    assert 1 <= d <= MAX_SHIFT, d
    return SHIFT_OFF_ID + d - 1


def is_note(tok) -> bool:
    return NOTE_OFF_ID <= tok < SHIFT_OFF_ID


def is_shift(tok) -> bool:
    return SHIFT_OFF_ID <= tok < VOCAB_SIZE


def tok_pitch(tok) -> int:
    return int(tok) - NOTE_OFF_ID


def tok_shift(tok) -> int:
    return int(tok) - SHIFT_OFF_ID + 1


def _emit_shift(out: list, g: int) -> None:
    """Codifica un gap arbitrario g >= 1 encadenando SHIFT(<=64)."""
    while g > MAX_SHIFT:
        out.append(shift_token(MAX_SHIFT))
        g -= MAX_SHIFT
    if g > 0:
        out.append(shift_token(g))


def encode_roll(roll: np.ndarray, add_bos: bool = True, add_eos: bool = True) -> np.ndarray:
    """[T,88] binario -> secuencia de tokens uint8.

    El tiempo se mide como gap entre pasos ACTIVOS. El silencio final se
    codifica con un ultimo shift para preservar T exactamente.
    """
    roll = np.asarray(roll)
    T = roll.shape[0]
    active = np.flatnonzero(roll.any(1))
    out: list[int] = [BOS] if add_bos else []
    prev = 0
    for t in active:
        if t > prev:
            _emit_shift(out, int(t - prev))
        for p in np.flatnonzero(roll[t]):
            out.append(note_token(int(p)))
        prev = int(t)
    if T > prev:                                   # silencio de cola -> conserva T
        _emit_shift(out, int(T - prev))
    if add_eos:
        out.append(EOS)
    return np.asarray(out, dtype=np.uint8)


def decode_tokens(tokens, n_positions: int = N_PITCH, max_steps: int | None = None) -> np.ndarray:
    """Secuencia de tokens -> [T,88] uint8. Inversa exacta de encode_roll."""
    events: list[tuple[int, int]] = []             # (t, pitch)
    t = 0
    for tok in np.asarray(tokens).ravel():
        tok = int(tok)
        if tok in (PAD, BOS):
            continue
        if tok == EOS:
            break
        if is_shift(tok):
            t += tok_shift(tok)
            if max_steps is not None and t >= max_steps:
                t = max_steps
                break
        elif is_note(tok):
            events.append((t, tok_pitch(tok)))
    T = t + 1 if events and events[-1][0] == t else t
    T = max(T, (events[-1][0] + 1) if events else 0)
    if max_steps is not None:
        T = min(T, max_steps)
    roll = np.zeros((T, n_positions), dtype=np.uint8)
    for tt, p in events:
        if tt < T:
            roll[tt, p] = 1
    return roll


def n_timesteps(tokens) -> int:
    """Duracion en pasos de 50 ms que representa una secuencia de tokens."""
    t = 0
    for tok in np.asarray(tokens).ravel():
        tok = int(tok)
        if tok == EOS:
            break
        if is_shift(tok):
            t += tok_shift(tok)
    return t


def token_names() -> list[str]:
    names = ["PAD", "BOS", "EOS"]
    names += [f"NOTE_{NOTE_MIN_MIDI + p}" for p in range(N_PITCH)]
    names += [f"SHIFT_{d}" for d in range(1, MAX_SHIFT + 1)]
    return names


def encode_roll_fast(roll: np.ndarray, add_bos: bool = True, add_eos: bool = True) -> np.ndarray:
    """Version vectorizada de encode_roll (identica salida, ~50x mas rapida)."""
    roll = np.asarray(roll)
    T = int(roll.shape[0])
    ts, ps = np.nonzero(roll)
    n_notes = len(ts)
    head = [BOS] if add_bos else []
    if n_notes == 0:
        out = list(head)
        if T > 0:
            _emit_shift(out, T)
        if add_eos:
            out.append(EOS)
        return np.asarray(out, dtype=np.uint8)

    new = np.empty(n_notes, bool); new[0] = True; new[1:] = ts[1:] != ts[:-1]
    grp_start = np.flatnonzero(new)
    uniq = ts[grp_start].astype(np.int64)
    cnt = np.diff(np.r_[grp_start, n_notes])
    gaps = np.diff(np.r_[0, uniq])
    k = np.where(gaps > 0, (gaps + MAX_SHIFT - 1) // MAX_SHIFT, 0)
    G = len(uniq)
    body = np.empty(int(k.sum() + n_notes), np.uint8)
    starts = np.r_[0, np.cumsum(k + cnt)[:-1]]

    if k.sum() > 0:                                  # tokens de tiempo
        gi = np.repeat(np.arange(G), k)
        j = np.arange(int(k.sum())) - np.repeat(np.r_[0, np.cumsum(k)[:-1]], k)
        last = j == (k[gi] - 1)
        val = np.where(last, gaps[gi] - (k[gi] - 1) * MAX_SHIFT, MAX_SHIFT)
        body[starts[gi] + j] = SHIFT_OFF_ID + val - 1
    ni = np.repeat(np.arange(G), cnt)                # tokens de nota
    jn = np.arange(n_notes) - np.repeat(grp_start, cnt)
    body[starts[ni] + k[ni] + jn] = NOTE_OFF_ID + ps

    tail: list[int] = []
    if T > int(uniq[-1]):
        _emit_shift(tail, T - int(uniq[-1]))
    pieces = [np.asarray(head, np.uint8), body, np.asarray(tail, np.uint8)]
    if add_eos:
        pieces.append(np.asarray([EOS], np.uint8))
    return np.concatenate([p for p in pieces if p.size])
