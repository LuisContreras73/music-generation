"""Datasets de ventanas: token-level (LM de eventos) y frame-level (piano-roll).

Los memmap se abren de forma PEREZOSA, uno por proceso, y se excluyen del
pickle. En Windows el DataLoader arranca los workers con spawn, asi que tiene
que serializar el objeto Dataset entero: con el memmap como atributo, pickle
intenta volcar los 566 MB de rolls_packed.bin por el pipe al proceso hijo y
falla con "OSError: [Errno 22] Invalid argument". Abrirlo perezosamente
tambien evita copiar esos datos a cada worker.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import PAD, N_PITCH

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_meta():
    m = np.load(PROC / "tokens_meta.npz", allow_pickle=True)
    splits = json.load(open(PROC / "splits.json"))
    return m, splits


def _fixed_starts(bases, lens, win, n_windows):
    """Ventanas deterministas stride=win dentro de cada pieza -> [(start, take)]."""
    starts = []
    for base, ln in zip(bases, lens):
        base, ln = int(base), int(ln)
        for s in range(0, max(ln - 1, 1), win):
            starts.append((base + s, min(win + 1, ln - s)))
    if n_windows and len(starts) > n_windows:
        sel = np.linspace(0, len(starts) - 1, n_windows).astype(int)
        starts = [starts[i] for i in sel]
    return starts


class TokenWindows(Dataset):
    """Ventanas de L+1 tokens que nunca cruzan el limite de una pieza.

    train: muestreo aleatorio con prob ~ longitud de la pieza (epoca virtual).
    val/test: ventanas deterministas, stride = L, cubriendo cada pieza.
    """

    def __init__(self, split: str, seq_len: int = 1024, n_windows: int = 0, seed: int = 0,
                 augment_cfg=None):
        m, splits = load_meta()
        self.tok_off = m["tok_offsets"].astype(np.int64)
        # La augmentacion SOLO se aplica en train: en val/test falsearia la medida.
        self.augment_cfg = augment_cfg if split == "train" else None
        self.idx = np.asarray(splits[split], dtype=np.int64)
        self.L = int(seq_len); self.split = split; self.seed = int(seed)
        self._tokens = None                       # se abre en cada proceso
        self.lens = (self.tok_off[self.idx + 1] - self.tok_off[self.idx]).astype(np.int64)
        if split == "train":
            p = self.lens.astype(np.float64); self.p = p / p.sum()
            self.n = int(n_windows) if n_windows else 200_000
            self._random = True
        else:
            self.starts = _fixed_starts(self.tok_off[self.idx], self.lens, self.L, n_windows)
            self.n = len(self.starts); self._random = False

    def _augment(self, chunk, rng):
        """Augmentacion en el worker. Nunca debe tumbar el entrenamiento."""
        try:
            from augment import augment_tokens
            return np.asarray(augment_tokens(chunk, rng, self.augment_cfg), np.int64)
        except Exception:
            return chunk

    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = np.memmap(PROC / "tokens.bin", dtype=np.uint8, mode="r")
        return self._tokens

    def __getstate__(self):
        d = self.__dict__.copy(); d["_tokens"] = None      # no viaja al worker
        return d

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._random:
            rng = np.random.default_rng((self.seed * 1_000_003 + i) & 0xFFFFFFFF)
            si = int(self.idx[int(rng.choice(len(self.idx), p=self.p))])
            a, b = int(self.tok_off[si]), int(self.tok_off[si + 1])
            st = a + (int(rng.integers(0, b - a - 1)) if (b - a) > self.L + 1 else 0)
            # margen extra: la augmentacion puede acortar la secuencia (thin_voices
            # borra notas, time_stretch redondea), y despues se recorta a L+1
            take = self.L + 1 if self.augment_cfg is None else int((self.L + 1) * 1.6)
            chunk = np.asarray(self.tokens[st: min(st + take, b)], np.int64)
            if self.augment_cfg is not None and len(chunk):
                chunk = self._augment(chunk, rng)[: self.L + 1]
        else:
            st, take = self.starts[i]
            chunk = np.asarray(self.tokens[st: st + take], np.int64)
        x = np.full(self.L + 1, PAD, np.int64)
        x[: len(chunk)] = chunk
        t = torch.from_numpy(x)
        return t[:-1].contiguous(), t[1:].contiguous()


class FrameWindows(Dataset):
    """Ventanas de T+1 frames binarios [T,88] desde rolls_packed.bin (bitpacked)."""

    def __init__(self, split: str, seq_len: int = 1024, n_windows: int = 0, seed: int = 0):
        m, splits = load_meta()
        self.roll_lens = m["roll_lens"].astype(np.int64)
        self.roll_off = np.r_[0, np.cumsum(self.roll_lens)]
        self.idx = np.asarray(splits[split], dtype=np.int64)
        self.T = int(seq_len); self.split = split; self.seed = int(seed)
        self._packed = None                       # se abre en cada proceso
        self.lens = self.roll_lens[self.idx]
        if split == "train":
            p = self.lens.astype(np.float64); self.p = p / p.sum()
            self.n = int(n_windows) if n_windows else 200_000
            self._random = True
        else:
            self.starts = _fixed_starts(self.roll_off[self.idx], self.lens, self.T, n_windows)
            self.n = len(self.starts); self._random = False

    @property
    def packed(self):
        if self._packed is None:
            self._packed = np.memmap(PROC / "rolls_packed.bin", dtype=np.uint8,
                                     mode="r").reshape(-1, 11)
        return self._packed

    def __getstate__(self):
        d = self.__dict__.copy(); d["_packed"] = None       # no viaja al worker
        return d

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._random:
            rng = np.random.default_rng((self.seed * 7_919 + i) & 0xFFFFFFFF)
            si = int(self.idx[int(rng.choice(len(self.idx), p=self.p))])
            a, ln = int(self.roll_off[si]), int(self.roll_lens[si])
            st = a + (int(rng.integers(0, ln - 1)) if ln > self.T + 1 else 0)
            take = min(self.T + 1, a + ln - st)
        else:
            st, take = self.starts[i]
        blk = np.unpackbits(np.asarray(self.packed[st: st + take]), axis=1)[:, :N_PITCH]
        out = np.zeros((self.T + 1, N_PITCH), np.float32)
        mask = np.zeros(self.T + 1, np.float32)
        out[: len(blk)] = blk; mask[: len(blk)] = 1.0
        x = torch.from_numpy(out); mk = torch.from_numpy(mask)
        return x[:-1].contiguous(), x[1:].contiguous(), mk[1:].contiguous()


def get_roll(seq_index: int) -> np.ndarray:
    """Piano-roll completo [T,88] de una pieza por indice global."""
    m, _ = load_meta()
    rl = m["roll_lens"].astype(np.int64); off = np.r_[0, np.cumsum(rl)]
    packed = np.memmap(PROC / "rolls_packed.bin", dtype=np.uint8, mode="r").reshape(-1, 11)
    a, ln = int(off[seq_index]), int(rl[seq_index])
    return np.unpackbits(np.asarray(packed[a:a + ln]), axis=1)[:, :N_PITCH]


def get_tokens(seq_index: int) -> np.ndarray:
    m, _ = load_meta()
    off = m["tok_offsets"].astype(np.int64)
    tokens = np.memmap(PROC / "tokens.bin", dtype=np.uint8, mode="r")
    return np.asarray(tokens[int(off[seq_index]):int(off[seq_index + 1])], np.int64)
