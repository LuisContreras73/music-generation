"""Alimentacion de datos ENTERAMENTE EN GPU: sin DataLoader, sin workers, sin CPU.

Por que existe
--------------
El corpus tokenizado ocupa 35.87 M tokens en uint8, es decir **36 MB**. Eso cabe
sobrado en los 12.9 GB de la GPU. No hay ninguna razon para pasarlo por CPU en
cada paso: se sube una vez al arrancar y a partir de ahi el muestreo de ventanas
y la augmentacion son operaciones de tensores en el propio dispositivo.

Que problemas resuelve, todos medidos en este laboratorio:
  * la augmentacion en CPU hundia el entrenamiento de 3.20 a 0.65 it/s (5x);
  * los workers del DataLoader fallaban de forma intermitente en Windows con
    Python 3.14 (OSError WinError 1114 al cargar shm.dll en el worker,
    _queue.Empty al cerrar, y un segmentation fault que mato una tanda entera);
  * serializar un memmap hacia los workers con spawn reventaba con OSError 22.
Con este modulo no hay workers, ni spawn, ni transferencias por paso.

Augmentacion vectorizada en GPU
-------------------------------
transponer  : sumar un desplazamiento a los tokens NOTE_ON del batch. Se calcula
              el rango valido por muestra a partir de su nota minima y maxima,
              asi que nunca se sale de [0,88).
adelgazar   : el truco esta en identificar el paso temporal de cada nota sin
              bucles. cumsum sobre los tokens SHIFT da un identificador de grupo
              por posicion; dentro de cada grupo se conserva la nota mas aguda y
              las demas se eliminan con probabilidad p_drop. La eliminacion se
              hace compactando con argsort estable y rellenando con PAD, que es
              O(L log L) en GPU en vez de un bucle en Python.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
PAD, BOS, EOS = 0, 1, 2
NOTE_OFF, N_PITCH = 3, 88
SHIFT_OFF, MAX_SHIFT = 91, 64
VOCAB = 155


class GPUTokenStream:
    """Muestreador de ventanas de tokens que vive en la GPU.

    train: ventanas aleatorias, pieza elegida con probabilidad ~ su longitud.
    val/test: ventanas deterministas, para que la medida sea reproducible.
    """

    def __init__(self, split: str, seq_len: int, batch_size: int, device: str = "cuda",
                 seed: int = 0, augment: dict | None = None, max_windows: int = 0):
        m = np.load(PROC / "tokens_meta.npz", allow_pickle=True)
        splits = json.load(open(PROC / "splits.json"))
        off = m["tok_offsets"].astype(np.int64)
        idx = np.asarray(splits[split], dtype=np.int64)
        tokens = np.memmap(PROC / "tokens.bin", dtype=np.uint8, mode="r")

        # Se suben SOLO las piezas de esta particion, concatenadas, y se guardan
        # sus limites: asi una ventana nunca puede cruzar de una pieza a otra.
        chunks, starts, lens = [], [], []
        pos = 0
        for si in idx:
            a, b = int(off[si]), int(off[si + 1])
            n = b - a
            if n < 16:
                continue
            chunks.append(np.asarray(tokens[a:b]))
            starts.append(pos); lens.append(n); pos += n
        flat = np.concatenate(chunks)
        self.device = device
        self.tokens = torch.from_numpy(flat).to(device)             # uint8, ~36 MB
        self.starts = torch.tensor(starts, dtype=torch.long, device=device)
        self.lens = torch.tensor(lens, dtype=torch.long, device=device)
        self.L = int(seq_len); self.B = int(batch_size); self.split = split
        self.augment = augment or {}
        self.gen = torch.Generator(device=device); self.gen.manual_seed(int(seed))

        usable = (self.lens - 1).clamp(min=1).double()
        self.p = (usable / usable.sum()).float()
        self.n_pieces = len(lens)
        self.n_tokens = int(flat.size)

        if split != "train":                                        # ventanas fijas
            s_list = []
            for st, ln in zip(starts, lens):
                for o in range(0, max(ln - 1, 1), self.L):
                    s_list.append(st + o)
            if max_windows and len(s_list) > max_windows:
                sel = np.linspace(0, len(s_list) - 1, max_windows).astype(int)
                s_list = [s_list[i] for i in sel]
            self.fixed = torch.tensor(s_list, dtype=torch.long, device=device)
            self.fixed_end = torch.tensor(
                [min(st + self.L + 1, starts[np.searchsorted(starts, st, "right") - 1]
                     + lens[np.searchsorted(starts, st, "right") - 1]) for st in s_list],
                dtype=torch.long, device=device)
            self.cursor = 0

    # ------------------------------------------------------------------ muestreo
    def _gather(self, base: torch.Tensor, take: torch.Tensor | None = None):
        """base [B] posiciones de inicio -> ventana [B, L+1] con PAD donde falte."""
        ar = torch.arange(self.L + 1, device=self.device)
        pos = base[:, None] + ar[None, :]
        lim = self.tokens.numel() - 1
        out = self.tokens[pos.clamp(max=lim)].long()
        if take is not None:
            out = torch.where(ar[None, :] < take[:, None], out,
                              torch.full_like(out, PAD))
        return out

    def train_batch(self):
        pi = torch.multinomial(self.p, self.B, replacement=True, generator=self.gen)
        ln = self.lens[pi]
        span = (ln - 1).clamp(min=1)
        r = torch.rand(self.B, device=self.device, generator=self.gen)
        base = self.starts[pi] + (r * span.float()).long()
        take = (self.starts[pi] + ln - base).clamp(max=self.L + 1)
        x = self._gather(base, take)
        if self.augment:
            x = augment_batch(x, self.augment, self.gen)
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()

    def eval_batches(self):
        """Itera las ventanas fijas de val/test en lotes de B."""
        n = self.fixed.numel()
        for i in range(0, n, self.B):
            base = self.fixed[i:i + self.B]
            take = (self.fixed_end[i:i + self.B] - base).clamp(max=self.L + 1)
            x = self._gather(base, take)
            yield x[:, :-1].contiguous(), x[:, 1:].contiguous()

    def __iter__(self):
        while True:
            yield self.train_batch()


# ---------------------------------------------------------------- augmentacion
def augment_batch(x: torch.Tensor, cfg: dict, gen: torch.Generator) -> torch.Tensor:
    """Augmentacion vectorizada sobre [B, L] de tokens. Todo en GPU."""
    B, L = x.shape
    dev = x.device
    is_note = (x >= NOTE_OFF) & (x < SHIFT_OFF)
    is_shift = x >= SHIFT_OFF

    # --- transposicion: desplaza los NOTE_ON dentro del rango valido por muestra
    p_tr = float(cfg.get("p_transpose", 0.0))
    if p_tr > 0:
        pitch = torch.where(is_note, x - NOTE_OFF, torch.zeros_like(x))
        big = torch.full_like(x, N_PITCH)
        lo_p = torch.where(is_note, pitch, big).amin(1)             # nota mas grave
        hi_p = torch.where(is_note, pitch, torch.zeros_like(x)).amax(1)
        max_s = int(cfg.get("max_semitones", 6))
        lo = (-lo_p).clamp(min=-max_s, max=0)
        hi = (N_PITCH - 1 - hi_p).clamp(min=0, max=max_s)
        u = torch.rand(B, device=dev, generator=gen)
        span = (hi - lo + 1).clamp(min=1)
        shift = lo + (u * span.float()).long()
        apply = (torch.rand(B, device=dev, generator=gen) < p_tr) & (hi_p >= lo_p)
        shift = torch.where(apply, shift, torch.zeros_like(shift))
        x = torch.where(is_note, x + shift[:, None], x)

    # --- adelgazado de voces: conserva la nota mas aguda de cada paso temporal
    p_thin = float(cfg.get("p_thin", 0.0))
    if p_thin > 0:
        sel = torch.rand(B, device=dev, generator=gen) < p_thin
        if bool(sel.any()):
            lo_d, hi_d = float(cfg.get("thin_lo", 0.3)), float(cfg.get("thin_hi", 1.0))
            pdrop = lo_d + torch.rand(B, device=dev, generator=gen) * (hi_d - lo_d)
            pdrop = torch.where(sel, pdrop, torch.zeros_like(pdrop))
            # grupo temporal de cada posicion: cuantos SHIFT quedan detras
            grp = torch.cumsum(is_shift.long(), dim=1)
            # nota mas aguda de cada grupo, por muestra
            key = torch.where(is_note, x, torch.zeros_like(x))
            n_g = int(grp.max().item()) + 1
            top = torch.zeros(B, n_g, dtype=x.dtype, device=dev)
            top.scatter_reduce_(1, grp, key, reduce="amax", include_self=True)
            is_top = is_note & (x == top.gather(1, grp))
            drop = (is_note & ~is_top &
                    (torch.rand(B, L, device=dev, generator=gen) < pdrop[:, None]))
            keep = ~drop
            # compactar: argsort estable deja los conservados delante
            order = torch.argsort((~keep).long(), dim=1, stable=True)
            x = torch.gather(x, 1, order)
            n_keep = keep.sum(1)
            ar = torch.arange(L, device=dev)
            x = torch.where(ar[None, :] < n_keep[:, None], x, torch.full_like(x, PAD))
    return x
