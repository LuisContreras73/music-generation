"""Exportacion de piano-roll de onsets a MIDI, sin dependencias externas.

El dataset es de ONSETS: marca cuando se pulsa cada tecla, pero no cuanto dura.
Para poder escuchar el resultado hay que asignar una duracion, y la eleccion se
hace explicita aqui en vez de esconderla:

  * cada nota suena hasta el siguiente ataque de LA MISMA nota, con un tope de
    `max_dur_sec` (por defecto 1.5 s). Es la reconstruccion mas fiel posible sin
    informacion de duracion: respeta las repeticiones rapidas y deja resonar las
    notas aisladas.
  * se emite un pedal de sustain (CC64) opcional, porque el corpus es piano
    interpretado y sin sustain suena artificialmente seco.

Se escribe un Standard MIDI File de tipo 0 a mano (cabecera MThd + pista MTrk
con eventos de tiempo delta en formato de cantidad de longitud variable).
"""
from __future__ import annotations
import struct
from pathlib import Path

import numpy as np

NOTE_MIN_MIDI = 21
STEP_SEC = 0.05
TICKS_PER_BEAT = 480
DEFAULT_BPM = 120.0


def _vlq(n: int) -> bytes:
    """Cantidad de longitud variable, como exige el formato MIDI."""
    n = int(max(n, 0))
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def roll_to_midi(roll: np.ndarray, path, *, step_sec: float = STEP_SEC,
                 max_dur_sec: float = 1.5, velocity: int = 78,
                 sustain: bool = True, program: int = 0,
                 note_min: int = NOTE_MIN_MIDI) -> Path:
    """Escribe [T,88] de onsets como MIDI tipo 0. Devuelve la ruta."""
    roll = np.asarray(roll)
    if roll.ndim != 2:
        raise ValueError("roll debe ser [T, n_notas]")
    T, P = roll.shape
    sec_per_tick = 60.0 / (DEFAULT_BPM * TICKS_PER_BEAT)
    max_dur_ticks = int(round(max_dur_sec / sec_per_tick))
    step_ticks = max(1, int(round(step_sec / sec_per_tick)))

    # (tick, tipo, nota) con tipo 1 = note_on, 0 = note_off
    events: list[tuple[int, int, int]] = []
    for p in range(P):
        onsets = np.flatnonzero(roll[:, p])
        for i, t in enumerate(onsets):
            start = int(t) * step_ticks
            nxt = int(onsets[i + 1]) * step_ticks if i + 1 < len(onsets) else None
            end = min(start + max_dur_ticks, nxt - 1) if nxt else start + max_dur_ticks
            end = max(end, start + step_ticks // 2)
            events.append((start, 1, note_min + p))
            events.append((end, 0, note_min + p))
    # note_off antes de note_on en el mismo tick, para no cortar la nota nueva
    events.sort(key=lambda e: (e[0], e[1]))

    track = bytearray()
    track += b"\x00\xff\x51\x03" + struct.pack(">I", int(6e7 / DEFAULT_BPM))[1:]  # tempo
    track += b"\x00\xc0" + bytes([program & 0x7F])                               # programa
    if sustain:
        track += b"\x00\xb0\x40\x7f"                                             # CC64 on
    prev = 0
    for tick, kind, note in events:
        track += _vlq(tick - prev)
        track += bytes([0x90 if kind else 0x80, note & 0x7F,
                        velocity if kind else 0])
        prev = tick
    if sustain:
        track += _vlq(step_ticks) + b"\xb0\x40\x00"                              # CC64 off
    track += b"\x00\xff\x2f\x00"                                                 # fin de pista

    data = (b"MThd" + struct.pack(">IHHH", 6, 0, 1, TICKS_PER_BEAT)
            + b"MTrk" + struct.pack(">I", len(track)) + bytes(track))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def export_generations(npz_path, out_dir, limit: int = 8, **kw) -> list:
    """Convierte a MIDI las generaciones guardadas en un .npz de un experimento."""
    npz_path = Path(npz_path)
    z = np.load(npz_path)
    rolls = z["rolls"]
    out_dir = Path(out_dir)
    out = []
    for i, r in enumerate(rolls[:limit]):
        out.append(roll_to_midi(r, out_dir / ("%s_muestra%02d.mid" % (npz_path.stem, i + 1)), **kw))
    return out


if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description="piano-roll .npz -> archivos MIDI")
    ap.add_argument("npz", help="ruta a un gen_step*.npz de experiments/<name>/generations/")
    ap.add_argument("--out", default=None, help="directorio de salida")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--max-dur", type=float, default=1.5)
    a = ap.parse_args()
    dest = Path(a.out) if a.out else Path(a.npz).parent / "midi"
    for p in export_generations(a.npz, dest, limit=a.limit, max_dur_sec=a.max_dur):
        print(p)
