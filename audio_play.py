from __future__ import annotations

import math
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, list]


# Conversión nota -> frecuencia
def midi_to_hz(midi_note: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convierte una nota MIDI a frecuencia en Hz.

    Ejemplo:
        MIDI 69 -> 440 Hz
        MIDI 60 -> C4
    """
    return 440.0 * (2.0 ** ((np.asarray(midi_note) - 69.0) / 12.0))

# Síntesis
def make_musicbox_note(
    frequency: float,
    duration_sec: float,
    sample_rate: int = 22050,
    decay: float = 6.0,
    brightness: float = 0.35,
) -> np.ndarray:
    n = max(1, int(round(duration_sec * sample_rate)))
    t = np.arange(n, dtype=np.float32) / float(sample_rate)

    # Ataque breve para evitar clicks.
    attack_sec = min(0.008, duration_sec * 0.1)
    attack_n = max(1, int(round(attack_sec * sample_rate)))

    envelope = np.exp(-decay * t).astype(np.float32)
    envelope[:attack_n] *= np.linspace(0.0, 1.0, attack_n, dtype=np.float32)

    # Timbre simple tipo campanilla/caja musical:
    # fundamental + armónicos con decaimiento ligeramente distinto.
    phase = 2.0 * np.pi * frequency * t
    tone = (
        np.sin(phase)
        + brightness * np.sin(2.0 * phase)
        + 0.18 * brightness * np.sin(3.0 * phase)
    )

    audio = tone.astype(np.float32) * envelope
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32)


def synthesize_musicbox_roll(
    roll: np.ndarray,
    step_sec: float = 0.05,
    note_min: int = 21,
    sample_rate: int = 22050,
    representation: str = "onset",
    note_duration_sec: Optional[float] = None,
    decay: float = 6.0,
    gain: float = 0.25,
    normalize: bool = True,
    max_events: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    roll = np.asarray(roll)

    if roll.ndim != 2:
        raise ValueError(f"roll debe tener forma [T, N], pero llegó con shape {roll.shape}")

    if roll.size == 0:
        return np.zeros(1, dtype=np.float32), sample_rate

    T, N = roll.shape
    representation = representation.lower().strip()

    if note_duration_sec is None:
        note_duration_sec = max(0.25, 8.0 * step_sec)

    if representation == "onset":
        events = np.argwhere(roll > 0)

        if max_events is not None and len(events) > max_events:
            events = events[:max_events]

        total_duration_sec = T * step_sec + note_duration_sec
        audio = np.zeros(int(math.ceil(total_duration_sec * sample_rate)), dtype=np.float32)

        # Cache de notas por pitch para no sintetizar la misma onda muchas veces.
        note_cache = {}

        for t_idx, pitch_idx in events:
            midi_note = int(note_min + pitch_idx)
            frequency = float(midi_to_hz(midi_note))

            if midi_note not in note_cache:
                note_cache[midi_note] = make_musicbox_note(
                    frequency=frequency,
                    duration_sec=note_duration_sec,
                    sample_rate=sample_rate,
                    decay=decay,
                )

            note_audio = note_cache[midi_note]
            start = int(round(float(t_idx) * step_sec * sample_rate))
            end = min(start + len(note_audio), len(audio))

            if end > start:
                audio[start:end] += gain * note_audio[: end - start]

    elif representation == "active":
        # Interpretación sostenida: cada celda activa produce una onda durante ese frame.
        total_duration_sec = T * step_sec
        audio = np.zeros(int(math.ceil(total_duration_sec * sample_rate)), dtype=np.float32)

        frame_n = max(1, int(round(step_sec * sample_rate)))
        t_frame = np.arange(frame_n, dtype=np.float32) / float(sample_rate)

        events_done = 0

        for t_idx in range(T):
            active = np.flatnonzero(roll[t_idx] > 0)

            for pitch_idx in active:
                if max_events is not None and events_done >= max_events:
                    break

                midi_note = int(note_min + pitch_idx)
                frequency = float(midi_to_hz(midi_note))

                tone = np.sin(2.0 * np.pi * frequency * t_frame).astype(np.float32)

                # Mini envolvente para reducir clicks por frame.
                fade_n = min(32, frame_n // 4)
                if fade_n > 1:
                    fade = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
                    tone[:fade_n] *= fade
                    tone[-fade_n:] *= fade[::-1]

                start = t_idx * frame_n
                end = min(start + frame_n, len(audio))

                if end > start:
                    audio[start:end] += gain * tone[: end - start]

                events_done += 1

            if max_events is not None and events_done >= max_events:
                break

    else:
        raise ValueError("representation debe ser 'onset' o 'active'")

    if normalize:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = 0.95 * audio / peak

    return audio.astype(np.float32), sample_rate


# Guardar y reproducir 
def save_wav(
    path: Union[str, Path],
    audio: np.ndarray,
    sample_rate: int = 22050,
) -> Path:
    """
    Guarda audio mono float32 como WAV PCM 16-bit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1:
        raise ValueError("save_wav espera audio mono 1D.")

    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())

    return path


def play_audio(
    audio: np.ndarray,
    sample_rate: int = 22050,
    prefer_notebook: bool = True,
    open_external_player: bool = False,
):
    if prefer_notebook:
        try:
            from IPython.display import Audio, display
            obj = Audio(audio, rate=sample_rate)
            display(obj)
            return obj
        except Exception:
            pass

    if open_external_player:
        tmp = Path(tempfile.gettempdir()) / "musicbox_preview.wav"
        save_wav(tmp, audio, sample_rate)

        if os.name == "nt":
            os.startfile(str(tmp))  # type: ignore[attr-defined]
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(tmp)])

        return tmp

    return audio


# Cargar 
def load_musicbox_dataset(npz_path: Union[str, Path]):
    """
    Carga train.npz o cualquier archivo .npz con el formato del dataset.
    """
    return np.load(str(npz_path), allow_pickle=True)


def get_sequence_from_dataset(data, index: int) -> np.ndarray:
    """
    Extrae una secuencia individual desde un .npz cargado.

    Soporta:
        - rolls_flat + offsets
        - sequences
    """
    if "rolls_flat" in data.files and "offsets" in data.files:
        rolls_flat = data["rolls_flat"]
        offsets = data["offsets"]
        start = int(offsets[index])
        end = int(offsets[index + 1])
        return np.asarray(rolls_flat[start:end], dtype=np.uint8)

    if "sequences" in data.files:
        return np.asarray(data["sequences"][index], dtype=np.uint8)

    raise KeyError("No encontré formato válido. Esperaba rolls_flat+offsets o sequences.")


def get_dataset_defaults(data) -> dict:
    """
    Lee parámetros guardados en train.npz si existen.
    """
    defaults = {
        "step_sec": 0.05,
        "note_min": 21,
        "representation": "onset",
        "num_positions": None,
    }

    if "step_sec" in data.files:
        defaults["step_sec"] = float(np.asarray(data["step_sec"]).ravel()[0])

    if "note_min" in data.files:
        defaults["note_min"] = int(np.asarray(data["note_min"]).ravel()[0])

    if "representation" in data.files:
        defaults["representation"] = str(np.asarray(data["representation"]).ravel()[0])

    if "num_positions" in data.files:
        defaults["num_positions"] = int(np.asarray(data["num_positions"]).ravel()[0])

    return defaults


# Utils
def synthesize_dataset_sequence(
    npz_path: Union[str, Path],
    index: int,
    start_step: int = 0,
    num_steps: Optional[int] = None,
    sample_rate: int = 22050,
    note_duration_sec: Optional[float] = None,
    decay: float = 6.0,
    gain: float = 0.25,
    normalize: bool = True,
) -> Tuple[np.ndarray, int, np.ndarray]:
    data = load_musicbox_dataset(npz_path)
    defaults = get_dataset_defaults(data)

    roll = get_sequence_from_dataset(data, index)

    start_step = max(0, int(start_step))
    if num_steps is None:
        frag = roll[start_step:]
    else:
        frag = roll[start_step : start_step + int(num_steps)]

    audio, sr = synthesize_musicbox_roll(
        frag,
        step_sec=defaults["step_sec"],
        note_min=defaults["note_min"],
        sample_rate=sample_rate,
        representation=defaults["representation"],
        note_duration_sec=note_duration_sec,
        decay=decay,
        gain=gain,
        normalize=normalize,
    )

    return audio, sr, frag


def play_dataset_sequence(
    npz_path: Union[str, Path],
    index: int,
    start_step: int = 0,
    num_steps: Optional[int] = 512,
    sample_rate: int = 22050,
    note_duration_sec: Optional[float] = None,
    decay: float = 6.0,
    gain: float = 0.25,
    prefer_notebook: bool = True,
    open_external_player: bool = False,
):
    audio, sr, _ = synthesize_dataset_sequence(
        npz_path=npz_path,
        index=index,
        start_step=start_step,
        num_steps=num_steps,
        sample_rate=sample_rate,
        note_duration_sec=note_duration_sec,
        decay=decay,
        gain=gain,
        normalize=True,
    )

    return play_audio(
        audio,
        sample_rate=sr,
        prefer_notebook=prefer_notebook,
        open_external_player=open_external_player,
    )


def export_dataset_sequence_wav(
    npz_path: Union[str, Path],
    index: int,
    out_path: Union[str, Path],
    start_step: int = 0,
    num_steps: Optional[int] = 512,
    sample_rate: int = 22050,
    note_duration_sec: Optional[float] = None,
    decay: float = 6.0,
    gain: float = 0.25,
) -> Path:
    audio, sr, _ = synthesize_dataset_sequence(
        npz_path=npz_path,
        index=index,
        start_step=start_step,
        num_steps=num_steps,
        sample_rate=sample_rate,
        note_duration_sec=note_duration_sec,
        decay=decay,
        gain=gain,
        normalize=True,
    )

    return save_wav(out_path, audio, sr)


if __name__ == "__main__":
    npz_path = r"musicbox/train.npz"
    out = export_dataset_sequence_wav(
        npz_path=npz_path,
        index=1,
        out_path=r"musicbox/example.wav",
        start_step=0,
        num_steps=512,
    )
