"""Audio conversion utilities (numpy-based)."""

import math
from typing import Any

import numpy as np
from scipy.signal import lfilter, resample_poly

from pocket_tts_mlx.default_parameters import (
    BRIGHTEN_SHELF_DB,
    BRIGHTEN_SHELF_HZ,
    NORMALIZE_PEAK_DBFS,
    NORMALIZE_TARGET_DBFS,
)


def _as_2d(wav: np.ndarray) -> np.ndarray:
    """Ensure audio has shape [C, T]."""
    return wav[None, :] if wav.ndim == 1 else wav


def convert_audio(
    wav: Any, from_rate: int | float, to_rate: int | float, to_channels: int
) -> np.ndarray:
    """Convert sample rate and channels using polyphase resampling."""
    wav_np = np.asarray(wav)
    wav_np = _as_2d(wav_np)

    if wav_np.shape[0] != to_channels:
        # Channel conversion via averaging or tiling.
        if to_channels == 1:
            wav_np = wav_np.mean(axis=0, keepdims=True)
        elif wav_np.shape[0] == 1:
            wav_np = np.tile(wav_np, (to_channels, 1))
        else:
            raise ValueError(
                f"Cannot convert from {wav_np.shape[0]} channels to {to_channels} channels"
            )

    from_rate_i = int(round(from_rate))
    to_rate_i = int(round(to_rate))
    if from_rate_i != to_rate_i:
        gcd = math.gcd(from_rate_i, to_rate_i)
        up = to_rate_i // gcd
        down = from_rate_i // gcd
        wav_np = resample_poly(wav_np, up, down, axis=-1)

    return wav_np.astype(np.float32)


def end_on_pause(
    wav: np.ndarray,
    sample_rate: int,
    pause_sec: float = 0.08,
    fade_sec: float = 0.02,
    floor_db: float = 35.0,
) -> np.ndarray:
    """End a voice prompt `[..., T]` on exactly `pause_sec` of silence.

    Training prompts stop inside the pause between words. A prompt that ends on
    speech makes the model continue that utterance, and a long trailing silence
    delays the onset. Trailing frames more than `floor_db` below the loudest
    frame are cut, the last `fade_sec` of what remains is faded out, and
    `pause_sec` of zeros is appended.
    """
    wav = np.asarray(wav)
    frame = max(1, int(0.02 * sample_rate))
    n_frames = wav.shape[-1] // frame
    if n_frames == 0:
        return wav
    framed = wav[..., : n_frames * frame].reshape(-1, n_frames, frame)
    rms = np.sqrt(np.square(framed).mean(axis=(0, 2)))
    db = 20 * np.log10(rms + 1e-12)
    loud = np.flatnonzero(db > db.max() - floor_db)
    if loud.size == 0:
        return wav
    end = (int(loud[-1]) + 1) * frame
    wav = np.array(wav[..., :end], copy=True)
    fade = min(int(fade_sec * sample_rate), end)
    if fade > 0:
        ramp = np.linspace(1, 0, fade, dtype=wav.dtype)
        wav[..., end - fade :] *= ramp
    pause = np.zeros((*wav.shape[:-1], int(pause_sec * sample_rate)), dtype=wav.dtype)
    return np.concatenate([wav, pause], axis=-1)


def normalize_loudness(
    wav: np.ndarray,
    sample_rate: int,
    target_dbfs: float = NORMALIZE_TARGET_DBFS,
    peak_dbfs: float = NORMALIZE_PEAK_DBFS,
) -> np.ndarray:
    """Scale mono speech `[T]` to `target_dbfs` active RMS with peaks under `peak_dbfs`.

    Active RMS covers the 20 ms frames within 20 dB of the loudest frame, so
    pauses do not count. After the gain, a limiter pulls samples above the
    ceiling down. It works on 10 ms blocks: each block takes the smallest gain
    that it or a neighbour needs, and the gain moves linearly between block
    centres, so no sample ends above the ceiling. Silence is returned unchanged.
    """
    wav = np.asarray(wav, dtype=np.float32)
    frame = max(1, int(0.02 * sample_rate))
    n_frames = wav.shape[-1] // frame
    if n_frames == 0:
        return wav
    rms = np.sqrt(np.square(wav[: n_frames * frame].reshape(n_frames, frame)).mean(axis=1))
    if rms.max() <= 1e-6:
        return wav
    active = rms[rms >= 0.1 * rms.max()]
    out = wav * (10 ** (target_dbfs / 20) / np.sqrt(np.square(active).mean()))

    ceiling = 10 ** (peak_dbfs / 20)
    block = max(1, sample_rate // 100)
    n_blocks = -(-out.shape[-1] // block)
    magnitude = np.zeros(n_blocks * block, dtype=np.float32)
    magnitude[: out.shape[-1]] = np.abs(out)
    need = np.minimum(1.0, ceiling / np.maximum(magnitude, 1e-12)).reshape(n_blocks, block).min(axis=1)
    if need.min() >= 1.0:
        return out.astype(np.float32)
    edged = np.pad(need, 1, mode="edge")
    need = np.minimum(np.minimum(edged[:-2], edged[1:-1]), edged[2:])
    centres = np.arange(n_blocks) * block + (block - 1) / 2
    gain = np.interp(np.arange(out.shape[-1]), centres, need)
    return (out * gain).astype(np.float32)


def brighten_speech(
    wav: np.ndarray,
    sample_rate: int,
    freq_hz: float = BRIGHTEN_SHELF_HZ,
    gain_db: float = BRIGHTEN_SHELF_DB,
) -> np.ndarray:
    """Apply a second-order high shelf (RBJ cookbook) of `gain_db` above `freq_hz`.

    It lifts dark voices such as french `estelle` without touching the low
    harmonics. The output can be louder than the input and is not limited;
    combine it with `normalize_loudness` to set the level.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if gain_db == 0 or wav.shape[-1] == 0:
        return wav
    amp = 10 ** (gain_db / 40)
    w = 2 * math.pi * freq_hz / sample_rate
    cos, sin = math.cos(w), math.sin(w)
    q = 2 * math.sqrt(amp) * (sin / 2 * math.sqrt(2.0))
    b = np.array(
        [
            amp * ((amp + 1) + (amp - 1) * cos + q),
            -2 * amp * ((amp - 1) + (amp + 1) * cos),
            amp * ((amp + 1) + (amp - 1) * cos - q),
        ]
    )
    a = np.array(
        [
            (amp + 1) - (amp - 1) * cos + q,
            2 * ((amp - 1) - (amp + 1) * cos),
            (amp + 1) - (amp - 1) * cos - q,
        ]
    )
    return lfilter(b / a[0], a / a[0], wav).astype(np.float32)
