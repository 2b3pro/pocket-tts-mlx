"""Audio conversion utilities (numpy-based)."""

import math
from typing import Any

import numpy as np
from scipy.signal import resample_poly


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
