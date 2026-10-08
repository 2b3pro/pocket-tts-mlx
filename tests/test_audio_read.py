"""Audio readers keep 16-bit WAV on the stdlib and send other encodings to soundfile."""

import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from pocket_tts_mlx.data.audio import audio_read


def test_int16_wav_uses_the_stdlib_scale(tmp_path: Path):
    sample_rate = 8_000
    samples = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    path = tmp_path / "pcm.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())

    audio, rate = audio_read(path)
    assert rate == sample_rate
    assert np.allclose(audio[0], samples.astype(np.float32) / 32768.0, atol=1e-6)


def test_float_wav_is_not_decoded_as_int16(tmp_path: Path):
    sample_rate = 16_000
    tone = (0.5 * np.sin(np.linspace(0, 8, 64, dtype=np.float32))).astype(np.float32)
    path = tmp_path / "float.wav"
    sf.write(path, tone, sample_rate, subtype="FLOAT")

    audio, rate = audio_read(path)
    assert rate == sample_rate
    assert audio.shape[-1] == tone.shape[0]
    assert np.max(np.abs(audio)) < 1.0
    assert np.allclose(audio[0], tone, atol=1e-5)
