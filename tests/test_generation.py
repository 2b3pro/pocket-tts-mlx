"""Generation behavior tests (no weights, no network)."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

import pocket_tts_mlx
from pocket_tts_mlx.data.audio_utils import end_on_pause, normalize_loudness
from pocket_tts_mlx.models.mimi import build_mimi
from pocket_tts_mlx.models.tts_model import (
    TTSModel,
    apply_chunk_onset_fade,
    mimi_steps_per_latent,
    note_eos_step,
    prepend_voice_bos,
    resolve_frames_after_eos,
)
from pocket_tts_mlx.utils.config import load_config

CONFIG_DIR = Path(pocket_tts_mlx.__file__).parent / "config"


def test_eos_on_frame_0_is_ignored_and_frame_6_is_kept():
    assert note_eos_step(True, 0, None) is None
    assert note_eos_step(True, 5, None) is None
    assert note_eos_step(False, 6, None) is None
    assert note_eos_step(True, 6, None) == 6
    assert note_eos_step(True, 9, 6) == 6


def test_chunk_fade_covers_120_samples_once_at_24khz():
    fade = mx.linspace(0.0, 1.0, 24_000 // 200)
    assert int(fade.shape[0]) == 120
    audio = mx.ones((2_400,))
    faded, rest = apply_chunk_onset_fade(audio, fade)
    faded_np = np.array(faded)
    assert faded_np[0] == 0.0
    assert np.allclose(faded_np[120:], np.ones(2_400 - 120))
    later = mx.ones((2_400,))
    again, _ = apply_chunk_onset_fade(later, rest)
    assert rest is None
    assert np.allclose(np.array(again), np.ones(2_400))


def test_end_on_pause_ends_on_80ms_of_zeros():
    sample_rate = 24_000
    pause = int(0.08 * sample_rate)
    loud = np.ones((1, sample_rate), dtype=np.float32)
    silence = np.zeros((1, sample_rate), dtype=np.float32)
    out = end_on_pause(np.concatenate([loud, silence], axis=-1), sample_rate)
    assert np.all(out[..., -pause:] == 0.0)
    assert out.shape[-1] == sample_rate + pause


def test_bos_is_prepended_only_when_the_flag_is_set():
    prompt = mx.ones((1, 3, 4))
    bos = mx.full((1, 1, 4), 2.0)
    with_bos = prepend_voice_bos(prompt, bos, True)
    assert tuple(with_bos.shape) == (1, 4, 4)
    assert float(np.array(with_bos)[0, 0, 0]) == 2.0
    without = prepend_voice_bos(prompt, bos, False)
    assert tuple(without.shape) == (1, 3, 4)
    assert np.allclose(np.array(without), np.array(prompt))


def test_frames_after_eos_follows_caller_then_recommendation_then_guess():
    assert resolve_frames_after_eos(7, 4, 1) == 7
    assert resolve_frames_after_eos(None, 4, 1) == 4
    assert resolve_frames_after_eos(None, None, 1) == 3
    assert resolve_frames_after_eos(None, None, 3) == 5


def test_english_mimi_step_is_the_encoder_ratio():
    mimi = build_mimi(load_config(CONFIG_DIR / "english.yaml").mimi)
    assert mimi_steps_per_latent(mimi.encoder_frame_rate, mimi.frame_rate) == 16


def test_warmup_trim_and_fade_defaults_are_zero():
    generate = inspect.signature(TTSModel.generate_audio)
    assert generate.parameters["warmup_frames"].default == 0
    assert generate.parameters["trim_start_ms"].default == 0
    assert generate.parameters["fade_in_ms"].default == 0


def _active_dbfs(x: np.ndarray, sample_rate: int = 24_000) -> float:
    frame = sample_rate // 50
    rms = np.sqrt(np.square(x[: len(x) // frame * frame].reshape(-1, frame)).mean(axis=1))
    active = rms[rms >= 0.1 * rms.max()]
    return float(20 * np.log10(np.sqrt(np.square(active).mean())))


def _tone_with_pause(amplitude: float, sample_rate: int = 24_000) -> np.ndarray:
    tone = np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate).astype(np.float32)
    pause = np.zeros(sample_rate // 4, dtype=np.float32)
    return np.concatenate([tone, pause, tone]) * amplitude


def test_normalize_lifts_quiet_speech_to_the_target():
    quiet = _tone_with_pause(10 ** (-30 / 20) * np.sqrt(2))
    assert abs(_active_dbfs(quiet) + 30.0) < 0.1
    out = normalize_loudness(quiet, 24_000)
    assert abs(_active_dbfs(out) + 18.0) < 0.1
    assert np.abs(out).max() <= 10 ** (-1 / 20) + 1e-6


def test_normalize_limits_a_spike_and_keeps_the_gain_elsewhere():
    audio = _tone_with_pause(0.01)
    audio[12_000] = 0.9
    out = normalize_loudness(audio, 24_000)
    assert np.abs(out).max() <= 10 ** (-1 / 20) + 1e-6
    far = [1_000, 40_000]
    gains = out[far] / audio[far]
    assert np.allclose(gains, gains[0], rtol=1e-5)
    assert gains[0] > 10.0


def test_normalize_returns_silence_and_short_input_unchanged():
    silence = np.zeros(24_000, dtype=np.float32)
    assert np.array_equal(normalize_loudness(silence, 24_000), silence)
    short = np.full(100, 0.5, dtype=np.float32)
    assert np.array_equal(normalize_loudness(short, 24_000), short)


def test_normalize_is_off_by_default():
    generate = inspect.signature(TTSModel.generate_audio)
    assert generate.parameters["normalize"].default is False


def test_generate_audio_normalizes_only_when_asked():
    quiet = mx.array(_tone_with_pause(0.01))
    model = SimpleNamespace(
        sample_rate=24_000,
        generate_audio_stream=lambda **kwargs: iter([quiet]),
        _postprocess_audio_start=lambda audio, trim_start_ms, fade_in_ms: audio,
    )
    plain = np.array(TTSModel.generate_audio(model, {}, "Hallo."))
    normalized = np.array(TTSModel.generate_audio(model, {}, "Hallo.", normalize=True))
    assert np.array_equal(plain, np.array(quiet))
    assert abs(_active_dbfs(normalized) + 18.0) < 0.1
