"""Strict bundle loads and short clips. Skips when Hugging Face is unreachable."""

import os

import mlx.core as mx
import numpy as np
import pytest

from pocket_tts_mlx.models.tts_model import CONFIGS_DIR, TTSModel
from pocket_tts_mlx.utils.config import load_config


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("POCKET_TTS_RUN_INTEGRATION") != "1",
                       reason="set POCKET_TTS_RUN_INTEGRATION=1 to download and run real models"),
]


def _network_failure(exc: BaseException) -> bool:
    if isinstance(exc, ValueError):
        return False
    module = type(exc).__module__
    if module.startswith(("huggingface_hub", "requests", "urllib3", "httpx")):
        return True
    return isinstance(exc, OSError)


def _load_public(stem: str) -> TTSModel:
    """Load the voice-cloning-free bundle. The gated repo is not required."""
    config_path = CONFIGS_DIR / f"{stem}.yaml"
    config = load_config(config_path)
    config.weights_path = config.weights_path_without_voice_cloning
    model = TTSModel._from_pydantic_config_with_weights(
        config, temp=0.3, lsd_decode_steps=1, noise_clamp=None, eos_threshold=config.default_eos_threshold
    )
    model.origin = config_path.resolve()
    return model


def _load_or_skip(stem: str) -> TTSModel:
    try:
        return _load_public(stem)
    except Exception as exc:
        if _network_failure(exc):
            pytest.skip(f"{stem} bundle unavailable: {exc}")
        raise


def _speak(model: TTSModel, voice: str, text: str) -> None:
    try:
        state = model.get_state_for_audio_prompt(voice)
    except Exception as exc:
        if _network_failure(exc):
            pytest.skip(f"voice {voice} unavailable: {exc}")
        raise
    assert any("current_end" in module_state for module_state in state.values())
    audio = model.generate_audio(state, text, max_tokens=30)
    mx.eval(audio)
    assert model.sample_rate == 24_000
    assert int(audio.shape[-1]) > 1000
    assert float(mx.max(mx.abs(audio))) > 1e-3
    assert bool(mx.all(mx.isfinite(audio)))


def test_live_b6369a24_bundle_loads():
    _load_or_skip("b6369a24")


def test_live_english_bundle_loads():
    _load_or_skip("english")


def test_live_english_alba_speaks_from_a_voice_state():
    _speak(_load_or_skip("english"), "alba", "Hello from MLX.")


def test_live_b6369a24_alba_still_speaks():
    _speak(_load_or_skip("b6369a24"), "alba", "Hello from MLX.")


def test_live_french_estelle_speaks():
    _speak(_load_or_skip("french"), "estelle", "Bonjour.")


def test_live_english_drifting_alba_speaks():
    _speak(_load_or_skip("english_drifting_26-09"), "alba", "Hello from MLX.")


@pytest.mark.parametrize("size", ["2m", "2m_refl", "7m", "7m_refl", "18m", "18m_refl"])
def test_live_small_english_alba_speaks(size):
    _speak(_load_or_skip(f"english_{size}_2026-10"), "alba", "Hello from MLX.")


@pytest.mark.parametrize("stem", ["english", "english_2m_2026-10"])
def test_live_seeded_generation_and_batched_decoder_agree(stem):
    model = _load_or_skip(stem)
    state = model.get_state_for_audio_prompt("alba")
    text = "The new English model reads the year twenty twenty six."
    first = np.array(model.generate_audio(state, text, seed=42))
    second = np.array(model.generate_audio(state, text, seed=42))
    batched = np.array(model.generate_audio(state, text, seed=42, decode_batch_size=4))
    np.testing.assert_array_equal(first, second)
    assert batched.shape == first.shape
    np.testing.assert_allclose(batched, first, atol=1e-3, rtol=1e-2)


def test_live_reference_voice_round_trip():
    model = TTSModel.load_model("english")
    if not model.has_voice_cloning:
        pytest.skip("gated voice-cloning weights are not available")
    voice = model.get_state_for_audio_prompt("alba")
    reference = model.generate_audio(voice, "This is a voice conditioning test.", seed=42)
    cloned = model.get_state_for_audio_prompt(reference[None, :])
    audio = model.generate_audio(cloned, "Hello from the reference voice.", seed=42)
    assert audio.size > 1000
    assert bool(mx.all(mx.isfinite(audio)))
