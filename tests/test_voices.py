"""Voice catalog, state import, and strict bundle checks."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from safetensors.numpy import save_file

from pocket_tts_mlx.models.tts_model import (
    CONFIGS_DIR,
    assign_bundle_weights,
    is_shipped_config,
    predefined_voice_source,
)
from pocket_tts_mlx.utils.weight_conversion import (
    PREDEFINED_VOICE_NAMES,
    VOICE_STATE_REVISION,
    import_flow_state,
    is_safetensors_source,
    language_voice_state_url,
)

MODULE = "transformer.layers.0.self_attn"


def _save(path: Path, tensors: dict[str, np.ndarray]) -> None:
    save_file(tensors, path)


def test_catalog_covers_the_upstream_names_and_per_language_urls():
    for name in ("alba", "estelle", "giovanni", "juergen", "lola", "rafael", "daan"):
        assert name in PREDEFINED_VOICE_NAMES
    assert len(PREDEFINED_VOICE_NAMES) == 27
    assert language_voice_state_url("english", "alba") == (
        "hf://kyutai/pocket-tts-without-voice-cloning/languages/english/"
        f"embeddings/alba.safetensors@{VOICE_STATE_REVISION}"
    )
    assert predefined_voice_source(CONFIGS_DIR / "english.yaml", "alba") == language_voice_state_url(
        "english", "alba"
    )
    legacy = predefined_voice_source(CONFIGS_DIR / "b6369a24.yaml", "alba")
    assert legacy == "legacy"


def test_custom_config_refuses_predefined_voices():
    with pytest.raises(ValueError, match="not loaded from a config associated with a language"):
        predefined_voice_source(Path("/tmp/custom.yaml"), "alba")
    assert is_shipped_config(CONFIGS_DIR / "french.yaml")
    assert not is_shipped_config(Path("/tmp/french.yaml"))


def test_b6369a24_catalog_is_the_eight_legacy_names():
    with pytest.raises(ValueError, match="not found"):
        predefined_voice_source(CONFIGS_DIR / "b6369a24.yaml", "estelle")


def test_safetensors_source_ignores_revision_and_query():
    assert is_safetensors_source("hf://kyutai/pocket-tts/voices/alba.safetensors@1e08e6a")
    assert is_safetensors_source("https://example.com/voice.safetensors?download=1")
    assert not is_safetensors_source("https://example.com/voice.wav?file=voice.safetensors")
    assert is_safetensors_source(Path("alba.safetensors"))
    assert not is_safetensors_source("alba")


def test_offset_becomes_current_end_and_zero_pad_is_dropped(tmp_path: Path):
    path = tmp_path / "voice.safetensors"
    cache = np.zeros((2, 1, 8, 2, 2), dtype=np.float32)
    cache[0, 0, 0, 0, 0] = 3.5
    _save(
        path,
        {
            f"{MODULE}/offset": np.array([5], dtype=np.int64),
            f"{MODULE}/pad": np.array([0], dtype=np.int64),
            f"{MODULE}/cache": cache,
        },
    )
    state = import_flow_state(path, {MODULE})
    current_end = state[MODULE]["current_end"]
    assert tuple(current_end.shape) == (5,)
    assert current_end.dtype == mx.int64
    assert "pad" not in state[MODULE]
    assert float(np.array(state[MODULE]["cache"])[0, 0, 0, 0, 0]) == 3.5


def test_nonzero_pad_is_rejected(tmp_path: Path):
    path = tmp_path / "voice.safetensors"
    _save(path, {f"{MODULE}/pad": np.array([1], dtype=np.int64)})
    with pytest.raises(ValueError, match="pad must be 0"):
        import_flow_state(path, {MODULE})


def test_unknown_module_is_rejected(tmp_path: Path):
    path = tmp_path / "voice.safetensors"
    _save(path, {f"{MODULE}/cache": np.zeros((2, 1, 1, 1, 1), dtype=np.float32)})
    with pytest.raises(ValueError, match="outside the flow LM"):
        import_flow_state(path, {"other.module"})


def test_legacy_current_end_is_kept(tmp_path: Path):
    path = tmp_path / "voice.safetensors"
    current = np.arange(4, dtype=np.int64)
    _save(path, {f"{MODULE}/current_end": current})
    state = import_flow_state(path, {MODULE})
    assert np.array_equal(np.array(state[MODULE]["current_end"]), current)


def test_audio_prompt_tensor_is_not_a_voice_state(tmp_path: Path):
    path = tmp_path / "voice.safetensors"
    _save(path, {"audio_prompt": np.zeros((1, 4, 8), dtype=np.float32)})
    with pytest.raises(ValueError, match="audio_prompt"):
        import_flow_state(path, {MODULE})


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 3)


def _matching_bundle(model: nn.Module) -> dict[str, np.ndarray]:
    from mlx.utils import tree_flatten

    return {
        name: np.zeros(tuple(int(dim) for dim in value.shape), dtype=np.float32)
        for name, value in tree_flatten(model.parameters())
    }


def test_strict_bundle_rejects_missing_shape_and_unplaceable_keys():
    model = _Tiny()
    with pytest.raises(ValueError, match="missing="):
        assign_bundle_weights(model, {})

    bad_shape = _matching_bundle(model)
    first = next(iter(bad_shape))
    bad_shape[first] = np.zeros((1,), dtype=np.float32)
    with pytest.raises(ValueError, match="shape_mismatch="):
        assign_bundle_weights(model, bad_shape)

    extra = _matching_bundle(model)
    extra["not.a.parameter"] = np.zeros((1,), dtype=np.float32)
    with pytest.raises(ValueError, match="unplaceable="):
        assign_bundle_weights(model, extra)

    assign_bundle_weights(model, _matching_bundle(model))
    assert tuple(model.proj.weight.shape) == (3, 2)
