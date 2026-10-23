"""Mimi width and latent-layout tests (no weight download)."""

import json
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

import pocket_tts_mlx
from pocket_tts_mlx.models.mimi import build_mimi
from pocket_tts_mlx.models.tts_model import TTSModel, speaker_proj_shape
from pocket_tts_mlx.modules.stateful_module import init_states
from pocket_tts_mlx.utils.config import load_config

CONFIG_DIR = Path(pocket_tts_mlx.__file__).parent / "config"
BUNDLE_REPO = "kyutai/pocket-tts-without-voice-cloning"
BUNDLE_REVISION = "d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"


def _mimi(name: str):
    return build_mimi(load_config(CONFIG_DIR / name).mimi)


def test_english_narrows_the_bottleneck_and_keeps_a_wide_upsample():
    mimi = _mimi("english.yaml")
    # MLX conv weight is (out, kernel, in). Stride 16 → kernel 32.
    assert tuple(mimi.downsample.conv.conv.weight.shape) == (32, 32, 512)
    # Depthwise upsample: in_channels/groups == 1, kernel 32, out 512.
    assert tuple(mimi.upsample.convtr.convtr.weight.shape) == (512, 32, 1)
    assert speaker_proj_shape(load_config(CONFIG_DIR / "english.yaml")) == (1024, 32)


def test_missing_inner_and_outer_stay_on_the_seanet_width():
    config = load_config(CONFIG_DIR / "b6369a24.yaml")
    assert config.mimi.inner_dim is None
    assert config.mimi.outer_dim is None
    mimi = build_mimi(config.mimi)
    assert tuple(mimi.downsample.conv.conv.weight.shape) == (512, 32, 512)
    assert tuple(mimi.upsample.convtr.convtr.weight.shape) == (512, 32, 1)
    assert speaker_proj_shape(config) == (1024, 512)


def test_latent_round_trip_uses_batch_time_channel():
    mimi = _mimi("english.yaml")
    audio = mx.zeros((1, mimi.channels, mimi.frame_size))
    latent = mimi.encode_to_latent(audio)
    mx.eval(latent)
    assert tuple(latent.shape) == (1, 1, 32)

    state = init_states(mimi, batch_size=1, sequence_length=250)
    decoded = mimi.decode_from_latent(mx.zeros((1, 1, mimi.quantizer.dimension)), state)
    mx.eval(decoded)
    assert decoded.ndim == 3
    assert tuple(decoded.shape[:2]) == (1, 1)
    assert decoded.shape[-1] > 0
    assert bool(mx.all(mx.isfinite(decoded)))


def _cached_hub_file(filename: str) -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(BUNDLE_REPO, filename, revision=BUNDLE_REVISION)
    if not isinstance(found, str):
        return None
    path = Path(found)
    return path if path.is_file() else None


def _header_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    with path.open("rb") as handle:
        nbytes = struct.unpack("<Q", handle.read(8))[0]
        meta = json.loads(handle.read(nbytes))
    meta.pop("__metadata__", None)
    return {key: tuple(value["shape"]) for key, value in meta.items()}


def _mlx_shape(obj, param_name: str, bundle_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Apply the bundle walker's conv layout transform to a header shape."""
    if param_name == "weight" and isinstance(obj, nn.Conv1d) and len(bundle_shape) == 3:
        return (bundle_shape[0], bundle_shape[2], bundle_shape[1])
    if param_name == "weight" and (
        isinstance(obj, nn.ConvTranspose1d) or hasattr(obj, "is_conv_transpose")
    ) and len(bundle_shape) == 3:
        if bundle_shape[1] == 1:
            return (bundle_shape[0], bundle_shape[2], bundle_shape[1])
        return (bundle_shape[1], bundle_shape[2], bundle_shape[0])
    return bundle_shape


def test_b6369a24_parameter_tree_matches_the_bundle_header():
    tokenizer = _cached_hub_file("tokenizer.model")
    bundle = _cached_hub_file("tts_b6369a24.safetensors")
    if tokenizer is None or bundle is None:
        pytest.skip("cached b6369a24 tokenizer or bundle header is unavailable")

    config = load_config(CONFIG_DIR / "b6369a24.yaml")
    config.flow_lm.lookup_table.tokenizer_path = str(tokenizer)
    model = TTSModel._assemble(config, temp=0.7, lsd_decode_steps=1, noise_clamp=None, eos_threshold=0.0)
    params = {name: tuple(int(dim) for dim in value.shape) for name, value in tree_flatten(model.parameters())}
    header = _header_shapes(bundle)

    assert set(params) == set(header)
    assert len(params) == 213
    for key, bundle_shape in header.items():
        obj = model
        parts = key.split(".")
        for part in parts[:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        expected = _mlx_shape(obj, parts[-1], bundle_shape)
        assert params[key] == expected, key
