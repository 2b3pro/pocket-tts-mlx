"""Flow head and sampler tests (no weights, no network)."""

import math

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from pocket_tts_mlx.models.flow_lm import FlowLMModel, drifting_decode, lsd_decode, ot_decode
from pocket_tts_mlx.modules.mimi_transformer import StreamingTransformer, tanh_gelu
from pocket_tts_mlx.modules.mlp import SimpleMLPAdaLN
from pocket_tts_mlx.utils.config import FlowConfig


class _FlowCfg:
    def __init__(self, flow_type: str):
        self.flow = FlowConfig(dim=8, depth=1, type=flow_type)


def _mlp(flow_type: str, cond_dim: int = 4) -> SimpleMLPAdaLN:
    return SimpleMLPAdaLN.from_pydantic_config(_FlowCfg(flow_type), latent_dim=4, cond_dim=cond_dim)


def _flow_lm(flow_type: str, insert_bos: bool) -> FlowLMModel:
    return FlowLMModel(
        conditioner=nn.Identity(),
        flow_net=_mlp(flow_type, cond_dim=8),
        transformer=StreamingTransformer(
            d_model=8, num_heads=2, num_layers=1, dim_feedforward=16, kind="flow_lm"
        ),
        dim=8,
        ldim=4,
        dtype=mx.float32,
        insert_bos_before_voice=insert_bos,
        flow_type=flow_type,
    )


def _param_names(module) -> set[str]:
    return {name for name, _ in tree_flatten(module.parameters())}


def test_flow_type_selects_time_embed_count():
    assert len(_mlp("lsd").time_embed) == 2
    assert len(_mlp("flow_matching").time_embed) == 1
    drifting = _mlp("drifting")
    assert len(drifting.time_embed) == 0
    assert not any(name.startswith("time_embed") for name in _param_names(drifting))


def test_unknown_flow_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown flow type"):
        _mlp("nope")


def test_drifting_head_does_not_divide_by_zero():
    mlp = _mlp("drifting")
    c = mx.zeros((2, 4))
    x = mx.ones((2, 4))
    y = mlp(c, x)
    mx.eval(y)
    assert y.shape == (2, 4)
    assert bool(mx.all(mx.isfinite(y)))


def test_each_head_accepts_only_its_time_count():
    batch = 2
    c = mx.zeros((batch, 4))
    x = mx.ones((batch, 4))
    s = mx.zeros((batch, 1))
    t = mx.ones((batch, 1))

    lsd = _mlp("lsd")(c, s, t, x)
    matching = _mlp("flow_matching")(c, t, x)
    mx.eval(lsd, matching)
    assert bool(mx.all(mx.isfinite(lsd)))
    assert bool(mx.all(mx.isfinite(matching)))

    with pytest.raises(ValueError, match="time conditions"):
        _mlp("drifting")(c, t, x)
    with pytest.raises(ValueError, match="time conditions"):
        _mlp("flow_matching")(c, s, t, x)


def test_decoders_pass_the_matching_time_count():
    def recorder(expected_times: int):
        def v_t(*args):
            assert len(args) == expected_times + 1
            return mx.zeros_like(args[-1])

        return v_t

    x = mx.zeros((1, 4))
    assert lsd_decode(recorder(2), x, num_steps=2).shape == (1, 4)
    assert ot_decode(recorder(1), x, num_steps=3).shape == (1, 4)
    assert drifting_decode(recorder(0), x, num_steps=5).shape == (1, 4)


def test_flow_lm_creates_bos_only_when_asked():
    with_bos = _flow_lm("lsd", insert_bos=True)
    without = _flow_lm("drifting", insert_bos=False)
    bos = dict(tree_flatten(with_bos.parameters()))["bos_before_voice"]
    assert bos.shape == (1, 1, 8)
    assert "bos_before_voice" not in _param_names(without)
    assert with_bos.flow_type == "lsd"
    assert without.flow_type == "drifting"


def test_flow_lm_dispatches_sampler():
    seen = []

    def run(flow_type: str):
        model = _flow_lm(flow_type, insert_bos=False)

        def backbone(input_, text_embeddings, sequence, model_state):
            del input_, text_embeddings, model_state
            return mx.zeros((sequence.shape[0], sequence.shape[1], model.dim))

        flow_net = model.flow_net

        def wrapped(c, *args):
            seen.append((flow_type, len(args) - 1))
            return flow_net(c, *args)

        model.backbone = backbone
        model.flow_net = wrapped
        sequence = mx.zeros((1, 1, 4))
        text = mx.zeros((1, 0, 8))
        output, _ = model(
            sequence,
            text,
            {},
            lsd_decode_steps=1,
            temp=0.0,
            noise_clamp=None,
            eos_threshold=0.0,
        )
        mx.eval(output)
        assert output.shape == (1, 4)
        assert bool(mx.all(mx.isfinite(output)))

    run("lsd")
    run("flow_matching")
    run("drifting")
    assert seen == [("lsd", 2), ("flow_matching", 1), ("drifting", 0)]


def test_unknown_sampler_is_rejected():
    model = _flow_lm("lsd", insert_bos=False)
    model.flow_type = "nope"

    def backbone(input_, text_embeddings, sequence, model_state):
        del input_, text_embeddings, model_state
        return mx.zeros((sequence.shape[0], sequence.shape[1], model.dim))

    model.backbone = backbone
    with pytest.raises(ValueError, match="Unknown flow type"):
        model(
            mx.zeros((1, 1, 4)),
            mx.zeros((1, 0, 8)),
            {},
            lsd_decode_steps=1,
            temp=0.0,
            noise_clamp=None,
            eos_threshold=0.0,
        )


def test_tanh_gelu_matches_the_written_formula():
    grid = [i * 0.25 for i in range(-16, 17)]

    def reference(value: float) -> float:
        scale = math.sqrt(2.0 / math.pi)
        return 0.5 * value * (1.0 + math.tanh(scale * (value + 0.044715 * value**3)))

    got = tanh_gelu(mx.array(grid, dtype=mx.float32))
    mx.eval(got)
    for value, actual in zip(grid, got.tolist()):
        assert abs(actual - reference(value)) < 1e-6
