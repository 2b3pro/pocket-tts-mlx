"""October model compatibility and numerical streaming oracles, without downloads."""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from pocket_tts_mlx.models.flow_lm import FlowLMModel, drifting_decode, pointwise_decode
from pocket_tts_mlx.models.mimi import build_mimi
from pocket_tts_mlx.models.text_chunking import _is_decimal_period_boundary
from pocket_tts_mlx.models.tts_model import CONFIGS_DIR, TTSModel
from pocket_tts_mlx.modules.attention import StreamingMultiheadAttention
from pocket_tts_mlx.modules.rope import apply_rope
from pocket_tts_mlx.modules.stateful_module import increment_steps, init_states
from pocket_tts_mlx.utils.config import load_config


SMALL_MODELS = [f"english_{size}{suffix}_2026-10" for size in ("2m", "7m", "18m")
                for suffix in ("", "_refl")]


@pytest.mark.parametrize("stem", SMALL_MODELS)
def test_small_model_parameter_contract_and_forward_pass(stem):
    config = load_config(CONFIGS_DIR / f"{stem}.yaml")
    with patch("pocket_tts_mlx.conditioners.text.build_tokenizer", return_value=None):
        flow = FlowLMModel.from_pydantic_config(config.flow_lm, config.mimi.quantizer.dimension)
    params = dict(tree_flatten(flow.parameters()))
    if config.flow_lm.lookup_table.rank is None:
        assert "conditioner.proj.weight" not in params
    else:
        assert params["conditioner.proj.weight"].shape == (
            config.flow_lm.lookup_table.dim, config.flow_lm.lookup_table.rank)
    assert params["z_proj.weight"].shape == (64, config.flow_lm.transformer.d_model)
    assert flow.flow_net.cond_embed.weight.shape[-1] == 64
    assert flow.flow_net.num_time_conds == 0
    state = init_states(flow, batch_size=1, sequence_length=4)
    first_layer = flow.transformer.layers[0].self_attn
    assert next(iter(state.values()))["cache"].shape[-2:] == (
        first_layer.num_kv_heads, first_layer.head_dim)
    latent, eos = flow(mx.zeros((1, 1, flow.ldim)), mx.zeros((1, 0, flow.dim)), state,
                       lsd_decode_steps=1, temp=0.3, noise_clamp=None,
                       eos_threshold=config.default_eos_threshold, sampling_key=mx.random.key(42))
    assert latent.shape == (1, flow.ldim)
    assert eos.shape == (1, 1)
    assert bool(mx.all(mx.isfinite(latent)))


def test_pointwise_adds_noise_while_drifting_replaces_it():
    noise = mx.array([[1.0, -2.0]])
    head = lambda x: mx.full(x.shape, 3.0)
    np.testing.assert_array_equal(pointwise_decode(head, noise), [[4.0, 1.0]])
    np.testing.assert_array_equal(drifting_decode(head, noise), [[3.0, 3.0]])


class _NoRoPE:
    def __call__(self, q, k, offset):
        return q, k


def test_grouped_attention_prefill_and_incremental_match_numpy():
    attention = StreamingMultiheadAttention(8, 3, _NoRoPE(), attn_dim=12, num_kv_heads=1)
    query = mx.random.normal((1, 4, 8), key=mx.random.key(10))
    state = init_states(attention, batch_size=1, sequence_length=4)
    prefill = attention(query[:, :3], state)
    increment_steps(attention, state, 3)
    incremental = attention(query[:, 3:], state)
    projected = np.asarray(attention.in_proj(query))[0]
    q, k, v = np.split(projected, [12, 16], axis=-1)
    q, k, v = q.reshape(4, 3, 4), k.reshape(4, 1, 4), v.reshape(4, 1, 4)
    scores = np.einsum("thd,shd->hts", q, np.repeat(k, 3, axis=1)) / 2
    scores = np.where(np.tril(np.ones((4, 4), dtype=bool)), scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    expected = np.einsum("hts,shd->thd", weights, np.repeat(v, 3, axis=1)).reshape(1, 4, 12)
    expected = expected @ np.asarray(attention.out_proj.weight).T
    actual = mx.concatenate([prefill, incremental], axis=1)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


def test_grouped_rope_matches_independent_query_and_key_rotations():
    q = mx.random.normal((1, 3, 3, 4), key=mx.random.key(1))
    k = mx.random.normal((1, 3, 1, 4), key=mx.random.key(2))
    actual_q, actual_k = apply_rope(q, k, offset=7)
    expected_q, _ = apply_rope(q, q, offset=7)
    expected_k, _ = apply_rope(k, k, offset=7)
    np.testing.assert_allclose(actual_q, expected_q, atol=1e-6)
    np.testing.assert_allclose(actual_k, expected_k, atol=1e-6)


@pytest.mark.parametrize("stem", SMALL_MODELS)
def test_small_model_decoder_uses_its_own_width_and_context(stem):
    config = load_config(CONFIGS_DIR / f"{stem}.yaml")
    mimi = build_mimi(config.mimi)
    assert mimi.encoder.dimension == 512
    assert mimi.decoder.dimension == config.mimi.decoder_seanet.dimension
    assert mimi.upsample.convtr.convtr.weight.shape[0] == mimi.decoder.dimension
    assert mimi.decoder_transformer.transformer.layers[0].linear1.weight.shape == (
        config.mimi.decoder_transformer.dim_feedforward,
        config.mimi.decoder_transformer.d_model)
    state = init_states(mimi, 1, config.mimi.decoder_transformer.context)
    audio = mimi.decode_from_latent(mx.zeros((1, 1, config.mimi.quantizer.dimension)), state)
    assert audio.shape == (1, 1, mimi.frame_size)
    assert bool(mx.all(mx.isfinite(audio)))


def test_model_eos_recommendation_and_explicit_override():
    with patch.object(TTSModel, "_from_pydantic_config_with_weights",
                      return_value=SimpleNamespace(origin=None)) as build:
        TTSModel.load_model("english_2m_2026-10")
        assert build.call_args.args[-1] == -1.0
        TTSModel.load_model("english_2m_2026-10", eos_threshold=-3.0)
        assert build.call_args.args[-1] == -3.0


def test_decimal_boundary_check_decodes_only_adjacent_tokens():
    calls = []

    class Tokenizer:
        def decode(self, ids):
            calls.append(len(ids))
            return "".join(chr(token) for token in ids)

    tokens = list(map(ord, "x" * 1000 + "3.14 then"))
    assert _is_decimal_period_boundary(tokens, 1002, Tokenizer())
    assert max(calls) <= 2
    assert not _is_decimal_period_boundary(list(map(ord, "3. 14 people.")), 2, Tokenizer())
