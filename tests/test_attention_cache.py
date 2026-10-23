import unittest

import mlx.core as mx
import numpy as np

from pocket_tts_mlx.modules.attention import MimiStreamingMultiheadAttention, complete_mimi_kv
from pocket_tts_mlx.modules.rope import RotaryEmbedding
from pocket_tts_mlx.modules.stateful_module import increment_steps, init_states


class MimiCacheTests(unittest.TestCase):
    def test_batched_attention_matches_full_windowed_attention_after_wrap(self):
        attention = MimiStreamingMultiheadAttention(8, 2, 5, RotaryEmbedding())
        query = mx.random.normal((1, 17, 8), key=mx.random.key(31))
        expected = attention(query, None)
        state = init_states(attention, batch_size=1, sequence_length=5)
        outputs = []
        # The last call is bigger than the ring, as with small models and decode_batch_size=4.
        for start, end in ((0, 3), (3, 7), (7, 17)):
            outputs.append(attention(query[:, start:end], state))
            increment_steps(attention, state, end - start)
        actual = mx.concatenate(outputs, axis=1)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
        layer_state = next(iter(state.values()))
        self.assertEqual(layer_state["cache"].shape, (2, 1, 2, 5, 4))
        self.assertEqual(layer_state["end_offset"].item(), 17)

    def test_vectorized_update_matches_ring_buffer_positions(self):
        batch, heads, steps, dimension, capacity = 2, 2, 4, 3, 8
        cache = mx.zeros((2, batch, heads, capacity, dimension))
        keys = mx.arange(batch * heads * steps * dimension).reshape(
            batch, heads, steps, dimension
        )
        values = keys + 1000
        offsets = mx.array([2, 7])

        result, new_offsets = complete_mimi_kv(cache, offsets, keys, values)
        mx.eval(result.keys, result.values, result.positions, new_offsets)

        expected_keys = np.zeros((batch, heads, capacity, dimension), dtype=np.float32)
        expected_values = np.zeros_like(expected_keys)
        keys_np = np.asarray(keys)
        values_np = np.asarray(values)
        for b, offset in enumerate((2, 7)):
            for step in range(steps):
                index = (offset + step) % capacity
                expected_keys[b, :, index, :] = keys_np[b, :, step, :]
                expected_values[b, :, index, :] = values_np[b, :, step, :]

        np.testing.assert_array_equal(np.asarray(result.keys), expected_keys)
        np.testing.assert_array_equal(np.asarray(result.values), expected_values)
        np.testing.assert_array_equal(np.asarray(new_offsets), np.array([6, 11]))

    def test_rejects_update_larger_than_cache(self):
        cache = mx.zeros((2, 1, 1, 2, 1))
        update = mx.zeros((1, 1, 3, 1))

        with self.assertRaisesRegex(ValueError, "cannot exceed cache capacity"):
            complete_mimi_kv(cache, mx.array([0]), update, update)


if __name__ == "__main__":
    unittest.main()
