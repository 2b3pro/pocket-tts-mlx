from types import SimpleNamespace
import unittest

import mlx.core as mx
import numpy as np

from pocket_tts_mlx.models.tts_model import GenerationDidNotReachEOS, TTSModel


class _RetryHarness:
    generate_audio = TTSModel.generate_audio

    def __init__(self, failures_before_success):
        self.failures_before_success = failures_before_success
        self.attempts = 0
        self.seeds = []
        self.raise_flags = []
        self.config = SimpleNamespace(mimi=SimpleNamespace(sample_rate=4))

    def generate_audio_stream(self, **kwargs):
        self.seeds.append(kwargs["seed"])
        self.raise_flags.append(kwargs["_raise_on_missing_eos"])
        self.attempts += 1
        yield mx.array([99.0])
        if self.attempts <= self.failures_before_success:
            raise GenerationDidNotReachEOS("missing EOS")
        yield mx.array([1.0, 2.0])

    def _postprocess_audio_start(self, audio, trim_start_ms, fade_in_ms):
        del trim_start_ms, fade_in_ms
        return audio


class BufferedRecoveryTests(unittest.TestCase):
    def test_discards_failed_attempt_and_retries_with_fresh_seed(self):
        harness = _RetryHarness(failures_before_success=1)

        audio = harness.generate_audio(
            model_state={},
            text_to_generate="Hello.",
            seed=42,
            max_retries=1,
        )

        np.testing.assert_array_equal(np.asarray(audio), np.array([99.0, 1.0, 2.0]))
        self.assertEqual(harness.seeds, [42, 43])
        self.assertEqual(harness.raise_flags, [True, True])

    def test_raises_after_retry_budget_is_exhausted(self):
        harness = _RetryHarness(failures_before_success=10)

        with self.assertRaises(GenerationDidNotReachEOS):
            harness.generate_audio(
                model_state={},
                text_to_generate="Hello.",
                seed=10,
                max_retries=2,
            )

        self.assertEqual(harness.seeds, [10, 11, 12])

    def test_rejects_negative_retry_count(self):
        harness = _RetryHarness(failures_before_success=0)

        with self.assertRaisesRegex(ValueError, "max_retries"):
            harness.generate_audio(
                model_state={},
                text_to_generate="Hello.",
                max_retries=-1,
            )


if __name__ == "__main__":
    unittest.main()
