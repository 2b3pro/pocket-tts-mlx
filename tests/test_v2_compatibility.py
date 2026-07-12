from pathlib import Path
import tempfile
import unittest

import mlx.core as mx

from pocket_tts_mlx.default_parameters import DEFAULT_VARIANT
from pocket_tts_mlx.models.tts_model import prepare_text_prompt
from pocket_tts_mlx.modules.resample import ConvDownsample1d, ConvTrUpsample1d
from pocket_tts_mlx.utils.config import load_config
from pocket_tts_mlx.utils.weight_conversion import load_model_state_mlx


CONFIG_DIR = Path(__file__).parents[1] / "pocket_tts_mlx" / "config"


class V2ConfigTests(unittest.TestCase):
    def test_v2_english_is_the_default_variant(self):
        self.assertEqual(DEFAULT_VARIANT, "english_2026-04")

    def test_v2_english_dimensions_and_voice_bos(self):
        config = load_config(CONFIG_DIR / "english_2026-04.yaml")

        self.assertTrue(config.flow_lm.insert_bos_before_voice)
        self.assertEqual(config.mimi.inner_dim, 32)
        self.assertEqual(config.mimi.outer_dim, 512)
        self.assertFalse(config.pad_with_spaces_for_short_inputs)

    def test_legacy_config_keeps_short_prompt_padding(self):
        config = load_config(CONFIG_DIR / "b6369a24.yaml")

        self.assertTrue(config.pad_with_spaces_for_short_inputs)
        self.assertFalse(config.flow_lm.insert_bos_before_voice)


class V2ResamplerTests(unittest.TestCase):
    def test_resampler_supports_distinct_inner_and_outer_dimensions(self):
        downsample = ConvDownsample1d(stride=16, dimension=512, out_dimension=32)
        upsample = ConvTrUpsample1d(stride=16, dimension=512, in_dimension=512)

        self.assertEqual(downsample.conv.conv.weight.shape, (32, 32, 512))
        self.assertEqual(upsample.convtr.convtr.weight.shape, (512, 32, 1))


class V2VoiceStateTests(unittest.TestCase):
    def test_scalar_offset_is_converted_to_legacy_current_end(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as state_file:
            mx.save_safetensors(
                state_file.name,
                {
                    "transformer.layers.0.self_attn/offset": mx.array([3], dtype=mx.int64),
                    "transformer.layers.0.self_attn/cache": mx.zeros((2, 1, 3, 1, 2)),
                },
            )
            state = load_model_state_mlx(state_file.name)

        layer = state["transformer.layers.0.self_attn"]
        self.assertEqual(layer["current_end"].shape, (3,))
        self.assertEqual(layer["cache"].shape, (2, 1, 3, 1, 2))


class V2PromptBehaviorTests(unittest.TestCase):
    def test_short_prompt_padding_can_be_disabled(self):
        padded, _ = prepare_text_prompt("hello", pad_with_spaces_for_short_inputs=True)
        unpadded, _ = prepare_text_prompt("hello", pad_with_spaces_for_short_inputs=False)

        self.assertTrue(padded.startswith(" " * 8))
        self.assertEqual(unpadded, "Hello.")

    def test_semicolons_can_be_normalized_per_model(self):
        text, _ = prepare_text_prompt(
            "hello; world",
            pad_with_spaces_for_short_inputs=False,
            remove_semicolons=True,
        )

        self.assertEqual(text, "Hello, world.")


if __name__ == "__main__":
    unittest.main()
