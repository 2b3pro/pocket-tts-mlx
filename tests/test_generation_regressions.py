import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

import pocket_tts_mlx.models.tts_model as tts_model_module
from pocket_tts_mlx.models.tts_model import TTSModel, _is_safetensors_source


class GenerationRegressionTests(unittest.TestCase):
    def test_generate_audio_stream_uses_prepared_chunk_text(self):
        calls = []

        def fake_generate_audio_stream_short_text(**kwargs):
            calls.append(kwargs)
            yield mx.zeros((1,))

        model = SimpleNamespace(
            flow_lm=SimpleNamespace(conditioner=SimpleNamespace(tokenizer=object())),
            config=SimpleNamespace(
                model_recommended_frames_after_eos=None,
                pad_with_spaces_for_short_inputs=True,
                remove_semicolons=False,
            ),
            _generate_audio_stream_short_text=fake_generate_audio_stream_short_text,
        )

        with patch.object(
            tts_model_module, "split_into_best_sentences", return_value=["hi"]
        ):
            chunks = list(TTSModel.generate_audio_stream(model, {}, "hi"))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(calls[0]["text_to_generate"], "        Hi.")
        self.assertEqual(calls[0]["frames_after_eos"], 5)

    def test_safetensors_sources_accept_revisions_and_query_strings(self):
        cases = {
            "voice.safetensors": True,
            "hf://owner/repo/voices/voice.safetensors@abcdef": True,
            "https://example.com/voice.safetensors?download=1": True,
            "https://example.com/voice.wav?format=safetensors": False,
        }

        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(_is_safetensors_source(source), expected)


if __name__ == "__main__":
    unittest.main()
