"""Text prompt and chunking tests (no weights, no network).

The b6369a24 cases pin the v0.2.1 short-prompt behavior that file must keep.
"""

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import pocket_tts_mlx
from pocket_tts_mlx.conditioners.text import JsonTokenizer, build_tokenizer
from pocket_tts_mlx.models.text_chunking import prepare_text_prompt, split_into_best_sentences
from pocket_tts_mlx.models.tts_model import TTSModel
from pocket_tts_mlx.utils.config import load_config

CONFIG_DIR = Path(pocket_tts_mlx.__file__).parent / "config"

# SentencePiece probes start with a ▁ piece. Dropping that first id leaves the punctuation.
_PROBE_BOS = 0


def _flags(config_name: str = "b6369a24.yaml", **overrides) -> dict:
    config = load_config(CONFIG_DIR / config_name)
    flags = {
        "pad_with_spaces_for_short_inputs": config.pad_with_spaces_for_short_inputs,
        "remove_semicolons": config.remove_semicolons,
        "append_terminal_punctuation": config.append_terminal_punctuation,
        "capitalize_first_letter": config.capitalize_first_letter,
        "replace_characters": config.replace_characters,
    }
    flags.update(overrides)
    return flags


class _ProbeTokenizer:
    """Char tokenizer with a leading piece, so punctuation probes match SentencePiece."""

    def encode(self, text: str) -> list[int]:
        return [_PROBE_BOS] + [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens if token != _PROBE_BOS)

    def __call__(self, text: str) -> mx.array:
        return mx.array(self.encode(text))[None, :]


def test_short_prompt_is_capitalized_terminated_and_padded():
    text, frames_after_eos_guess = prepare_text_prompt("hello world", **_flags())
    assert text == " " * 8 + "Hello world."
    assert frames_after_eos_guess == 3


def test_long_prompt_is_not_padded():
    text, frames_after_eos_guess = prepare_text_prompt(
        "this sentence has more than five words", **_flags()
    )
    assert text == "This sentence has more than five words."
    assert frames_after_eos_guess == 1


def test_existing_terminal_punctuation_is_kept():
    text, _ = prepare_text_prompt("Is this the sixth word here?", **_flags())
    assert text == "Is this the sixth word here?"


def test_newlines_become_spaces():
    text, _ = prepare_text_prompt("One two three\nfour five six", **_flags())
    assert text == "One two three four five six."


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError):
        prepare_text_prompt("   ", **_flags())


def test_padding_happens_only_when_the_flag_is_set():
    padded, _ = prepare_text_prompt("hello world", **_flags())
    plain, _ = prepare_text_prompt(
        "hello world", **_flags(pad_with_spaces_for_short_inputs=False)
    )
    assert padded == " " * 8 + "Hello world."
    assert plain == "Hello world."


def test_capitalize_first_letter_can_be_off():
    text, _ = prepare_text_prompt(
        "hello world and more words",
        **_flags(capitalize_first_letter=False, pad_with_spaces_for_short_inputs=False),
    )
    assert text == "hello world and more words."


def test_trailing_comma_becomes_a_period():
    text, _ = prepare_text_prompt(
        "Hi,", **_flags(pad_with_spaces_for_short_inputs=False)
    )
    assert text == "Hi."


def test_french_quote_rewrite_deletes_quotes():
    text, _ = prepare_text_prompt('"Hi?", she said more words', **_flags("french.yaml"))
    assert '"' not in text
    assert "«" not in text
    assert text == "Hi? she said more words."


def test_decimals_are_spoken_and_sentences_still_split():
    chunks = split_into_best_sentences(
        _ProbeTokenizer(),
        "See the value 3.14 right now. Then more words follow after.",
        max_tokens=40,
        **_flags(pad_with_spaces_for_short_inputs=False, capitalize_first_letter=False),
    )
    assert chunks == [
        "See the value 3 point 14 right now.",
        "Then more words follow after.",
    ]


def test_oversized_sentence_splits_on_commas():
    chunks = split_into_best_sentences(
        _ProbeTokenizer(),
        "alpha beta gamma delta epsilon, zeta eta theta iota kappa.",
        max_tokens=30,
        **_flags(pad_with_spaces_for_short_inputs=False, capitalize_first_letter=False),
    )
    assert chunks == [
        "alpha beta gamma delta epsilon,",
        "zeta eta theta iota kappa.",
    ]


def test_prepared_text_reaches_generation():
    captured = []
    model = TTSModel.__new__(TTSModel)
    model.config = load_config(CONFIG_DIR / "b6369a24.yaml")
    model.flow_lm = SimpleNamespace(conditioner=SimpleNamespace(tokenizer=_ProbeTokenizer()))

    def capture(**kwargs):
        captured.append(kwargs["text_to_generate"])
        return iter(())

    model._generate_audio_stream_short_text = capture
    assert list(model.generate_audio_stream({}, "hello world")) == []
    assert captured == [" " * 8 + "Hello world."]


def test_tokenizer_kind_selects_json_or_sentencepiece(monkeypatch):
    seen = []

    def json_tokenizer(n_bins, path):
        seen.append(("json", n_bins, path))
        return "json"

    def sentencepiece_tokenizer(n_bins, path):
        seen.append(("sentencepiece", n_bins, path))
        return "sentencepiece"

    monkeypatch.setattr("pocket_tts_mlx.conditioners.text.JsonTokenizer", json_tokenizer)
    monkeypatch.setattr(
        "pocket_tts_mlx.conditioners.text.SentencePieceTokenizer", sentencepiece_tokenizer
    )

    assert build_tokenizer(4000, "hf://kyutai/repo/tokenizer.json@abc", "tokenizers") == "json"
    assert build_tokenizer(4000, "hf://kyutai/repo/tokenizer.model@abc", "sentencepiece") == (
        "sentencepiece"
    )
    assert build_tokenizer(4000, "hf://kyutai/repo/tokenizer.json@abc", "sentencepiece") == "json"
    assert build_tokenizer(4000, "/tmp/voices/tokenizer.json", "sentencepiece") == "json"
    assert build_tokenizer(4000, "hf://kyutai/repo/tokenizer.model@abc", "json") == "json"
    assert seen == [
        ("json", 4000, "hf://kyutai/repo/tokenizer.json@abc"),
        ("sentencepiece", 4000, "hf://kyutai/repo/tokenizer.model@abc"),
        ("json", 4000, "hf://kyutai/repo/tokenizer.json@abc"),
        ("json", 4000, "/tmp/voices/tokenizer.json"),
        ("json", 4000, "hf://kyutai/repo/tokenizer.model@abc"),
    ]


def test_json_tokenizer_round_trips_without_network(tmp_path):
    vocab = {"Hello": 1, "world": 2, "[UNK]": 0}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))

    loaded = JsonTokenizer(n_bins=len(vocab), tokenizer_path=str(path))
    assert loaded.encode("Hello world") == [1, 2]
    assert loaded.decode([1, 2]) == "Hello world"
    tokens = loaded("Hello world")
    assert tuple(tokens.shape) == (1, 2)
    assert tokens[0].tolist() == [1, 2]

    with pytest.raises(AssertionError):
        JsonTokenizer(n_bins=1, tokenizer_path=str(path))
