"""Text prompt tests (no weights, no network).

These pin the v0.2.1 behavior that `b6369a24` must keep after the update.
"""

import pytest

from pocket_tts_mlx.models.tts_model import prepare_text_prompt


def test_short_prompt_is_capitalized_terminated_and_padded():
    text, frames_after_eos_guess = prepare_text_prompt("hello world")
    assert text == " " * 8 + "Hello world."
    assert frames_after_eos_guess == 3


def test_long_prompt_is_not_padded():
    text, frames_after_eos_guess = prepare_text_prompt("this sentence has more than five words")
    assert text == "This sentence has more than five words."
    assert frames_after_eos_guess == 1


def test_existing_terminal_punctuation_is_kept():
    text, _ = prepare_text_prompt("Is this the sixth word here?")
    assert text == "Is this the sixth word here?"


def test_newlines_become_spaces():
    text, _ = prepare_text_prompt("One two three\nfour five six")
    assert text == "One two three four five six."


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError):
        prepare_text_prompt("   ")
