"""API and CLI defaults that do not need weights."""

import io
import subprocess
import sys

import pytest

from pocket_tts_mlx import __version__
from pocket_tts_mlx.default_parameters import ALBA_SOURCE_WAV, get_default_voice_for_language
from pocket_tts_mlx.main import cli_voice, read_cli_text
from pocket_tts_mlx.models.tts_model import (
    resolve_sampler_steps,
    resolve_temperature,
    select_model_request,
)
from pocket_tts_mlx.utils.config import load_config
from pocket_tts_mlx.models.tts_model import CONFIGS_DIR


def test_version_is_0_3_0():
    assert __version__ == "0.3.0"


def test_omitting_config_and_language_selects_english():
    assert select_model_request(None, None) == "english"
    assert select_model_request("b6369a24", None) == "b6369a24"
    assert select_model_request(None, "french") == "french"
    with pytest.raises(ValueError, match="both"):
        select_model_request("b6369a24", "english")


def test_omitted_temperature_uses_the_config_value():
    legacy = load_config(CONFIGS_DIR / "b6369a24.yaml")
    assert resolve_temperature(None, legacy.default_temperature) == 0.3
    assert resolve_temperature(0.7, legacy.default_temperature) == 0.7


def test_lsd_decode_steps_overrides_sampler_steps(caplog):
    with caplog.at_level("WARNING"):
        assert resolve_sampler_steps(1, 4) == 4
    assert "deprecated" in caplog.text
    assert resolve_sampler_steps(3, None) == 3


def test_default_voice_follows_the_language_stem():
    assert get_default_voice_for_language("french") == "estelle"
    assert get_default_voice_for_language("french_24l") == "estelle"
    assert get_default_voice_for_language("italian") == "giovanni"
    assert get_default_voice_for_language("german") == "juergen"
    assert get_default_voice_for_language("spanish") == "lola"
    assert get_default_voice_for_language("portuguese") == "rafael"
    assert get_default_voice_for_language("dutch") == "daan"
    assert get_default_voice_for_language("english") == "alba"
    assert get_default_voice_for_language(None) == "alba"
    assert get_default_voice_for_language("english", custom_config=True) == ALBA_SOURCE_WAV


def test_cli_voice_uses_alba_audio_for_a_custom_config():
    assert cli_voice(None, "french", None) == "estelle"
    assert cli_voice("marius", "french", None) == "marius"
    assert cli_voice(None, None, "/tmp/custom.yaml") == ALBA_SOURCE_WAV
    assert cli_voice(None, None, str(CONFIGS_DIR / "dutch.yaml")) == "daan"


def test_dash_reads_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("Hello from stdin."))
    assert read_cli_text("-") == "Hello from stdin."
    assert read_cli_text("keep") == "keep"


def test_help_shows_the_new_flags():
    result = subprocess.run(
        [sys.executable, "-m", "pocket_tts_mlx", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for flag in ("--language", "--config", "--temperature", "--warmup-frames", "--frames-after-eos"):
        assert flag in result.stdout
    assert "default: 50" in result.stdout or "Max tokens" in result.stdout
