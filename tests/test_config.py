"""Config parsing tests (no weights, no network)."""

from pathlib import Path

import pocket_tts_mlx
from pocket_tts_mlx.utils.config import load_config

CONFIG_DIR = Path(pocket_tts_mlx.__file__).parent / "config"

# Upstream pocket_tts/config at 41cbc84, plus the v0.2.1 regression file.
EXPECTED_CONFIGS = {
    "b6369a24.yaml",
    "dutch.yaml",
    "dutch_24l.yaml",
    "english.yaml",
    "english_2026-01.yaml",
    "english_2026-04.yaml",
    "english_2026-04_24l.yaml",
    "english_2026-09.yaml",
    "english_2026-09_24l.yaml",
    "english_drifting_26-09.yaml",
    "french.yaml",
    "french_24l.yaml",
    "german.yaml",
    "german_24l.yaml",
    "italian.yaml",
    "italian_24l.yaml",
    "portuguese.yaml",
    "portuguese_24l.yaml",
    "spanish.yaml",
    "spanish_24l.yaml",
}

ENGLISH_WEIGHTS_REVISION = "983151f13aaeab1b13c1e5e3c2c383d49a9edf3f"


def _weights_revision(url: str) -> str:
    return url.rsplit("@", 1)[1]


def test_b6369a24_parses():
    config = load_config(CONFIG_DIR / "b6369a24.yaml")
    assert config.flow_lm.lookup_table.tokenizer == "sentencepiece"
    assert config.flow_lm.transformer.num_layers == 6
    assert config.mimi.seanet.dimension == 512
    assert config.mimi.quantizer.dimension == 32
    # v0.2.1 text handling stays on; the new schema defaults fill the rest.
    assert config.pad_with_spaces_for_short_inputs is True
    assert config.capitalize_first_letter is True
    assert config.append_terminal_punctuation is True
    assert config.flow_lm.flow.type == "lsd"
    assert config.flow_lm.insert_bos_before_voice is False
    assert config.mimi.inner_dim is None
    assert config.mimi.outer_dim is None
    assert config.default_temperature == 0.3


def test_every_shipped_config_parses():
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert {path.name for path in paths} == EXPECTED_CONFIGS
    for path in paths:
        load_config(path)


def test_english_and_english_2026_09_share_weights_revision():
    english = load_config(CONFIG_DIR / "english.yaml")
    september = load_config(CONFIG_DIR / "english_2026-09.yaml")
    assert _weights_revision(english.weights_path) == ENGLISH_WEIGHTS_REVISION
    assert _weights_revision(september.weights_path) == ENGLISH_WEIGHTS_REVISION
    assert english.weights_path != september.weights_path


def test_nondefault_schema_fields_round_trip():
    drifting = load_config(CONFIG_DIR / "english_drifting_26-09.yaml")
    assert drifting.flow_lm.flow.type == "drifting"
    assert drifting.mimi.inner_dim == 32
    assert drifting.flow_lm.insert_bos_before_voice is True

    french = load_config(CONFIG_DIR / "french.yaml")
    assert french.remove_semicolons is True
    assert french.replace_characters['"'] == ""
    assert french.replace_characters["«"] == ""
