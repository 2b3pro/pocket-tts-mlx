"""Config parsing tests (no weights, no network)."""

from pathlib import Path

import pocket_tts_mlx
from pocket_tts_mlx.utils.config import load_config

CONFIG_DIR = Path(pocket_tts_mlx.__file__).parent / "config"


def test_b6369a24_parses():
    config = load_config(CONFIG_DIR / "b6369a24.yaml")
    assert config.flow_lm.lookup_table.tokenizer == "sentencepiece"
    assert config.flow_lm.transformer.num_layers == 6
    assert config.mimi.seanet.dimension == 512
    assert config.mimi.quantizer.dimension == 32


def test_every_shipped_config_parses():
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert paths
    for path in paths:
        load_config(path)
