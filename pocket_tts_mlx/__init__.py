"""pocket-tts-mlx: MLX backend for pocket-tts."""

__version__ = "0.3.1.dev0"

from pocket_tts_mlx.models.tts_model import GenerationDidNotReachEOS, TTSModel
from pocket_tts_mlx.text_normalization import DictionaryEntry, UserDictionary, normalize_text

__all__ = [
    "DictionaryEntry",
    "GenerationDidNotReachEOS",
    "TTSModel",
    "UserDictionary",
    "normalize_text",
]
