"""Text conditioner for MLX using SentencePiece or tokenizer.json."""

import logging

import mlx.core as mx
import mlx.nn as nn
import sentencepiece
import tokenizers

from pocket_tts_mlx.conditioners.base import BaseConditioner, TokenizedText
from pocket_tts_mlx.utils.utils import download_if_necessary

logger = logging.getLogger(__name__)


class SentencePieceTokenizer:
    """SentencePiece tokenizer wrapper with cached model download."""

    def __init__(self, n_bins: int, tokenizer_path: str):
        logger.info("Loading SentencePiece tokenizer from %s", tokenizer_path)
        tokenizer_path = download_if_necessary(tokenizer_path)
        self.sp = sentencepiece.SentencePieceProcessor(str(tokenizer_path))
        assert n_bins == self.sp.vocab_size(), (
            f"sentencepiece tokenizer has vocab_size={self.sp.vocab_size()} "
            f"but n_bins={n_bins} was specified"
        )

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens: list[int]) -> str:
        return self.sp.decode(tokens)

    def __call__(self, text: str) -> mx.array:
        """Tokenize text into a batch of integer ids, shape [1, T]."""
        return mx.array(self.encode(text))[None, :]


class JsonTokenizer:
    """Hugging Face `tokenizer.json` with the same encode/decode surface."""

    def __init__(self, n_bins: int, tokenizer_path: str):
        logger.info("Loading tokenizers json from %s", tokenizer_path)
        tokenizer_path = download_if_necessary(tokenizer_path)
        self.tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        vocab_size = self.tokenizer.get_vocab_size()
        assert n_bins == vocab_size, (
            f"tokenizer has vocab_size={vocab_size} but n_bins={n_bins} was specified"
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def __call__(self, text: str) -> mx.array:
        """Tokenize text into a batch of integer ids, shape [1, T]."""
        return mx.array(self.encode(text))[None, :]


def build_tokenizer(n_bins: int, tokenizer_path: str, kind: str = "sentencepiece"):
    """Pick the backend from the config kind, or from a `.json` path."""
    suffix = str(tokenizer_path).split("@")[0]
    if kind in ("tokenizers", "json") or suffix.endswith(".json"):
        return JsonTokenizer(n_bins, tokenizer_path)
    return SentencePieceTokenizer(n_bins, tokenizer_path)


class LUTConditioner(BaseConditioner):
    """Lookup-table conditioner that maps tokens to embeddings."""

    def __init__(
        self,
        n_bins: int,
        tokenizer_path: str,
        dim: int,
        output_dim: int,
        tokenizer: str = "sentencepiece",
        rank: int | None = None,
    ):
        super().__init__(dim=dim, output_dim=output_dim)
        self.tokenizer = build_tokenizer(n_bins, tokenizer_path, tokenizer)
        self.embed = nn.Embedding(n_bins + 1, rank or self.dim)
        self.proj = None if rank is None else nn.Linear(rank, self.dim, bias=False)

    def prepare(self, x: str) -> TokenizedText:
        """Tokenize raw text into TokenizedText."""
        return TokenizedText(self.tokenizer(x))

    def _get_condition(self, inputs: TokenizedText) -> mx.array:
        """Embed token ids into conditioning vectors."""
        embeddings = self.embed(inputs.tokens)
        return embeddings if self.proj is None else self.proj(embeddings)
