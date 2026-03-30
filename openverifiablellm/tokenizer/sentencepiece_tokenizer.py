from pathlib import Path
from typing import Dict, Sequence

import sentencepiece as spm

from .base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    """SentencePiece tokenizer implementation."""

    def __init__(self, vocab_size: int, min_frequency: int):
        super().__init__(vocab_size, min_frequency)
        self._processor: spm.SentencePieceProcessor | None = None

    def train(self, text_file: Path, save_path: Path) -> None:
        model_prefix = save_path / "spm"
        spm.SentencePieceTrainer.train(
            input=str(text_file),
            model_prefix=str(model_prefix),
            vocab_size=self.vocab_size,
            character_coverage=1.0,
            hard_vocab_limit=False,
            shuffle_input_sentence=False,
            input_sentence_size=0,
            num_threads=1,
        )

    def load(self, tokenizer_dir: Path) -> "SentencePieceTokenizer":
        model_path = self.get_model_path(tokenizer_dir)
        vocab_path = self.get_vocab_path(tokenizer_dir)
        if not model_path.is_file():
            raise FileNotFoundError(f"SentencePiece model missing at {model_path}")
        if not vocab_path.is_file():
            raise FileNotFoundError(f"SentencePiece vocab missing at {vocab_path}")

        processor = spm.SentencePieceProcessor()
        if not processor.Load(str(model_path)):
            raise RuntimeError(f"Failed to load SentencePiece model at {model_path}")
        self._processor = processor
        return self

    def encode(self, text: str) -> list[int]:
        if self._processor is None:
            raise RuntimeError("Tokenizer is not loaded. Call load() first.")
        return list(self._processor.EncodeAsIds(text))

    def decode(self, token_ids: Sequence[int]) -> str:
        if self._processor is None:
            raise RuntimeError("Tokenizer is not loaded. Call load() first.")
        return self._processor.DecodeIds(list(token_ids))

    def get_model_path(self, tokenizer_dir: Path) -> Path:
        return tokenizer_dir / "spm.model"

    def get_vocab_path(self, tokenizer_dir: Path) -> Path:
        return tokenizer_dir / "spm.vocab"

    def get_merges_path(self, tokenizer_dir: Path) -> None:
        # SentencePiece does not use merges.
        return None

    def artifact_paths(self, tokenizer_dir: Path) -> Dict[str, Path]:
        return {
            "model": self.get_model_path(tokenizer_dir),
            "vocab": self.get_vocab_path(tokenizer_dir),
        }

    @staticmethod
    def backend_identity() -> Dict[str, str]:
        return {
            "backend": "sentencepiece",
            "library": "sentencepiece",
            "library_version": getattr(spm, "__version__", "unknown"),
        }
