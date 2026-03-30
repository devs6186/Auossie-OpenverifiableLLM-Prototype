from pathlib import Path
from typing import Dict, Sequence

from tokenizers import ByteLevelBPETokenizer
from tokenizers import __version__ as tokenizers_version

from .base import BaseTokenizer

SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]


class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int, min_frequency: int):
        super().__init__(vocab_size, min_frequency)
        self._tokenizer: ByteLevelBPETokenizer | None = None

    def train(self, text_file: Path, save_path: Path) -> None:
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[str(text_file)],
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.save_model(str(save_path))
        self._tokenizer = tokenizer

    def load(self, tokenizer_dir: Path) -> "BPETokenizer":
        vocab_path = self.get_vocab_path(tokenizer_dir)
        merges_path = self.get_merges_path(tokenizer_dir)
        if merges_path is None:
            raise FileNotFoundError("BPE merges artifact path cannot be None")
        if not vocab_path.is_file():
            raise FileNotFoundError(f"BPE vocab artifact missing at {vocab_path}")
        if not merges_path.is_file():
            raise FileNotFoundError(f"BPE merges artifact missing at {merges_path}")

        self._tokenizer = ByteLevelBPETokenizer(
            vocab=str(vocab_path),
            merges=str(merges_path),
        )
        return self

    def encode(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded. Call load() first.")
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: Sequence[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded. Call load() first.")
        return self._tokenizer.decode(list(token_ids))

    def get_vocab_path(self, tokenizer_dir: Path) -> Path:
        return tokenizer_dir / "vocab.json"

    def get_merges_path(self, tokenizer_dir: Path) -> Path:
        return tokenizer_dir / "merges.txt"

    def artifact_paths(self, tokenizer_dir: Path) -> Dict[str, Path]:
        return {
            "vocab": self.get_vocab_path(tokenizer_dir),
            "merges": self.get_merges_path(tokenizer_dir),
        }

    @staticmethod
    def backend_identity() -> Dict[str, str]:
        return {"backend": "bpe", "library": "tokenizers", "library_version": tokenizers_version}
