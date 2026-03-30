from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Sequence


class BaseTokenizer(ABC):
    """Abstract base class for deterministic tokenizer implementations."""

    def __init__(self, vocab_size: int, min_frequency: int):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be > 0")

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    @abstractmethod
    def train(self, text_file: Path, save_path: Path) -> None:
        """Train tokenizer and save artifacts."""

    @abstractmethod
    def load(self, tokenizer_dir: Path) -> "BaseTokenizer":
        """Load persisted artifacts into memory."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Deterministic encoding."""

    @abstractmethod
    def decode(self, token_ids: Sequence[int]) -> str:
        """Deterministic decoding."""

    @abstractmethod
    def get_vocab_path(self, tokenizer_dir: Path) -> Path:
        """Return path to vocab artifact."""

    @abstractmethod
    def get_merges_path(self, tokenizer_dir: Path) -> Path | None:
        """Return path to merges artifact (or None for backends without merges)."""

    @abstractmethod
    def artifact_paths(self, tokenizer_dir: Path) -> Dict[str, Path]:
        """Return backend artifact mapping used in hash parity checks."""

    @staticmethod
    @abstractmethod
    def backend_identity() -> Dict[str, str]:
        """Return backend metadata included in tokenizer hash payload."""
