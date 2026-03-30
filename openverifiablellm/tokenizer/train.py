import json
import logging
from pathlib import Path
from typing import Union

from .factory import create_tokenizer
from .verify import compute_tokenizer_manifest

logger = logging.getLogger(__name__)

TOKENIZER_VOCAB_SIZE = 32000
TOKENIZER_MIN_FREQUENCY = 2


def train_tokenizer(
    text_file: Union[str, Path],
    save_path: Union[str, Path] = "data/tokenizer",
    tokenizer_type: str = "bpe",
    vocab_size: int = TOKENIZER_VOCAB_SIZE,
    min_frequency: int = TOKENIZER_MIN_FREQUENCY,
) -> Path:
    """
    Train a tokenizer on preprocessed text.

    Currently supports:
    - BPE
    - SentencePiece

    Reproducibility depends on:
    - Stable input data
    - Stable file ordering
    - Pinned tokenizer library versions
    - Consistent execution environment
    """

    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")

    if min_frequency <= 0:
        raise ValueError("min_frequency must be > 0")

    text_file = Path(text_file)
    save_path = Path(save_path)

    if not text_file.is_file():
        raise FileNotFoundError(f"Text file not found at {text_file}. Run preprocessing first.")

    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Training %s tokenizer on %s", tokenizer_type, text_file)

    tokenizer = create_tokenizer(
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    tokenizer.train(text_file, save_path)

    logger.info("Tokenizer saved to %s", save_path)

    return save_path


def hash_tokenizer_config(tokenizer_path: Union[str, Path], tokenizer_type: str = "bpe") -> dict:
    """
    Compute SHA256 hashes of tokenizer configuration files with backend identity.
    """

    tokenizer_path = Path(tokenizer_path)
    tokenizer = create_tokenizer(
        tokenizer_type=tokenizer_type,
        vocab_size=TOKENIZER_VOCAB_SIZE,
        min_frequency=TOKENIZER_MIN_FREQUENCY,
    )
    manifest = compute_tokenizer_manifest(tokenizer, tokenizer_path)
    artifacts = manifest["artifacts"]
    backend = manifest["backend_metadata"]["backend"]
    vocab_path = tokenizer.get_vocab_path(tokenizer_path)
    if not vocab_path.is_file():
        raise FileNotFoundError(f"vocab artifact not found at {vocab_path}")

    if backend == "bpe":
        actual_vocab_size = len(json.loads(vocab_path.read_text(encoding="utf-8")))
    else:
        # SentencePiece vocab is a TSV-like text artifact, not JSON.
        actual_vocab_size = sum(
            1 for line in vocab_path.read_text(encoding="utf-8").splitlines() if line
        )

    logger.info("Tokenizer config hashed successfully")

    response = {
        "tokenizer_backend": manifest["backend_metadata"]["backend"],
        "tokenizer_manifest_hash": manifest["tokenizer_manifest_hash"],
        "tokenizer_vocab_hash": artifacts.get("vocab", ""),
        "tokenizer_vocab_size": actual_vocab_size,
        "tokenizer_artifact_hashes": artifacts,
        "tokenizer_backend_metadata": manifest["backend_metadata"],
    }
    if "merges" in artifacts:
        response["tokenizer_merges_hash"] = artifacts["merges"]
    if "model" in artifacts:
        response["tokenizer_model_hash"] = artifacts["model"]
    return response
