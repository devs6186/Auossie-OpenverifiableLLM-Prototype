import json

import pytest

from openverifiablellm.tokenizer import (
    compute_tokenizer_manifest,
    hash_tokenizer_config,
    train_tokenizer,
    verify_backend_hash_parity,
    verify_deterministic_contract,
)
from openverifiablellm.tokenizer.factory import create_tokenizer

SAMPLE_TEXT = "OpenVerifiableLLM requires deterministic tokenization.\n" * 500


@pytest.fixture
def sample_text_file(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text(SAMPLE_TEXT, encoding="utf-8")
    return text_file


@pytest.fixture
def trained_bpe_tokenizer(tmp_path, sample_text_file):
    tokenizer_path = tmp_path / "tokenizer_bpe"
    train_tokenizer(
        text_file=sample_text_file,
        save_path=tokenizer_path,
        tokenizer_type="bpe",
        vocab_size=300,
        min_frequency=2,
    )
    return tokenizer_path


@pytest.fixture
def trained_sentencepiece_tokenizer(tmp_path, sample_text_file):
    tokenizer_path = tmp_path / "tokenizer_spm"
    train_tokenizer(
        text_file=sample_text_file,
        save_path=tokenizer_path,
        tokenizer_type="sentencepiece",
        vocab_size=128,
        min_frequency=2,
    )
    return tokenizer_path


def test_train_tokenizer_creates_bpe_files(trained_bpe_tokenizer):
    assert (trained_bpe_tokenizer / "vocab.json").is_file()
    assert (trained_bpe_tokenizer / "merges.txt").is_file()


def test_train_tokenizer_creates_sentencepiece_files(trained_sentencepiece_tokenizer):
    assert (trained_sentencepiece_tokenizer / "spm.model").is_file()
    assert (trained_sentencepiece_tokenizer / "spm.vocab").is_file()


def test_train_bpe_tokenizer_is_deterministic(tmp_path, sample_text_file):
    path1 = tmp_path / "tokenizer1"
    path2 = tmp_path / "tokenizer2"
    train_tokenizer(sample_text_file, path1, tokenizer_type="bpe", vocab_size=300)
    train_tokenizer(sample_text_file, path2, tokenizer_type="bpe", vocab_size=300)

    vocab1 = (path1 / "vocab.json").read_text(encoding="utf-8")
    vocab2 = (path2 / "vocab.json").read_text(encoding="utf-8")
    assert vocab1 == vocab2

    merges1 = (path1 / "merges.txt").read_text(encoding="utf-8")
    merges2 = (path2 / "merges.txt").read_text(encoding="utf-8")
    assert merges1 == merges2


def test_bpe_deterministic_contract(trained_bpe_tokenizer):
    tokenizer = create_tokenizer("bpe", vocab_size=300, min_frequency=2)
    report = verify_deterministic_contract(
        tokenizer,
        trained_bpe_tokenizer,
        "determinism check for bpe backend",
    )
    assert report.all_passed, report.to_dict()


def test_sentencepiece_deterministic_contract(trained_sentencepiece_tokenizer):
    tokenizer = create_tokenizer("sentencepiece", vocab_size=128, min_frequency=2)
    report = verify_deterministic_contract(
        tokenizer,
        trained_sentencepiece_tokenizer,
        "determinism check for sentencepiece backend",
    )
    assert report.all_passed, report.to_dict()


def test_hash_tokenizer_config_returns_backend_aware_fields(trained_bpe_tokenizer):
    hashes = hash_tokenizer_config(trained_bpe_tokenizer, tokenizer_type="bpe")

    assert hashes["tokenizer_backend"] == "bpe"
    assert "tokenizer_manifest_hash" in hashes
    assert "tokenizer_vocab_hash" in hashes
    assert "tokenizer_merges_hash" in hashes
    assert "tokenizer_backend_metadata" in hashes
    assert "tokenizer_artifact_hashes" in hashes
    assert hashes["tokenizer_vocab_size"] > 0


def test_hash_tokenizer_config_sentencepiece_contains_model_hash(trained_sentencepiece_tokenizer):
    hashes = hash_tokenizer_config(trained_sentencepiece_tokenizer, tokenizer_type="sentencepiece")

    assert hashes["tokenizer_backend"] == "sentencepiece"
    assert "tokenizer_manifest_hash" in hashes
    assert "tokenizer_model_hash" in hashes
    assert "tokenizer_vocab_hash" in hashes
    assert "tokenizer_merges_hash" not in hashes


def test_hash_changes_when_bpe_vocab_changes(trained_bpe_tokenizer):
    before = hash_tokenizer_config(trained_bpe_tokenizer, tokenizer_type="bpe")

    vocab_path = trained_bpe_tokenizer / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab["new_test_token"] = 99999
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    after = hash_tokenizer_config(trained_bpe_tokenizer, tokenizer_type="bpe")
    assert before["tokenizer_manifest_hash"] != after["tokenizer_manifest_hash"]
    assert before["tokenizer_vocab_hash"] != after["tokenizer_vocab_hash"]


def test_hash_changes_when_bpe_merges_change(trained_bpe_tokenizer):
    before = hash_tokenizer_config(trained_bpe_tokenizer, tokenizer_type="bpe")

    merges_path = trained_bpe_tokenizer / "merges.txt"
    original = merges_path.read_text(encoding="utf-8")
    merges_path.write_text(original + "\nxx yy", encoding="utf-8")

    after = hash_tokenizer_config(trained_bpe_tokenizer, tokenizer_type="bpe")
    assert before["tokenizer_manifest_hash"] != after["tokenizer_manifest_hash"]
    assert before["tokenizer_merges_hash"] != after["tokenizer_merges_hash"]


def test_compute_tokenizer_manifest_is_stable(trained_bpe_tokenizer):
    tokenizer = create_tokenizer("bpe", vocab_size=300, min_frequency=2)
    m1 = compute_tokenizer_manifest(tokenizer, trained_bpe_tokenizer)
    m2 = compute_tokenizer_manifest(tokenizer, trained_bpe_tokenizer)
    assert m1["tokenizer_manifest_hash"] == m2["tokenizer_manifest_hash"]
    assert m1["backend_metadata"]["backend"] == "bpe"


def test_backend_hash_parity_reports_success(
    trained_bpe_tokenizer, trained_sentencepiece_tokenizer
):
    bpe_manifest = compute_tokenizer_manifest(
        create_tokenizer("bpe", vocab_size=300, min_frequency=2),
        trained_bpe_tokenizer,
    )
    sp_manifest = compute_tokenizer_manifest(
        create_tokenizer("sentencepiece", vocab_size=128, min_frequency=2),
        trained_sentencepiece_tokenizer,
    )
    report = verify_backend_hash_parity(bpe_manifest, sp_manifest)
    assert report.all_passed, report.to_dict()


def test_bpe_load_fails_when_merges_missing(tmp_path):
    tokenizer_path = tmp_path / "tok"
    tokenizer_path.mkdir()
    (tokenizer_path / "vocab.json").write_text("{}", encoding="utf-8")

    tokenizer = create_tokenizer("bpe", vocab_size=128, min_frequency=2)
    with pytest.raises(FileNotFoundError):
        tokenizer.load(tokenizer_path)


def test_sentencepiece_load_fails_when_model_missing(tmp_path):
    tokenizer_path = tmp_path / "tok"
    tokenizer_path.mkdir()
    (tokenizer_path / "spm.vocab").write_text("dummy", encoding="utf-8")

    tokenizer = create_tokenizer("sentencepiece", vocab_size=128, min_frequency=2)
    with pytest.raises(FileNotFoundError):
        tokenizer.load(tokenizer_path)


def test_train_tokenizer_invalid_vocab_size(sample_text_file, tmp_path):
    with pytest.raises(ValueError, match="vocab_size must be > 0"):
        train_tokenizer(sample_text_file, tmp_path / "tok", vocab_size=0)


def test_train_tokenizer_invalid_min_frequency(sample_text_file, tmp_path):
    with pytest.raises(ValueError, match="min_frequency must be > 0"):
        train_tokenizer(sample_text_file, tmp_path / "tok", min_frequency=0)


def test_train_tokenizer_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_tokenizer(tmp_path / "does_not_exist.txt", tmp_path / "tok")
