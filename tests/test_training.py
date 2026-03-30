import copy
from dataclasses import replace

import pytest
import torch

from openverifiablellm.training import (
    FailureCode,
    TrainingConfig,
    canonical_training_config_hash,
    capture_rng_snapshot,
    hash_checkpoint_tensors,
    load_checkpoint_verified,
    restore_rng_snapshot,
    run_training_with_receipts,
    save_checkpoint_deterministic,
    verify_receipt_chain,
)


def _sample_config() -> TrainingConfig:
    return TrainingConfig(
        model_name="toy-mlp",
        optimizer="sgd",
        learning_rate=0.1,
        batch_size=4,
        max_steps=3,
        seed=42,
        data_manifest_hash="data_hash",
        tokenizer_manifest_hash="tok_hash",
        extra={"momentum": 0.0},
    )


def _sample_state(step: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(100 + step)
    return {
        "linear.weight": torch.randn(2, 2),
        "linear.bias": torch.randn(2),
    }


def test_training_config_hash_deterministic():
    cfg = _sample_config()
    assert canonical_training_config_hash(cfg) == canonical_training_config_hash(cfg)


def test_training_config_hash_changes_on_mutation():
    cfg = _sample_config()
    mutated = replace(cfg, learning_rate=0.2)
    assert canonical_training_config_hash(cfg) != canonical_training_config_hash(mutated)


def test_checkpoint_tensor_hash_same_for_same_values():
    state_a = _sample_state(0)
    state_b = {k: v.clone() for k, v in state_a.items()}
    assert hash_checkpoint_tensors(state_a) == hash_checkpoint_tensors(state_b)


def test_checkpoint_hash_changes_when_tensor_changes():
    state = _sample_state(0)
    before = hash_checkpoint_tensors(state)
    state["linear.bias"][0] += 1.0
    after = hash_checkpoint_tensors(state)
    assert before != after


def test_save_and_load_checkpoint_verified(tmp_path):
    state = _sample_state(1)
    ckpt_path = tmp_path / "step_1.safetensors"
    identity = save_checkpoint_deterministic(state, ckpt_path, step=1)

    loaded = load_checkpoint_verified(ckpt_path, identity.tensor_hash)
    assert hash_checkpoint_tensors(loaded) == identity.tensor_hash


def test_load_checkpoint_verified_fails_on_wrong_hash(tmp_path):
    state = _sample_state(1)
    ckpt_path = tmp_path / "step_1.safetensors"
    identity = save_checkpoint_deterministic(state, ckpt_path, step=1)
    wrong = "0" * 64
    assert wrong != identity.tensor_hash
    with pytest.raises(ValueError, match="FAIL_IDENTITY_CHECKPOINT"):
        load_checkpoint_verified(ckpt_path, wrong)


def test_rng_capture_restore_reproducible():
    torch.manual_seed(123)
    snapshot = capture_rng_snapshot()
    a = torch.rand(5)
    restore_rng_snapshot(snapshot)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_receipt_chain_passes_for_valid_sequence(tmp_path):
    cfg_hash = canonical_training_config_hash(_sample_config())
    receipts = run_training_with_receipts(
        max_steps=3,
        step_fn=lambda step: _sample_state(step),
        checkpoint_path_fn=lambda step: str(tmp_path / f"step_{step}.safetensors"),
        training_config_hash=cfg_hash,
        data_chunk_hash_fn=lambda step: f"data_chunk_{step}",
    )
    result = verify_receipt_chain(receipts)
    assert result.passed
    assert result.failure_code is None


def test_receipt_chain_detects_missing_receipt(tmp_path):
    cfg_hash = canonical_training_config_hash(_sample_config())
    receipts = run_training_with_receipts(
        max_steps=4,
        step_fn=lambda step: _sample_state(step),
        checkpoint_path_fn=lambda step: str(tmp_path / f"step_{step}.safetensors"),
        training_config_hash=cfg_hash,
        data_chunk_hash_fn=lambda step: f"data_chunk_{step}",
    )
    missing = [receipts[0], receipts[1], receipts[3]]
    result = verify_receipt_chain(missing)
    assert not result.passed
    assert result.failure_code == FailureCode.FAIL_RECEIPT_CHAIN_MISSING


def test_receipt_chain_detects_reordered_receipts(tmp_path):
    cfg_hash = canonical_training_config_hash(_sample_config())
    receipts = run_training_with_receipts(
        max_steps=3,
        step_fn=lambda step: _sample_state(step),
        checkpoint_path_fn=lambda step: str(tmp_path / f"step_{step}.safetensors"),
        training_config_hash=cfg_hash,
        data_chunk_hash_fn=lambda step: f"data_chunk_{step}",
    )
    reordered = [receipts[1], receipts[0], receipts[2]]
    result = verify_receipt_chain(reordered)
    assert not result.passed
    assert result.failure_code == FailureCode.FAIL_RECEIPT_CHAIN_REORDERED


def test_receipt_chain_detects_tampered_receipt(tmp_path):
    cfg_hash = canonical_training_config_hash(_sample_config())
    receipts = run_training_with_receipts(
        max_steps=3,
        step_fn=lambda step: _sample_state(step),
        checkpoint_path_fn=lambda step: str(tmp_path / f"step_{step}.safetensors"),
        training_config_hash=cfg_hash,
        data_chunk_hash_fn=lambda step: f"data_chunk_{step}",
    )
    tampered = copy.deepcopy(receipts)
    tampered[1] = replace(tampered[1], checkpoint_hash="bad_hash_value")
    result = verify_receipt_chain(tampered)
    assert not result.passed
    assert result.failure_code == FailureCode.FAIL_RECEIPT_CHAIN_TAMPERED


def test_training_hooks_emit_checkpoint_files(tmp_path):
    cfg_hash = canonical_training_config_hash(_sample_config())
    _ = run_training_with_receipts(
        max_steps=2,
        step_fn=lambda step: _sample_state(step),
        checkpoint_path_fn=lambda step: str(tmp_path / f"step_{step}.safetensors"),
        training_config_hash=cfg_hash,
        data_chunk_hash_fn=lambda step: f"data_chunk_{step}",
    )
    assert (tmp_path / "step_0.safetensors").is_file()
    assert (tmp_path / "step_1.safetensors").is_file()
