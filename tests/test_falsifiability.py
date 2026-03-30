import copy
import hashlib
from dataclasses import replace

import pytest
import torch

from openverifiablellm.eval import (
    EvaluationConfig,
    EvaluationFailureCode,
    EvaluationReport,
    calibrate_tolerance_policy,
    canonical_eval_config_hash,
    verify_evaluation,
)
from openverifiablellm.training import (
    FailureCode,
    TrainingConfig,
    canonical_training_config_hash,
    load_checkpoint_verified,
    run_training_with_receipts,
    save_checkpoint_deterministic,
    verify_receipt_chain,
)


def _config_hash() -> str:
    cfg = TrainingConfig(
        model_name="toy",
        optimizer="sgd",
        learning_rate=0.1,
        batch_size=2,
        max_steps=3,
        seed=123,
        data_manifest_hash="data_hash",
        tokenizer_manifest_hash="tok_hash",
    )
    return canonical_training_config_hash(cfg)


def _state(step: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(100 + step)
    return {
        "w": torch.randn(4, 4),
        "b": torch.randn(4),
    }


def _calibration_policy():
    runs = [{"pairwise_accuracy": 0.81 + (i * 0.001)} for i in range(10)]
    return calibrate_tolerance_policy(runs, safety_margin=2.0)


def _benchmark_hash() -> str:
    return hashlib.sha256(b"pairwise-qa-benchmark").hexdigest()


def _base_eval_report(
    checkpoint_hash: str, benchmark_hash: str, accuracy: float
) -> EvaluationReport:
    config = EvaluationConfig(
        benchmark_name="pairwise_qa",
        benchmark_hash=benchmark_hash,
        checkpoint_hash=checkpoint_hash,
        metrics=["pairwise_accuracy"],
    )
    policy = _calibration_policy()
    return EvaluationReport(
        checkpoint_hash=checkpoint_hash,
        benchmark_hash=benchmark_hash,
        eval_config_hash=canonical_eval_config_hash(config),
        tolerance_policy_hash=policy.policy_hash,
        metrics={"pairwise_accuracy": accuracy},
        verdict="PASS",
        failure_code=None,
        reason=None,
    )


def test_clean_audit_passes(tmp_path):
    cfg_hash = _config_hash()
    receipts = run_training_with_receipts(
        max_steps=3,
        step_fn=lambda step: _state(step),
        checkpoint_path_fn=lambda step: str(tmp_path / f"clean_{step}.safetensors"),
        training_config_hash=cfg_hash,
        data_chunk_hash_fn=lambda step: f"chunk_{step}",
    )
    chain_result = verify_receipt_chain(receipts)
    assert chain_result.passed

    final_identity = save_checkpoint_deterministic(
        _state(99), tmp_path / "final.safetensors", step=99
    )
    _ = load_checkpoint_verified(tmp_path / "final.safetensors", final_identity.tensor_hash)

    benchmark_hash = _benchmark_hash()
    config = EvaluationConfig(
        benchmark_name="pairwise_qa",
        benchmark_hash=benchmark_hash,
        checkpoint_hash=final_identity.tensor_hash,
        metrics=["pairwise_accuracy"],
    )
    policy = _calibration_policy()
    report = _base_eval_report(
        checkpoint_hash=final_identity.tensor_hash,
        benchmark_hash=benchmark_hash,
        accuracy=policy.calibration_summary["pairwise_accuracy"]["mean"],
    )
    verdict = verify_evaluation(
        report,
        config,
        policy,
        expected_checkpoint_hash=final_identity.tensor_hash,
        expected_benchmark_hash=benchmark_hash,
    )
    assert verdict.verdict == "PASS"


def test_bad_seed_fails_with_trajectory_divergence(tmp_path):
    cfg_hash = _config_hash()
    clean_receipts = run_training_with_receipts(
        max_steps=3,
        step_fn=lambda step: _state(step),
        checkpoint_path_fn=lambda step: str(tmp_path / f"clean_{step}.safetensors"),
        training_config_hash=cfg_hash,
        data_chunk_hash_fn=lambda step: f"chunk_{step}",
    )

    def bad_seed_state(step: int):
        torch.manual_seed(999 + step)
        return {
            "w": torch.randn(4, 4),
            "b": torch.randn(4),
        }

    bad_receipts = run_training_with_receipts(
        max_steps=3,
        step_fn=bad_seed_state,
        checkpoint_path_fn=lambda step: str(tmp_path / f"badseed_{step}.safetensors"),
        training_config_hash=cfg_hash,
        data_chunk_hash_fn=lambda step: f"chunk_{step}",
    )

    assert clean_receipts[-1].checkpoint_hash != bad_receipts[-1].checkpoint_hash


def test_noise_injected_fails_with_hash_mismatch(tmp_path):
    clean_state = _state(10)
    clean_identity = save_checkpoint_deterministic(
        clean_state, tmp_path / "clean.safetensors", step=10
    )
    noisy_state = {k: v.clone() for k, v in clean_state.items()}
    noisy_state["w"][0, 0] += 0.0001
    noisy_identity = save_checkpoint_deterministic(
        noisy_state, tmp_path / "noisy.safetensors", step=10
    )
    assert clean_identity.tensor_hash != noisy_identity.tensor_hash
    with pytest.raises(ValueError, match="FAIL_IDENTITY_CHECKPOINT"):
        load_checkpoint_verified(tmp_path / "noisy.safetensors", clean_identity.tensor_hash)


def test_post_training_sabotage_fails_hash_only(tmp_path):
    state = _state(20)
    identity = save_checkpoint_deterministic(state, tmp_path / "sabotage.safetensors", step=20)
    bytes_data = bytearray((tmp_path / "sabotage.safetensors").read_bytes())
    bytes_data[-1] ^= 0x01
    (tmp_path / "sabotage.safetensors").write_bytes(bytes(bytes_data))
    with pytest.raises(ValueError, match="FAIL_IDENTITY_CHECKPOINT"):
        load_checkpoint_verified(tmp_path / "sabotage.safetensors", identity.tensor_hash)


def test_minimal_corruption_fails_hash_only_within_metric_noise(tmp_path):
    clean_state = _state(30)
    clean_identity = save_checkpoint_deterministic(
        clean_state, tmp_path / "clean.safetensors", step=30
    )
    corrupt_state = copy.deepcopy(clean_state)
    corrupt_state["w"][0, 0] += 1e-7
    corrupt_identity = save_checkpoint_deterministic(
        corrupt_state, tmp_path / "corrupt.safetensors", step=30
    )

    # Simulate metrics still within tolerance envelope.
    benchmark_hash = _benchmark_hash()
    policy = _calibration_policy()
    config = EvaluationConfig(
        benchmark_name="pairwise_qa",
        benchmark_hash=benchmark_hash,
        checkpoint_hash=clean_identity.tensor_hash,
        metrics=["pairwise_accuracy"],
    )
    report = _base_eval_report(
        checkpoint_hash=clean_identity.tensor_hash,
        benchmark_hash=benchmark_hash,
        accuracy=policy.calibration_summary["pairwise_accuracy"]["mean"],
    )
    verdict = verify_evaluation(
        report,
        config,
        policy,
        expected_checkpoint_hash=clean_identity.tensor_hash,
        expected_benchmark_hash=benchmark_hash,
    )
    assert verdict.verdict == "PASS"
    assert clean_identity.tensor_hash != corrupt_identity.tensor_hash
    with pytest.raises(ValueError, match="FAIL_IDENTITY_CHECKPOINT"):
        load_checkpoint_verified(tmp_path / "corrupt.safetensors", clean_identity.tensor_hash)


def test_receipt_chain_failure_codes_missing_reordered_tampered(tmp_path):
    cfg_hash = _config_hash()
    receipts = run_training_with_receipts(
        max_steps=4,
        step_fn=lambda step: _state(step),
        checkpoint_path_fn=lambda step: str(tmp_path / f"chain_{step}.safetensors"),
        training_config_hash=cfg_hash,
        data_chunk_hash_fn=lambda step: f"chunk_{step}",
    )
    missing_result = verify_receipt_chain([receipts[0], receipts[1], receipts[3]])
    assert missing_result.failure_code == FailureCode.FAIL_RECEIPT_CHAIN_MISSING

    reordered_result = verify_receipt_chain([receipts[1], receipts[0], receipts[2], receipts[3]])
    assert reordered_result.failure_code == FailureCode.FAIL_RECEIPT_CHAIN_REORDERED

    tampered = copy.deepcopy(receipts)
    tampered[2] = replace(tampered[2], checkpoint_hash="tampered")
    tampered_result = verify_receipt_chain(tampered)
    assert tampered_result.failure_code == FailureCode.FAIL_RECEIPT_CHAIN_TAMPERED


def test_policy_integrity_and_metric_bound_failure_codes():
    checkpoint_hash = "c" * 64
    benchmark_hash = _benchmark_hash()
    config = EvaluationConfig(
        benchmark_name="pairwise_qa",
        benchmark_hash=benchmark_hash,
        checkpoint_hash=checkpoint_hash,
        metrics=["pairwise_accuracy"],
    )
    policy = _calibration_policy()

    base_report = EvaluationReport(
        checkpoint_hash=checkpoint_hash,
        benchmark_hash=benchmark_hash,
        eval_config_hash=canonical_eval_config_hash(config),
        tolerance_policy_hash=policy.policy_hash,
        metrics={"pairwise_accuracy": 0.95},
        verdict="PASS",
        failure_code=None,
        reason=None,
    )

    # FAIL_POLICY_INTEGRITY
    tampered_policy = replace(
        policy,
        metric_bounds={"pairwise_accuracy": policy.metric_bounds["pairwise_accuracy"] + 0.5},
    )
    policy_result = verify_evaluation(
        base_report,
        config,
        tampered_policy,
        expected_checkpoint_hash=checkpoint_hash,
        expected_benchmark_hash=benchmark_hash,
    )
    assert policy_result.failure_code == EvaluationFailureCode.FAIL_POLICY_INTEGRITY.value

    # FAIL_METRIC_BOUND
    metric_report = replace(
        base_report,
        tolerance_policy_hash=policy.policy_hash,
        metrics={"pairwise_accuracy": 2.0},
    )
    metric_result = verify_evaluation(
        metric_report,
        config,
        policy,
        expected_checkpoint_hash=checkpoint_hash,
        expected_benchmark_hash=benchmark_hash,
    )
    assert metric_result.failure_code == EvaluationFailureCode.FAIL_METRIC_BOUND.value
