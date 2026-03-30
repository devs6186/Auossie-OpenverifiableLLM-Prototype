from dataclasses import replace

import pytest

from openverifiablellm.eval import (
    EvaluationConfig,
    EvaluationFailureCode,
    EvaluationReport,
    calibrate_tolerance_policy,
    canonical_eval_config_hash,
    run_pairwise_qa_harness,
    verify_evaluation,
)


def _calibration_runs() -> list[dict[str, float]]:
    return [{"pairwise_accuracy": 0.8 + (i * 0.001)} for i in range(10)]


def _policy():
    return calibrate_tolerance_policy(_calibration_runs(), safety_margin=2.0)


def _config(benchmark_hash: str, checkpoint_hash: str) -> EvaluationConfig:
    return EvaluationConfig(
        benchmark_name="pairwise_qa_mvp",
        benchmark_hash=benchmark_hash,
        checkpoint_hash=checkpoint_hash,
        metrics=["pairwise_accuracy"],
        mode="bounded",
    )


def test_calibration_requires_at_least_10_runs():
    with pytest.raises(ValueError, match="N >= 10"):
        calibrate_tolerance_policy([{"pairwise_accuracy": 0.8}] * 9)


def test_calibration_policy_hash_stable():
    p1 = _policy()
    p2 = _policy()
    assert p1.policy_hash == p2.policy_hash


def test_pairwise_harness_returns_accuracy_and_hash(tmp_path):
    benchmark = tmp_path / "pairwise.jsonl"
    benchmark.write_text(
        "\n".join(
            [
                '{"question":"q1","answer":"a1"}',
                '{"question":"q2","answer":"a2"}',
            ]
        ),
        encoding="utf-8",
    )
    outputs = {"q1": "a1", "q2": "a2"}
    metrics = run_pairwise_qa_harness(benchmark, lambda q: outputs[q])
    assert metrics["pairwise_accuracy"] == 1.0
    assert "benchmark_hash" in metrics


def test_verify_evaluation_passes_with_valid_inputs(tmp_path):
    benchmark = tmp_path / "pairwise.jsonl"
    benchmark.write_text('{"question":"q1","answer":"a1"}\n', encoding="utf-8")
    metrics = run_pairwise_qa_harness(benchmark, lambda _: "a1")
    checkpoint_hash = "c" * 64
    cfg = _config(metrics["benchmark_hash"], checkpoint_hash)
    policy = _policy()
    report = EvaluationReport(
        checkpoint_hash=checkpoint_hash,
        benchmark_hash=metrics["benchmark_hash"],
        eval_config_hash=canonical_eval_config_hash(cfg),
        tolerance_policy_hash=policy.policy_hash,
        metrics={"pairwise_accuracy": policy.calibration_summary["pairwise_accuracy"]["mean"]},
        verdict="PASS",
        failure_code=None,
        reason=None,
    )
    result = verify_evaluation(
        report,
        cfg,
        policy,
        expected_checkpoint_hash=checkpoint_hash,
        expected_benchmark_hash=metrics["benchmark_hash"],
    )
    assert result.verdict == "PASS"
    assert result.failure_code is None


def test_verify_evaluation_fails_checkpoint_precheck(tmp_path):
    benchmark = tmp_path / "pairwise.jsonl"
    benchmark.write_text('{"question":"q1","answer":"a1"}\n', encoding="utf-8")
    metrics = run_pairwise_qa_harness(benchmark, lambda _: "a1")
    checkpoint_hash = "c" * 64
    cfg = _config(metrics["benchmark_hash"], checkpoint_hash)
    policy = _policy()
    report = EvaluationReport(
        checkpoint_hash="d" * 64,
        benchmark_hash=metrics["benchmark_hash"],
        eval_config_hash=canonical_eval_config_hash(cfg),
        tolerance_policy_hash=policy.policy_hash,
        metrics={"pairwise_accuracy": 0.8},
        verdict="PASS",
        failure_code=None,
        reason=None,
    )
    result = verify_evaluation(
        report,
        cfg,
        policy,
        expected_checkpoint_hash=checkpoint_hash,
        expected_benchmark_hash=metrics["benchmark_hash"],
    )
    assert result.failure_code == EvaluationFailureCode.FAIL_IDENTITY_CHECKPOINT.value


def test_verify_evaluation_fails_benchmark_precheck(tmp_path):
    benchmark = tmp_path / "pairwise.jsonl"
    benchmark.write_text('{"question":"q1","answer":"a1"}\n', encoding="utf-8")
    metrics = run_pairwise_qa_harness(benchmark, lambda _: "a1")
    checkpoint_hash = "c" * 64
    cfg = _config(metrics["benchmark_hash"], checkpoint_hash)
    policy = _policy()
    report = EvaluationReport(
        checkpoint_hash=checkpoint_hash,
        benchmark_hash="b" * 64,
        eval_config_hash=canonical_eval_config_hash(cfg),
        tolerance_policy_hash=policy.policy_hash,
        metrics={"pairwise_accuracy": 0.8},
        verdict="PASS",
        failure_code=None,
        reason=None,
    )
    result = verify_evaluation(
        report,
        cfg,
        policy,
        expected_checkpoint_hash=checkpoint_hash,
        expected_benchmark_hash=metrics["benchmark_hash"],
    )
    assert result.failure_code == EvaluationFailureCode.FAIL_IDENTITY_BENCHMARK.value


def test_verify_evaluation_fails_eval_config_precheck(tmp_path):
    benchmark = tmp_path / "pairwise.jsonl"
    benchmark.write_text('{"question":"q1","answer":"a1"}\n', encoding="utf-8")
    metrics = run_pairwise_qa_harness(benchmark, lambda _: "a1")
    checkpoint_hash = "c" * 64
    cfg = _config(metrics["benchmark_hash"], checkpoint_hash)
    policy = _policy()
    report = EvaluationReport(
        checkpoint_hash=checkpoint_hash,
        benchmark_hash=metrics["benchmark_hash"],
        eval_config_hash="e" * 64,
        tolerance_policy_hash=policy.policy_hash,
        metrics={"pairwise_accuracy": 0.8},
        verdict="PASS",
        failure_code=None,
        reason=None,
    )
    result = verify_evaluation(
        report,
        cfg,
        policy,
        expected_checkpoint_hash=checkpoint_hash,
        expected_benchmark_hash=metrics["benchmark_hash"],
    )
    assert result.failure_code == EvaluationFailureCode.FAIL_IDENTITY_EVAL_CONFIG.value


def test_verify_evaluation_fails_policy_integrity(tmp_path):
    benchmark = tmp_path / "pairwise.jsonl"
    benchmark.write_text('{"question":"q1","answer":"a1"}\n', encoding="utf-8")
    metrics = run_pairwise_qa_harness(benchmark, lambda _: "a1")
    checkpoint_hash = "c" * 64
    cfg = _config(metrics["benchmark_hash"], checkpoint_hash)
    policy = _policy()
    tampered_policy = replace(
        policy,
        metric_bounds={"pairwise_accuracy": policy.metric_bounds["pairwise_accuracy"] + 0.1},
    )
    report = EvaluationReport(
        checkpoint_hash=checkpoint_hash,
        benchmark_hash=metrics["benchmark_hash"],
        eval_config_hash=canonical_eval_config_hash(cfg),
        tolerance_policy_hash=policy.policy_hash,
        metrics={"pairwise_accuracy": 0.8},
        verdict="PASS",
        failure_code=None,
        reason=None,
    )
    result = verify_evaluation(
        report,
        cfg,
        tampered_policy,
        expected_checkpoint_hash=checkpoint_hash,
        expected_benchmark_hash=metrics["benchmark_hash"],
    )
    assert result.failure_code == EvaluationFailureCode.FAIL_POLICY_INTEGRITY.value


def test_verify_evaluation_fails_metric_bound(tmp_path):
    benchmark = tmp_path / "pairwise.jsonl"
    benchmark.write_text('{"question":"q1","answer":"a1"}\n', encoding="utf-8")
    metrics = run_pairwise_qa_harness(benchmark, lambda _: "a1")
    checkpoint_hash = "c" * 64
    cfg = _config(metrics["benchmark_hash"], checkpoint_hash)
    policy = _policy()
    mean_acc = policy.calibration_summary["pairwise_accuracy"]["mean"]
    bound = policy.metric_bounds["pairwise_accuracy"]
    report = EvaluationReport(
        checkpoint_hash=checkpoint_hash,
        benchmark_hash=metrics["benchmark_hash"],
        eval_config_hash=canonical_eval_config_hash(cfg),
        tolerance_policy_hash=policy.policy_hash,
        metrics={"pairwise_accuracy": mean_acc + bound + 0.05},
        verdict="PASS",
        failure_code=None,
        reason=None,
    )
    result = verify_evaluation(
        report,
        cfg,
        policy,
        expected_checkpoint_hash=checkpoint_hash,
        expected_benchmark_hash=metrics["benchmark_hash"],
    )
    assert result.failure_code == EvaluationFailureCode.FAIL_METRIC_BOUND.value
