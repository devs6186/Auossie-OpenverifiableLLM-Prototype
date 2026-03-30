from dataclasses import asdict
from enum import Enum

from .config import EvaluationConfig, canonical_eval_config_hash
from .policy import TolerancePolicy, compute_policy_hash
from .report import EvaluationReport


class EvaluationFailureCode(str, Enum):
    FAIL_IDENTITY_CHECKPOINT = "FAIL_IDENTITY_CHECKPOINT"
    FAIL_IDENTITY_BENCHMARK = "FAIL_IDENTITY_BENCHMARK"
    FAIL_IDENTITY_EVAL_CONFIG = "FAIL_IDENTITY_EVAL_CONFIG"
    FAIL_POLICY_INTEGRITY = "FAIL_POLICY_INTEGRITY"
    FAIL_METRIC_BOUND = "FAIL_METRIC_BOUND"


def _failed(code: EvaluationFailureCode, reason: str, report: EvaluationReport) -> EvaluationReport:
    return EvaluationReport(
        checkpoint_hash=report.checkpoint_hash,
        benchmark_hash=report.benchmark_hash,
        eval_config_hash=report.eval_config_hash,
        tolerance_policy_hash=report.tolerance_policy_hash,
        metrics=report.metrics,
        verdict="FAIL",
        failure_code=code.value,
        reason=reason,
    )


def verify_evaluation(
    report: EvaluationReport,
    config: EvaluationConfig,
    policy: TolerancePolicy,
    *,
    expected_checkpoint_hash: str,
    expected_benchmark_hash: str,
) -> EvaluationReport:
    # Strict-before-bounded ordering is non-negotiable.
    if report.checkpoint_hash != expected_checkpoint_hash:
        return _failed(
            EvaluationFailureCode.FAIL_IDENTITY_CHECKPOINT,
            "Checkpoint hash mismatch",
            report,
        )

    if report.benchmark_hash != expected_benchmark_hash:
        return _failed(
            EvaluationFailureCode.FAIL_IDENTITY_BENCHMARK,
            "Benchmark hash mismatch",
            report,
        )

    expected_eval_hash = canonical_eval_config_hash(config)
    if report.eval_config_hash != expected_eval_hash:
        return _failed(
            EvaluationFailureCode.FAIL_IDENTITY_EVAL_CONFIG,
            "Evaluation config hash mismatch",
            report,
        )

    recomputed_policy_hash = compute_policy_hash(policy)
    if report.tolerance_policy_hash != recomputed_policy_hash:
        return _failed(
            EvaluationFailureCode.FAIL_POLICY_INTEGRITY,
            "Policy hash mismatch",
            report,
        )

    for metric, observed in report.metrics.items():
        if metric not in policy.calibration_summary:
            continue
        baseline_mean = policy.calibration_summary[metric]["mean"]
        bound = policy.metric_bounds[metric]
        deviation = abs(observed - baseline_mean)
        if deviation > bound:
            return _failed(
                EvaluationFailureCode.FAIL_METRIC_BOUND,
                (
                    f"{metric} out of bound: observed={observed}, "
                    f"baseline_mean={baseline_mean}, bound={bound}"
                ),
                report,
            )

    return EvaluationReport(
        **{
            **asdict(report),
            "verdict": "PASS",
            "failure_code": None,
            "reason": None,
        }
    )
