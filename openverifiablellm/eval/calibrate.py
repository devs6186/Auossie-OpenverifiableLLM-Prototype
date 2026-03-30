from dataclasses import asdict
from statistics import mean, pstdev
from typing import Dict, List

from .policy import TolerancePolicy, compute_policy_hash


def calibrate_tolerance_policy(
    runs: List[Dict[str, float]],
    safety_margin: float = 1.5,
) -> TolerancePolicy:
    if len(runs) < 10:
        raise ValueError("Calibration requires N >= 10 runs.")
    if safety_margin <= 0:
        raise ValueError("safety_margin must be > 0")

    metrics = sorted(runs[0].keys())
    for idx, run in enumerate(runs):
        if sorted(run.keys()) != metrics:
            raise ValueError(f"Inconsistent metrics at run index {idx}")

    calibration_summary: Dict[str, Dict[str, float]] = {}
    metric_bounds: Dict[str, float] = {}
    for metric in metrics:
        values = [run[metric] for run in runs]
        std = pstdev(values)
        calibration_summary[metric] = {
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
            "std": std,
        }
        metric_bounds[metric] = std * safety_margin

    provisional = TolerancePolicy(
        n_runs=len(runs),
        metric_bounds=metric_bounds,
        calibration_summary=calibration_summary,
        policy_hash="",
    )
    policy_hash = compute_policy_hash(provisional)
    return TolerancePolicy(**{**asdict(provisional), "policy_hash": policy_hash})
