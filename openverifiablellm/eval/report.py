from dataclasses import dataclass
from typing import Dict, Literal, Optional


@dataclass(frozen=True)
class EvaluationReport:
    checkpoint_hash: str
    benchmark_hash: str
    eval_config_hash: str
    tolerance_policy_hash: str
    metrics: Dict[str, float]
    verdict: Literal["PASS", "FAIL"]
    failure_code: Optional[str]
    reason: Optional[str]
