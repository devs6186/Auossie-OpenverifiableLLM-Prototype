import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class TolerancePolicy:
    n_runs: int
    metric_bounds: Dict[str, float]
    calibration_summary: Dict[str, Dict[str, float]]
    policy_hash: str


def compute_policy_hash(policy: TolerancePolicy) -> str:
    payload = asdict(policy).copy()
    payload.pop("policy_hash", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
