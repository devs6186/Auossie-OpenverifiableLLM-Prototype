import hashlib
import json
from dataclasses import asdict, dataclass
from typing import List, Literal


@dataclass(frozen=True)
class EvaluationConfig:
    benchmark_name: str
    benchmark_hash: str
    checkpoint_hash: str
    metrics: List[str]
    mode: Literal["bounded", "strict"] = "bounded"


def canonical_eval_config_hash(config: EvaluationConfig) -> str:
    payload = asdict(config)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
