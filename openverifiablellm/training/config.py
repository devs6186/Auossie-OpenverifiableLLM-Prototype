import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    optimizer: str
    learning_rate: float
    batch_size: int
    max_steps: int
    seed: int
    data_manifest_hash: str
    tokenizer_manifest_hash: str
    extra: Dict[str, Any] = field(default_factory=dict)


def _canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def canonical_training_config_hash(config: TrainingConfig) -> str:
    payload = asdict(config)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
