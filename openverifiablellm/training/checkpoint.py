import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors


@dataclass(frozen=True)
class CheckpointIdentity:
    step: int
    tensor_hash: str
    serializer: str
    serializer_version: str
    checkpoint_path: str


def hash_checkpoint_tensors(state_dict: Dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()


def save_checkpoint_deterministic(
    state_dict: Dict[str, torch.Tensor],
    path: Path,
    step: int,
) -> CheckpointIdentity:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {k: state_dict[k].detach().cpu().contiguous() for k in sorted(state_dict)}
    save_safetensors(normalized, str(path))
    tensor_hash = hash_checkpoint_tensors(normalized)
    return CheckpointIdentity(
        step=step,
        tensor_hash=tensor_hash,
        serializer="safetensors-canonical",
        serializer_version="1",
        checkpoint_path=str(path),
    )


def load_checkpoint_verified(path: Path, expected_hash: str) -> Dict[str, torch.Tensor]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    state_dict = load_safetensors(str(path))
    actual_hash = hash_checkpoint_tensors(state_dict)
    if actual_hash != expected_hash:
        raise ValueError(
            f"FAIL_IDENTITY_CHECKPOINT: expected {expected_hash}, got {actual_hash} for {path}"
        )
    return state_dict
