import base64
import hashlib
import json
import pickle
import random
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class RNGSnapshot:
    python_state_b64: str
    numpy_state_b64: str
    torch_cpu_state_b64: str
    torch_cuda_state_b64: Optional[str]
    rng_hash: str


def _b64_pickle(obj) -> str:
    return base64.b64encode(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")


def _b64_unpickle(data: str):
    return pickle.loads(base64.b64decode(data.encode("ascii")))


def hash_rng_snapshot(snapshot: RNGSnapshot) -> str:
    payload = {
        "python_state_b64": snapshot.python_state_b64,
        "numpy_state_b64": snapshot.numpy_state_b64,
        "torch_cpu_state_b64": snapshot.torch_cpu_state_b64,
        "torch_cuda_state_b64": snapshot.torch_cuda_state_b64,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def capture_rng_snapshot() -> RNGSnapshot:
    python_state = _b64_pickle(random.getstate())
    numpy_state = _b64_pickle(np.random.get_state())
    torch_cpu_state = base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode("ascii")
    torch_cuda_state = None
    if torch.cuda.is_available():
        cuda_states = [state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all()]
        torch_cuda_state = _b64_pickle(cuda_states)

    snapshot = RNGSnapshot(
        python_state_b64=python_state,
        numpy_state_b64=numpy_state,
        torch_cpu_state_b64=torch_cpu_state,
        torch_cuda_state_b64=torch_cuda_state,
        rng_hash="",
    )
    return RNGSnapshot(**{**asdict(snapshot), "rng_hash": hash_rng_snapshot(snapshot)})


def restore_rng_snapshot(snapshot: RNGSnapshot) -> None:
    random.setstate(_b64_unpickle(snapshot.python_state_b64))
    np.random.set_state(_b64_unpickle(snapshot.numpy_state_b64))
    cpu_bytes = base64.b64decode(snapshot.torch_cpu_state_b64.encode("ascii"))
    cpu_state = torch.tensor(list(cpu_bytes), dtype=torch.uint8)
    torch.set_rng_state(cpu_state)
    if snapshot.torch_cuda_state_b64 is not None and torch.cuda.is_available():
        cuda_states = _b64_unpickle(snapshot.torch_cuda_state_b64)
        tensors = [
            torch.tensor(list(state), dtype=torch.uint8, device="cpu") for state in cuda_states
        ]
        torch.cuda.set_rng_state_all(tensors)

    recomputed = hash_rng_snapshot(snapshot)
    if recomputed != snapshot.rng_hash:
        raise ValueError(
            f"RNG snapshot integrity mismatch: expected {snapshot.rng_hash}, got {recomputed}"
        )
