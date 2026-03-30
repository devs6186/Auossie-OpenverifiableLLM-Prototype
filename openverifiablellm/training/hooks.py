from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import torch

from .checkpoint import save_checkpoint_deterministic
from .receipt import StepReceipt, make_step_receipt
from .rng import capture_rng_snapshot


@dataclass(frozen=True)
class ReceiptHookContext:
    step: int
    model_state_dict: Dict[str, torch.Tensor]
    checkpoint_path: str
    training_config_hash: str
    data_chunk_hash: str
    event_type: str = "step_end"


def emit_step_receipt(
    context: ReceiptHookContext, parent_receipt_hash: Optional[str]
) -> StepReceipt:
    identity = save_checkpoint_deterministic(
        context.model_state_dict,
        context.checkpoint_path,
        context.step,
    )
    rng_snapshot = capture_rng_snapshot()
    return make_step_receipt(
        step=context.step,
        parent_receipt_hash=parent_receipt_hash,
        training_config_hash=context.training_config_hash,
        checkpoint_hash=identity.tensor_hash,
        rng_hash=rng_snapshot.rng_hash,
        data_chunk_hash=context.data_chunk_hash,
        event_type=context.event_type,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def run_training_with_receipts(
    *,
    max_steps: int,
    step_fn: Callable[[int], Dict[str, torch.Tensor]],
    checkpoint_path_fn: Callable[[int], str],
    training_config_hash: str,
    data_chunk_hash_fn: Callable[[int], str],
) -> list[StepReceipt]:
    receipts: list[StepReceipt] = []
    parent_hash: Optional[str] = None
    for step in range(max_steps):
        state_dict = step_fn(step)
        context = ReceiptHookContext(
            step=step,
            model_state_dict=state_dict,
            checkpoint_path=checkpoint_path_fn(step),
            training_config_hash=training_config_hash,
            data_chunk_hash=data_chunk_hash_fn(step),
        )
        receipt = emit_step_receipt(context, parent_hash)
        receipts.append(receipt)
        parent_hash = receipt.receipt_hash
    return receipts
