import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional


class FailureCode(str, Enum):
    FAIL_RECEIPT_CHAIN_MISSING = "FAIL_RECEIPT_CHAIN_MISSING"
    FAIL_RECEIPT_CHAIN_REORDERED = "FAIL_RECEIPT_CHAIN_REORDERED"
    FAIL_RECEIPT_CHAIN_TAMPERED = "FAIL_RECEIPT_CHAIN_TAMPERED"


@dataclass(frozen=True)
class StepReceipt:
    step: int
    parent_receipt_hash: Optional[str]
    receipt_hash: str
    training_config_hash: str
    checkpoint_hash: str
    rng_hash: str
    data_chunk_hash: str
    event_type: str
    timestamp_utc: str


@dataclass(frozen=True)
class ChainVerificationResult:
    passed: bool
    failure_code: Optional[FailureCode]
    reason: Optional[str]
    offending_step: Optional[int]


def _canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_receipt_hash(receipt: StepReceipt) -> str:
    payload = asdict(receipt).copy()
    payload.pop("receipt_hash", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def make_step_receipt(
    *,
    step: int,
    parent_receipt_hash: Optional[str],
    training_config_hash: str,
    checkpoint_hash: str,
    rng_hash: str,
    data_chunk_hash: str,
    event_type: str,
    timestamp_utc: str,
) -> StepReceipt:
    provisional = StepReceipt(
        step=step,
        parent_receipt_hash=parent_receipt_hash,
        receipt_hash="",
        training_config_hash=training_config_hash,
        checkpoint_hash=checkpoint_hash,
        rng_hash=rng_hash,
        data_chunk_hash=data_chunk_hash,
        event_type=event_type,
        timestamp_utc=timestamp_utc,
    )
    receipt_hash = compute_receipt_hash(provisional)
    return StepReceipt(**{**asdict(provisional), "receipt_hash": receipt_hash})


def verify_receipt_chain(receipts: list[StepReceipt]) -> ChainVerificationResult:
    if not receipts:
        return ChainVerificationResult(
            passed=False,
            failure_code=FailureCode.FAIL_RECEIPT_CHAIN_MISSING,
            reason="Receipt sequence is empty",
            offending_step=None,
        )

    steps = [r.step for r in receipts]
    if steps != sorted(steps):
        return ChainVerificationResult(
            passed=False,
            failure_code=FailureCode.FAIL_RECEIPT_CHAIN_REORDERED,
            reason=f"Receipt steps are not monotonic increasing: {steps}",
            offending_step=receipts[0].step,
        )

    expected = list(range(steps[0], steps[0] + len(steps)))
    if steps != expected:
        missing_steps = sorted(set(expected).difference(set(steps)))
        return ChainVerificationResult(
            passed=False,
            failure_code=FailureCode.FAIL_RECEIPT_CHAIN_MISSING,
            reason=f"Missing steps in chain: {missing_steps}",
            offending_step=missing_steps[0] if missing_steps else None,
        )

    for i, receipt in enumerate(receipts):
        recomputed = compute_receipt_hash(receipt)
        if recomputed != receipt.receipt_hash:
            return ChainVerificationResult(
                passed=False,
                failure_code=FailureCode.FAIL_RECEIPT_CHAIN_TAMPERED,
                reason=(
                    f"Receipt hash mismatch at step {receipt.step}: "
                    f"expected {receipt.receipt_hash}, got {recomputed}"
                ),
                offending_step=receipt.step,
            )

        expected_parent = None if i == 0 else receipts[i - 1].receipt_hash
        if receipt.parent_receipt_hash != expected_parent:
            return ChainVerificationResult(
                passed=False,
                failure_code=FailureCode.FAIL_RECEIPT_CHAIN_TAMPERED,
                reason=(
                    f"Broken parent link at step {receipt.step}: "
                    f"expected parent {expected_parent}, got {receipt.parent_receipt_hash}"
                ),
                offending_step=receipt.step,
            )

    return ChainVerificationResult(
        passed=True,
        failure_code=None,
        reason=None,
        offending_step=None,
    )
