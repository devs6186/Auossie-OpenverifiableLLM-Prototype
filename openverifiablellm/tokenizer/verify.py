import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

from openverifiablellm.utils import compute_sha256

from .base import BaseTokenizer


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    expected: Optional[str] = None
    actual: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class TokenizerVerificationReport:
    backend: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.PASS]

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    @property
    def skipped(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.SKIP]

    @property
    def all_passed(self) -> bool:
        return len(self.failed) == 0

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "all_passed": self.all_passed,
            "counts": {
                "total": len(self.checks),
                "passed": len(self.passed),
                "failed": len(self.failed),
                "skipped": len(self.skipped),
            },
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "expected": c.expected,
                    "actual": c.actual,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


def _canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_tokenizer_manifest(
    tokenizer: BaseTokenizer,
    tokenizer_dir: Union[str, Path],
) -> dict:
    tokenizer_dir = Path(tokenizer_dir)
    artifact_hashes: Dict[str, str] = {}
    artifact_sizes: Dict[str, int] = {}

    artifacts = tokenizer.artifact_paths(tokenizer_dir)
    for artifact_name, artifact_path in artifacts.items():
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer artifact '{artifact_name}' not found at {artifact_path}"
            )
        artifact_hashes[artifact_name] = compute_sha256(file_path=artifact_path)
        artifact_sizes[artifact_name] = artifact_path.stat().st_size

    backend_meta = tokenizer.backend_identity()
    payload = {
        "backend_metadata": backend_meta,
        "artifacts": artifact_hashes,
        "artifact_sizes": artifact_sizes,
    }
    payload_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    return {
        "backend_metadata": backend_meta,
        "artifacts": artifact_hashes,
        "artifact_sizes": artifact_sizes,
        "tokenizer_manifest_hash": payload_hash,
    }


def verify_deterministic_contract(
    tokenizer: BaseTokenizer,
    tokenizer_dir: Union[str, Path],
    sample_text: str,
) -> TokenizerVerificationReport:
    report = TokenizerVerificationReport(backend=tokenizer.backend_identity()["backend"])

    loaded = tokenizer.load(Path(tokenizer_dir))
    ids_once = loaded.encode(sample_text)
    ids_twice = loaded.encode(sample_text)
    report.add(
        CheckResult(
            name="encode_determinism",
            status=CheckStatus.PASS if ids_once == ids_twice else CheckStatus.FAIL,
            detail="Repeated encode calls must produce identical token IDs",
        )
    )

    decode_once = loaded.decode(ids_once)
    decode_twice = loaded.decode(ids_once)
    report.add(
        CheckResult(
            name="decode_determinism",
            status=CheckStatus.PASS if decode_once == decode_twice else CheckStatus.FAIL,
            detail="Repeated decode calls must produce identical text",
        )
    )

    reloaded = tokenizer.load(Path(tokenizer_dir))
    ids_reloaded = reloaded.encode(sample_text)
    report.add(
        CheckResult(
            name="load_determinism",
            status=CheckStatus.PASS if ids_once == ids_reloaded else CheckStatus.FAIL,
            detail="Loading artifacts again must preserve encode output",
        )
    )

    return report


def verify_backend_hash_parity(
    bpe_manifest: dict,
    sentencepiece_manifest: dict,
) -> TokenizerVerificationReport:
    report = TokenizerVerificationReport(backend="cross-backend-parity")

    for manifest_name, manifest in (
        ("bpe", bpe_manifest),
        ("sentencepiece", sentencepiece_manifest),
    ):
        has_metadata = "backend_metadata" in manifest and "backend" in manifest["backend_metadata"]
        has_hash = "tokenizer_manifest_hash" in manifest
        report.add(
            CheckResult(
                name=f"{manifest_name}_metadata_present",
                status=CheckStatus.PASS if has_metadata else CheckStatus.FAIL,
                detail="Manifest must include backend metadata",
            )
        )
        report.add(
            CheckResult(
                name=f"{manifest_name}_manifest_hash_present",
                status=CheckStatus.PASS if has_hash else CheckStatus.FAIL,
                detail="Manifest must include tokenizer_manifest_hash",
            )
        )

    bpe_backend = bpe_manifest.get("backend_metadata", {}).get("backend")
    sp_backend = sentencepiece_manifest.get("backend_metadata", {}).get("backend")
    report.add(
        CheckResult(
            name="backend_identity_distinct",
            status=CheckStatus.PASS
            if bpe_backend == "bpe" and sp_backend == "sentencepiece"
            else CheckStatus.FAIL,
            expected="bpe vs sentencepiece",
            actual=f"{bpe_backend} vs {sp_backend}",
            detail="Equivalent hash semantics must still record backend identity",
        )
    )

    bpe_payload = {
        "backend_metadata": bpe_manifest.get("backend_metadata", {}),
        "artifacts": bpe_manifest.get("artifacts", {}),
        "artifact_sizes": bpe_manifest.get("artifact_sizes", {}),
    }
    sp_payload = {
        "backend_metadata": sentencepiece_manifest.get("backend_metadata", {}),
        "artifacts": sentencepiece_manifest.get("artifacts", {}),
        "artifact_sizes": sentencepiece_manifest.get("artifact_sizes", {}),
    }
    bpe_recomputed = hashlib.sha256(_canonical_json(bpe_payload).encode("utf-8")).hexdigest()
    sp_recomputed = hashlib.sha256(_canonical_json(sp_payload).encode("utf-8")).hexdigest()

    report.add(
        CheckResult(
            name="bpe_manifest_hash_stable",
            status=CheckStatus.PASS
            if bpe_recomputed == bpe_manifest.get("tokenizer_manifest_hash")
            else CheckStatus.FAIL,
            detail="BPE manifest hash must match canonical payload hash",
        )
    )
    report.add(
        CheckResult(
            name="sentencepiece_manifest_hash_stable",
            status=CheckStatus.PASS
            if sp_recomputed == sentencepiece_manifest.get("tokenizer_manifest_hash")
            else CheckStatus.FAIL,
            detail="SentencePiece manifest hash must match canonical payload hash",
        )
    )

    return report
