import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from openverifiablellm.eval import (
    EvaluationConfig,
    EvaluationReport,
    calibrate_tolerance_policy,
    canonical_eval_config_hash,
    run_pairwise_qa_harness,
    verify_evaluation,
)
from openverifiablellm.tokenizer import (
    compute_tokenizer_manifest,
    train_tokenizer,
    verify_deterministic_contract,
)
from openverifiablellm.tokenizer.factory import create_tokenizer
from openverifiablellm.training import (
    TrainingConfig,
    canonical_training_config_hash,
    load_checkpoint_verified,
    run_training_with_receipts,
    save_checkpoint_deterministic,
    verify_receipt_chain,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _toy_state(step: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(2026 + step)
    return {
        "w": torch.randn(3, 3),
        "b": torch.randn(3),
    }


def run_mvp_demo(project_root: Path) -> dict:
    project_root = Path(project_root)
    artifacts_root = project_root / "artifacts"
    tokenizer_root = artifacts_root / "tokenizer"
    training_root = artifacts_root / "training"
    eval_root = artifacts_root / "evaluation"

    sample_text = (
        "OpenVerifiableLLM emphasizes deterministic, tamper-evident ML verification.\n" * 200
    )
    sample_text_path = artifacts_root / "sample_corpus.txt"
    sample_text_path.parent.mkdir(parents=True, exist_ok=True)
    sample_text_path.write_text(sample_text, encoding="utf-8")

    # Component A: tokenizer trust-chain.
    bpe_dir = tokenizer_root / "bpe"
    train_tokenizer(
        sample_text_path, bpe_dir, tokenizer_type="bpe", vocab_size=200, min_frequency=2
    )
    bpe_tokenizer = create_tokenizer("bpe", vocab_size=200, min_frequency=2)
    tokenizer_manifest = compute_tokenizer_manifest(bpe_tokenizer, bpe_dir)
    tokenizer_contract = verify_deterministic_contract(
        bpe_tokenizer,
        bpe_dir,
        "deterministic tokenizer contract validation",
    )
    _write_json(tokenizer_root / "tokenizer_manifest.json", tokenizer_manifest)
    _write_json(tokenizer_root / "tokenizer_verification_report.json", tokenizer_contract.to_dict())

    # Component B: training lineage receipts.
    train_cfg = TrainingConfig(
        model_name="toy-linear",
        optimizer="sgd",
        learning_rate=0.01,
        batch_size=2,
        max_steps=4,
        seed=2026,
        data_manifest_hash=hashlib.sha256(sample_text.encode("utf-8")).hexdigest(),
        tokenizer_manifest_hash=tokenizer_manifest["tokenizer_manifest_hash"],
    )
    train_cfg_hash = canonical_training_config_hash(train_cfg)
    _write_json(training_root / "training_config.json", asdict(train_cfg))
    _write_json(
        training_root / "training_config.hash.json", {"training_config_hash": train_cfg_hash}
    )

    receipts = run_training_with_receipts(
        max_steps=train_cfg.max_steps,
        step_fn=_toy_state,
        checkpoint_path_fn=lambda step: str(training_root / f"checkpoint_step_{step}.safetensors"),
        training_config_hash=train_cfg_hash,
        data_chunk_hash_fn=lambda step: hashlib.sha256(f"chunk-{step}".encode("utf-8")).hexdigest(),
    )
    _write_json(training_root / "step_receipts.json", [asdict(r) for r in receipts])
    chain_result = verify_receipt_chain(receipts)
    _write_json(training_root / "receipt_chain_verification.json", asdict(chain_result))

    final_ckpt = save_checkpoint_deterministic(
        _toy_state(999),
        training_root / "final_checkpoint.safetensors",
        step=999,
    )
    _ = load_checkpoint_verified(
        training_root / "final_checkpoint.safetensors", final_ckpt.tensor_hash
    )
    _write_json(training_root / "checkpoint_identity.json", asdict(final_ckpt))

    # Component C: evaluation verification layer.
    benchmark_path = project_root / "benchmarks" / "pairwise_qa_mvp.jsonl"
    qa_knowledge = {
        "What color is the sky on a clear day?": "blue",
        "How many days are in a week?": "7",
        "What is 2 + 2?": "4",
        "What is the capital of France?": "Paris",
        "What do bees produce?": "honey",
    }
    harness_metrics = run_pairwise_qa_harness(benchmark_path, lambda q: qa_knowledge[q])
    benchmark_hash = harness_metrics.pop("benchmark_hash")

    # Locked-environment calibration in MVP can legitimately observe near-zero variance.
    calibration_runs = [
        {"pairwise_accuracy": harness_metrics["pairwise_accuracy"]} for _ in range(10)
    ]
    policy = calibrate_tolerance_policy(calibration_runs, safety_margin=2.0)
    _write_json(eval_root / "tolerance_policy.json", asdict(policy))

    eval_cfg = EvaluationConfig(
        benchmark_name="pairwise_qa_mvp",
        benchmark_hash=benchmark_hash,
        checkpoint_hash=final_ckpt.tensor_hash,
        metrics=["pairwise_accuracy"],
        mode="bounded",
    )
    eval_cfg_hash = canonical_eval_config_hash(eval_cfg)
    _write_json(eval_root / "evaluation_config.json", asdict(eval_cfg))

    report = EvaluationReport(
        checkpoint_hash=final_ckpt.tensor_hash,
        benchmark_hash=benchmark_hash,
        eval_config_hash=eval_cfg_hash,
        tolerance_policy_hash=policy.policy_hash,
        metrics={"pairwise_accuracy": harness_metrics["pairwise_accuracy"]},
        verdict="PASS",
        failure_code=None,
        reason=None,
    )
    _write_json(eval_root / "evaluation_report.json", asdict(report))

    evaluation_verdict = verify_evaluation(
        report,
        eval_cfg,
        policy,
        expected_checkpoint_hash=final_ckpt.tensor_hash,
        expected_benchmark_hash=benchmark_hash,
    )
    _write_json(eval_root / "evaluation_verification.json", asdict(evaluation_verdict))

    final_verdict = {
        "tokenizer_all_passed": tokenizer_contract.all_passed,
        "receipt_chain_passed": chain_result.passed,
        "evaluation_verdict": evaluation_verdict.verdict,
        "checkpoint_hash": final_ckpt.tensor_hash,
        "benchmark_hash": benchmark_hash,
        "policy_hash": policy.policy_hash,
    }
    _write_json(artifacts_root / "final_verdict.json", final_verdict)

    return final_verdict
