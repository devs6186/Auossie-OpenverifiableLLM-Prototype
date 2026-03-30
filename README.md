# Auossie OpenVerifiableLLM Prototype

This repository is an MVP/prototype implementation of the GSoC-26 proposal:
**OpenVerifiableLLM — Deterministic Training Lineage and Policy-Governed Evaluation Verification**.

It extends the baseline project with three proposal-aligned components:

- **Component A (Core)**: Tokenizer trust-chain completion (deterministic `encode/decode/load`, backend-aware hash parity, report-aligned checks).
- **Component B (Core)**: Training lineage MVP (canonical training config hash, deterministic tensor-level checkpoint hashing, RNG provenance, parent-linked step receipts, chain verifier).
- **Component C (Target)**: Evaluation verification layer (strict identity prechecks, policy-sealed tolerance calibration, fail-closed ordering).

## Scope Boundaries

- No cross-hardware universal bitwise reproducibility claim.
- Artifact integrity checks are strict hash checks (no tolerance).
- Tolerance only applies to metric checks **after** identity checks pass.

## Prototype Architecture

### Tokenizer Layer (`openverifiablellm/tokenizer`)

- `base.py`: deterministic tokenizer interface.
- `bpe_tokenizer.py`: deterministic BPE backend implementation.
- `sentencepiece_tokenizer.py`: deterministic SentencePiece backend implementation.
- `verify.py`: tokenizer manifests, deterministic contract checks, backend parity verification.
- `train.py`: training + backend-aware tokenizer hash manifest API.

### Training Lineage Layer (`openverifiablellm/training`)

- `config.py`: immutable `TrainingConfig` + canonical hash.
- `checkpoint.py`: deterministic safetensors checkpoint save/load/hash.
- `rng.py`: RNG capture/restore/hash utilities.
- `receipt.py`: `StepReceipt`, parent-link hashing, chain verification and failure codes.
- `hooks.py`: hook-style receipt emission during step execution.

### Evaluation Verification Layer (`openverifiablellm/eval`)

- `config.py`: `EvaluationConfig` identity model.
- `report.py`: `EvaluationReport`.
- `policy.py` + `calibrate.py`: N>=10 calibration + policy hash sealing.
- `harness.py`: pairwise QA benchmark harness with pinned benchmark hash.
- `verifier.py`: strict-before-bounded fail-closed verification.

### Demo Pipeline

- `openverifiablellm/pipeline/mvp_demo.py`: end-to-end tokenizer → training lineage → evaluation verification run.

## Failure Taxonomy Implemented

- `FAIL_IDENTITY_CHECKPOINT`
- `FAIL_IDENTITY_BENCHMARK`
- `FAIL_IDENTITY_EVAL_CONFIG`
- `FAIL_POLICY_INTEGRITY`
- `FAIL_RECEIPT_CHAIN_MISSING`
- `FAIL_RECEIPT_CHAIN_REORDERED`
- `FAIL_RECEIPT_CHAIN_TAMPERED`
- `FAIL_METRIC_BOUND`

`FAIL_REPLAY_DIVERGENCE` is represented in adversarial tests by explicit divergence assertions on checkpoint identity and trajectory mismatch behavior.

## Test Coverage Added

- `tests/test_tokenizer.py`
- `tests/test_training.py`
- `tests/test_eval.py`
- `tests/test_falsifiability.py`
- `tests/test_pipeline_demo.py`

These are in addition to existing baseline tests.

## Quick Start

```bash
python -m pip install -e .
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

## Run the MVP Demo

```bash
python -c "from pathlib import Path; from openverifiablellm.pipeline import run_mvp_demo; print(run_mvp_demo(Path('.')))"
```

Generated artifacts:

- `artifacts/tokenizer/*`
- `artifacts/training/*`
- `artifacts/evaluation/*`
- `artifacts/final_verdict.json`

Example prototype run output:

```json
{
  "tokenizer_all_passed": true,
  "receipt_chain_passed": true,
  "evaluation_verdict": "PASS",
  "checkpoint_hash": "32e968f30a3631096b6106e04b312d1c8bca6d1a49b9e831fbf0b01f5ff044d1",
  "benchmark_hash": "657c45798c7be501020bbe5087ac0bc426c86d733af4069496e3c714a62c3186",
  "policy_hash": "26ddfabed3b5fa0947a51f3bdfab1df25fc35a5299b17f67d4a725a0329e2393"
}
```

Pairwise QA benchmark hash used in the MVP:

`657c45798c7be501020bbe5087ac0bc426c86d733af4069496e3c714a62c3186`

## Notes

- Deterministic checkpoint identity uses tensor-level hashing over safetensors content, not `.pt` container hash.
- Evaluation verification is intentionally fail-closed: identity failures short-circuit metric checks.
