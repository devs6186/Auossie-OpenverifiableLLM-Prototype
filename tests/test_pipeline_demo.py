from pathlib import Path

from openverifiablellm.pipeline import run_mvp_demo


def test_mvp_demo_generates_expected_artifacts(tmp_path):
    project_root = Path(tmp_path)
    benchmark_dir = project_root / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = benchmark_dir / "pairwise_qa_mvp.jsonl"
    benchmark_path.write_text(
        "\n".join(
            [
                '{"question":"What color is the sky on a clear day?","answer":"blue"}',
                '{"question":"How many days are in a week?","answer":"7"}',
                '{"question":"What is 2 + 2?","answer":"4"}',
                '{"question":"What is the capital of France?","answer":"Paris"}',
                '{"question":"What do bees produce?","answer":"honey"}',
            ]
        ),
        encoding="utf-8",
    )

    verdict = run_mvp_demo(project_root)

    assert verdict["tokenizer_all_passed"] is True
    assert verdict["receipt_chain_passed"] is True
    assert verdict["evaluation_verdict"] == "PASS"

    artifacts = project_root / "artifacts"
    expected_files = [
        artifacts / "tokenizer" / "tokenizer_manifest.json",
        artifacts / "tokenizer" / "tokenizer_verification_report.json",
        artifacts / "training" / "training_config.json",
        artifacts / "training" / "training_config.hash.json",
        artifacts / "training" / "step_receipts.json",
        artifacts / "training" / "receipt_chain_verification.json",
        artifacts / "training" / "checkpoint_identity.json",
        artifacts / "evaluation" / "evaluation_config.json",
        artifacts / "evaluation" / "tolerance_policy.json",
        artifacts / "evaluation" / "evaluation_report.json",
        artifacts / "evaluation" / "evaluation_verification.json",
        artifacts / "final_verdict.json",
    ]
    for file_path in expected_files:
        assert file_path.is_file(), f"Expected artifact missing: {file_path}"
