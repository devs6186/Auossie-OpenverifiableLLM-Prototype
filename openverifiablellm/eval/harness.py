import hashlib
import json
from pathlib import Path
from typing import Callable, Dict


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_pairwise_qa_harness(
    benchmark_path: Path,
    predict_fn: Callable[[str], str],
) -> Dict[str, float]:
    benchmark_path = Path(benchmark_path)
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark file not found at {benchmark_path}")

    rows = [
        json.loads(line) for line in benchmark_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if not rows:
        raise ValueError("Benchmark file is empty")

    exact = 0
    for row in rows:
        predicted = predict_fn(row["question"])
        if predicted.strip() == row["answer"].strip():
            exact += 1

    accuracy = exact / len(rows)
    return {
        "pairwise_accuracy": accuracy,
        "pairwise_total": float(len(rows)),
        "pairwise_correct": float(exact),
        "benchmark_hash": _sha256_file(benchmark_path),
    }
