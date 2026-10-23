import csv
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
BENCHMARK = REPOSITORY / "scripts" / "benchmark_h3_optimization.py"


def test_h3_release_benchmark_smoke(tmp_path):
    output = tmp_path / "raw.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK),
            "--warmups",
            "0",
            "--samples",
            "1",
            "--loops",
            "1",
            "--output",
            str(output),
        ],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "H^3 objective-throughput benchmark" in completed.stdout

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 6
    assert {int(row["n_vertices"]) for row in rows} == {7, 20, 50}
    assert {row["implementation"] for row in rows} == {
        "current_vectorized_bloch_wigner",
        "legacy_0_3_torch_series96",
    }
    assert all(row["compatible"] == "True" for row in rows)
    assert all(int(row["elapsed_ns"]) > 0 for row in rows)
    assert all(row["git_dirty"] in {"True", "False", ""} for row in rows)
    assert all(len(row["benchmark_sha256"]) == 64 for row in rows)
    assert all(len(row["geometry_sha256"]) == 64 for row in rows)
    assert all(len(row["optimization_sha256"]) == 64 for row in rows)
