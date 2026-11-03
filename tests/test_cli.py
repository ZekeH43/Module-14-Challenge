"""Run the public command from another directory and inspect its actual output files."""

import hashlib
import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from trading_analysis.data import DEFAULT_DATA, ROOT


def run_cli(directory, *args):
    return subprocess.run(
        [sys.executable, "-m", "trading_analysis", *args],
        cwd=directory,
        env={**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_writes_reproducible_tables_provenance_and_plots(tmp_path):
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "artifacts"
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["source"]["sha256"] == hashlib.sha256(DEFAULT_DATA.read_bytes()).hexdigest()
    assert metadata["source"]["observations"] == 4323
    assert (
        metadata["selected_candidate"] in pd.read_csv(output / "validation.csv").candidate.tolist()
    )
    holdout = pd.read_csv(output / "holdout.csv")
    assert len(holdout) == 4
    reports = json.loads((output / "classification_reports.json").read_text())
    for name in holdout.model:
        predictions = pd.read_csv(output / f"{name}_predictions.csv")
        assert len(predictions) == 639 and not predictions.isna().any().any()
        assert reports[name]["down"]["support"] + reports[name]["up_or_flat"]["support"] == 639
    assert (output / "equity.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in (output / "equity.svg").read_text()


@pytest.mark.parametrize(
    "args",
    [
        ("--cost-bps", "-1"),
        ("--cost-bps", "nan"),
        ("--validation-start", "bad"),
        ("--test-start", "2018-01-01"),
        ("--seed", "-1"),
        ("--data", "missing.csv"),
    ],
)
def test_cli_errors_are_actionable_and_create_no_outputs(tmp_path, args):
    result = run_cli(tmp_path, *args)
    assert result.returncode == 2
    assert "error:" in result.stderr and "Traceback" not in result.stderr
    assert not (tmp_path / "artifacts").exists()
