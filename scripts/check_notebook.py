"""Execute the documented notebook in a fresh kernel without modifying the source."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Use the caller's environment and save an executed copy under artifacts/."""
    output = ROOT / "artifacts" / "notebooks"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(ROOT / "Starter_Code" / "machine_learning_trading_bot.ipynb"),
            "--output-dir",
            str(output),
            "--ExecutePreprocessor.timeout=180",
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
