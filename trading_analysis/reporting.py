"""Save traceable text results and readable charts for a completed experiment."""

import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from trading_analysis.experiment import ExperimentResult


def plot_equity(result: ExperimentResult):
    """Draw all comparisons over identical holdout timestamps, including initial capital."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5.5))
    colors = {
        "svc_baseline": "#3263A4",
        "adaboost_baseline": "#D07B31",
        "selected": "#2B8C62",
        "always_long": "#515B66",
    }
    for name, frame in result.predictions.items():
        label = name.replace("_", " ")
        if name == "selected":
            label += f" ({result.metadata['selected_candidate']})"
        # Starting equity is placed at the preceding observed close, not a made-up interval.
        timestamps = [pd.Timestamp(result.metadata["holdout_window"]["entry"]), *frame.index]
        axis.plot(
            timestamps,
            [1, *frame.net_equity],
            label=label,
            color=colors[name],
            linestyle="--" if name == "selected" else "-",
            linewidth=1.8,
        )
    axis.axhline(1, color="#9AA2AA", linestyle=":", linewidth=1)
    axis.set(
        title="Historical holdout: net growth of 1 unit",
        xlabel="Observed timestamp",
        ylabel="Compounded value after modeled turnover costs",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.text(
        0.5,
        0.01,
        "Historical classroom data; idealized fills and costs; not live performance.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    return figure


def save_results(result: ExperimentResult, output: Path, *, plots=True):
    """Write summaries, per-bar predictions, class reports, metadata, and optional PNG/SVG."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result.validation.to_csv(output / "validation.csv", float_format="%.10g")
    result.holdout.to_csv(output / "holdout.csv", float_format="%.10g")
    for name, frame in result.predictions.items():
        frame.to_csv(output / f"{name}_predictions.csv", float_format="%.10g")
    metadata = {
        **result.metadata,
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "matplotlib")},
        },
    }
    for name, content in (
        ("metadata.json", metadata),
        ("classification_reports.json", result.reports),
    ):
        (output / name).write_text(
            json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure = plot_equity(result)
        figure.savefig(output / "equity.png", dpi=140)
        figure.savefig(output / "equity.svg")
        plt.close(figure)
