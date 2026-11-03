"""Select on a validation period, freeze the choice, then evaluate a later holdout."""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from trading_analysis.data import DEFAULT_DATA, build_features, load_close
from trading_analysis.evaluation import ModelConfig, backtest, evaluate, summarize

# Declared before looking at holdout results. Ties keep the first candidate.
CANDIDATES = (
    ModelConfig("svc_baseline"),
    ModelConfig("svc_six_months", train_months=6),
    ModelConfig("svc_shorter_smas", fast=10, slow=50),
    ModelConfig("adaboost_baseline", classifier="adaboost"),
)


@dataclass
class ExperimentResult:
    """Reviewable tables and provenance, with no broker connection or saved executable model."""

    validation: pd.DataFrame
    holdout: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    reports: dict
    metadata: dict


def _boundary(value, name: str) -> pd.Timestamp:
    """Accept a timestamp without silently inventing a timezone or missing date."""
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date.") from exc
    if pd.isna(result) or result.tz is not None:
        raise ValueError(f"{name} must be a valid timezone-naive date.")
    return result


def run_experiment(
    data_path: Path = DEFAULT_DATA,
    *,
    validation_start="2019-01-01",
    test_start="2020-01-01",
    seed: int = 42,
    cost_bps: float = 5,
) -> ExperimentResult:
    """Compare four fixed configurations on common validation and holdout timestamps.

    Training uses the trailing 3 or 6 calendar months before each evaluation
    boundary. Selection uses validation balanced accuracy only. Selected and
    baseline models are refit before the holdout and then remain frozen.
    """
    validation_start = _boundary(validation_start, "Validation start")
    test_start = _boundary(test_start, "Test start")
    if validation_start >= test_start:
        raise ValueError("Validation start must precede test start.")
    data_path = Path(data_path)
    # Hash the exact bytes we read, and reject a file changed during the read.
    source_bytes = data_path.read_bytes()
    close = load_close(data_path)
    if source_bytes != data_path.read_bytes():
        raise ValueError("The input file changed during loading; rerun with a stable file.")
    feature_sets = {c.name: build_features(close, c.fast, c.slow) for c in CANDIDATES}
    common = feature_sets[CANDIDATES[0].name].index
    for frame in feature_sets.values():
        common = common.intersection(frame.index, sort=False)
    feature_sets = {name: frame.loc[common] for name, frame in feature_sets.items()}
    validation_index = common[(common >= validation_start) & (common < test_start)]
    validation_rows, training_windows = [], {}
    for config in CANDIDATES:
        result = evaluate(
            feature_sets[config.name],
            config,
            validation_start,
            test_start,
            seed=seed,
            cost_bps=cost_bps,
        )
        validation_rows.append({"candidate": config.name, **asdict(config), **result.metrics})
        training_windows[config.name] = {
            "start": result.train_index[0].isoformat(),
            "end": result.train_index[-1].isoformat(),
            "observations": len(result.train_index),
        }
    validation = pd.DataFrame(validation_rows).set_index("candidate").drop(columns="name")
    selected_name = validation.balanced_accuracy.idxmax()
    selected = next(c for c in CANDIDATES if c.name == selected_name)

    # Nothing below this point feeds back into the chosen candidate.
    comparisons = {
        "svc_baseline": CANDIDATES[0],
        "adaboost_baseline": CANDIDATES[-1],
        "selected": selected,
    }
    predictions, reports, holdout_rows, holdout_training = {}, {}, [], {}
    for label, config in comparisons.items():
        result = evaluate(
            feature_sets[config.name], config, test_start, seed=seed, cost_bps=cost_bps
        )
        predictions[label] = result.predictions
        reports[label] = result.report
        holdout_rows.append({"model": label, "configuration": config.name, **result.metrics})
        holdout_training[label] = {
            "start": result.train_index[0].isoformat(),
            "end": result.train_index[-1].isoformat(),
            "observations": len(result.train_index),
        }
    test = feature_sets[CANDIDATES[0].name].loc[common >= test_start]
    always_long = backtest(test.actual_return, pd.Series(1, index=test.index), cost_bps)
    always_long["actual_direction"] = test.actual_direction
    metrics, report = summarize(always_long, test.actual_direction)
    predictions["always_long"] = always_long
    reports["always_long"] = report
    holdout_rows.append({"model": "always_long", "configuration": "constant_long", **metrics})
    holdout = pd.DataFrame(holdout_rows).set_index("model")
    assert all(p.index.equals(always_long.index) for p in predictions.values())
    metadata = {
        "experiment": "chronological_sma_classifiers_v1",
        "source": {
            "file": data_path.name,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "observations": len(close),
            "start": close.index[0].isoformat(),
            "end": close.index[-1].isoformat(),
        },
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "validation_window": {
            "start": validation_index[0].isoformat(),
            "end": validation_index[-1].isoformat(),
            "observations": len(validation_index),
        },
        "validation_training": training_windows,
        "holdout_training": holdout_training,
        "holdout_window": {
            "entry": close.index[close.index.get_loc(test.index[0]) - 1].isoformat(),
            "start": test.index[0].isoformat(),
            "end": test.index[-1].isoformat(),
        },
        "selected_candidate": selected_name,
        "selection_metric": "validation balanced_accuracy; first candidate wins ties",
        "candidates": [asdict(c) for c in CANDIDATES],
        "seed": seed,
        "cost_bps_per_turnover_unit": float(cost_bps),
        "timing": "Prior-observation SMAs predict the next observed close-to-close return",
        "assumptions": [
            "SMA windows count observed bars, not days; no resampling or forward filling",
            "Idealized execution at the previous observed close, without latency or spread",
            "Constant +1/-1 exposure, proportional turnover costs, initial entry and final exit",
            "No borrow fees, taxes, dividends, liquidity limits, or margin modeling",
            "Historical symbol, timezone, adjustment policy, and data license are unverified",
            "The historical holdout is not new live data or evidence of future profitability",
        ],
    }
    return ExperimentResult(validation, holdout, predictions, reports, metadata)
