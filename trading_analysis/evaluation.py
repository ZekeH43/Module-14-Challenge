"""Fit classifiers on past observations and evaluate correctly aligned predictions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from trading_analysis.data import FEATURES


@dataclass(frozen=True)
class ModelConfig:
    """One predeclared model, feature-window, and calendar training-lookback choice."""

    name: str
    classifier: str = "svc"
    fast: int = 4
    slow: int = 100
    train_months: int = 3

    def __post_init__(self):
        if not self.name or self.classifier not in ("svc", "adaboost"):
            raise ValueError("Provide a name and a classifier of svc or adaboost.")
        values = (self.fast, self.slow, self.train_months)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
            raise ValueError("Windows and training months must be positive integers.")
        if self.fast >= self.slow:
            raise ValueError("Fast SMA window must be shorter than slow SMA window.")


def build_model(classifier: str, seed: int = 42) -> Pipeline:
    """Keep scaling and prediction together so test data cannot skip preprocessing."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer from 0 through 4294967295.")
    if classifier == "svc":
        estimator = SVC(C=1.0, kernel="rbf", gamma="scale")
    elif classifier == "adaboost":
        estimator = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=seed)
    else:
        raise ValueError("Classifier must be svc or adaboost.")
    return Pipeline([("scaler", StandardScaler()), ("classifier", estimator)])


def backtest(actual_returns: pd.Series, positions: pd.Series, cost_bps: float = 5) -> pd.DataFrame:
    """Compound idealized long/short returns with a cost per unit of turnover.

    Enter from cash, charge two units when reversing +1/-1, and liquidate at
    the final close. Positions already align to their return interval; do not
    shift them again. Execution at the previous observed close is assumed.
    """
    if not isinstance(actual_returns, pd.Series) or not isinstance(positions, pd.Series):
        raise ValueError("Returns and positions must be indexed pandas Series.")
    if actual_returns.empty or not actual_returns.index.equals(positions.index):
        raise ValueError("Returns and positions need the same nonempty timestamp index.")
    index = actual_returns.index
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans:
        raise ValueError("Backtest rows need valid timestamps.")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Backtest timestamps must be unique and chronological.")
    try:
        cost_bps = float(cost_bps)
        returns = actual_returns.astype(float)
        position = positions.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Returns, positions, and costs must be numeric.") from exc
    if not np.isfinite(cost_bps) or not 0 <= cost_bps <= 100:
        raise ValueError("Cost must be finite and between 0 and 100 basis points.")
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Returns must be finite and greater than -1.")
    if not position.isin([-1, 1]).all():
        raise ValueError("Positions must be -1 (short) or +1 (long).")
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    turnover.iloc[-1] += abs(position.iloc[-1])
    gross = returns * position
    costs = turnover * cost_bps / 10_000
    net = gross - costs
    if (net <= -1).any():
        raise ValueError(
            "A modeled position loses all capital; this simple backtest cannot continue."
        )
    result = pd.DataFrame(
        {
            "position": position.astype(int),
            "actual_return": returns,
            "turnover": turnover,
            "gross_return": gross,
            "cost": costs,
            "net_return": net,
            "gross_equity": (1 + gross).cumprod(),
            "net_equity": (1 + net).cumprod(),
        }
    )
    if not np.isfinite(result.to_numpy()).all() or (result.net_equity <= 0).any():
        raise ValueError("Compounded results exceed numeric limits.")
    result.index = result.index.rename("timestamp")
    return result


def summarize(backtested: pd.DataFrame, actual: pd.Series) -> tuple[dict, dict]:
    """Report true-label-first classification metrics and nonannualized performance."""
    if not backtested.index.equals(actual.index):
        raise ValueError("Actual labels must align exactly with backtest timestamps.")
    report = classification_report(
        actual,
        backtested.position,
        labels=[-1, 1],
        target_names=["down", "up_or_flat"],
        output_dict=True,
        zero_division=0,
    )
    # Include the starting capital of 1 in the peak: a first-bar loss is a drawdown.
    equity = backtested.net_equity.to_numpy()
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    metrics = {
        "accuracy": float(accuracy_score(actual, backtested.position)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, backtested.position)),
        "gross_total_return": float(backtested.gross_equity.iloc[-1] - 1),
        "net_total_return": float(equity[-1] - 1),
        "max_drawdown": float(np.min(equity / peak - 1)),
        "turnover_units": float(backtested.turnover.sum()),
        "long_fraction": float((backtested.position == 1).mean()),
        "observations": len(backtested),
    }
    return metrics, report


@dataclass
class Evaluation:
    """A fitted pipeline, timestamped outcomes, metrics, and exact training dates."""

    model: Pipeline
    predictions: pd.DataFrame
    metrics: dict
    report: dict
    train_index: pd.DatetimeIndex


def evaluate(
    features: pd.DataFrame,
    config: ModelConfig,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    *,
    seed: int = 42,
    cost_bps: float = 5,
) -> Evaluation:
    """Train strictly before start, then predict [start, end) without refitting."""
    training_start = start - pd.DateOffset(months=config.train_months)
    train = features.loc[(features.index >= training_start) & (features.index < start)]
    test_mask = features.index >= start
    if end is not None:
        test_mask &= features.index < end
    test = features.loc[test_mask]
    if len(train) < 2 or test.empty:
        raise ValueError(f"{config.name}: insufficient training or evaluation observations.")
    if train.actual_direction.nunique() != 2:
        raise ValueError(f"{config.name}: training needs both down and up/flat labels.")
    model = build_model(config.classifier, seed)
    model.fit(train[FEATURES], train.actual_direction)
    positions = pd.Series(model.predict(test[FEATURES]), index=test.index, name="position")
    predictions = backtest(test.actual_return, positions, cost_bps)
    predictions["actual_direction"] = test.actual_direction
    metrics, report = summarize(predictions, test.actual_direction)
    return Evaluation(model, predictions, metrics, report, train.index.copy())
