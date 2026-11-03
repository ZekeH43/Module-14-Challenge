"""Regression tests for scaling, alignment, model identity, report labels, and costs."""

import numpy as np
import pandas as pd
import pytest

from trading_analysis.data import FEATURES, build_features
from trading_analysis.evaluation import ModelConfig, backtest, build_model, evaluate, summarize


@pytest.mark.parametrize("classifier", ["svc", "adaboost"])
def test_training_only_scaler_and_correct_model_predictions(close, classifier):
    frame = build_features(close, 4, 20)
    start, end = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")
    config = ModelConfig("example", classifier=classifier, slow=20)
    result = evaluate(frame, config, start, end)
    train = frame.loc[result.train_index]
    test = frame.loc[(frame.index >= start) & (frame.index < end)]
    np.testing.assert_allclose(result.model.named_steps["scaler"].mean_, train[FEATURES].mean())
    assert result.train_index.min() >= start - pd.DateOffset(months=3)
    assert result.train_index.max() < start <= result.predictions.index.min()
    assert result.predictions.index.max() < end
    np.testing.assert_array_equal(result.predictions.position, result.model.predict(test[FEATURES]))
    np.testing.assert_allclose(
        result.predictions.gross_return, result.predictions.position * test.actual_return
    )
    assert not result.predictions.isna().any().any()


def test_costs_charge_entry_reversals_and_final_exit_without_double_shifting():
    dates = pd.date_range("2020-01-01", periods=4)
    actual = pd.Series([0.1, -0.1, 0.2, 0.1], index=dates)
    positions = pd.Series([1, -1, -1, 1], index=dates)
    original = actual.copy()
    result = backtest(actual, positions, cost_bps=10)
    np.testing.assert_array_equal(result.turnover, [1, 2, 0, 3])
    np.testing.assert_allclose(result.gross_return, [0.1, 0.1, -0.2, 0.1])
    expected = np.array([0.099, 0.098, -0.2, 0.097])
    np.testing.assert_allclose(result.net_return, expected)
    np.testing.assert_allclose(result.net_equity, np.cumprod(1 + expected))
    pd.testing.assert_series_equal(original, actual)


def test_single_bar_pays_entry_and_exit():
    dates = pd.date_range("2020-01-01", periods=1)
    result = backtest(pd.Series([0.1], index=dates), pd.Series([1], index=dates), 10)
    assert result.turnover.iloc[0] == 2
    assert result.net_return.iloc[0] == pytest.approx(0.098)


def test_report_uses_true_labels_first_and_drawdown_includes_initial_capital():
    dates = pd.date_range("2020-01-01", periods=4)
    actual = pd.Series([-0.1, -0.05, -0.02, 0.05], index=dates)
    labels = pd.Series([-1, -1, -1, 1], index=dates)
    result = backtest(actual, pd.Series([1, 1, -1, 1], index=dates), 0)
    metrics, report = summarize(result, labels)
    assert report["down"]["support"] == 3
    assert report["up_or_flat"]["support"] == 1
    assert report["down"]["recall"] == pytest.approx(1 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx((1 / 3 + 1) / 2)
    assert metrics["max_drawdown"] == pytest.approx(-0.145)


@pytest.mark.parametrize("cost", [-1, 101, float("nan"), float("inf"), "bad"])
def test_invalid_costs_are_rejected(cost):
    index = pd.date_range("2020-01-01", periods=2)
    with pytest.raises(ValueError):
        backtest(pd.Series([0.1, 0.1], index=index), pd.Series([1, 1], index=index), cost)


def test_different_indexes_cannot_silently_create_nan_returns():
    dates = pd.date_range("2020-01-01", periods=2)
    with pytest.raises(ValueError, match="same nonempty timestamp index"):
        backtest(pd.Series([0.1, -0.1], index=dates), pd.Series([1, -1]))


@pytest.mark.parametrize(
    "returns,positions",
    [
        ([np.nan, 0.1], [1, 1]),
        ([-1, 0.1], [1, 1]),
        ([0.1, 0.1], [0, 1]),
        ([0.1, 0.1], [np.inf, 1]),
        ([1.1, 0.1], [-1, 1]),
    ],
)
def test_invalid_or_ruined_paths_are_rejected(returns, positions):
    dates = pd.date_range("2020-01-01", periods=2)
    with pytest.raises(ValueError):
        backtest(pd.Series(returns, index=dates), pd.Series(positions, index=dates))


@pytest.mark.parametrize(
    "params",
    [
        {"fast": 0},
        {"fast": 100},
        {"train_months": 0},
        {"train_months": True},
        {"classifier": "unknown"},
    ],
)
def test_invalid_model_settings(params):
    with pytest.raises(ValueError):
        ModelConfig("example", **params)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 2.5])
def test_invalid_seeds(seed):
    with pytest.raises(ValueError):
        build_model("adaboost", seed)


def test_single_class_training_is_explained(close):
    features = build_features(close, 4, 20)
    features["actual_direction"] = 1
    with pytest.raises(ValueError, match="both down and up/flat"):
        evaluate(features, ModelConfig("example", slow=20), pd.Timestamp("2020-01-01"))
