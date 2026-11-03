"""Protect chronological parsing, strict validation, and prior-observation features."""

import numpy as np
import pandas as pd
import pytest

from trading_analysis.data import FEATURES, build_features, load_close, validate_close


def test_loader_sorts_dates_without_resampling(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "date,close,volume\n1/3/20 15:00,103,N/A\n1/1/20 9:30,100,N/A\n1/1/20 9:45,101,N/A\n"
    )
    result = load_close(path)
    assert result.tolist() == [100, 101, 103]
    assert len(result) == 3 and result.index.is_monotonic_increasing


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, "invalid"])
def test_invalid_prices_are_rejected(close, value):
    bad = close.astype(object)
    bad.iloc[3] = value
    with pytest.raises(ValueError):
        validate_close(bad)


@pytest.mark.parametrize(
    "kind", ["duplicate", "missing", "not_dates", "timezone", "reversed", "short"]
)
def test_invalid_price_indexes_are_rejected(close, kind):
    if kind == "duplicate":
        close.index = pd.DatetimeIndex([close.index[0]] * len(close))
    elif kind == "missing":
        close.index = pd.DatetimeIndex([pd.NaT, *close.index[1:]])
    elif kind == "not_dates":
        close.index = range(len(close))
    elif kind == "timezone":
        close.index = close.index.tz_localize("UTC")
    elif kind == "reversed":
        close = close.iloc[::-1]
    else:
        close = close.iloc[:2]
    with pytest.raises(ValueError):
        validate_close(close)


@pytest.mark.parametrize(
    "content",
    [
        "date,price\n1/1/20 9:30,100\n",
        "date,close\nnot-a-date,100\n",
        "date,close\n1/1/20 9:30,100\n1/1/20 9:30,101\n1/2/20 9:30,102\n",
    ],
)
def test_csv_schema_and_dates_are_validated(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_close(path)


def test_features_count_bars_and_use_only_previous_closes():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-10", "2020-02-01", "2020-02-10"])
    prices = pd.Series([10.0, 12, 14, 11, 11], index=dates)
    frame = build_features(prices, fast=2, slow=3)
    assert frame.index.equals(dates[3:])
    assert frame.iloc[0].sma_fast == 13
    assert frame.iloc[0].sma_slow == 12
    assert frame.iloc[0].actual_return == pytest.approx(11 / 14 - 1)
    assert frame.actual_direction.tolist() == [-1, 1]


def test_changing_current_and_future_prices_cannot_change_current_features(close):
    original = close.copy()
    reference = build_features(close, fast=4, slow=20)
    cutoff = close.index[100]
    changed = close.copy()
    changed.loc[cutoff:] *= 3
    altered = build_features(changed, fast=4, slow=20)
    pd.testing.assert_frame_equal(reference.loc[:cutoff, FEATURES], altered.loc[:cutoff, FEATURES])
    assert reference.loc[cutoff, "actual_return"] != altered.loc[cutoff, "actual_return"]
    pd.testing.assert_series_equal(original, close)


@pytest.mark.parametrize(
    "fast,slow", [(0, 10), (10, 10), (20, 10), (True, 10), (2.5, 10), (1, 10000)]
)
def test_invalid_windows(close, fast, slow):
    with pytest.raises(ValueError):
        build_features(close, fast, slow)


def test_original_snapshot_is_readable():
    prices = load_close()
    assert len(prices) == 4323
    assert prices.index[0] == pd.Timestamp("2015-01-21 09:30")
    assert prices.index[-1] == pd.Timestamp("2021-01-22 15:45")
