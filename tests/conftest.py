"""Small deterministic historical-price fixtures with both direction classes."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def close():
    dates = pd.date_range("2018-01-01", "2020-08-01", freq="B")
    rng = np.random.default_rng(7)
    moves = rng.normal(0.0003, 0.015, len(dates))
    return pd.Series(100 * np.exp(np.cumsum(moves)), index=dates, name="close")


@pytest.fixture
def csv_path(tmp_path, close):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {"date": close.index.strftime("%m/%d/%y %H:%M"), "close": close.to_numpy()}
    ).to_csv(path, index=False)
    return path
