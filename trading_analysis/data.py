"""Read historical bars and build features using only previously observed closes."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "Starter_Code" / "Resources" / "emerging_markets_ohlcv.csv"
FEATURES = ["sma_fast", "sma_slow"]


def validate_close(close: pd.Series) -> pd.Series:
    """Copy a positive chronological price series; reject missing or duplicate bars."""
    if not isinstance(close, pd.Series) or len(close) < 3:
        raise ValueError("Provide a price Series with at least three observations.")
    if not isinstance(close.index, pd.DatetimeIndex) or close.index.hasnans:
        raise ValueError("Prices need a valid DatetimeIndex.")
    if close.index.tz is not None:
        raise ValueError("Use timezone-naive timestamps matching the source CSV convention.")
    if not close.index.is_unique or not close.index.is_monotonic_increasing:
        raise ValueError("Price timestamps must be unique and sorted oldest first.")
    try:
        numeric = pd.to_numeric(close, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Closing prices must be numeric.") from exc
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError("Closing prices must be finite, positive, and present on every row.")
    return numeric.rename("close").copy()


def load_close(path: Path = DEFAULT_DATA) -> pd.Series:
    """Read the original month/day/two-digit-year timestamps without guessing dates.

    Only date and close are used. The irregular intraday observations are kept
    as supplied: no resampling, artificial calendar, or forward filling.
    """
    frame = pd.read_csv(path)
    if not {"date", "close"}.issubset(frame.columns):
        raise ValueError("The CSV must contain date and close columns.")
    try:
        dates = pd.to_datetime(frame["date"], format="%m/%d/%y %H:%M", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Dates must use month/day/two-digit-year hour:minute format.") from exc
    close = pd.Series(frame["close"].to_numpy(), index=pd.DatetimeIndex(dates))
    return validate_close(close.sort_index())


def build_features(close: pd.Series, fast: int = 4, slow: int = 100) -> pd.DataFrame:
    """Pair prior-bar SMAs with the current bar's return and direction label.

    Row t uses closes through t-1, never close[t] or later. Windows count
    observed bars, not calendar days. Zero returns belong to the +1 class.
    """
    close = validate_close(close)
    if any(isinstance(w, bool) or not isinstance(w, int) for w in (fast, slow)):
        raise ValueError("SMA windows must be integers.")
    if not 1 <= fast < slow < len(close):
        raise ValueError("Require 1 <= fast < slow < number of price observations.")
    frame = pd.DataFrame({"actual_return": close.pct_change(fill_method=None)})
    frame["sma_fast"] = close.rolling(fast).mean().shift(1)
    frame["sma_slow"] = close.rolling(slow).mean().shift(1)
    frame["actual_direction"] = np.where(frame.actual_return >= 0, 1, -1)
    frame = frame.iloc[slow:].copy()
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Features contain missing or nonfinite values; check source prices.")
    return frame
