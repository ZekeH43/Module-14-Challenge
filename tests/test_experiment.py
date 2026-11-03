"""Ensure model selection stays independent of future outcomes and comparisons align."""

import pandas as pd
import pytest

from trading_analysis.experiment import run_experiment


def test_holdout_price_changes_cannot_change_validation_or_selected_candidate(csv_path, tmp_path):
    first = run_experiment(csv_path)
    frame = pd.read_csv(csv_path)
    dates = pd.to_datetime(frame.date, format="%m/%d/%y %H:%M")
    frame.loc[dates >= "2020-01-01", "close"] *= 1.4
    changed_path = tmp_path / "changed.csv"
    frame.to_csv(changed_path, index=False)
    changed = run_experiment(changed_path)
    pd.testing.assert_frame_equal(first.validation, changed.validation)
    assert first.metadata["selected_candidate"] == changed.metadata["selected_candidate"]
    assert first.metadata["source"]["sha256"] != changed.metadata["source"]["sha256"]


def test_holdout_comparisons_share_dates_and_reproduce(csv_path):
    first = run_experiment(csv_path)
    repeated = run_experiment(csv_path)
    pd.testing.assert_frame_equal(first.validation, repeated.validation)
    pd.testing.assert_frame_equal(first.holdout, repeated.holdout)
    selected_name = first.metadata["selected_candidate"]
    assert (
        first.validation.loc[selected_name, "balanced_accuracy"]
        == first.validation.balanced_accuracy.max()
    )
    index = first.predictions["always_long"].index
    for name, prediction in first.predictions.items():
        assert prediction.index.equals(index)
        assert not prediction.isna().any().any()
        pd.testing.assert_frame_equal(prediction, repeated.predictions[name])
    for record in first.metadata["holdout_training"].values():
        assert pd.Timestamp(record["end"]) < pd.Timestamp(first.metadata["test_start"])


@pytest.mark.parametrize(
    "options",
    [
        {"validation_start": "bad"},
        {"test_start": "NaT"},
        {"validation_start": "2021-01-01"},
        {"validation_start": "1900-01-01", "test_start": "1901-01-01"},
        {"test_start": "2100-01-01"},
    ],
)
def test_invalid_or_empty_periods(csv_path, options):
    with pytest.raises(ValueError):
        run_experiment(csv_path, **options)
