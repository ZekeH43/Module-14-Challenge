# Historical input and its limits

The active input is `Starter_Code/Resources/emerging_markets_ohlcv.csv`. Its original bytes are unchanged. It contains **4,323 observations**, from **January 21, 2015 at 09:30** to **January 22, 2021 at 15:45**.

The repository does not establish a definitive instrument identifier, publisher, retrieval date, timezone, split/dividend adjustment policy, or redistribution license for this file. The filename is descriptive, not verified instrument provenance. The refactor does not invent those missing details or claim the values are current quotes.

## Columns and validation

| Column | Format | Treatment |
| --- | --- | --- |
| `date` | `month/day/two-digit-year hour:minute`, for example `1/21/15 9:30` | Parsed explicitly, sorted chronologically, duplicates/missing timestamps rejected |
| `close` | Numeric closing value | Must be finite and strictly positive |
| `open`, `high`, `low`, `volume` | Original source fields | Retained unchanged, not used or independently validated by the model |

Custom inputs use the same schema through `--data path/to/prices.csv`. At least three valid observations are required to load a series; the model needs enough additional observations to warm up its SMA features and populate both chronological evaluation periods. Training must contain both direction classes. Empty periods produce an error instead of an empty success report.

## Bars are not days

The observations are irregular: consecutive rows may be minutes, days, or weeks apart. The loader does not resample, forward-fill, or check an exchange calendar. Moving-average windows count observed rows. Returns span adjacent observed closes, regardless of elapsed time.

A particularly important gap runs from **December 3, 2019 at 15:30** to **January 2, 2020 at 10:00**. The first holdout return spans that interval. The default holdout model has only **50 training observations** in its three-month lookback, ending December 3. Sparse data makes both fitting and execution assumptions fragile.

The plotted initial value is at the preceding observed close, December 3, even though the first return is timestamped January 2. Do not interpret it as a continuous, verified trading record for every market session.

## Traceability and preservation

Each run records the source SHA-256, observation count, first/last timestamps, actual training windows, actual evaluation windows, and settings in `metadata.json`. A matching hash establishes matching file bytes, not market-data accuracy.

- The duplicate active-CSV checkpoint and duplicate notebook checkpoint were removed after confirming they were byte-identical to their main files.
- `Starter_Code/Resources/archive/ohlcv.csv` preserves a **different 12,069-row dataset** that had been stored only as `ohlcv-checkpoint.csv`. It is not used by the default experiment, and its provenance is also unverified.
- The [original assignment](ORIGINAL_ASSIGNMENT.md) remains available for educational context.

Before extending this into a more realistic analysis, obtain documented, appropriately licensed price data with stable instrument identifiers and a consistent adjustment policy. Keep that new dataset separate and record how its timestamps and missing sessions are handled.
