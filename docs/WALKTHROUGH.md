# Learn to explain the project

Run the [README](../README.md) example, open the notebook, and trace one request through these files. Be able to explain the calculations yourself before using them as interview examples.

## Follow the code

1. **`__main__.py` parses options.** The command accepts an input file, evaluation dates, transaction-cost assumption, seed, and output directory. Invalid inputs return a readable error and nonzero exit status.
2. **`data.py` builds a chronological table.** `load_close` validates dates and prices. `build_features` calculates returns and shifts each moving average so it uses only earlier closes.
3. **`evaluation.py` fits one pipeline.** `evaluate` slices a past training window and a later evaluation window. `StandardScaler` learns only from training features, and the pipeline applies the same transformation during prediction.
4. **`experiment.py` makes the comparison fair.** It aligns candidate timestamps, selects using validation balanced accuracy, freezes the choice, and evaluates the later holdout and baselines.
5. **`evaluation.py` computes indexed outcomes.** Predictions keep the test timestamps. `backtest` multiplies each position by its matching return, charges modeled turnover, and compounds from 1. `summarize` calculates classification and drawdown metrics.
6. **`reporting.py` makes the experiment reviewable.** Saved tables show outcomes; metadata identifies the input hash and settings; the chart displays the same holdout for every comparison.

## Bugs worth understanding

| Original problem | Why it changes the result | Current safeguard |
| --- | --- | --- |
| Train SVC on scaled values, predict on raw values | The classifier receives features in a different numerical space | One pipeline owns fitting and prediction |
| Reverse classification-report arguments | Precision, recall, and support describe the wrong roles | True labels first, with a regression test for support counts |
| Assign predictions to a new integer-indexed DataFrame | Adding timestamp-indexed returns aligns unrelated indexes and creates missing values | Predictions are created with the exact test index; mismatches raise an error |
| Use SVM predictions in the AdaBoost backtest | The chart does not describe the alternative model | Each model's evaluation carries its own predictions and returns |
| Choose parameters after repeatedly looking at test outcomes | The test becomes part of model selection | Separate earlier validation and later holdout periods |

The feature-timing test changes the current and future prices and verifies that the current features stay unchanged. Another test changes holdout prices and verifies that validation results and the selected configuration stay unchanged. These are useful examples of tests that protect a modeling promise rather than just checking that a function returns something.

## Small calculations to try by hand

- With closes `10, 12, 14`, the next observation's lagged two-bar SMA is `13` and three-bar SMA is `12`.
- If the next close is `11`, its return is `11 / 14 - 1`; its direction is `-1`. That close must not appear in its own input features.
- A long-to-short reversal has turnover `abs(-1 - 1) = 2`. At 5 basis points per unit, its modeled cost is `0.001`, or 0.1%.
- If equity falls from its starting value of `1` to `0.9`, the drawdown is 10% even when that is the first scored observation.

## Interview practice

Explain these without reading the implementation:

- Why does pandas align by labels, and how did that break the old prediction table?
- Why fit a scaler on training data only? How does a pipeline help?
- What information is available when a position is chosen?
- Why are validation and holdout different, and when does a holdout become exploratory data?
- Why can high returns coexist with weak classification performance?
- What do irregular timestamps and the 50-row training window prevent you from claiming?

Describe the educational starter-code origin and the portions you have personally studied, modified, and tested. The evidence here is a reproducible software experiment, not a claim of independent financial research or a production bot.

## Manageable extensions

1. Add a lagged percentage-return feature and a test proving it cannot read future prices.
2. Implement several walk-forward evaluation folds, keeping parameter selection inside the earlier data of each fold.
3. Compare cost assumptions in separate output directories and explain which metrics change.

Make one change at a time, update its documentation, and keep the current example as a reproducible reference.
