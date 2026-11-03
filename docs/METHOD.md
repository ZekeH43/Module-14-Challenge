# Evaluation method and assumptions

This implementation is an offline historical classification exercise. The test dates are already public historical data; separating them in code prevents specific implementation leaks but does not make the experiment a prospective or untouched live evaluation.

## Feature and target timing

For observed closing prices `P[t]`:

```text
return[t] = P[t] / P[t-1] - 1
direction[t] = +1 if return[t] >= 0, otherwise -1
fast[t] = mean of the most recent fast_window closes ending at t-1
slow[t] = mean of the most recent slow_window closes ending at t-1
```

The rolling averages are shifted by one observed row. Row `t` never uses `P[t]` or later prices as a feature. The features can use earlier holdout closes as time moves forward, since those observations are then in the past; fitted model parameters are not updated during that holdout.

Prediction at the previous close and execution at that exact close assume zero latency. There is no independently observed executable quote. This optimistic convention is explicit and particularly questionable across missing-data gaps. Positions already align to the return they predict, so shifting them again would change the strategy.

## Training, validation, and holdout

The default cutoffs are January 1, 2019 and January 1, 2020. Training windows include observations at or after `evaluation_start - train_months`, and strictly before `evaluation_start`. Validation includes timestamps at or after its start and strictly before the holdout start. The holdout includes subsequent available observations.

The four configurations in `experiment.CANDIDATES` are declared before selection. All feature sets are aligned to a common index so each candidate is evaluated on identical timestamps. The pipeline fits `StandardScaler` and the classifier only on the relevant training rows; prediction applies that fitted scaler automatically.

Select the largest validation balanced accuracy, breaking ties by candidate order. Then freeze the configuration and refit it on the specified lookback before the holdout. The original SVC and AdaBoost baselines are also refit at that boundary. There is no shuffled split, random cross-validation, or automatic search over holdout scores.

If a baseline is selected, its baseline and selected rows intentionally match. Looking at all comparison results is descriptive evaluation, not permission to pick a new winner using the holdout.

## Classifiers and reproducibility

SVC uses an RBF kernel with `C=1.0` and `gamma="scale"`. AdaBoost uses up to 50 default decision stumps, learning rate 1.0, and seed 42 by default. Both are wrapped in a pipeline to keep preprocessing consistent. Scaling is important for SVC; keeping it in the AdaBoost pipeline provides a common interface.

The dependencies are pinned and run metadata records versions. Repeated runs in the same numerical environment reproduce the tables. Exact floating-point results may differ across environments or dependency changes. The seed controls AdaBoost's randomness; it does not certify model quality, and SVC's configured deterministic path does not use it.

## Backtest arithmetic

Each prediction becomes a `+1` long or `-1` short exposure over the matching return interval. The portfolio starts in cash and liquidates at the final close:

```text
gross_return[t] = position[t] * return[t]
turnover[0] = abs(position[0])
turnover[t] = abs(position[t] - position[t-1])  for t > 0
turnover[last] += abs(position[last])
cost[t] = turnover[t] * cost_bps / 10,000
net_return[t] = gross_return[t] - cost[t]
net_equity[t] = product of (1 + net_return) through t
```

Entering costs one turnover unit; reversing long/short costs two; the final exit costs one. A single-bar run pays both entry and exit. The 5-basis-point default is an illustrative proportional cost, not a quote from a broker. Costs are modeled as a simple deduction from interval returns. Constant exposure implies rebalancing, but the approximation does not separately charge notional adjustments within an unchanged position.

The always-long benchmark uses the same dates, entry/exit charges, and compounding. Returns at or below minus 100% stop the simple strategy calculation rather than continuing through insolvency.

## Metrics

| Metric | Interpretation |
| --- | --- |
| Accuracy | Fraction of direction labels classified correctly |
| Balanced accuracy | Mean recall of the observed classes; 0.5 for a constant prediction when both classes occur |
| Gross/net total return | Final compounded value minus the starting value of 1 |
| Maximum drawdown | Most negative `equity / prior_peak - 1`, including the initial value of 1 in the peak |
| Turnover units | Total entry, reversal, and exit exposure changes |
| Long fraction | Fraction of predictions equal to `+1` |

Classification reports receive `y_true` first and `y_pred` second; undefined precision/recall terms are reported as zero. A zero price change belongs to the up/flat class. The source timestamps are irregular, so the project does not invent a daily frequency to calculate annualized Sharpe ratios or returns.

## Limits and next experiments

The input's identity, corporate-action adjustments, and gaps are unverified. The three-month holdout training window contains only 50 observations. The setup omits borrowing costs, taxes, dividends, spread, latency, liquidity, margin rules, and realistic execution. A single validation period and single holdout cannot establish a stable edge; classification performance and compounded returns answer different questions.

Useful extensions include documented adjusted data, several chronological walk-forward folds, sensitivity to explicit costs, and better lagged features. Reserve a new evaluation period before selecting additional settings. Treat repeated changes based on the current holdout as exploration, and do not present the historical chart as evidence of expected profits.

## Primary library references

- [scikit-learn: common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), including consistent preprocessing and training-only fitting.
- [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html), including argument order and class support.
- [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html), including estimator settings and random state.
