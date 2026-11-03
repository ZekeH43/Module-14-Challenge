"""Reproduce the offline experiment with `python -m trading_analysis`."""

import argparse
from pathlib import Path

from trading_analysis.data import DEFAULT_DATA
from trading_analysis.experiment import run_experiment
from trading_analysis.reporting import save_results


def main(argv=None):
    """Expose documented dates, costs, seed, and output paths without contacting a broker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--validation-start", default="2019-01-01")
    parser.add_argument("--test-start", default="2020-01-01")
    parser.add_argument("--cost-bps", type=float, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_experiment(
            args.data,
            validation_start=args.validation_start,
            test_start=args.test_start,
            seed=args.seed,
            cost_bps=args.cost_bps,
        )
        save_results(result, args.output, plots=not args.no_plots)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("Selected using validation only:", result.metadata["selected_candidate"])
    print("Historical holdout (fractions, not percentages):")
    print(result.holdout.round(4).to_string())
    print("\nThis is an offline historical experiment, not a live trading bot.")
    print(f"Saved results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
