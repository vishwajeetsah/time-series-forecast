"""CLI demo: generate a synthetic series, backtest every model, print a
comparison table, and save forecast + decomposition plots.

Usage:
    python -m tsforecast.cli [--n 240] [--m 12] [--horizon 12] [--out DIR]
"""

import argparse
import os

from .data import make_series
from .models import SES, Drift, Holt, HoltWintersAdditive, Naive, SeasonalNaive
from .backtest import compare_models, rolling_origin_backtest
from .decomposition import additive_decompose


def main():
    ap = argparse.ArgumentParser(description="tsforecast demo")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()

    y = make_series(n=args.n, m=args.m, seed=7)
    models = {
        "Naive": Naive,
        "SeasonalNaive": lambda: SeasonalNaive(m=args.m),
        "Drift": Drift,
        "SES": SES,
        "Holt": Holt,
        "HoltWinters": lambda: HoltWintersAdditive(m=args.m),
    }
    rows = compare_models(models, y, initial_train=args.n - 48, horizon=args.horizon, m=args.m)

    print(f"{'model':<14}{'folds':>6}{'mean_mae':>11}{'mean_rmse':>12}{'mean_mape':>11}{'mean_mase':>11}")
    print("-" * 65)
    for r in rows:
        print(
            f"{r['model']:<14}{r['n_folds']:>6}{r['mean_mae']:>11.2f}"
            f"{r['mean_rmse']:>12.2f}{r['mean_mape']:>11.2f}{r['mean_mase']:>11.3f}"
        )
    print("\nBest model (lowest mean MASE):", rows[0]["model"])

    os.makedirs(args.out, exist_ok=True)
    try:
        from .plot import plot_forecast, plot_decomposition

        best = {"Naive": Naive, "SeasonalNaive": lambda: SeasonalNaive(m=args.m),
                "Drift": Drift, "SES": SES, "Holt": Holt,
                "HoltWinters": lambda: HoltWintersAdditive(m=args.m)}[rows[0]["model"]]()
        split = args.n - args.horizon
        fc = best.fit(y[:split]).forecast(args.horizon)
        plot_forecast(y, fc, train_end=split, path=os.path.join(args.out, "forecast.png"),
                      title=f"Best model: {rows[0]['model']}")
        plot_decomposition(y, additive_decompose(y, m=args.m),
                           path=os.path.join(args.out, "decomposition.png"))
        print("Plots written to", args.out + "/")
    except ImportError:
        print("matplotlib not installed; skipping plots.")


if __name__ == "__main__":
    main()
