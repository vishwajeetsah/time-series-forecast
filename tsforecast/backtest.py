"""Rolling-origin (time-series) backtesting.

Splits the series into successive train/test windows with an expanding
training set: the model is refit on all data up to each origin and asked
to forecast `horizon` steps ahead. Metrics are averaged across origins.
"""

import numpy as np

from . import metrics as _m


def rolling_origin_backtest(model_fn, y, initial_train, horizon, step=1, m=12):
    """Backtest a model across rolling origins.

    model_fn: zero-argument-agnostic callable -> unfitted model instance
              (called fresh for every origin, so state never leaks).
    y: full series (1-D array-like).
    initial_train: number of observations in the first training window.
    horizon: forecast horizon h at each origin.
    step: how far the origin advances between folds.
    m: seasonal period used for MASE.

    Returns a dict with per-fold errors and mean MAE/RMSE/MAPE/SMAPE/MASE.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if initial_train < 2 or initial_train + horizon > n:
        raise ValueError("initial_train + horizon must be <= len(y)")
    folds = []
    for end in range(initial_train, n - horizon + 1, step):
        train, test = y[:end], y[end : end + horizon]
        model = model_fn().fit(train)
        pred = model.forecast(horizon)
        folds.append(
            {
                "origin": end,
                "mae": _m.mae(test, pred),
                "rmse": _m.rmse(test, pred),
                "mape": _m.mape(test, pred),
                "smape": _m.smape(test, pred),
                "mase": _m.mase(train, test, pred, m=m),
            }
        )
    summary = {
        "n_folds": len(folds),
        "horizon": horizon,
        "mean_mae": float(np.mean([f["mae"] for f in folds])),
        "mean_rmse": float(np.mean([f["rmse"] for f in folds])),
        "mean_mape": float(np.mean([f["mape"] for f in folds])),
        "mean_smape": float(np.mean([f["smape"] for f in folds])),
        "mean_mase": float(np.mean([f["mase"] for f in folds])),
    }
    return {"folds": folds, "summary": summary}


def compare_models(model_fns, y, initial_train, horizon, step=1, m=12):
    """Backtest several named models and return a ranked summary table.

    model_fns: dict {name: model_fn}. Ranked by mean MASE (lower is better).
    """
    rows = []
    for name, fn in model_fns.items():
        res = rolling_origin_backtest(fn, y, initial_train, horizon, step, m)
        s = res["summary"]
        rows.append(
            {
                "model": name,
                "n_folds": s["n_folds"],
                "mean_mae": s["mean_mae"],
                "mean_rmse": s["mean_rmse"],
                "mean_mape": s["mean_mape"],
                "mean_mase": s["mean_mase"],
            }
        )
    rows.sort(key=lambda r: r["mean_mase"])
    return rows
