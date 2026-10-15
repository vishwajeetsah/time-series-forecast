"""Forecast accuracy metrics."""

import numpy as np


def _as_arrays(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return yt, yp


def mae(y_true, y_pred):
    """Mean absolute error."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def mse(y_true, y_pred):
    """Mean squared error."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred):
    """Mean absolute percentage error (in %). Raises if any y_true is 0."""
    yt, yp = _as_arrays(y_true, y_pred)
    if np.any(yt == 0):
        raise ValueError("mape is undefined when y_true contains zeros; use smape")
    return float(np.mean(np.abs((yt - yp) / yt)) * 100)


def smape(y_true, y_pred):
    """Symmetric MAPE (in %, 0-200 scale)."""
    yt, yp = _as_arrays(y_true, y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    mask = denom != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / denom[mask]) * 100)


def mase(y_train, y_true, y_pred, m=1):
    """Mean absolute scaled error.

    Scales MAE by the in-sample MAE of the seasonal-naive (m=1: naive)
    one-step forecast, so MASE < 1 beats the naive benchmark.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    train = np.asarray(y_train, dtype=float)
    if len(train) <= m:
        raise ValueError("y_train must be longer than m")
    scale = np.mean(np.abs(train[m:] - train[:-m]))
    if scale == 0:
        raise ValueError("cannot compute mase: naive benchmark has zero error")
    return float(np.mean(np.abs(yt - yp)) / scale)
