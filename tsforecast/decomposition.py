"""Classical additive decomposition: y = trend + seasonal + residual.

Trend via a centred moving average of order m, seasonal as the average
detrended value per season, residual as the remainder.
"""

import numpy as np


def additive_decompose(y, m=12):
    """Decompose a 1-D series into trend, seasonal, residual components.

    Returns a dict with keys 'trend', 'seasonal', 'resid', 'm'.
    Edges of trend/resid are NaN where the centred window is incomplete;
    the seasonal pattern is tiled across the whole series.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) < 2 * m:
        raise ValueError(f"need at least 2*m={2*m} observations to decompose")

    # centred moving average (works for even m via the 0.5/1/0.5 trick)
    if m % 2 == 0:
        w = np.r_[0.5, np.ones(m - 1), 0.5] / m
    else:
        w = np.ones(m) / m
    trend = np.convolve(y, w, mode="same")
    edge = m // 2
    trend[:edge] = np.nan
    trend[-edge:] = np.nan

    detrended = y - trend
    seasonal_pattern = np.array(
        [np.nanmean(detrended[i::m]) for i in range(m)]
    )
    seasonal_pattern -= np.nanmean(seasonal_pattern)  # centre on zero
    seasonal = np.tile(seasonal_pattern, len(y) // m + 1)[: len(y)]
    resid = y - trend - seasonal
    return {"trend": trend, "seasonal": seasonal, "resid": resid, "m": m}
