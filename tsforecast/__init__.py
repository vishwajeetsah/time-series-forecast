"""tsforecast: a small, dependency-light time-series forecasting toolkit.

From-scratch exponential smoothing models, forecast accuracy metrics,
rolling-origin backtesting, and classical additive decomposition.
Built for learning and for small, real forecasting jobs.
"""

from .models import Drift, Holt, HoltWintersAdditive, Naive, SeasonalNaive, SES
from .metrics import mae, mape, mase, mse, rmse, smape
from .backtest import rolling_origin_backtest, compare_models
from .decomposition import additive_decompose
from .data import make_series, make_ar1, make_step_series

__all__ = [
    "Naive",
    "SeasonalNaive",
    "Drift",
    "SES",
    "Holt",
    "HoltWintersAdditive",
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "mase",
    "rolling_origin_backtest",
    "compare_models",
    "additive_decompose",
    "make_series",
    "make_ar1",
    "make_step_series",
]

__version__ = "0.1.0"
