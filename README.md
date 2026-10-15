# time-series-forecast

A small, dependency-light Python toolkit for time-series forecasting — built from scratch with NumPy, so every algorithm is readable and every number is reproducible. Useful for learning forecasting properly, and for small real jobs (sales, traffic, demand) where you want transparent baselines before reaching for heavy models.

## What's inside

**Models** (`tsforecast/models.py`) — all implemented from scratch, no statsmodels:
- `Naive` — last observation carried forward
- `SeasonalNaive(m)` — repeats the value from one season ago
- `Drift` — random walk with drift (linear extrapolation of average change)
- `SES` — simple exponential smoothing (grid-searched α)
- `Holt` — Holt's linear trend method (grid-searched α, β)
- `HoltWintersAdditive(m)` — additive Holt–Winters with level, trend, and seasonal components (grid-searched α, β, γ)

Smoothing parameters are chosen by grid search minimising the sum of squared one-step-ahead errors — no optimiser dependency needed.

**Evaluation** (`tsforecast/metrics.py`): MAE, MSE, RMSE, MAPE, sMAPE, and MASE (scaled so MASE < 1 beats the naive benchmark).

**Backtesting** (`tsforecast/backtest.py`): rolling-origin (expanding-window) cross-validation — refits the model at every origin, averages metrics across folds, and `compare_models()` ranks models by mean MASE.

**Decomposition** (`tsforecast/decomposition.py`): classical additive decomposition (centred moving-average trend + averaged seasonal pattern + residual).

**Data** (`tsforecast/data.py`): seeded synthetic generators — trend+seasonality+noise, AR(1), and level-shift series.

**Plots** (`tsforecast/plot.py`, needs matplotlib): forecast and decomposition charts.

## Quickstart

```bash
pip install -r requirements.txt

# backtest every model on a synthetic monthly series and save plots
python -m tsforecast.cli --n 240 --m 12 --horizon 12 --out demo_out
```

```python
import numpy as np
from tsforecast import HoltWintersAdditive, SeasonalNaive, mape
from tsforecast import rolling_origin_backtest, make_series

y = make_series(n=240, m=12, seed=7)          # monthly series: trend + seasonality
res = rolling_origin_backtest(lambda: HoltWintersAdditive(m=12),
                              y, initial_train=192, horizon=12, m=12)
print(res["summary"])
# {'n_folds': 4, 'horizon': 12, 'mean_mae': ..., 'mean_rmse': ...,
#  'mean_mape': ..., 'mean_smape': ..., 'mean_mase': ...}

split = len(y) - 12
fc = HoltWintersAdditive(m=12).fit(y[:split]).forecast(12)
print("12-month MAPE:", mape(y[split:], fc))
```

Compare models head-to-head:

```python
from tsforecast import compare_models, Naive, SES, Holt
rows = compare_models({"Naive": Naive, "SES": SES, "Holt": Holt},
                      y, initial_train=192, horizon=12, m=12)
for r in rows:
    print(r["model"], round(r["mean_mase"], 3))
```

## How it works

1. **Fit.** Each model's `fit(y)` learns its state (level, trend, seasonal pattern). Exponential-smoothing parameters come from a coarse grid search over the training SSE — transparent, deterministic, dependency-free.
2. **Forecast.** `forecast(h)` projects the state `h` steps ahead (Holt–Winters: `level + h·trend + seasonal[(t+h) mod m]`).
3. **Evaluate honestly.** `rolling_origin_backtest` never peeks at the future: it refits on data up to each origin and averages error metrics across origins. MASE scales errors against the naive benchmark, so a MASE below 1 means the model genuinely beats "repeat last value".

## Project structure

```
tsforecast/
  __init__.py        public API
  models.py          Naive, SeasonalNaive, Drift, SES, Holt, HoltWintersAdditive
  metrics.py         mae, mse, rmse, mape, smape, mase
  backtest.py        rolling_origin_backtest, compare_models
  decomposition.py   classical additive decomposition
  data.py            synthetic series generators
  plot.py            matplotlib forecast/decomposition charts (optional)
  cli.py             demo CLI: backtests all models, prints table, saves plots
tests/
  test_tsforecast.py 20 tests covering metrics, models, decomposition, backtesting
```

## Limitations (honest)

- Only the **additive** Holt–Winters form is implemented; multiplicative seasonality (series whose seasonal swings grow with the level) is not modelled — log-transform the series first in that case.
- Grid search is coarse (α in 0.05 steps, β/γ in wider steps), so parameters are good but not optimal; fine grids get slow in backtests.
- No prediction intervals — only point forecasts.
- No exogenous regressors, no ARIMA/SARIMA, no automatic seasonality detection (you set `m`).
- Classical decomposition needs at least `2·m` observations and is descriptive, not predictive.

## What I Learned

Building every smoother from scratch forced me to learn the recursions instead of hiding behind a library: the one-step-ahead error is what actually drives parameter choice, and a bad initialisation (e.g. an uncentred seasonal pattern) silently biases every forecast. The biggest lesson was methodological, not mathematical — writing the rolling-origin backtest showed how easy it is to leak the future (fit once, evaluate everywhere) versus the honest alternative (refit at every origin). MASE turned out to be the metric that makes model comparison fair across scales: a "5% MAPE" means nothing until you know whether naive already gets 4%. If I extend this, prediction intervals and a multiplicative form are next.

## License

MIT — see [LICENSE](LICENSE).
