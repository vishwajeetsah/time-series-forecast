"""Forecasting models, implemented from scratch with numpy.

Each model exposes:
    fit(y)        -> self (learns smoothing/state parameters)
    forecast(h)   -> np.ndarray of length h, the h-step-ahead forecast
    fitted()      -> in-sample one-step-ahead fitted values (np.ndarray)

Smoothing parameters are chosen by a coarse grid search minimising the
sum of squared one-step-ahead errors on the training data. No optimiser
dependency is needed; the grids are small and each fit is O(n).
"""

import itertools

import numpy as np

_ALPHA_GRID = np.round(np.arange(0.05, 1.0, 0.1), 2)
_BETA_GRID = np.round(np.arange(0.05, 1.0, 0.15), 2)
_GAMMA_GRID = np.round(np.arange(0.05, 1.0, 0.15), 2)


class _Base:
    def fit(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or len(y) < 2:
            raise ValueError("y must be a 1-D array with at least 2 observations")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values")
        self.y_ = y
        self._fit(y)
        return self

    def forecast(self, h):
        if not hasattr(self, "y_"):
            raise RuntimeError("call fit() before forecast()")
        if h < 1:
            raise ValueError("h must be >= 1")
        return self._forecast(int(h))


class Naive(_Base):
    """Last observation carried forward."""

    def _fit(self, y):
        pass

    def _forecast(self, h):
        return np.full(h, self.y_[-1])

    def fitted(self):
        return np.r_[self.y_[0], self.y_[:-1]]


class SeasonalNaive(_Base):
    """Repeat the observation from one season ago."""

    def __init__(self, m=12):
        if m < 2:
            raise ValueError("m must be >= 2")
        self.m = m

    def _fit(self, y):
        if len(y) < self.m:
            raise ValueError("need at least m observations for SeasonalNaive")

    def _forecast(self, h):
        m = self.m
        tail = self.y_[-m:]
        idx = np.arange(h) % m
        return tail[idx]

    def fitted(self):
        m = self.m
        out = np.full_like(self.y_, np.nan)
        out[m:] = self.y_[:-m]
        return out


class Drift(_Base):
    """Linear extrapolation of the average change (random walk with drift)."""

    def _fit(self, y):
        n = len(y)
        self.slope_ = (y[-1] - y[0]) / (n - 1)

    def _forecast(self, h):
        return self.y_[-1] + self.slope_ * np.arange(1, h + 1)

    def fitted(self):
        n = len(self.y_)
        slope = (self.y_[-1] - self.y_[0]) / (n - 1)
        return self.y_[0] + slope * np.arange(n)


class SES(_Base):
    """Simple exponential smoothing: level_t = a*y_t + (1-a)*level_{t-1}.

    Flat forecast = last level. Parameter alpha is grid-searched.
    """

    def _fit(self, y):
        best_sse, best = np.inf, None
        for a in _ALPHA_GRID:
            level, sse = _ses_states(y, a)
            if sse < best_sse:
                best_sse, best = sse, (a, level)
        self.alpha_, self.level_ = best

    def _forecast(self, h):
        return np.full(h, self.level_)

    def fitted(self):
        # one-step-ahead fitted values from the recursive level
        y = self.y_
        a = self.alpha_
        level = y[0]
        out = np.empty_like(y)
        out[0] = np.nan
        for t in range(1, len(y)):
            level = a * y[t - 1] + (1 - a) * level
            out[t] = level
        return out


def _ses_states(y, alpha):
    """Return (final level, SSE of one-step-ahead errors) for SES(alpha)."""
    level = y[0]
    sse = 0.0
    for t in range(1, len(y)):
        err = y[t] - level
        sse += err * err
        level = alpha * y[t] + (1 - alpha) * level
    return level, sse


class Holt(_Base):
    """Holt's linear trend method: level + trend, both exponentially smoothed."""

    def _fit(self, y):
        best_sse, best = np.inf, None
        for a, b in itertools.product(_ALPHA_GRID, _BETA_GRID):
            (level, trend), sse = _holt_states(y, a, b)
            if sse < best_sse:
                best_sse, best = sse, (a, b, level, trend)
        self.alpha_, self.beta_, self.level_, self.trend_ = best

    def _forecast(self, h):
        return self.level_ + self.trend_ * np.arange(1, h + 1)

    def fitted(self):
        y = self.y_
        a, b = self.alpha_, self.beta_
        level, trend = y[0], y[1] - y[0]
        out = np.empty_like(y)
        out[0] = np.nan
        for t in range(1, len(y)):
            out[t] = level + trend
            new_level = a * y[t] + (1 - a) * (level + trend)
            trend = b * (new_level - level) + (1 - b) * trend
            level = new_level
        return out


def _holt_states(y, alpha, beta):
    level, trend = y[0], y[1] - y[0]
    sse = 0.0
    for t in range(1, len(y)):
        err = y[t] - (level + trend)
        sse += err * err
        new_level = alpha * y[t] + (1 - alpha) * (level + trend)
        trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
    return (level, trend), sse


class HoltWintersAdditive(_Base):
    """Additive Holt-Winters: level + trend + seasonal component (period m).

    Forecast(h) = level + h*trend + seasonal[(t + h) mod m].
    """

    def __init__(self, m=12):
        if m < 2:
            raise ValueError("m must be >= 2")
        self.m = m

    def _fit(self, y):
        m = self.m
        if len(y) < 2 * m:
            raise ValueError(f"need at least 2*m={2*m} observations for Holt-Winters")
        best_sse, best = np.inf, None
        for a, b, g in itertools.product(_ALPHA_GRID, _BETA_GRID, _GAMMA_GRID):
            state, sse = _hw_states(y, m, a, b, g)
            if sse < best_sse:
                best_sse, best = sse, (a, b, g) + state
        self.alpha_, self.beta_, self.gamma_ = best[:3]
        self.level_, self.trend_, self.seasonal_ = best[3:]

    def _forecast(self, h):
        m = self.m
        steps = np.arange(1, h + 1)
        seas_idx = (len(self.y_) - m + steps - 1) % m
        return self.level_ + self.trend_ * steps + self.seasonal_[seas_idx]

    def fitted(self):
        y, m = self.y_, self.m
        a, b, g = self.alpha_, self.beta_, self.gamma_
        level, trend, seas = _hw_init(y, m)
        out = np.empty_like(y)
        out[:m] = np.nan
        for t in range(m, len(y)):
            out[t] = level + trend + seas[t % m]
            new_level = a * (y[t] - seas[t % m]) + (1 - a) * (level + trend)
            new_trend = b * (new_level - level) + (1 - b) * trend
            seas[t % m] = g * (y[t] - new_level) + (1 - g) * seas[t % m]
            level, trend = new_level, new_trend
        return out


def _hw_init(y, m):
    level = y[:m].mean()
    trend = (y[m : 2 * m].mean() - y[:m].mean()) / m
    seasonal = y[:m] - level
    seasonal = seasonal - seasonal.mean()  # centre so they sum to ~0
    return level, trend, seasonal


def _hw_states(y, m, alpha, beta, gamma):
    level, trend, seas = _hw_init(y, m)
    sse = 0.0
    for t in range(m, len(y)):
        err = y[t] - (level + trend + seas[t % m])
        sse += err * err
        new_level = alpha * (y[t] - seas[t % m]) + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        seas[t % m] = gamma * (y[t] - new_level) + (1 - gamma) * seas[t % m]
        level, trend = new_level, new_trend
    return (level, trend, seas), sse
