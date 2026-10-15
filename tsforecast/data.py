"""Synthetic time-series generators for demos, tests, and experiments."""

import numpy as np


def make_series(
    n=240,
    trend=0.5,
    season_amplitude=10.0,
    m=12,
    noise=2.0,
    level=100.0,
    seed=42,
):
    """Trend + additive seasonality + Gaussian noise.

    n: length, trend: linear drift per step, m: seasonal period,
    noise: std of the noise term.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    seasonal = season_amplitude * np.sin(2 * np.pi * t / m)
    y = level + trend * t + seasonal + rng.normal(0.0, noise, n)
    return y


def make_ar1(n=500, phi=0.8, sigma=1.0, seed=42):
    """First-order autoregressive process: y_t = phi * y_{t-1} + e_t."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal(0.0, sigma)
    return y


def make_step_series(n=200, step_at=120, step_size=25.0, noise=1.5, seed=42):
    """A level series with a single structural break (level shift)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[step_at:] = step_size
    return y + rng.normal(0.0, noise, n)
