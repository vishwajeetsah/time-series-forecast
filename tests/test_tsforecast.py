import numpy as np
import pytest

from tsforecast import (
    Drift, Holt, HoltWintersAdditive, Naive, SeasonalNaive, SES,
    additive_decompose, compare_models, mae, mape, mase, mse,
    rmse, rolling_origin_backtest, smape, make_ar1, make_series,
    make_step_series,
)


# ---------- metrics ----------

def test_metrics_hand_computed():
    yt = np.array([100.0, 200.0, 300.0])
    yp = np.array([110.0, 190.0, 300.0])
    assert mae(yt, yp) == pytest.approx(20 / 3)
    assert mse(yt, yp) == pytest.approx((100 + 100 + 0) / 3)
    assert rmse(yt, yp) == pytest.approx(np.sqrt(200 / 3))
    assert mape(yt, yp) == pytest.approx((10 + 5 + 0) / 3)
    assert smape(yt, yp) == pytest.approx(
        (10 / 105 + 10 / 195 + 0) / 3 * 100
    )


def test_metrics_shape_mismatch():
    with pytest.raises(ValueError):
        mae([1, 2], [1])


def test_mape_rejects_zeros():
    with pytest.raises(ValueError):
        mape([0.0, 1.0], [1.0, 1.0])


def test_mase_below_one_beats_naive():
    train = np.arange(1.0, 21.0)          # naive one-step MAE = 1
    true = np.array([21.0, 22.0, 23.0])
    perfect = np.array([21.0, 22.0, 23.0])
    assert mase(train, true, perfect) == pytest.approx(0.0)
    naive_pred = np.array([20.0, 21.0, 22.0])  # same as naive forecast
    assert mase(train, true, naive_pred, m=1) == pytest.approx(1.0)


# ---------- models ----------

def test_naive_carries_last_value():
    y = make_series(n=60, seed=1)
    fc = Naive().fit(y).forecast(5)
    assert np.allclose(fc, y[-1])


def test_seasonalnaive_repeats_season():
    m = 4
    y = np.tile([1.0, 2.0, 3.0, 4.0], 10)
    fc = SeasonalNaive(m=m).fit(y).forecast(6)
    assert np.allclose(fc, [1.0, 2.0, 3.0, 4.0, 1.0, 2.0])


def test_drift_extrapolates_linear_trend():
    y = np.arange(10.0) * 2.0 + 5.0
    fc = Drift().fit(y).forecast(3)
    assert np.allclose(fc, [25.0, 27.0, 29.0])


def test_ses_flat_forecast_on_constant_series():
    y = np.full(50, 7.5)
    m = SES().fit(y)
    assert np.allclose(m.forecast(4), 7.5, atol=1e-9)
    assert 0.0 < m.alpha_ <= 1.0


def test_holt_recovers_linear_trend():
    y = 3.0 * np.arange(80) + 10.0
    fc = Holt().fit(y).forecast(5)
    assert np.allclose(fc, 3.0 * np.arange(80, 85) + 10.0, rtol=1e-3)


def test_holtwinters_beats_naive_on_seasonal_data():
    y = make_series(n=120, trend=0.8, season_amplitude=12.0, m=12, noise=1.0, seed=3)
    split = 108
    train, test = y[:split], y[split:]
    hw = HoltWintersAdditive(m=12).fit(train).forecast(12)
    nv = Naive().fit(train).forecast(12)
    assert mape(test, hw) < mape(test, nv)


def test_holtwinters_needs_two_seasons():
    with pytest.raises(ValueError):
        HoltWintersAdditive(m=12).fit(np.arange(20.0))


def test_models_reject_bad_input():
    with pytest.raises(ValueError):
        SES().fit([1.0])
    with pytest.raises(ValueError):
        SES().fit([1.0, np.nan, 3.0])
    with pytest.raises(RuntimeError):
        Holt().forecast(3)


def test_fitted_values_shapes():
    y = make_series(n=60, seed=5)
    for cls, kw in [(Naive, {}), (SeasonalNaive, {"m": 12}), (Drift, {}),
                    (SES, {}), (Holt, {}), (HoltWintersAdditive, {"m": 12})]:
        m = cls(**kw).fit(y)
        assert m.fitted().shape == y.shape


# ---------- decomposition ----------

def test_decompose_recovers_pure_seasonal():
    m = 12
    t = np.arange(120, dtype=float)
    y = 5.0 * np.sin(2 * np.pi * t / m) + 100.0
    comp = additive_decompose(y, m=m)
    mid = comp["resid"][m:-m]
    assert np.nanmean(np.abs(mid)) < 0.05
    assert np.allclose(comp["seasonal"][:m] - comp["seasonal"][:m].mean(),
                       comp["seasonal"][m:2 * m] - comp["seasonal"][m:2 * m].mean(),
                       atol=1e-9)


def test_decompose_sums_to_observed():
    y = make_series(n=120, seed=9)
    comp = additive_decompose(y, m=12)
    mid = slice(12, -12)
    recon = comp["trend"][mid] + comp["seasonal"][mid] + comp["resid"][mid]
    assert np.allclose(recon, y[mid])


# ---------- backtest ----------

def test_rolling_origin_fold_count_and_monotonic_mae():
    y = make_series(n=100, seed=11)
    res = rolling_origin_backtest(Naive, y, initial_train=60, horizon=5, step=5, m=12)
    # origins: 60, 65, ..., 95 -> (95-60)/5 + 1 = 8 folds
    assert res["summary"]["n_folds"] == 8
    origins = [f["origin"] for f in res["folds"]]
    assert origins == sorted(origins)
    for key in ("mean_mae", "mean_rmse", "mean_mape", "mean_mase"):
        assert res["summary"][key] > 0


def test_backtest_refits_each_origin():
    # a step change: an honest backtest must adapt, so error must be finite
    # and folds must each see a longer training set
    y = make_step_series(n=120, step_at=80, seed=2)
    res = rolling_origin_backtest(SES, y, initial_train=90, horizon=4, step=10, m=12)
    assert res["summary"]["n_folds"] == 3
    assert np.isfinite(res["summary"]["mean_mae"])


def test_compare_models_ranks_by_mase():
    y = make_series(n=96, trend=0.5, season_amplitude=8.0, m=12, noise=1.0, seed=4)
    rows = compare_models(
        {"Naive": Naive, "HW": lambda: HoltWintersAdditive(m=12)},
        y, initial_train=72, horizon=12, m=12,
    )
    assert rows[0]["model"] == "HW"  # seasonal data: HW should win
    assert rows[0]["mean_mase"] <= rows[1]["mean_mase"]


def test_backtest_rejects_impossible_split():
    with pytest.raises(ValueError):
        rolling_origin_backtest(Naive, np.arange(10.0), initial_train=8, horizon=5)


# ---------- data ----------

def test_generators_deterministic_and_shaped():
    a = make_series(n=50, seed=0)
    b = make_series(n=50, seed=0)
    assert a.shape == (50,) and np.array_equal(a, b)
    assert make_ar1(n=30, seed=0).shape == (30,)
    s = make_step_series(n=100, step_at=50, step_size=25.0, noise=0.0, seed=0)
    assert s[60] - s[40] == pytest.approx(25.0)
