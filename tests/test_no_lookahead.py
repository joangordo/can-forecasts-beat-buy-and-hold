"""Causality tests.

If a forecast, feature, position or weight for day t used information from day t
or later, replacing the data from a cut-off date onwards with noise would change
it. Each test scrambles the 'future' and checks that the past stays identical,
and that the test is not vacuous (later values do change)."""
import numpy as np
import pandas as pd

from beatbh.data import split_indices
from beatbh.models import make_logit, technical_features, wf_arima, wf_classifier, wf_garch, wf_var
from beatbh.strategies import allocation_rules, historical_vol, signal_positions, vol_positions

START, STOP, CUT, EVERY = 500, 600, 550, 21


def scramble(values, cut=CUT, seed=1):
    out = np.array(values, dtype=float, copy=True)
    out[cut:] = np.random.default_rng(seed).normal(0, 5, out[cut:].shape)
    return out


def scrambled(df):
    return pd.DataFrame(scramble(df.values), index=df.index, columns=df.columns)


def assert_past_unchanged(a, b, name, last=CUT):
    """Values up to `last` must be identical; values after it must differ."""
    np.testing.assert_allclose(a[START:last + 1], b[START:last + 1], err_msg=f"{name} uses future data")
    assert not np.allclose(a[last + 1:STOP], b[last + 1:STOP]), f"{name}: vacuous test"


# Forecasts for day t may use data up to t-1, so scrambling from CUT leaves day CUT intact.
def test_arima(returns):
    y = returns["AAA"].values
    assert_past_unchanged(wf_arima(y, START, STOP, (1, 0, 1), EVERY),
                          wf_arima(scramble(y), START, STOP, (1, 0, 1), EVERY), "ARIMA")


def test_garch(returns):
    y = returns["AAA"].values
    (m1, v1), (m2, v2) = wf_garch(y, START, STOP, 1, EVERY), wf_garch(scramble(y), START, STOP, 1, EVERY)
    assert_past_unchanged(m1, m2, "AR-GARCH mean")
    assert_past_unchanged(v1, v2, "AR-GARCH variance")


def test_var(returns):
    Y = returns.values
    assert_past_unchanged(wf_var(Y, START, STOP, 2, EVERY), wf_var(scramble(Y), START, STOP, 2, EVERY), "VAR")


def test_technical_features(returns):
    f1, f2 = technical_features(returns, "AAA"), technical_features(scrambled(returns), "AAA")
    pd.testing.assert_frame_equal(f1.iloc[:CUT + 1], f2.iloc[:CUT + 1])
    assert not f1.iloc[CUT + 1:].equals(f2.iloc[CUT + 1:])


def test_classifier(returns):
    def run(r):
        X = technical_features(r, "AAA").replace([np.inf, -np.inf], np.nan).values
        return wf_classifier(make_logit, X, (r["AAA"].values > 0).astype(int), START, STOP, EVERY)

    assert_past_unchanged(run(returns), run(scrambled(returns)), "Logit")


def test_historical_volatility(returns):
    assert_past_unchanged(historical_vol(returns).values, historical_vol(scrambled(returns)).values,
                          "historical volatility")


# Positions for day t use the forecast for day t (made at t-1), so they may change from CUT on.
def test_positions(cfg):
    rng = np.random.default_rng(0)
    signal, sigma = rng.normal(0, 0.05, 700), rng.uniform(0.5, 2.0, 700)
    other_sigma = sigma.copy()
    other_sigma[CUT:] = rng.uniform(0.5, 2.0, 700 - CUT)
    assert_past_unchanged(signal_positions(signal, 0.01, cfg),
                          signal_positions(scramble(signal), 0.01, cfg), "signal positions", CUT - 1)
    assert_past_unchanged(vol_positions(sigma, np.ones(700), 1.0, cfg),
                          vol_positions(other_sigma, np.ones(700), 1.0, cfg),
                          "vol. managed positions", CUT - 1)


def test_allocation_weights(returns, cfg):
    sigma = returns.rolling(20).std().shift(1)  # any causal volatility forecast
    w1 = allocation_rules(returns, sigma, (START, STOP), cfg)
    w2 = allocation_rules(scrambled(returns), sigma, (START, STOP), cfg)
    for name in ("Inverse vol (GARCH)", "Risk parity (GARCH)", "Risk parity (historical)"):
        np.testing.assert_allclose(w1[name]["weights"][:CUT - START + 1], w2[name]["weights"][:CUT - START + 1],
                                   err_msg=f"{name} weights use future data")


def test_split_is_chronological_and_complete(returns):
    s = split_indices(returns.index, "2015-12-31", "2016-12-30")
    assert s["train"][0] == 0 and s["test"][1] == len(returns)
    assert s["train"][1] == s["validation"][0] and s["validation"][1] == s["test"][0]
    assert returns.index[s["train"][1] - 1] <= pd.Timestamp("2015-12-31") < returns.index[s["validation"][0]]
