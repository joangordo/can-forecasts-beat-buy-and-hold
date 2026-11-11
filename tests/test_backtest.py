"""Backtest accounting and allocation rules."""
import numpy as np

from beatbh.strategies import apply_band, erc_weights, simulate, simulate_portfolio


def unit_cost(cfg):
    return cfg["costs"]["commission"] + cfg["costs"]["slippage"]


def test_buy_and_hold_tracks_the_price(cfg):
    r = np.random.default_rng(0).normal(0.03, 1, 250)
    sim = simulate(np.ones(250), r, np.zeros(250), cfg, 1000.0)
    first = np.expm1(r[0] / 100)
    expected = 1000 * np.exp(r.sum() / 100) * (1 + first - unit_cost(cfg)) / (1 + first)  # one purchase cost
    assert np.isclose(sim["equity"][-1], expected)


def test_cash_earns_the_risk_free_rate(cfg):
    sim = simulate(np.zeros(100), np.full(100, 5.0), np.full(100, 1e-4), cfg, 1000.0)
    assert np.isclose(sim["equity"][-1], 1000 * (1 + 1e-4) ** 100)


def test_costs_scale_with_turnover(cfg):
    sim = simulate(np.tile([1.0, 0.0], 10), np.zeros(20), np.zeros(20), cfg, 1000.0)
    assert np.isclose(sim["equity"][-1], 1000 * (1 - unit_cost(cfg)) ** 20)


def test_band_skips_small_changes_but_always_returns_to_full_exposure(cfg):
    assert list(apply_band(np.array([1.0, 0.85, 0.93, 1.0, 0.0]), cfg)) == [1.0, 0.85, 0.85, 1.0, 0.0]


def test_portfolio_without_rebalancing_is_the_sum_of_sleeves(cfg):
    r = np.random.default_rng(1).normal(0.03, 1, (300, 3))
    port = simulate_portfolio(lambda t: np.full(3, 1 / 3), np.expm1(r / 100), 0, 300, cfg, 3000.0)
    sleeves = sum(simulate(np.ones(300), r[:, j], np.zeros(300), cfg, 1000.0)["equity"] for j in range(3))
    np.testing.assert_allclose(port["equity"], sleeves, rtol=1e-4)


def test_risk_parity_equalises_risk_contributions():
    vol = np.array([1.0, 1.4, 0.9, 0.8])
    corr = np.array([[1, .9, 0, -.4], [.9, 1, 0, -.4], [0, 0, 1, .2], [-.4, -.4, .2, 1]])
    cov = np.outer(vol, vol) * corr
    w = erc_weights(cov)
    contributions = w * (cov @ w)
    np.testing.assert_allclose(contributions / contributions.sum(), 0.25, atol=1e-4)
    assert np.isclose(w.sum(), 1) and (w > 0).all()
