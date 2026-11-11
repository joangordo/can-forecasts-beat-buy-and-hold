"""Statistical tests: Holm, Sharpe-difference tests and Diebold-Mariano."""
import numpy as np

from beatbh.models import diebold_mariano
from beatbh.sharpe_tests import bootstrap_test, hac_test, holm


def test_holm_matches_hand_calculation():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, np.nan]), [0.03, 0.06, 0.06, np.nan])


def test_identical_strategies_are_not_different():
    x = np.random.default_rng(0).normal(0, 0.01, 500)
    assert hac_test(x, x) == 1.0 and bootstrap_test(x, x, reps=200) == 1.0


def test_a_clearly_better_strategy_is_detected():
    rng = np.random.default_rng(3)
    y = rng.normal(0.0002, 0.01, 2000)
    x = 0.8 * y + rng.normal(0.0012, 0.004, 2000)  # correlated with y, much higher Sharpe
    assert hac_test(x, y) < 0.01 and bootstrap_test(x, y, reps=500) < 0.01


def test_hac_test_holds_its_size_under_the_null():
    """Equal Sharpe ratios, fat tails, correlation 0.9: about 5 % false rejections."""
    rng, rejections = np.random.default_rng(5), 0
    for _ in range(200):
        z = rng.standard_t(5, (1000, 2)) / np.sqrt(5 / 3)
        z[:, 1] = 0.9 * z[:, 0] + np.sqrt(1 - 0.81) * z[:, 1]
        x = 0.0004 + 0.01 * z
        rejections += hac_test(x[:, 0], x[:, 1]) < 0.05
    assert 0.01 <= rejections / 200 <= 0.10


def test_diebold_mariano_prefers_the_informative_forecast():
    rng = np.random.default_rng(2)
    signal = rng.normal(0, 1, 1500)
    y = signal + rng.normal(0, 1, 1500)
    dm, p = diebold_mariano(y - signal, y)  # informative forecast vs random walk (forecast 0)
    assert dm < 0 and p < 0.01
