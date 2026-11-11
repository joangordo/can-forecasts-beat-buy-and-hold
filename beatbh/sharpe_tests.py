"""Tests for the difference between two Sharpe ratios, and Holm's correction.

Both tests work on daily excess returns and allow for fat tails and serial
dependence:
* HAC: delta method with a Newey-West covariance matrix (Ledoit and Wolf, 2008).
* Circular block bootstrap of the difference (Politis and Romano, 1992).
"""
import numpy as np
import pandas as pd
from scipy import stats


def sharpe(x):
    sd = x.std(ddof=1)
    return x.mean() / sd if sd > 0 else np.nan


def newey_west(Z, lags):
    T = len(Z)
    S = Z.T @ Z / T
    for lag in range(1, lags + 1):
        G = Z[lag:].T @ Z[:-lag] / T
        S += (1 - lag / (lags + 1)) * (G + G.T)
    return S


def hac_test(x, y):
    """Two-sided p-value for H0: Sharpe(x) = Sharpe(y)."""
    if np.allclose(x, y):
        return 1.0
    T = len(x)
    mu = np.array([x.mean(), y.mean()])
    m2 = np.array([(x ** 2).mean(), (y ** 2).mean()])
    s3 = (m2 - mu ** 2) ** 1.5
    diff = mu[0] / np.sqrt(m2[0] - mu[0] ** 2) - mu[1] / np.sqrt(m2[1] - mu[1] ** 2)
    grad = np.array([m2[0] / s3[0], -m2[1] / s3[1], -mu[0] / (2 * s3[0]), mu[1] / (2 * s3[1])])
    Z = np.column_stack([x - mu[0], y - mu[1], x ** 2 - m2[0], y ** 2 - m2[1]])
    se = np.sqrt(grad @ newey_west(Z, int(4 * (T / 100) ** (2 / 9))) @ grad / T)
    return 2 * stats.norm.sf(abs(diff / se)) if se > 0 else np.nan


def bootstrap_test(x, y, reps=2000, block=20, seed=42):
    """Two-sided p-value for H0: Sharpe(x) = Sharpe(y): resample days in circular
    blocks (keeping serial dependence) and centre the bootstrap differences."""
    if np.allclose(x, y):
        return 1.0
    T, rng = len(x), np.random.default_rng(seed)
    observed = sharpe(x) - sharpe(y)
    n_blocks = -(-T // block)
    diffs = []
    for done in range(0, reps, 250):  # in batches to keep memory low
        b = min(250, reps - done)
        idx = (rng.integers(0, T, (b, n_blocks))[:, :, None] + np.arange(block)) % T
        idx = idx.reshape(b, -1)[:, :T]
        xs, ys = x[idx], y[idx]
        diffs.append(xs.mean(1) / xs.std(1, ddof=1) - ys.mean(1) / ys.std(1, ddof=1))
    return float(np.mean(np.abs(np.concatenate(diffs) - observed) >= abs(observed)))


def holm(p):
    """Holm-Bonferroni adjusted p-values (NaN entries are ignored)."""
    p = np.asarray(p, dtype=float)
    valid = np.where(~np.isnan(p))[0]
    adjusted, running = np.full(len(p), np.nan), 0.0
    for rank, i in enumerate(valid[np.argsort(p[valid])]):
        running = max(running, min(1.0, (len(valid) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def compare(x, y, cfg):
    """x, y: daily excess returns over the same days."""
    s = cfg["stats"]
    return {"sharpe": np.sqrt(252) * sharpe(x), "sharpe_bh": np.sqrt(252) * sharpe(y),
            "delta_sharpe": np.sqrt(252) * (sharpe(x) - sharpe(y)),
            "p_hac": hac_test(x, y),
            "p_bootstrap": bootstrap_test(x, y, s["bootstrap_reps"], s["bootstrap_block"], cfg["seed"])}


def add_holm(df):
    df.insert(df.columns.get_loc("p_hac") + 1, "p_hac_holm", holm(df["p_hac"]))
    return df


def sharpe_tests(returns, rf, cfg, reference="Buy & hold"):
    """Every column of `returns` (daily, total) against the reference column."""
    excess = returns.sub(np.asarray(rf), axis=0)
    base = excess[reference].values
    return add_holm(pd.DataFrame({n: compare(excess[n].values, base, cfg)
                                  for n in excess.columns if n != reference}).T)
