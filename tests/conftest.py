import numpy as np
import pandas as pd
import pytest

from beatbh.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides={"models": {"refit_every": 21}, "stats": {"bootstrap_reps": 300}})


@pytest.fixture(scope="session")
def returns():
    """Synthetic daily % returns: three correlated assets with GARCH-type volatility."""
    rng = np.random.default_rng(7)
    n, k = 900, 3
    chol = np.linalg.cholesky(np.array([[1, .6, -.2], [.6, 1, -.1], [-.2, -.1, 1]]))
    out, var = np.zeros((n, k)), np.ones(k)
    for t in range(1, n):
        z = chol @ rng.standard_t(6, k) / np.sqrt(1.5)
        var = 0.05 + 0.08 * out[t - 1] ** 2 + 0.87 * var
        out[t] = 0.03 + 0.05 * out[t - 1] + np.sqrt(var) * z
    return pd.DataFrame(out, index=pd.bdate_range("2015-01-01", periods=n), columns=["AAA", "BBB", "CCC"])
