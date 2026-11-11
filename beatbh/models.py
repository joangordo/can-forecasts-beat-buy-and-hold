"""Model-order selection, walk-forward forecasts and forecast evaluation.

Every wf_* function returns an array as long as the whole series, holding at
position t the forecast for day t made with data up to day t-1 only (NaN
outside [start, stop)). Parameters are re-estimated every `every` days on an
expanding window; in between they stay fixed and only new observations are
filtered in. tests/test_no_lookahead.py checks this property directly.
"""
import logging
import time

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

log = logging.getLogger("beatbh")

BENCHMARK_MODELS = ["Random walk", "Historical mean"]
RETURN_MODELS = ["ARIMA", "AR-GARCH", "VAR"]   # forecast the return (%)
PROB_MODELS = ["Logit", "GBM"]                 # forecast P(return > 0)


# --- Order selection (training data only) --------------------------------------
def make_garch(y, lags, hold_back=None):
    mean = {"mean": "AR", "lags": lags} if lags > 0 else {"mean": "Constant"}
    return arch_model(y, vol="GARCH", p=1, q=1, dist="t", hold_back=hold_back, rescale=False, **mean)


def select_arima(y, max_p, max_q):
    best_order, best_aic = (0, 0, 0), np.inf
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                aic = ARIMA(y, order=(p, 0, q), trend="c").fit().aic
            except Exception:
                continue
            if aic < best_aic:
                best_order, best_aic = (p, 0, q), aic
    return best_order, best_aic


def select_garch(y, max_lags):
    best = (0, np.inf, None)
    for lags in range(max_lags + 1):
        res = make_garch(y, lags, hold_back=max_lags).fit(disp="off")  # same sample -> comparable BIC
        if res.bic < best[1]:
            best = (lags, res.bic, res)
    return best[0], best[2]


def select_specification(r_hist, cfg):
    """ARIMA orders (AIC), AR lags of the AR-GARCH mean (BIC) and VAR lags (AIC)."""
    m = cfg["models"]
    spec, rows = {"arima": {}, "garch": {}}, []
    for a in r_hist:
        y = r_hist[a].values
        order, aic = select_arima(y, m["arima_max_p"], m["arima_max_q"])
        lags, res = select_garch(y, m["garch_max_lags"])
        spec["arima"][a], spec["garch"][a] = order, lags
        p = res.params
        rows.append({"asset": a, "ARIMA": str(order), "AIC": aic, "AR_lags_GARCH": lags,
                     "omega": p["omega"], "alpha": p["alpha[1]"], "beta": p["beta[1]"],
                     "persistence": p["alpha[1]"] + p["beta[1]"], "t_dof": p["nu"]})
    spec["var"] = max(1, int(VAR(r_hist.reset_index(drop=True)).select_order(m["var_max_lags"]).aic))
    return spec, pd.DataFrame(rows).set_index("asset")


# --- Walk-forward forecasts ---------------------------------------------------
def wf_arima(y, start, stop, order, every):
    pred, params = np.full(len(y), np.nan), None
    for t0 in range(start, stop, every):
        t1 = min(t0 + every, stop)
        res = ARIMA(y[:t0], order=order, trend="c").fit(start_params=params)
        params = res.params
        # Append the block without re-estimating: in-sample one-step predictions
        # for each day use only the observations before it.
        pred[t0:t1] = res.append(y[t0:t1], refit=False).predict(start=t0, end=t1 - 1)
    return pred


def wf_garch(y, start, stop, lags, every):
    mean, var = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for t0 in range(start, stop, every):
        t1 = min(t0 + every, stop)
        res = make_garch(y[:t1], lags).fit(last_obs=t0, disp="off")   # estimated on y[:t0]
        f = res.forecast(start=t0 - 1, horizon=1, reindex=False)      # origins t0-1 ... t1-1
        mean[t0:t1] = f.mean["h.1"].values[:t1 - t0]
        var[t0:t1] = f.variance["h.1"].values[:t1 - t0]
    return mean, var


def wf_var(Y, start, stop, k, every):
    pred = np.full(Y.shape, np.nan)
    for t0 in range(start, stop, every):
        t1 = min(t0 + every, stop)
        res = VAR(Y[:t0]).fit(k, trend="c")
        block = np.tile(res.intercept, (t1 - t0, 1))
        for i in range(k):  # y_t = c + sum_i A_i y_{t-1-i}
            block += Y[t0 - 1 - i:t1 - 1 - i] @ res.coefs[i].T
        pred[t0:t1] = block
    return pred


def technical_features(r, asset):
    """Features for predicting day t, built only from data up to the close of t-1:
    lagged returns, momentum, realised volatility, RSI, MACD, Bollinger %b, distance
    to moving averages and the other assets' latest returns."""
    x = r[asset]
    p = np.exp(x.cumsum() / 100)  # price index; every feature below is scale-free
    f = pd.DataFrame(index=r.index)
    for lag in (1, 2, 3, 5, 10):
        f[f"ret_lag{lag}"] = x.shift(lag - 1)
    for w in (5, 10, 20, 60):
        f[f"momentum_{w}"] = p / p.shift(w) - 1
    for w in (10, 20, 60):
        f[f"vol_{w}"] = x.rolling(w).std()
    delta = p.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss)
    macd = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
    f["macd"] = macd / p
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / p
    ma20, sd20 = p.rolling(20).mean(), p.rolling(20).std()
    f["bollinger_pct_b"] = (p - (ma20 - 2 * sd20)) / (4 * sd20)
    for w in (20, 50, 200):
        f[f"dist_ma_{w}"] = p / p.rolling(w).mean() - 1
    for other in r.columns:
        if other != asset:
            f[f"{other}_ret"] = r[other]
    return f.shift(1)  # everything known at the close of t-1


def make_logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))


def make_gbm(seed=42):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                          min_samples_leaf=50, l2_regularization=1.0,
                                          random_state=seed)


def wf_classifier(make_model, X, up, start, stop, every):
    valid = ~np.isnan(X).any(axis=1)
    prob = np.full(len(up), np.nan)
    for t0 in range(start, stop, every):
        t1 = min(t0 + every, stop)
        rows = np.where(valid[:t0])[0]
        model = make_model().fit(X[rows], up[rows])
        block, ok = np.full(t1 - t0, np.nan), valid[t0:t1]
        if ok.any():
            block[ok] = model.predict_proba(X[t0:t1][ok])[:, 1]
        prob[t0:t1] = block
    return prob


def generate_forecasts(r, start, stop, cfg, spec):
    """Walk-forward forecasts of every model on [start, stop) with a frozen specification.
    Returns (forecasts: model -> DataFrame, sigma: GARCH volatility forecasts in %)."""
    m, seed = cfg["models"], cfg["seed"]
    every, assets, Y = m["refit_every"], list(r.columns), r.values
    names = BENCHMARK_MODELS + RETURN_MODELS + (PROB_MODELS if m["use_ml"] else [])
    fc = {n: pd.DataFrame(np.nan, index=r.index, columns=assets) for n in names}
    sigma = pd.DataFrame(np.nan, index=r.index, columns=assets)
    fc["Random walk"].iloc[start:stop] = 0.0  # random walk in prices = zero expected return
    fc["Historical mean"].iloc[start:stop] = r.expanding().mean().shift(1).iloc[start:stop].values

    log.info(f"Walk-forward {r.index[start].date()} -> {r.index[stop - 1].date()} "
             f"(re-estimated every {every} days)")
    for j, a in enumerate(assets):
        t = time.time()
        fc["ARIMA"].iloc[:, j] = wf_arima(Y[:, j], start, stop, spec["arima"][a], every)
        mean, var = wf_garch(Y[:, j], start, stop, spec["garch"][a], every)
        fc["AR-GARCH"].iloc[:, j] = mean
        sigma.iloc[:, j] = np.sqrt(var)
        if m["use_ml"]:
            X = technical_features(r, a).replace([np.inf, -np.inf], np.nan).values
            up = (Y[:, j] > 0).astype(int)
            fc["Logit"].iloc[:, j] = wf_classifier(make_logit, X, up, start, stop, every)
            fc["GBM"].iloc[:, j] = wf_classifier(lambda: make_gbm(seed), X, up, start, stop, every)
        log.info(f"  {a}: done ({time.time() - t:.0f} s)")
    fc["VAR"].iloc[:, :] = wf_var(Y, start, stop, spec["var"], every)
    return fc, sigma


# --- Forecast evaluation ------------------------------------------------------
def diebold_mariano(e1, e2):
    """Diebold-Mariano test (h = 1, squared-error loss) with the Harvey-Leybourne-Newbold
    small-sample correction. DM < 0 means forecast 1 is more accurate."""
    d = e1 ** 2 - e2 ** 2
    n, var = len(d), np.var(d)
    if var == 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var / n) * np.sqrt((n - 1) / n)
    return dm, 2 * stats.t.sf(abs(dm), df=n - 1)


def evaluate_forecasts(fc, r, split):
    """Return models: RMSE, MAE, hit rate, Diebold-Mariano vs random walk.
    Classifiers: hit rate and Brier skill score vs the historical frequency of up days."""
    start, stop = split
    rows = []
    for a in r.columns:
        y = r[a].values[start:stop]
        up = (y > 0).astype(float)
        base = (r[a] > 0).astype(float).expanding().mean().shift(1).values[start:stop]
        for name, df in fc.items():
            p = df[a].values[start:stop]
            row = {"asset": a, "model": name}
            if name in PROB_MODELS:
                row["hit_rate_%"] = 100 * np.mean((p > 0.5) == (y > 0))
                row["brier_skill"] = 1 - np.mean((p - up) ** 2) / np.mean((base - up) ** 2)
            else:
                e, nonzero = y - p, p != 0
                row.update({"RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)),
                            "hit_rate_%": 100 * np.mean(np.sign(p[nonzero]) == np.sign(y[nonzero]))
                            if nonzero.any() else np.nan})
                if name != "Random walk":
                    row["DM_vs_RW"], row["DM_p"] = diebold_mariano(e, y)
            rows.append(row)
    cols = ["RMSE", "MAE", "hit_rate_%", "DM_vs_RW", "DM_p", "brier_skill"]
    return pd.DataFrame(rows).set_index(["asset", "model"]).reindex(columns=cols)
