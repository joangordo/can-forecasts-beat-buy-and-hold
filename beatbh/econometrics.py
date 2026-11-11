"""Classical time-series diagnostics, run on the training sample only."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def descriptives(r):
    rows = {}
    for a in r:
        x = r[a]
        rows[a] = {"mean_ann_%": x.mean() * 252, "vol_ann_%": x.std() * np.sqrt(252),
                   "skewness": stats.skew(x), "excess_kurtosis": stats.kurtosis(x),
                   "jarque_bera_p": stats.jarque_bera(x).pvalue, "min_%": x.min(), "max_%": x.max()}
    return pd.DataFrame(rows).T


def stationarity(log_prices, r):
    rows = []
    for a in r:
        for name, s in (("log price", log_prices[a]), ("return", r[a])):
            adf_p = adfuller(s, autolag="AIC")[1]
            kpss_p = kpss(s, regression="c", nlags="auto")[1]
            if adf_p < 0.05 and kpss_p > 0.05:
                verdict = "stationary"
            elif adf_p > 0.05 and kpss_p < 0.05:
                verdict = "non-stationary"
            else:
                verdict = "ambiguous"
            rows.append({"asset": a, "series": name, "ADF_p": adf_p, "KPSS_p": kpss_p, "verdict": verdict})
    return pd.DataFrame(rows).set_index(["asset", "series"])


def dependence(r, lags=10):
    rows = {}
    for a in r:
        x = r[a]
        lb = acorr_ljungbox(x, lags=[lags], return_df=True)["lb_pvalue"].iloc[0]
        lb2 = acorr_ljungbox(x ** 2, lags=[lags], return_df=True)["lb_pvalue"].iloc[0]
        arch_p = het_arch(x - x.mean(), nlags=lags)[1]
        rows[a] = {"ljung_box_r_p": lb, "ljung_box_r2_p": lb2, "arch_lm_p": arch_p,
                   "memory_in_mean": "yes" if lb < 0.05 else "no",
                   "arch_effects": "yes" if arch_p < 0.05 else "no"}
    return pd.DataFrame(rows).T


def granger(r, max_lags):
    """p-values of Granger-causality F tests in a VAR (row = caused, column = causing)."""
    data = r.reset_index(drop=True)
    k = max(1, int(VAR(data).select_order(max_lags).aic))
    res = VAR(data).fit(k, trend="c")
    out = pd.DataFrame(np.nan, index=r.columns, columns=r.columns)
    for caused in r.columns:
        for causing in r.columns:
            if caused != causing:
                out.loc[caused, causing] = res.test_causality(caused, [causing], kind="f").pvalue
    out.index.name = "caused / causing"
    return out, k


def johansen(log_prices):
    jo = coint_johansen(log_prices.values, det_order=0, k_ar_diff=1)
    df = pd.DataFrame({"H0": [f"rank <= {i}" for i in range(len(jo.lr1))],
                       "trace": jo.lr1, "crit_95%": jo.cvt[:, 1]}).set_index("H0")
    df["reject"] = np.where(df["trace"] > df["crit_95%"], "yes", "no")
    rejected = (df["reject"] == "yes").values
    rank = int(np.argmin(rejected)) if not rejected.all() else len(rejected)
    return df, rank


def engle_granger(log_prices):
    rows = [{"pair": f"{a}-{b}", "EG_p": coint(log_prices[a], log_prices[b])[1]}
            for a, b in combinations(log_prices.columns, 2)]
    df = pd.DataFrame(rows).set_index("pair")
    df["cointegrated_5%"] = np.where(df["EG_p"] < 0.05, "yes", "no")
    return df
