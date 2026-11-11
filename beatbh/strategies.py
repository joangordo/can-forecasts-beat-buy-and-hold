"""Trading rules, backtest accounting, performance metrics and asset-allocation rules."""
import numpy as np
from scipy.optimize import minimize

BUY_HOLD = "Buy & hold"
VOL_MANAGED = "Vol. managed"
ALLOCATION_RULES = ["Equal weight (monthly)", "Inverse vol (GARCH)",
                    "Risk parity (GARCH)", "Risk parity (historical)"]


# --- Positions ------------------------------------------------------------------
def apply_band(target, cfg):
    """Turn the desired daily exposure into the executed one. Moves to 0 % or 100 %
    always execute; other changes only if larger than the rebalance band."""
    s = cfg["strategy"]
    pos, current = np.zeros(len(target)), 0.0
    for i, goal in enumerate(target):
        if goal in (0.0, 1.0) or current == 0 or abs(goal - current) >= s["rebalance_band"]:
            current = goal
        pos[i] = current
    lag = s["execution_lag"]
    return np.r_[np.zeros(lag), pos[:-lag]] if lag > 0 else pos


def signal_positions(signal, threshold, cfg):
    """Fully invested on day t if the forecast for t exceeds the threshold, else cash."""
    return apply_band(np.where(np.nan_to_num(signal, nan=-np.inf) > threshold, 1.0, 0.0), cfg)


def vol_positions(sigma, vol_base, factor, cfg):
    """Always invested; exposure shrinks when the GARCH forecast exceeds `factor`
    times the asset's historical volatility."""
    with np.errstate(divide="ignore", invalid="ignore"):
        target = np.minimum(1.0, factor * vol_base / sigma)
    return apply_band(np.where(np.isfinite(target), target, 1.0), cfg)


def historical_vol(r, min_days=252):
    """Volatility of all returns up to t-1 (expanding window)."""
    return r.expanding(min_periods=min_days).std().shift(1)


# --- Single-asset backtest ---------------------------------------------------
def simulate(pos, r_pct, rf, cfg, capital):
    """pos[t] is decided at the close of t-1 and earns day t's return; the uninvested
    fraction earns the risk-free rate; costs are charged on the traded fraction."""
    c = cfg["costs"]
    gross = np.expm1(r_pct / 100)
    cost = np.abs(np.diff(np.r_[0.0, pos])) * (c["commission"] + c["slippage"])
    net = pos * gross + (1 - pos) * rf - cost
    equity = capital * np.cumprod(1 + net)
    return {"pos": pos, "net": net, "equity": equity,
            "cost_usd": cost * np.r_[capital, equity[:-1]]}


def closed_trades(pos, net):
    """Return of each complete trade, from entry to exit (exit cost included)."""
    out, i, n = [], 0, len(pos)
    while i < n:
        if pos[i] > 0:
            j = i
            while j + 1 < n and pos[j + 1] > 0:
                j += 1
            out.append(np.prod(1 + net[i:min(j + 2, n)]) - 1)
            i = j + 1
        else:
            i += 1
    return out


def basic_metrics(equity, capital, rf, costs=np.nan):
    """Sharpe and Sortino use daily returns in excess of the risk-free rate."""
    net = equity / np.r_[capital, equity[:-1]] - 1
    excess = net - rf
    sd = excess.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    curve = np.r_[capital, equity]
    final = equity[-1]
    return {
        "final_value": final,
        "profit": final - capital,
        "total_return_%": 100 * (final / capital - 1),
        "CAGR_%": 100 * ((final / capital) ** (252 / len(equity)) - 1),
        "volatility_%": 100 * net.std(ddof=1) * np.sqrt(252),
        "sharpe": np.sqrt(252) * excess.mean() / sd if sd > 0 else np.nan,
        "sortino": np.sqrt(252) * excess.mean() / downside if downside > 0 else np.nan,
        "max_drawdown_%": 100 * np.min(curve / np.maximum.accumulate(curve) - 1),
        "costs": costs,
    }


def trade_metrics(sim, capital, rf):
    m = basic_metrics(sim["equity"], capital, rf)
    m.pop("costs")
    pos, trades = sim["pos"], closed_trades(sim["pos"], sim["net"])
    m.update({
        "orders": int(np.sum(np.abs(np.diff(np.r_[0.0, pos])) > 1e-12)),
        "trades": len(trades),
        "win_rate_%": 100 * np.mean(np.array(trades) > 0) if trades else np.nan,
        "time_in_market_%": 100 * np.mean(pos > 0),
        "avg_exposure_%": 100 * np.mean(pos),
        "costs": sim["cost_usd"].sum(),
    })
    return m


# --- Asset allocation -----------------------------------------------------------
def erc_weights(cov):
    """Equal-risk-contribution weights (Spinu, 2013): minimise
    0.5 y'Σy - (1/k) Σ log y_i over y > 0, then normalise y to sum to one."""
    k = len(cov)
    res = minimize(lambda y: 0.5 * y @ cov @ y - np.log(y).sum() / k,
                   1.0 / np.sqrt(np.diag(cov)),
                   jac=lambda y: cov @ y - 1.0 / (k * y),
                   method="L-BFGS-B", bounds=[(1e-12, None)] * k)
    return res.x / res.x.sum()


def simulate_portfolio(rule, R, start, stop, cfg, capital, every=None):
    """Fully invested multi-asset portfolio. rule(t) gives target weights for day t
    from data up to t-1. Rebalances on the first day and then every `every` days
    (None = never, i.e. buy & hold); in between, weights drift with prices."""
    k = R.shape[1]
    unit_cost = cfg["costs"]["commission"] + cfg["costs"]["slippage"]
    values, total, costs = np.zeros(k), capital, 0.0
    equity, weights = np.empty(stop - start), np.empty((stop - start, k))
    for j, t in enumerate(range(start, stop)):
        cost = 0.0
        if j == 0 or (every and j % every == 0):
            w = rule(t)
            cost = unit_cost * np.abs(w * total - values).sum()
            values = w * total
        weights[j] = values / values.sum()
        values = values * (1 + R[t])
        if cost:  # same cost treatment as simulate()
            values *= (values.sum() - cost) / values.sum()
        total = values.sum()
        costs += cost
        equity[j] = total
    return {"equity": equity, "costs": costs, "weights": weights}


def allocation_rules(r, sigma, split, cfg):
    """Parameter-free allocation rules. Weights for day t use the GARCH volatility
    forecast for t and correlations estimated on data up to t-1."""
    start, stop = split
    a = cfg["allocation"]
    Y, S = r.values, sigma.values
    R = np.expm1(Y / 100)
    k, every = Y.shape[1], a["rebalance_every"]
    nc, nv = a["corr_window"], a["hist_vol_window"]

    def cov(vol, t):
        return np.outer(vol, vol) * np.corrcoef(Y[t - nc:t].T)

    def inverse(vol):
        return (1 / vol) / (1 / vol).sum()

    rules = {
        BUY_HOLD: (lambda t: np.full(k, 1 / k), None),
        "Equal weight (monthly)": (lambda t: np.full(k, 1 / k), every),
        "Inverse vol (GARCH)": (lambda t: inverse(S[t]), every),
        "Risk parity (GARCH)": (lambda t: erc_weights(cov(S[t], t)), every),
        "Risk parity (historical)": (lambda t: erc_weights(cov(Y[t - nv:t].std(axis=0, ddof=1), t)), every),
    }
    capital = cfg["strategy"]["initial_capital"]
    return {name: simulate_portfolio(rule, R, start, stop, cfg, capital, n)
            for name, (rule, n) in rules.items()}
