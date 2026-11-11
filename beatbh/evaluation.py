"""Strategy candidates, validation-based selection and the out-of-sample test."""
import numpy as np
import pandas as pd

from .models import PROB_MODELS, RETURN_MODELS
from .sharpe_tests import add_holm, compare
from .strategies import (ALLOCATION_RULES, BUY_HOLD, VOL_MANAGED, basic_metrics,
                         signal_positions, simulate, trade_metrics, vol_positions)

SELECTED = "Selected strategy"
STRATEGY_ORDER = [BUY_HOLD, SELECTED, VOL_MANAGED] + ALLOCATION_RULES


def signal_models(cfg):
    return RETURN_MODELS + (PROB_MODELS if cfg["models"]["use_ml"] else [])


def candidates(asset, cfg):
    s = cfg["strategy"]
    rows = [{"asset": asset, "model": BUY_HOLD, "threshold": np.nan, "vol_factor": np.nan}]
    rows += [{"asset": asset, "model": m, "threshold": u, "vol_factor": np.nan}
             for m in RETURN_MODELS for u in s["return_thresholds"]]
    if cfg["models"]["use_ml"]:
        rows += [{"asset": asset, "model": m, "threshold": u, "vol_factor": np.nan}
                 for m in PROB_MODELS for u in s["prob_thresholds"]]
    rows += [{"asset": asset, "model": VOL_MANAGED, "threshold": np.nan, "vol_factor": f}
             for f in s["vol_factors"]]
    return rows


def positions_for(row, fc, sigma, vol_base, start, stop, cfg):
    a, m = row["asset"], row["model"]
    if m == BUY_HOLD:
        return np.ones(stop - start)
    if m == VOL_MANAGED:
        return vol_positions(sigma[a].values[start:stop], vol_base[a].values[start:stop],
                             row["vol_factor"], cfg)
    return signal_positions(fc[m][a].values[start:stop], row["threshold"], cfg)


def label(row):
    m = row["model"]
    if m == BUY_HOLD:
        return "Buy & hold (default)"
    if m == VOL_MANAGED:
        return f"Vol. managed (factor {row['vol_factor']:.2f})"
    if m in PROB_MODELS:
        return f"{m} (P(up) > {row['threshold']:.2f})"
    return f"{m} (forecast > {row['threshold']:.2f} %)"


def short_label(row):
    m = row["model"]
    if m == BUY_HOLD:
        return "B&H"
    if m == VOL_MANAGED:
        return f"Vol ({row['vol_factor']:.2f})"
    return f"{m} ({row['threshold']:.2f})"


def tune_on_validation(fc, sigma, vol_base, r, rf, split, cfg):
    """Backtest every candidate on validation, keep the best parameter per model and
    apply the selection rule (Sharpe margin over buy & hold, minimum trades)."""
    start, stop = split
    capital = cfg["strategy"]["initial_capital"] / r.shape[1]
    rf_v = rf.values[start:stop]
    rows = []
    for a in r.columns:
        for c in candidates(a, cfg):
            sim = simulate(positions_for(c, fc, sigma, vol_base, start, stop, cfg),
                           r[a].values[start:stop], rf_v, cfg, capital)
            rows.append({**c, **trade_metrics(sim, capital, rf_v)})
    every = pd.DataFrame(rows)
    every["_rank"] = every["sharpe"].fillna(-np.inf)
    bh = every[every["model"] == BUY_HOLD]
    others = every[every["model"] != BUY_HOLD]
    best = others.loc[others.groupby(["asset", "model"], sort=False)["_rank"].idxmax()].copy()

    rule = cfg["selection"]
    best["sharpe_bh"] = best["asset"].map(bh.set_index("asset")["sharpe"])
    beats = (best["sharpe"].fillna(-np.inf) >= best["sharpe_bh"] + rule["sharpe_margin"]).values
    enough = (~best["model"].isin(signal_models(cfg)) | (best["trades"] >= rule["min_trades"])).values
    best["accepted"] = beats & enough
    best["reason"] = np.select([~beats, ~enough], ["below Sharpe margin", "too few trades"],
                               default="accepted")
    selection = []
    for a in r.columns:
        ok = best[(best["asset"] == a) & best["accepted"]]
        selection.append(ok.loc[ok["_rank"].idxmax()] if len(ok) else bh[bh["asset"] == a].iloc[0])
    selection = pd.DataFrame(selection).reset_index(drop=True)

    order = {a: i for i, a in enumerate(r.columns)}
    view = pd.concat([bh.assign(reason="reference"), best])
    view = view.sort_values("asset", key=lambda s: s.map(order), kind="stable")
    return best, selection, every.drop(columns="_rank"), view


def order_log(sim, exec_dates, exec_prices, capital, asset, strategy):
    rows, before = [], 0.0
    value_before = np.r_[capital, sim["equity"][:-1]]
    for i, p in enumerate(sim["pos"]):
        if abs(p - before) > 1e-12:
            kind = "BUY" if before == 0 else ("SELL" if p == 0 else "RESIZE")
            rows.append({"date": exec_dates[i].date(), "asset": asset, "strategy": strategy,
                         "order": kind, "exposure_before": before, "exposure_after": p,
                         "price": exec_prices[i], "sleeve_value": value_before[i]})
        before = p
    return rows


def run_test(fc, sigma, vol_base, r, rf, prices, split, best, selection, cfg):
    start, stop = split
    capital = cfg["strategy"]["initial_capital"] / r.shape[1]
    rf_t = rf.values[start:stop]
    exec_dates = r.index[start - 1:stop - 1]  # day t's order executes at the close of t-1

    def run(row):
        pos = positions_for(row, fc, sigma, vol_base, start, stop, cfg)
        return simulate(pos, r[row["asset"]].values[start:stop], rf_t, cfg, capital)

    all_test = pd.DataFrame([{"asset": b["asset"], "model": b["model"], "threshold": b["threshold"],
                              "vol_factor": b["vol_factor"], **trade_metrics(run(b), capital, rf_t)}
                             for _, b in best.iterrows()]).set_index(["asset", "model"])
    vm = best[best["model"] == VOL_MANAGED].set_index("asset")
    sims, rows, orders = {"selected": {}, "vol": {}, "bh": {}}, [], []
    for _, row in selection.iterrows():
        a = row["asset"]
        sims["selected"][a] = run(row)
        sims["vol"][a] = run({**vm.loc[a].to_dict(), "asset": a})
        sims["bh"][a] = run({"asset": a, "model": BUY_HOLD})
        rows.append({"asset": a, "strategy": label(row),
                     **trade_metrics(sims["selected"][a], capital, rf_t)})
        if row["model"] != BUY_HOLD:
            rows.append({"asset": a, "strategy": "Buy & hold",
                         **trade_metrics(sims["bh"][a], capital, rf_t)})
        orders += order_log(sims["selected"][a], exec_dates, prices.loc[exec_dates, a].values,
                            capital, a, label(row))
    names = {"selected": SELECTED, "vol": VOL_MANAGED, "bh": BUY_HOLD}
    return {"all": all_test, "per_asset": pd.DataFrame(rows).set_index(["asset", "strategy"]),
            "sims": sims, "orders": pd.DataFrame(orders), "dates": r.index[start:stop], "vol": vm,
            "curves": {names[k]: np.sum([s["equity"] for s in v.values()], axis=0) for k, v in sims.items()},
            "costs": {names[k]: sum(s["cost_usd"].sum() for s in v.values()) for k, v in sims.items()}}


def per_asset_tests(res, selection, rf, cfg):
    """Vol. managed (and the selected strategy, if different) vs buy & hold, per asset."""
    capital = cfg["strategy"]["initial_capital"] / len(selection)

    def excess(sim):
        return sim["equity"] / np.r_[capital, sim["equity"][:-1]] - 1 - rf

    rows = []
    for _, row in selection.iterrows():
        a = row["asset"]
        tests = {VOL_MANAGED: res["sims"]["vol"][a]}
        if row["model"] not in (BUY_HOLD, VOL_MANAGED):
            tests[label(row)] = res["sims"]["selected"][a]
        for name, sim in tests.items():
            rows.append({"asset": a, "strategy": name,
                         **compare(excess(sim), excess(res["sims"]["bh"][a]), cfg)})
    return add_holm(pd.DataFrame(rows).set_index(["asset", "strategy"]))


def collect_strategies(res, alloc):
    """Equity curves and costs of every portfolio-level strategy, in a fixed order."""
    out = {n: {"equity": res["curves"][n], "costs": res["costs"][n]} for n in (BUY_HOLD, SELECTED, VOL_MANAGED)}
    out.update({n: {"equity": alloc[n]["equity"], "costs": alloc[n]["costs"]} for n in ALLOCATION_RULES})
    return {n: out[n] for n in STRATEGY_ORDER}


def returns_frame(strategies, dates, capital):
    return pd.DataFrame({n: s["equity"] / np.r_[capital, s["equity"][:-1]] - 1
                         for n, s in strategies.items()}, index=dates)


def summary_table(strategies, capital, rf):
    return pd.DataFrame({n: basic_metrics(s["equity"], capital, rf, s["costs"])
                         for n, s in strategies.items()})
