"""Repeat the whole procedure (orders, validation, selection, test) on several windows."""
import logging
import os
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .data import split_indices
from .evaluation import (SELECTED, collect_strategies, returns_frame, run_test,
                         short_label, tune_on_validation)
from .models import generate_forecasts, select_specification
from .report import quiet
from .sharpe_tests import sharpe, sharpe_tests
from .strategies import BUY_HOLD, allocation_rules, basic_metrics, historical_vol

log = logging.getLogger("beatbh")


def analyze_window(r, prices, rf, window, cfg):
    t0 = time.time()
    train_end, validation_end, test_end = window
    rv = r.loc[:test_end]
    rf_v = rf.loc[rv.index]
    splits = split_indices(rv.index, train_end, validation_end)
    start, stop = splits["validation"][0], splits["test"][1]
    spec, _ = select_specification(rv.iloc[:start], cfg)
    fc, sigma = generate_forecasts(rv, start, stop, cfg, spec)
    vol_base = historical_vol(rv)
    best, selection, _, _ = tune_on_validation(fc, sigma, vol_base, rv, rf_v, splits["validation"], cfg)
    res = run_test(fc, sigma, vol_base, rv, rf_v, prices, splits["test"], best, selection, cfg)
    strategies = collect_strategies(res, allocation_rules(rv, sigma, splits["test"], cfg))
    dates = res["dates"]
    return {"window": f"{dates[0].year}-{dates[-1].year}",
            "returns": returns_frame(strategies, dates, cfg["strategy"]["initial_capital"]),
            "rf": rf_v.loc[dates],
            "selection": {row["asset"]: short_label(row) for _, row in selection.iterrows()},
            "seconds": time.time() - t0}


def _worker(r, prices, rf, window, cfg):
    """Runs in a separate process: silent and single-threaded (avoids oversubscription)."""
    from threadpoolctl import threadpool_limits

    logging.getLogger("beatbh").disabled = True
    warnings.simplefilter("ignore")
    with threadpool_limits(limits=1):
        return analyze_window(r, prices, rf, window, cfg)


def run_robustness(r, prices, rf, cfg):
    windows = [tuple(w) for w in cfg["robustness"]["windows"]]
    n_proc = cfg["robustness"]["processes"]
    if n_proc == "auto":
        n_proc = max(1, min(len(windows), (os.cpu_count() or 2) - 1))
    log.info(f"{len(windows)} windows, {n_proc} parallel process(es). This can take a while.")
    done = {}
    if n_proc > 1:
        try:
            with ProcessPoolExecutor(max_workers=n_proc) as pool:
                futures = {pool.submit(_worker, r, prices, rf, w, cfg): w for w in windows}
                for future in as_completed(futures):
                    res = future.result()
                    done[futures[future]] = res
                    log.info(f"  Test {res['window']} done ({res['seconds'] / 60:.1f} min)")
        except Exception as exc:
            log.info(f"  Parallel run failed ({exc}); continuing sequentially.")
    for w in windows:
        if w not in done:
            with quiet():
                res = analyze_window(r, prices, rf, w, cfg)
            done[w] = res
            log.info(f"  Test {res['window']} done ({res['seconds'] / 60:.1f} min)")
    return [done[w] for w in windows]


def summarize_robustness(results, cfg):
    capital, n = cfg["strategy"]["initial_capital"], len(results)
    selection = pd.DataFrame({res["window"]: res["selection"] for res in results}).T
    raw = pd.DataFrame({res["window"]: res["returns"].sub(res["rf"].values, axis=0)
                        .apply(lambda x: np.sqrt(252) * sharpe(x.values)) for res in results})
    cagr = pd.DataFrame({res["window"]: 100 * ((1 + res["returns"]).prod() ** (252 / len(res["returns"])) - 1)
                         for res in results})
    table = raw.copy()
    table["mean"] = raw.mean(axis=1)
    table["beats_bh"] = [f"{int(v)} of {n}" for v in (raw.sub(raw.loc[BUY_HOLD], axis=1) > 0).sum(axis=1)]
    table.loc[BUY_HOLD, "beats_bh"] = "-"
    for df in (table, raw, cagr):
        df.index.name = "Strategy"

    stitched = pd.concat([res["returns"] for res in results]).sort_index()
    stitched = stitched[~stitched.index.duplicated()]
    rf = pd.concat([res["rf"] for res in results]).sort_index()
    rf = rf[~rf.index.duplicated()].reindex(stitched.index)
    summary = pd.DataFrame({n: basic_metrics(capital * np.cumprod(1 + stitched[n].values), capital, rf.values)
                            for n in stitched}).drop(index="costs")
    return {"selection": selection, "sharpe": table, "sharpe_raw": raw, "cagr": cagr,
            "stitched": stitched, "rf": rf, "summary": summary,
            "tests": sharpe_tests(stitched, rf.values, cfg),
            "starts": [res["returns"].index[0] for res in results],
            "selected_wins": int((raw.loc[SELECTED] > raw.loc[BUY_HOLD]).sum())}
