"""Entry point: python -m beatbh [--quick] [--no-robustness] [--config FILE]."""
import argparse
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd

from . import econometrics as econ
from . import plots
from .config import ROOT, load_config
from .data import load_prices, load_risk_free, prepare, split_indices
from .evaluation import (SELECTED, collect_strategies, label, per_asset_tests, returns_frame,
                         run_test, short_label, summary_table, tune_on_validation)
from .models import (PROB_MODELS, RETURN_MODELS, evaluate_forecasts, generate_forecasts,
                     select_specification)
from .report import Reporter, inject_readme, md_table
from .robustness import run_robustness, summarize_robustness
from .sharpe_tests import sharpe_tests
from .strategies import BUY_HOLD, allocation_rules, historical_vol

VALIDATION_COLS = ["threshold", "vol_factor", "final_value", "total_return_%", "sharpe",
                   "max_drawdown_%", "trades", "avg_exposure_%"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Can econometric forecasts beat buy & hold?")
    parser.add_argument("--config", help="alternative YAML configuration file")
    parser.add_argument("--quick", action="store_true",
                        help="fast end-to-end check; output goes to reports_quick/ and is not published")
    parser.add_argument("--no-robustness", action="store_true", help="skip the multi-window analysis")
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore")  # estimation routines emit many harmless convergence warnings
    cfg = load_config(args.config, quick=args.quick)
    if args.no_robustness:
        cfg["robustness"]["enabled"] = False

    out_dir = ROOT / (cfg["output_dir"] + ("_quick" if cfg["quick"] else ""))
    rep, t0 = Reporter(out_dir), time.time()
    figs, capital = rep.fig_dir, cfg["strategy"]["initial_capital"]
    if cfg["quick"]:
        rep.text("QUICK RUN: coarser settings; results are not published.")

    rep.section("1. DATA")
    prices, r = prepare(load_prices(cfg))
    rf = load_risk_free(cfg, r.index)
    rep.text(f"Average risk-free rate: {100 * 252 * rf.mean():.2f} % per year")

    rep.section("2. CHRONOLOGICAL SPLIT (no shuffling)")
    splits = split_indices(r.index, cfg["split"]["train_end"], cfg["split"]["validation_end"])
    for name, (a, b) in splits.items():
        rep.text(f"{name:<11} {r.index[a].date()} -> {r.index[b - 1].date()}  ({b - a} days)")
    plots.prices(prices, r, splits, figs)
    plots.returns(r, splits, figs)

    rep.section("3. ECONOMETRIC ANALYSIS (training data only)")
    a, b = splits["train"]
    r_tr = r.iloc[a:b]
    lp_tr = np.log(prices).loc[:r_tr.index[-1]]
    rep.text("\n3.1 Daily log returns (%); mean and volatility annualised")
    rep.table(econ.descriptives(r_tr), "03_1_descriptives")
    rep.text("\n3.2 Stationarity. ADF H0: unit root. KPSS H0: stationary "
             "(statsmodels bounds the KPSS p-value to [0.01, 0.10])")
    rep.table(econ.stationarity(lp_tr, r_tr), "03_2_stationarity")
    rep.text("\n3.3 Serial dependence, 10 lags. Returns: memory in the mean (what ARIMA exploits). "
             "Squared returns and ARCH-LM: memory in volatility (what GARCH models)")
    rep.table(econ.dependence(r_tr), "03_3_dependence")
    corr = r_tr.corr()
    rep.text("\n3.4 Return correlations")
    rep.table(corr, "03_4_correlations")
    granger, k = econ.granger(r_tr, cfg["models"]["var_max_lags"])
    rep.text(f"\n3.5 Granger causality p-values, VAR({k})")
    rep.table(granger, "03_5_granger")
    jo, rank = econ.johansen(lp_tr)
    rep.text("\n3.6 Johansen trace test on log prices")
    rep.table(jo, "03_6_johansen")
    rep.text(f"Estimated number of cointegrating relations: {rank}")
    rep.text("\nEngle-Granger, pairwise")
    rep.table(econ.engle_granger(lp_tr), "03_7_engle_granger")
    plots.acf(r_tr, figs)
    plots.correlation(corr, figs)

    rep.section("4. FORECASTS (specification chosen on train, then frozen)")
    start, stop = splits["validation"][0], splits["test"][1]
    spec, spec_table = select_specification(r.iloc[:start], cfg)
    rep.table(spec_table, "04_specification")
    rep.text(f"VAR lags (AIC): {spec['var']}. Persistence alpha + beta close to 1 means "
             "long-lived volatility shocks.")
    fc, sigma = generate_forecasts(r, start, stop, cfg, spec)
    vol_base = historical_vol(r)
    rep.text("\nForecast accuracy on validation. DM < 0 with DM_p < 0.05: better than a random walk. "
             "brier_skill > 0: the classifier beats the historical frequency of up days.")
    rep.table(evaluate_forecasts(fc, r, splits["validation"]), "04_forecast_accuracy_validation", 4)

    rep.section("5. STRATEGY SELECTION ON VALIDATION")
    c, rule = cfg["costs"], cfg["selection"]
    rep.text(f"Costs: {100 * c['commission']:.2f} % commission + {100 * c['slippage']:.2f} % slippage "
             "per traded dollar. Cash earns the T-bill rate. Sharpe ratios use excess returns.")
    rep.text(f"A strategy replaces buy & hold only if its validation Sharpe beats buy & hold's by "
             f"{rule['sharpe_margin']:.2f} and, for signal rules, it completed >= {rule['min_trades']} trades.")
    best, selection, every, view = tune_on_validation(fc, sigma, vol_base, r, rf, splits["validation"], cfg)
    rep.table(every.set_index(["asset", "model"]), "05_validation_all_candidates", show=False)
    rep.text("\nBest parameter of each strategy, per asset")
    rep.table(view.set_index(["asset", "model"])[VALIDATION_COLS + ["reason"]], "05_validation_best")
    rep.text("\nSelected strategy per asset")
    rep.table(selection.set_index("asset")[["model"] + VALIDATION_COLS], "05_validation_selected")

    rep.section("6. OUT-OF-SAMPLE TEST, PER ASSET")
    rep.text("Forecast accuracy on test")
    accuracy = evaluate_forecasts(fc, r, splits["test"])
    rep.table(accuracy, "06_forecast_accuracy_test", 4)
    res = run_test(fc, sigma, vol_base, r, rf, prices, splits["test"], best, selection, cfg)
    rep.text("\nEvery strategy on test with its validation parameter "
             "(information only: picking the best one here would be cheating)")
    rep.table(res["all"][["threshold", "vol_factor", "final_value", "total_return_%", "sharpe",
                          "max_drawdown_%", "trades"]], "06_test_all_strategies")
    rep.text("\nSelected strategy vs buy & hold, per asset")
    rep.table(res["per_asset"], "06_test_selected_vs_buy_hold")
    res["orders"].to_csv(rep.tab_dir / "06_test_orders.csv", index=False)
    plots.forecast_scatter(fc, r, splits["test"], best, figs)
    plots.equity_by_asset(res, selection, figs)
    plots.vol_exposure(res, sigma, figs)

    rep.section("7. ASSET ALLOCATION (fixed rules, monthly rebalancing)")
    rep.text("Equal weight (monthly):    1/N in each asset, rebalanced monthly.")
    rep.text("Inverse vol (GARCH):       weights proportional to 1 / GARCH volatility forecast.")
    rep.text("Risk parity (GARCH):       equal risk contributions (GARCH volatility, 1-year correlations).")
    rep.text("Risk parity (historical):  same with 3-month realised volatility: does GARCH add anything?")
    (va, vb), (ta, tb) = splits["validation"], splits["test"]
    alloc_val = allocation_rules(r, sigma, splits["validation"], cfg)
    alloc_test = allocation_rules(r, sigma, splits["test"], cfg)
    rep.text("\nValidation")
    rep.table(summary_table(alloc_val, capital, rf.values[va:vb]), "07_allocation_validation")
    rep.text("\nTest")
    rep.table(summary_table(alloc_test, capital, rf.values[ta:tb]), "07_allocation_test")
    rep.text("\nAverage weights on test (%)")
    rep.table(pd.DataFrame({n: 100 * s["weights"].mean(axis=0) for n, s in alloc_test.items()},
                           index=r.columns).T, "07_allocation_weights_test", 1)
    plots.allocation_weights(alloc_test, res["dates"], list(r.columns), figs)

    rep.section("8. RESULTS ON THE TEST PERIOD")
    strategies = collect_strategies(res, alloc_test)
    rets = returns_frame(strategies, res["dates"], capital)
    rf_test = rf.values[ta:tb]
    rep.text("Strategy selected on validation, per asset:")
    for _, row in selection.iterrows():
        rep.text(f"  {row['asset']:<5} -> {label(row)}")
    summary = summary_table(strategies, capital, rf_test)
    rep.text("")
    rep.table(summary, "08_test_summary")
    rep.text("\nIs the Sharpe difference vs buy & hold significant? p_hac: Ledoit-Wolf HAC test. "
             "p_hac_holm: adjusted for testing several strategies. p_bootstrap: circular block bootstrap.")
    tests = sharpe_tests(rets, rf_test, cfg)
    rep.table(tests, "08_test_sharpe_tests")
    rep.text("\nPer asset")
    rep.table(per_asset_tests(res, selection, rf_test, cfg), "08_test_sharpe_tests_per_asset")
    plots.portfolio(rets, capital, figs / "09_portfolio_test.png", "Portfolio value on the test period (USD)")

    rep.section("9. ROBUSTNESS: THE WHOLE PROCEDURE ON SEVERAL TEST WINDOWS")
    rob = None
    if cfg["robustness"]["enabled"]:
        rob = summarize_robustness(run_robustness(r, prices, rf, cfg), cfg)
        rep.text("\nStrategy selected on validation, per window")
        rep.table(rob["selection"], "09_robustness_selection")
        rep.text("\nSharpe ratio per test window")
        rep.table(rob["sharpe"], "09_robustness_sharpe", 2)
        rep.text("\nCAGR (%) per test window")
        rep.table(rob["cagr"], "09_robustness_cagr", 1)
        s = rob["stitched"]
        rep.text(f"\nAll test windows stitched ({s.index[0].date()} -> {s.index[-1].date()}, {len(s)} "
                 "out-of-sample days). Models and strategies are re-selected in every window.")
        rep.table(rob["summary"], "09_robustness_stitched")
        rep.text("\nSharpe tests on the stitched record")
        rep.table(rob["tests"], "09_robustness_sharpe_tests")
        plots.portfolio(s, capital, figs / "10_robustness_equity.png",
                        "All test windows stitched together (USD)", cuts=rob["starts"])
        plots.robustness_sharpe(rob["sharpe_raw"], figs)
    else:
        rep.text("Skipped (robustness.enabled = false or --no-robustness).")

    context = dict(cfg=cfg, r=r, splits=splits, accuracy=accuracy, selection=selection,
                   summary=summary, tests=tests, rob=rob)
    (out_dir / "results.md").write_text(results_markdown(context, "figures/"), encoding="utf-8")
    if cfg["quick"] or rob is None:
        rep.text("\nREADME.md not updated: only a full run (with robustness) publishes results.")
    elif inject_readme(results_markdown(context, f"{cfg['output_dir']}/figures/")):
        rep.text("\nREADME.md results section updated.")
    rep.text("\nNot investment advice: historical results of an educational project.")
    rep.text(f"\nTotal time: {(time.time() - t0) / 60:.1f} min. Output: {out_dir}")


def _names(df, column):
    return ", ".join(f"{n} ({row[column]:+.2f})" for n, row in df.iterrows()) or "none"


def results_markdown(ctx, prefix):
    """Results section of the README. Every sentence is computed from the run."""
    cfg, r, splits, tests, rob = ctx["cfg"], ctx["r"], ctx["splits"], ctx["tests"], ctx["rob"]
    ta, tb = splits["test"]
    d0, d1 = r.index[ta].date(), r.index[tb - 1].date()
    capital = cfg["strategy"]["initial_capital"]
    acc = ctx["accuracy"]
    models = acc.index.get_level_values("model")

    econ = acc[models.isin(RETURN_MODELS + ["Historical mean"])]
    better = econ[(econ["DM_vs_RW"] < 0) & (econ["DM_p"] < 0.05)]
    lines = [f"_Generated by `python -m beatbh` on {date.today()}. Out-of-sample test period: "
             f"{d0} to {d1}. USD {capital:,.0f} initial capital, costs included, cash earns T-bills, "
             f"Sharpe ratios in excess of T-bills._", "", "**Key results**", "",
             "- Return forecasts significantly more accurate than a random walk on the test period "
             "(Diebold-Mariano, 5 %): "
             + (", ".join(f"{m} on {a}" for a, m in better.index) or "none")
             + f", out of {len(econ)} model-asset pairs."]
    if cfg["models"]["use_ml"]:
        hits = acc[models.isin(PROB_MODELS)]["hit_rate_%"]
        base = 100 * (r.iloc[ta:tb].values > 0).mean()
        lines.append(f"- Direction classifiers (logistic regression, gradient boosting) called the sign of "
                     f"the next day's return correctly {hits.min():.1f}-{hits.max():.1f} % of the time; "
                     f"always answering 'up' scored {base:.1f} %.")
    chosen = ", ".join(f"{row['asset']}: {short_label(row)}" for _, row in ctx["selection"].iterrows())
    sel = tests.loc[SELECTED]
    lines += [f"- Strategy selected on validation ({chosen}) vs buy & hold on the test period: Sharpe "
              f"{sel['sharpe']:.2f} vs {sel['sharpe_bh']:.2f} (Holm-adjusted p = {sel['p_hac_holm']:.2f}).",
              "- Strategies whose Sharpe ratio differs significantly from buy & hold on the test period "
              "(HAC test, Holm-adjusted, 5 %): " + _names(tests[tests["p_hac_holm"] < 0.05], "delta_sharpe") + "."]
    if rob is not None:
        raw, s, st = rob["sharpe_raw"], rob["stitched"], rob["tests"]
        lines += [f"- Repeating everything on {raw.shape[1]} consecutive test windows "
                  f"({s.index[0].year}-{s.index[-1].year}), the selected strategy beat buy & hold's Sharpe "
                  f"in {rob['selected_wins']} of {raw.shape[1]}; on the stitched {len(s):,}-day out-of-sample "
                  f"record the difference is {st.loc[SELECTED, 'delta_sharpe']:+.2f} "
                  f"(Holm-adjusted p = {st.loc[SELECTED, 'p_hac_holm']:.2f}).",
                  "- Significant differences on the stitched record (Holm, 5 %): "
                  + _names(st[st["p_hac_holm"] < 0.05], "delta_sharpe") + "."]

    def table(summary, t):
        df = summary.T[["CAGR_%", "volatility_%", "sharpe", "max_drawdown_%"]].join(
            t[["delta_sharpe", "p_hac_holm", "p_bootstrap"]])
        df.columns = ["CAGR %", "Volatility %", "Sharpe", "Max drawdown %", "ΔSharpe vs B&H",
                      "p (HAC, Holm)", "p (bootstrap)"]
        df.index.name = "Strategy"
        return md_table(df)

    lines += ["", f"**Test period, {d0} to {d1}**", "", table(ctx["summary"], tests), "",
              f"![Portfolio value on the test period]({prefix}09_portfolio_test.png)"]
    if rob is not None:
        lines += ["", "**Robustness: the whole procedure repeated on consecutive test windows**", "",
                  "Sharpe ratio per test window (each window has its own train and validation periods):", "",
                  md_table(rob["sharpe"]), "", "All windows stitched together:", "",
                  table(rob["summary"], rob["tests"]), "",
                  f"![Stitched out-of-sample record]({prefix}10_robustness_equity.png)"]
    return "\n".join(lines)
