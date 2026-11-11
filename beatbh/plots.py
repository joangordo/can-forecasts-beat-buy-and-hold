"""Figures, saved as PNG (never shown on screen)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf  # noqa: E402

from .evaluation import label  # noqa: E402
from .models import RETURN_MODELS  # noqa: E402
from .strategies import BUY_HOLD  # noqa: E402


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _grid(n):
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, 3.6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def _shade(ax, dates, splits):
    for name, color in (("validation", "tab:orange"), ("test", "tab:green")):
        a, b = splits[name]
        ax.axvspan(dates[a], dates[b - 1], color=color, alpha=0.10, label=name)


def prices(prices, r, splits, d):
    fig, ax = plt.subplots(figsize=(13, 5))
    (prices / prices.iloc[0] * 100).plot(ax=ax, logy=True, lw=1)
    _shade(ax, r.index, splits)
    ax.set_title("Prices rebased to 100 (log scale)")
    ax.set_xlabel("")
    ax.legend(ncol=3)
    _save(fig, d / "01_prices.png")


def returns(r, splits, d):
    fig, axes = plt.subplots(r.shape[1], 1, figsize=(13, 2.4 * r.shape[1]), sharex=True, squeeze=False)
    for ax, a in zip(axes[:, 0], r.columns):
        ax.plot(r.index, r[a], lw=0.4)
        _shade(ax, r.index, splits)
        ax.set_ylabel(f"{a} (%)")
    axes[0, 0].set_title("Daily log returns: volatility clustering")
    _save(fig, d / "02_returns.png")


def acf(r_train, d):
    fig, axes = plt.subplots(r_train.shape[1], 2, figsize=(13, 2.6 * r_train.shape[1]), squeeze=False)
    for i, a in enumerate(r_train.columns):
        plot_acf(r_train[a], lags=30, zero=False, ax=axes[i, 0], title=f"ACF of returns, {a}")
        plot_acf(r_train[a] ** 2, lags=30, zero=False, ax=axes[i, 1], title=f"ACF of squared returns, {a}")
    _save(fig, d / "03_acf_train.png")


def correlation(corr, d):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns)
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    ax.set_title("Return correlations (train)")
    _save(fig, d / "04_correlation_train.png")


def forecast_scatter(fc, r, split, best, d):
    """Forecast vs actual return for the econometric model with the best validation Sharpe."""
    start, stop = split
    econ = best[best["model"].isin(RETURN_MODELS)]
    top = econ.loc[econ.groupby("asset", sort=False)["_rank"].idxmax()]
    fig, axes = _grid(len(top))
    for ax, (_, row) in zip(axes, top.iterrows()):
        a, m = row["asset"], row["model"]
        p, y = fc[m][a].values[start:stop], r[a].values[start:stop]
        ax.scatter(p, y, s=4, alpha=0.4)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_title(f"{a} · {m}: corr(forecast, actual) = {np.corrcoef(p, y)[0, 1]:.3f}")
        ax.set_xlabel("Forecast return (%)")
        ax.set_ylabel("Actual return (%)")
    _save(fig, d / "05_forecast_vs_actual_test.png")


def equity_by_asset(res, selection, d):
    fig, axes = _grid(len(selection))
    for ax, (_, row) in zip(axes, selection.iterrows()):
        a = row["asset"]
        ax.plot(res["dates"], res["sims"]["selected"][a]["equity"], label=label(row))
        ax.plot(res["dates"], res["sims"]["vol"][a]["equity"],
                label=f"Vol. managed (factor {res['vol'].loc[a, 'vol_factor']:.2f})")
        ax.plot(res["dates"], res["sims"]["bh"][a]["equity"], label="Buy & hold", color="grey", alpha=0.8)
        ax.set_title(f"{a}: sleeve value (USD)")
        ax.legend(fontsize=8)
    _save(fig, d / "06_equity_by_asset_test.png")


def vol_exposure(res, sigma, d):
    fig, axes = _grid(len(res["sims"]["vol"]))
    for ax, (a, sim) in zip(axes, res["sims"]["vol"].items()):
        ax.fill_between(res["dates"], 100 * sim["pos"], alpha=0.35)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Exposure (%)")
        ax2 = ax.twinx()
        ax2.plot(res["dates"], sigma.loc[res["dates"], a] * np.sqrt(252), color="tab:red", lw=0.7)
        ax2.set_ylabel("GARCH vol (% p.a.)", color="tab:red")
        ax.set_title(f"{a}: vol. managed exposure (factor {res['vol'].loc[a, 'vol_factor']:.2f})")
    _save(fig, d / "07_vol_managed_exposure_test.png")


def allocation_weights(alloc, dates, assets, d):
    names = ["Inverse vol (GARCH)", "Risk parity (GARCH)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, n in zip(axes, names):
        ax.stackplot(dates, 100 * alloc[n]["weights"].T, labels=assets, alpha=0.85)
        ax.set_title(f"Weights: {n}")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("% of portfolio")
    axes[1].legend(loc="upper left", fontsize=8)
    _save(fig, d / "08_allocation_weights_test.png")


def portfolio(rets, capital, path, title, cuts=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for n in rets.columns:
        style = {"lw": 2.2, "color": "black", "zorder": 5} if n == BUY_HOLD else {"lw": 1.1}
        curve = capital * np.cumprod(1 + rets[n].values)
        ax1.plot(rets.index, curve, label=n, **style)
        full = np.r_[capital, curve]
        ax2.plot(rets.index, 100 * (full / np.maximum.accumulate(full) - 1)[1:], **{**style, "lw": 0.9})
    for c in cuts or []:
        for ax in (ax1, ax2):
            ax.axvline(c, color="grey", lw=0.6, ls=":")
    ax1.axhline(capital, color="grey", lw=0.6, ls="--")
    ax1.set_title(title)
    ax1.set_ylabel("USD")
    ax1.legend(ncol=2, fontsize=8)
    ax2.set_title("Drawdown (%)")
    _save(fig, path)


def robustness_sharpe(raw, d):
    fig, ax = plt.subplots(figsize=(13, 5))
    width, x = 0.8 / len(raw), np.arange(raw.shape[1])
    for i, n in enumerate(raw.index):
        ax.bar(x + i * width, raw.loc[n], width, label=n, **({"color": "black"} if n == BUY_HOLD else {}))
    ax.set_xticks(x + 0.4 - width / 2, raw.columns)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("Sharpe ratio (excess of T-bills)")
    ax.set_title("Sharpe ratio by test window")
    ax.legend(ncol=4, fontsize=8)
    _save(fig, d / "11_robustness_sharpe.png")
