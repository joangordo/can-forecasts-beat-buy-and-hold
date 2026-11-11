"""Download (with cache), log returns and the chronological train/validation/test split."""
import logging
import time

import numpy as np
import pandas as pd

from .config import ROOT

log = logging.getLogger("beatbh")


def _cache_dir(cfg):
    path = ROOT / cfg["data"]["cache_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _use_private_yf_cache(cfg):
    """yfinance keeps a small SQLite cache; a private copy avoids 'database is locked' on Windows."""
    try:
        import yfinance as yf

        yf.set_tz_cache_location(str(_cache_dir(cfg) / "yfinance_cache"))
    except Exception:
        pass


def _download_close(ticker, start, end, retries=3):
    """Adjusted close of a single ticker (one at a time, with retries)."""
    import yfinance as yf

    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                             progress=False, threads=False)
            close = df["Close"] if df is not None and not df.empty else None
            if isinstance(close, pd.DataFrame):  # (Price, Ticker) column layout
                close = close.iloc[:, 0]
            if close is not None and close.notna().any():
                close = close.dropna()
                if close.index.tz is not None:
                    close.index = close.index.tz_localize(None)
                return close
        except Exception as exc:
            log.info(f"  {ticker}: attempt {attempt} failed ({exc})")
        if attempt < retries:
            time.sleep(3 * attempt)
    raise RuntimeError(f"Could not download {ticker} after {retries} attempts. Yahoo "
                       "rate-limits requests: wait a few minutes and run again.")


def _covers(obj, start, end):
    return obj.index.min() <= start + pd.Timedelta(days=10) and obj.index.max() >= end - pd.Timedelta(days=10)


def _incomplete(prices, tickers, min_coverage=0.9):
    if prices is None or prices.empty:
        return list(tickers)
    return [t for t in tickers if t not in prices or prices[t].notna().mean() < min_coverage]


def load_prices(cfg):
    d = cfg["data"]
    tickers, start, end = d["tickers"], pd.Timestamp(d["start"]), pd.Timestamp(d["end"])
    path = _cache_dir(cfg) / "prices.csv"
    if path.exists():
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if _covers(cached, start, end) and not _incomplete(cached.loc[start:end], tickers):
            log.info(f"Prices loaded from cache ({path.name})")
            return cached.loc[start:end, tickers]
        log.info("Cached prices do not cover the configuration: downloading again.")
    _use_private_yf_cache(cfg)
    log.info("Downloading prices from Yahoo Finance...")
    end_exclusive = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prices = {}
    for t in tickers:
        prices[t] = _download_close(t, d["start"], end_exclusive)
        log.info(f"  {t}: {len(prices[t])} days")
        time.sleep(1)
    prices = pd.DataFrame(prices)
    missing = _incomplete(prices, tickers)
    if missing:  # never cache incomplete data
        raise RuntimeError(f"Incomplete data for {missing}. Run again in a few minutes.")
    prices.to_csv(path)
    return prices.loc[start:end, tickers]


def load_risk_free(cfg, index):
    """Daily risk-free return (decimal) on `index`, from the 13-week T-bill yield.
    Falls back to 0 % (with a warning) if it cannot be obtained."""
    ticker = cfg["data"].get("risk_free_ticker")
    zero = pd.Series(0.0, index=index)
    if not ticker:
        log.info("Risk-free rate disabled: cash earns 0 %.")
        return zero
    path = _cache_dir(cfg) / "risk_free.csv"
    yields = None
    if path.exists():
        cached = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        if _covers(cached, index[0], index[-1]):
            yields = cached
    if yields is None:
        try:
            _use_private_yf_cache(cfg)
            end_exclusive = (index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            yields = _download_close(ticker, index[0].strftime("%Y-%m-%d"), end_exclusive)
            yields.rename(ticker).to_csv(path)
        except Exception as exc:
            log.info(f"WARNING: risk-free rate not available ({exc}). Cash earns 0 %.")
            return zero
    yields = yields.reindex(yields.index.union(index)).ffill().reindex(index).bfill()
    return yields / 100 / 252


def prepare(prices):
    """Keep days on which every asset trades; daily log returns in %."""
    n0 = len(prices)
    prices = prices.dropna()
    log.info(f"Days with all prices: {len(prices)} ({n0 - len(prices)} dropped)")
    return prices, 100 * np.log(prices).diff().dropna()


def split_indices(index, train_end, validation_end, min_days=252):
    """[start, stop) positions of each split; the cut-off date belongs to the earlier split."""
    i_val = index.searchsorted(pd.Timestamp(train_end), side="right")
    i_test = index.searchsorted(pd.Timestamp(validation_end), side="right")
    splits = {"train": (0, i_val), "validation": (i_val, i_test), "test": (i_test, len(index))}
    for name, (a, b) in splits.items():
        if b - a < min_days:
            raise ValueError(f"The {name} split has fewer than {min_days} trading days.")
    return splits
