#!/usr/bin/env python3
"""
Pairs Trading Strategy — Statistical Arbitrage System
======================================================
Uses Yahoo Finance data (historical + near real-time) to identify
cointegrated stock pairs, compute z-scores, and generate live
trading signals based on mean-reversion dynamics.
"""

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

# ─────────────────────────────────────────────────────────────────────────────
# Configuration defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_TICKER_X = "AAPL"
DEFAULT_TICKER_Y = "MSFT"
DEFAULT_LOOKBACK = "1y"
DEFAULT_ROLLING_WINDOW = 20
DEFAULT_REFRESH_INTERVAL = 60  # seconds
DEFAULT_Z_ENTRY = 1.0
DEFAULT_Z_EXIT = 0.5
DEFAULT_STOP_LOSS_Z = 3.0
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "signals.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("PairsTrader")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    return logger


log = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# CSV signal log
# ─────────────────────────────────────────────────────────────────────────────
def init_csv_log():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "ticker_x", "price_x",
                "ticker_y", "price_y", "hedge_ratio",
                "spread", "z_score", "signal", "pnl_cumulative",
            ])


def append_csv_log(row: list):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Fetching
# ─────────────────────────────────────────────────────────────────────────────
def fetch_historical_data(
    ticker_x: str,
    ticker_y: str,
    period: str = DEFAULT_LOOKBACK,
) -> pd.DataFrame:
    """Download adjusted close prices for the pair over *period*."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(
                "Fetching %s of daily data for %s & %s (attempt %d/%d)",
                period, ticker_x, ticker_y, attempt, MAX_RETRIES,
            )
            raw = yf.download(
                [ticker_x, ticker_y],
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                raise ValueError("yfinance returned an empty DataFrame")

            df = raw["Close"][[ticker_x, ticker_y]].copy()
            df.columns = [ticker_x, ticker_y]
            df.index = pd.to_datetime(df.index)
            df = df.ffill().dropna()

            if len(df) < DEFAULT_ROLLING_WINDOW + 10:
                raise ValueError(
                    f"Insufficient data: got {len(df)} rows, "
                    f"need at least {DEFAULT_ROLLING_WINDOW + 10}"
                )

            log.info("Loaded %d daily bars", len(df))
            return df

        except Exception as exc:
            log.warning("Fetch failed: %s", exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(
                    "Could not fetch historical data after "
                    f"{MAX_RETRIES} attempts"
                ) from exc


def fetch_live_data(
    ticker_x: str,
    ticker_y: str,
) -> tuple[float, float, pd.Timestamp] | None:
    """Return the latest (price_x, price_y, timestamp) from intraday feed."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = yf.download(
                [ticker_x, ticker_y],
                period="1d",
                interval="1m",
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                raise ValueError("Empty intraday response")

            close = raw["Close"][[ticker_x, ticker_y]]
            last = close.dropna().iloc[-1]
            ts = close.dropna().index[-1]
            return float(last[ticker_x]), float(last[ticker_y]), pd.Timestamp(ts)

        except Exception as exc:
            log.warning("Live fetch attempt %d failed: %s", attempt, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    log.error("All live-data fetch attempts failed — skipping this tick")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hedge Ratio (OLS)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS regression of Y on X (with constant); returns (beta, alpha)."""
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    alpha = float(model.params.iloc[0])
    beta = float(model.params.iloc[1])
    log.info("Hedge ratio (β): %.6f  |  Intercept (α): %.6f  |  R²: %.4f",
             beta, alpha, model.rsquared)
    return beta, alpha


# ─────────────────────────────────────────────────────────────────────────────
# 3. Spread
# ─────────────────────────────────────────────────────────────────────────────
def compute_spread(
    y: pd.Series, x: pd.Series, hedge_ratio: float, intercept: float,
) -> pd.Series:
    """Spread = Y − (β·X + α)"""
    return y - (hedge_ratio * x + intercept)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ADF stationarity test
# ─────────────────────────────────────────────────────────────────────────────
def perform_adf_test(spread: pd.Series) -> dict:
    """Run ADF test and return structured results."""
    result = adfuller(spread.dropna(), autolag="AIC")
    adf_stat, p_value, _, _, critical_values, _ = result

    log.info("─── ADF Test ───")
    log.info("  Statistic : %.6f", adf_stat)
    log.info("  p-value   : %.6f", p_value)
    for level, cv in critical_values.items():
        log.info("  Critical (%s): %.6f", level, cv)

    stationary = p_value < 0.05
    verdict = "STATIONARY ✓ — valid pair" if stationary else "NON-STATIONARY ✗ — pair may be unreliable"
    log.info("  Verdict   : %s", verdict)

    return {
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "critical_values": critical_values,
        "is_stationary": stationary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Z-Score
# ─────────────────────────────────────────────────────────────────────────────
def compute_zscore(spread: pd.Series, window: int = DEFAULT_ROLLING_WINDOW) -> pd.Series:
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    return (spread - rolling_mean) / rolling_std


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dataset update
# ─────────────────────────────────────────────────────────────────────────────
def update_dataset(
    df: pd.DataFrame,
    ticker_x: str,
    ticker_y: str,
    price_x: float,
    price_y: float,
    ts: pd.Timestamp,
) -> pd.DataFrame:
    """Append a new observation, avoiding duplicates."""
    new_row = pd.DataFrame(
        {ticker_x: [price_x], ticker_y: [price_y]},
        index=[ts],
    )
    df = pd.concat([df, new_row])
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Signal generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_signal(
    z: float,
    z_entry: float = DEFAULT_Z_ENTRY,
    z_exit: float = DEFAULT_Z_EXIT,
    z_stop: float = DEFAULT_STOP_LOSS_Z,
) -> str:
    """
    Map a z-score to a trading signal.

    |z| > z_stop → STOP_LOSS  (emergency exit)
    z   > z_entry → SHORT     (sell Y, buy X)
    z   < -z_entry → LONG     (buy Y, sell X)
    |z| < z_exit → EXIT       (close position)
    otherwise    → HOLD
    """
    abs_z = abs(z)
    if abs_z > z_stop:
        return "STOP_LOSS"
    if z > z_entry:
        return "SHORT"
    if z < -z_entry:
        return "LONG"
    if abs_z < z_exit:
        return "EXIT"
    return "HOLD"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Simple PnL tracker
# ─────────────────────────────────────────────────────────────────────────────
class PnLTracker:
    """Tracks a trivially simple mark-to-market PnL on the spread."""

    def __init__(self):
        self.position = 0       # +1 long spread, -1 short spread, 0 flat
        self.entry_spread = 0.0
        self.cumulative_pnl = 0.0

    def update(self, signal: str, current_spread: float) -> float:
        if signal in ("EXIT", "STOP_LOSS") and self.position != 0:
            pnl = self.position * (current_spread - self.entry_spread)
            self.cumulative_pnl += pnl
            self.position = 0
            self.entry_spread = 0.0
        elif signal == "LONG" and self.position <= 0:
            if self.position == -1:
                pnl = self.position * (current_spread - self.entry_spread)
                self.cumulative_pnl += pnl
            self.position = 1
            self.entry_spread = current_spread
        elif signal == "SHORT" and self.position >= 0:
            if self.position == 1:
                pnl = self.position * (current_spread - self.entry_spread)
                self.cumulative_pnl += pnl
            self.position = -1
            self.entry_spread = current_spread

        return self.cumulative_pnl


# ─────────────────────────────────────────────────────────────────────────────
# 9. Initial analysis summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(
    df: pd.DataFrame,
    ticker_x: str,
    ticker_y: str,
    hedge_ratio: float,
    intercept: float,
    adf: dict,
):
    corr = df[ticker_x].corr(df[ticker_y])
    print("\n" + "=" * 60)
    print("         PAIRS TRADING — INITIAL ANALYSIS")
    print("=" * 60)
    print(f"  Pair          : {ticker_y} / {ticker_x}")
    print(f"  Period        : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Observations  : {len(df)}")
    print(f"  Correlation   : {corr:.4f}")
    print(f"  Hedge ratio β : {hedge_ratio:.6f}")
    print(f"  Intercept α   : {intercept:.6f}")
    print(f"  ADF statistic : {adf['adf_statistic']:.6f}")
    print(f"  ADF p-value   : {adf['p_value']:.6f}")
    status = "VALID" if adf["is_stationary"] else "QUESTIONABLE"
    print(f"  Pair status   : {status}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Live display
# ─────────────────────────────────────────────────────────────────────────────
SIGNAL_COLORS = {
    "LONG": "\033[92m",       # green
    "SHORT": "\033[91m",      # red
    "EXIT": "\033[93m",       # yellow
    "HOLD": "\033[90m",       # grey
    "STOP_LOSS": "\033[95m",  # magenta
}
RESET = "\033[0m"


def print_live_signal(
    ts: pd.Timestamp,
    ticker_x: str, price_x: float,
    ticker_y: str, price_y: float,
    z: float,
    signal: str,
    pnl: float,
):
    color = SIGNAL_COLORS.get(signal, "")
    print(
        f"[{ts:%Y-%m-%d %H:%M:%S}]  "
        f"{ticker_x}={price_x:>9.2f}  "
        f"{ticker_y}={price_y:>9.2f}  "
        f"z={z:+.4f}  "
        f"{color}Signal={signal:<10}{RESET}  "
        f"PnL={pnl:+.2f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11. Main loop
# ─────────────────────────────────────────────────────────────────────────────
def run(
    ticker_x: str,
    ticker_y: str,
    period: str,
    window: int,
    interval: int,
    z_entry: float,
    z_exit: float,
    z_stop: float,
    recalc_hedge: bool,
):
    init_csv_log()

    # ── Historical bootstrap ────────────────────────────────────────────
    df = fetch_historical_data(ticker_x, ticker_y, period)
    hedge_ratio, intercept = calculate_hedge_ratio(df[ticker_y], df[ticker_x])
    spread = compute_spread(df[ticker_y], df[ticker_x], hedge_ratio, intercept)
    adf = perform_adf_test(spread)
    print_summary(df, ticker_x, ticker_y, hedge_ratio, intercept, adf)

    if not adf["is_stationary"]:
        log.warning(
            "Spread is NOT stationary (p=%.4f). "
            "Proceeding anyway — signals may be unreliable.",
            adf["p_value"],
        )

    tracker = PnLTracker()

    # ── Live loop ───────────────────────────────────────────────────────
    log.info(
        "Entering live loop  (refresh every %ds, Ctrl+C to stop)", interval,
    )
    tick_count = 0
    hedge_recalc_every = 50  # recalc hedge ratio every N ticks

    try:
        while True:
            result = fetch_live_data(ticker_x, ticker_y)
            if result is None:
                time.sleep(interval)
                continue

            price_x, price_y, ts = result
            df = update_dataset(df, ticker_x, ticker_y, price_x, price_y, ts)

            tick_count += 1
            if recalc_hedge and tick_count % hedge_recalc_every == 0:
                hedge_ratio, intercept = calculate_hedge_ratio(df[ticker_y], df[ticker_x])
                log.info("Recalculated hedge ratio: %.6f, intercept: %.6f",
                         hedge_ratio, intercept)

            spread = compute_spread(df[ticker_y], df[ticker_x], hedge_ratio, intercept)
            z_series = compute_zscore(spread, window)
            current_z = z_series.iloc[-1]

            if np.isnan(current_z):
                log.warning("Z-score is NaN (insufficient rolling window data)")
                time.sleep(interval)
                continue

            current_spread = spread.iloc[-1]
            signal = generate_signal(current_z, z_entry, z_exit, z_stop)
            pnl = tracker.update(signal, current_spread)

            print_live_signal(
                ts, ticker_x, price_x, ticker_y, price_y,
                current_z, signal, pnl,
            )

            append_csv_log([
                ts.isoformat(), ticker_x, f"{price_x:.4f}",
                ticker_y, f"{price_y:.4f}", f"{hedge_ratio:.6f}",
                f"{current_spread:.4f}", f"{current_z:.4f}",
                signal, f"{pnl:.4f}",
            ])

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n")
        log.info("Stopped by user")
        log.info("Final cumulative PnL: %.4f", tracker.cumulative_pnl)
        log.info("Signal log saved to %s", LOG_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pairs Trading — Statistical Arbitrage System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-x", "--ticker-x", default=DEFAULT_TICKER_X,
                   help=f"Ticker X (default: {DEFAULT_TICKER_X})")
    p.add_argument("-y", "--ticker-y", default=DEFAULT_TICKER_Y,
                   help=f"Ticker Y (default: {DEFAULT_TICKER_Y})")
    p.add_argument("-p", "--period", default=DEFAULT_LOOKBACK,
                   help=f"Historical lookback (default: {DEFAULT_LOOKBACK})")
    p.add_argument("-w", "--window", type=int, default=DEFAULT_ROLLING_WINDOW,
                   help=f"Rolling z-score window (default: {DEFAULT_ROLLING_WINDOW})")
    p.add_argument("-i", "--interval", type=int, default=DEFAULT_REFRESH_INTERVAL,
                   help=f"Refresh interval in seconds (default: {DEFAULT_REFRESH_INTERVAL})")
    p.add_argument("--z-entry", type=float, default=DEFAULT_Z_ENTRY,
                   help=f"Z-score entry threshold (default: {DEFAULT_Z_ENTRY})")
    p.add_argument("--z-exit", type=float, default=DEFAULT_Z_EXIT,
                   help=f"Z-score exit threshold (default: {DEFAULT_Z_EXIT})")
    p.add_argument("--z-stop", type=float, default=DEFAULT_STOP_LOSS_Z,
                   help=f"Z-score stop-loss threshold (default: {DEFAULT_STOP_LOSS_Z})")
    p.add_argument("--recalc-hedge", action="store_true",
                   help="Periodically recalculate the hedge ratio")
    return p.parse_args()


def main():
    args = parse_args()
    run(
        ticker_x=args.ticker_x.upper(),
        ticker_y=args.ticker_y.upper(),
        period=args.period,
        window=args.window,
        interval=args.interval,
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        z_stop=args.z_stop,
        recalc_hedge=args.recalc_hedge,
    )


if __name__ == "__main__":
    main()
