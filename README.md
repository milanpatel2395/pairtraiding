# Pairs Trading — Statistical Arbitrage System

A production-ready Python implementation of a pairs trading strategy using Yahoo Finance data. The system bootstraps on 1 year of daily history, validates the pair with the Augmented Dickey-Fuller test, then enters a continuous loop that fetches near real-time prices and emits live trading signals.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# ── Web Dashboard (recommended) ──
streamlit run app.py

# ── CLI mode ──
python pairs_trading.py                        # default pair: AAPL / MSFT
python pairs_trading.py -x GOOG -y META        # custom pair

# CLI with all options
python pairs_trading.py \
  -x AAPL -y MSFT \
  -p 1y \
  -w 20 \
  -i 60 \
  --z-entry 1.0 \
  --z-exit 0.5 \
  --z-stop 3.0 \
  --recalc-hedge
```

## Web Dashboard

Launch the interactive Streamlit UI:

```bash
streamlit run app.py
```

The dashboard provides:
- **Sidebar controls** — configure tickers, lookback period, rolling window, z-score thresholds, and refresh interval
- **KPI cards** — correlation, hedge ratio, ADF p-value, pair validity, latest z-score
- **Interactive charts** — price history (dual-axis), spread, z-score with threshold bands, and correlation scatter plot
- **ADF details tab** — full Dickey-Fuller test output with critical values
- **Live signal monitor** — start/stop real-time feed with live z-score chart and signal cards
- **Signal log table** — scrollable history of all generated signals, downloadable as CSV

## CLI Options

| Flag | Description | Default |
|---|---|---|
| `-x` | Ticker X (independent variable) | `AAPL` |
| `-y` | Ticker Y (dependent variable) | `MSFT` |
| `-p` | Historical lookback period | `1y` |
| `-w` | Rolling z-score window | `20` |
| `-i` | Refresh interval (seconds) | `60` |
| `--z-entry` | Z-score threshold to open a position | `1.0` |
| `--z-exit` | Z-score threshold to close a position | `0.5` |
| `--z-stop` | Z-score stop-loss threshold | `3.0` |
| `--recalc-hedge` | Periodically recalculate the hedge ratio | off |

## How It Works

1. **Historical bootstrap** — downloads 1 year of daily adjusted close prices via `yfinance`.
2. **OLS hedge ratio** — regresses Y on X to find β (the hedge ratio).
3. **Spread** — computes `spread = Y − β·X`.
4. **ADF test** — checks whether the spread is stationary (p < 0.05).
5. **Z-score** — rolling z-score over a configurable window.
6. **Live loop** — polls 1-minute intraday data, appends new observations, recomputes z-score, and generates signals every *interval* seconds.

## Signal Logic

| Condition | Signal | Action |
|---|---|---|
| z > entry | **SHORT** | Sell Y, Buy X |
| z < −entry | **LONG** | Buy Y, Sell X |
| \|z\| < exit | **EXIT** | Close position |
| \|z\| > stop | **STOP_LOSS** | Emergency exit |
| otherwise | **HOLD** | No action |

## Output

- **Console** — color-coded live signals with prices, z-score, and cumulative PnL.
- **CSV log** — every tick is appended to `logs/signals.csv` for later analysis.

## Project Structure

```
Pairtrading/
├── app.py              # Streamlit web dashboard
├── pairs_trading.py    # Core engine + CLI
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── logs/
    └── signals.csv     # Auto-generated signal log (CLI mode)
```

## Dependencies

- pandas
- numpy
- yfinance
- statsmodels
- streamlit
- plotly
