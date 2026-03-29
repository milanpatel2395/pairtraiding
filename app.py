#!/usr/bin/env python3
"""
Pairs Trading — Streamlit Dashboard
====================================
Clean, focused UI for analysing stock pair relationships
using Correlation, Cointegration (ADF), and Z-Score.

Launch:  python3 -m streamlit run app.py
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from pairs_trading import (
    PnLTracker,
    calculate_hedge_ratio,
    compute_spread,
    compute_zscore,
    fetch_live_data,
    generate_signal,
    perform_adf_test,
    update_dataset,
)

ROLLING_WINDOW = 20
Z_ENTRY = 1.0
Z_EXIT = 0.5
Z_STOP = 3.0
REFRESH_SEC = 60

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pairs Trading Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card .label {
        color: #9ca3af;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-card .value {
        color: #f1f5f9;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .metric-card .hint {
        color: #64748b;
        font-size: 0.72rem;
        margin-top: 4px;
    }
    .status-valid   { color: #22c55e; }
    .status-invalid { color: #ef4444; }
    .signal-long    { color: #22c55e; font-weight: 700; }
    .signal-short   { color: #ef4444; font-weight: 700; }
    .signal-exit    { color: #eab308; font-weight: 700; }
    .signal-hold    { color: #6b7280; font-weight: 700; }
    .signal-stop    { color: #d946ef; font-weight: 700; }
    .rec-box {
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 12px;
        text-align: center;
    }
    .rec-buy {
        background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
        border: 1px solid #166534;
    }
    .rec-sell {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        border: 1px solid #991b1b;
    }
    .rec-neutral {
        background: linear-gradient(135deg, #1c1917 0%, #292524 100%);
        border: 1px solid #44403c;
    }
    .rec-box .rec-ticker {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .rec-box .rec-action {
        font-size: 2rem;
        font-weight: 800;
        margin: 6px 0;
    }
    .rec-box .rec-action.buy  { color: #22c55e; }
    .rec-box .rec-action.sell { color: #ef4444; }
    .rec-box .rec-action.hold { color: #78716c; }
    .rec-box .rec-price {
        font-size: 1.1rem;
        color: #d1d5db;
        font-weight: 600;
    }
    .rec-box .rec-reason {
        font-size: 0.76rem;
        color: #6b7280;
        margin-top: 6px;
    }
    .confidence-bar {
        background: #1e293b;
        border-radius: 12px;
        padding: 18px 24px;
        text-align: center;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    .confidence-bar .conf-label {
        color: #9ca3af;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .confidence-bar .conf-value {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .conf-high   { color: #22c55e; }
    .conf-medium { color: #eab308; }
    .conf-low    { color: #ef4444; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    .explain {
        background: #1e293b;
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 10px 0 18px 0;
        color: #94a3b8;
        font-size: 0.88rem;
        line-height: 1.55;
    }
</style>
""", unsafe_allow_html=True)

SIGNAL_CSS = {
    "LONG": "signal-long",
    "SHORT": "signal-short",
    "EXIT": "signal-exit",
    "HOLD": "signal-hold",
    "STOP_LOSS": "signal-stop",
}


def card(label: str, value: str, hint: str = "", extra_class: str = "") -> str:
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    return (
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value {extra_class}">{value}</div>'
        f'{hint_html}</div>'
    )


def explain(text: str):
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)


def build_recommendation(z_val: float, signal: str, tx: str, ty: str,
                         price_x: float, price_y: float,
                         corr: float, is_cointegrated: bool, p_value: float):
    """Return (rec_x, rec_y, confidence, reasons) based on all indicators."""
    reasons = []

    # Confidence scoring
    score = 0
    if is_cointegrated:
        score += 40
        reasons.append(f"Spread is cointegrated (p={p_value:.4f})")
    else:
        reasons.append(f"Spread is NOT cointegrated (p={p_value:.4f}) — lower reliability")

    if abs(corr) > 0.9:
        score += 25
        reasons.append(f"Very strong correlation ({corr:.4f})")
    elif abs(corr) > 0.8:
        score += 15
        reasons.append(f"Strong correlation ({corr:.4f})")
    else:
        reasons.append(f"Weak correlation ({corr:.4f}) — pair may diverge")

    abs_z = abs(z_val)
    if abs_z > 2.0:
        score += 25
        reasons.append(f"Z-Score is extreme ({z_val:+.2f}) — strong mean-reversion signal")
    elif abs_z > 1.0:
        score += 15
        reasons.append(f"Z-Score outside ±1σ ({z_val:+.2f}) — entry signal triggered")
    elif abs_z > 0.5:
        score += 5
        reasons.append(f"Z-Score is moderate ({z_val:+.2f}) — no clear entry")
    else:
        reasons.append(f"Z-Score near zero ({z_val:+.2f}) — spread at equilibrium")

    score = min(score, 100)

    if score >= 65:
        conf_label, conf_cls = "HIGH", "conf-high"
    elif score >= 40:
        conf_label, conf_cls = "MEDIUM", "conf-medium"
    else:
        conf_label, conf_cls = "LOW", "conf-low"

    if signal == "SHORT":
        rec_x = ("BUY", "buy", f"Z={z_val:+.2f} > +1 → Spread overvalued → Buy {tx}")
        rec_y = ("SELL", "sell", f"Z={z_val:+.2f} > +1 → Spread overvalued → Sell {ty}")
    elif signal == "LONG":
        rec_x = ("SELL", "sell", f"Z={z_val:+.2f} < −1 → Spread undervalued → Sell {tx}")
        rec_y = ("BUY", "buy", f"Z={z_val:+.2f} < −1 → Spread undervalued → Buy {ty}")
    elif signal == "STOP_LOSS":
        rec_x = ("EXIT", "hold", "Stop-loss triggered — close all positions")
        rec_y = ("EXIT", "hold", "Stop-loss triggered — close all positions")
    elif signal == "EXIT":
        rec_x = ("HOLD", "hold", "Spread near equilibrium — no position")
        rec_y = ("HOLD", "hold", "Spread near equilibrium — no position")
    else:
        rec_x = ("HOLD", "hold", "No actionable signal")
        rec_y = ("HOLD", "hold", "No actionable signal")

    return rec_x, rec_y, (score, conf_label, conf_cls), reasons


def rec_card(ticker: str, price: float, action: str, action_cls: str, reason: str) -> str:
    box_cls = "rec-buy" if action_cls == "buy" else "rec-sell" if action_cls == "sell" else "rec-neutral"
    return (
        f'<div class="rec-box {box_cls}">'
        f'<div class="rec-ticker">{ticker}</div>'
        f'<div class="rec-action {action_cls}">{action}</div>'
        f'<div class="rec-price">@ {price:,.2f}</div>'
        f'<div class="rec-reason">{reason}</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cached data loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def load_historical(ticker_x: str, ticker_y: str, num_obs: int) -> pd.DataFrame:
    """Fetch enough history and return exactly *num_obs* trading days."""
    period = "5y" if num_obs > 500 else "3y" if num_obs > 248 else "2y"
    raw = yf.download(
        [ticker_x, ticker_y],
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError("yfinance returned empty data — check ticker symbols")
    df = raw["Close"][[ticker_x, ticker_y]].copy()
    df.columns = [ticker_x, ticker_y]
    df.index = pd.to_datetime(df.index)
    df = df.ffill().dropna()
    return df.tail(num_obs)


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────
LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=36, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    height=380,
)


def chart_prices(df: pd.DataFrame, tx: str, ty: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df.index, y=df[tx], name=tx,
                   line=dict(color="#3b82f6", width=1.5)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df[ty], name=ty,
                   line=dict(color="#f97316", width=1.5)),
        secondary_y=True,
    )
    fig.update_yaxes(title_text=tx, secondary_y=False, gridcolor="#1e293b")
    fig.update_yaxes(title_text=ty, secondary_y=True, gridcolor="#1e293b")
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_layout(title="Price History (Dual Axis)", **LAYOUT)
    return fig


def chart_spread(spread: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spread.index, y=spread, name="Spread",
        line=dict(color="#8b5cf6", width=1.5),
        fill="tozeroy", fillcolor="rgba(139,92,246,0.06)",
    ))
    fig.add_hline(y=spread.mean(), line_dash="dash", line_color="#64748b",
                  annotation_text="Mean", annotation_position="top left")
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    fig.update_layout(title="Spread  (Y − β · X)", **LAYOUT)
    return fig


def chart_zscore(z: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=z.index, y=z, name="Z-Score",
        line=dict(color="#06b6d4", width=1.8),
        fill="tozeroy", fillcolor="rgba(6,182,212,0.06)",
    ))
    for level, color, label in [
        (1.0, "#ef4444", "+1 σ  (Short zone)"),
        (-1.0, "#22c55e", "−1 σ  (Long zone)"),
    ]:
        fig.add_hline(
            y=level, line_dash="dot", line_color=color, line_width=1,
            annotation_text=label if level > 0 else "",
            annotation_position="top left",
            annotation_font_color=color,
        )
    fig.add_hline(y=0, line_color="#475569", line_width=0.8)
    fig.add_hrect(y0=-1, y1=1, fillcolor="#06b6d4", opacity=0.04,
                  annotation_text="Mean-reversion zone", annotation_position="top left",
                  annotation_font_color="#475569", annotation_font_size=10)
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b", title_text="Standard Deviations")
    fig.update_layout(title="Z-Score  (Rolling 20-day)", **LAYOUT)
    return fig


def chart_scatter(df: pd.DataFrame, tx: str, ty: str, hedge: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[tx], y=df[ty], mode="markers",
        marker=dict(color="#3b82f6", size=4, opacity=0.4),
        name="Daily prices",
    ))
    x_range = np.linspace(df[tx].min(), df[tx].max(), 100)
    m, b = np.polyfit(df[tx], df[ty], 1)
    fig.add_trace(go.Scatter(
        x=x_range, y=m * x_range + b, mode="lines",
        line=dict(color="#f97316", width=2, dash="dash"),
        name=f"OLS fit  (β = {hedge:.4f})",
    ))
    fig.update_xaxes(title_text=tx, gridcolor="#1e293b")
    fig.update_yaxes(title_text=ty, gridcolor="#1e293b")
    fig.update_layout(title="Regression Scatter", **LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — only tickers + period
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Pair Selection")

    ticker_x = st.text_input("Stock X (Independent)", value="HDFCBANK.NS", max_chars=20,
                             help="Yahoo Finance ticker, e.g. HDFCBANK.NS, AAPL").upper().strip()
    ticker_y = st.text_input("Stock Y (Dependent)", value="ICICIBANK.NS", max_chars=20,
                             help="Yahoo Finance ticker, e.g. ICICIBANK.NS, MSFT").upper().strip()

    num_observations = st.number_input("Observations", min_value=50, max_value=1000,
                                       value=248, step=1,
                                       help="Number of trading days to analyse (248 ≈ 1 year)")

    st.divider()
    run_analysis = st.button("Run Analysis", use_container_width=True, type="primary")

    st.divider()
    st.markdown(
        '<p style="color:#475569; font-size:0.78rem; line-height:1.5;">'
        "This tool analyses the statistical relationship between two stocks "
        "using <b>Correlation</b>, <b>Cointegration (ADF test)</b>, and "
        "<b>Z-Score</b> to determine if they form a valid pair for "
        "mean-reversion trading.</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0;'>Pairs Trading Analysis</h1>"
    f"<p style='text-align:center; color:#64748b;'>"
    f"{ticker_y} / {ticker_x}</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "live_running" not in st.session_state:
    st.session_state.live_running = False
if "signal_log" not in st.session_state:
    st.session_state.signal_log = []

# ─────────────────────────────────────────────────────────────────────────────
# Run analysis
# ─────────────────────────────────────────────────────────────────────────────
if run_analysis:
    if not ticker_x or not ticker_y:
        st.error("Please enter both ticker symbols.")
        st.stop()
    if ticker_x == ticker_y:
        st.error("Ticker X and Ticker Y must be different.")
        st.stop()

    with st.spinner(f"Fetching {num_observations} observations for {ticker_x} & {ticker_y} ..."):
        try:
            df = load_historical(ticker_x, ticker_y, num_observations)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    if len(df) < ROLLING_WINDOW + 10:
        st.error(f"Not enough data — got {len(df)} rows, need at least {ROLLING_WINDOW + 10}.")
        st.stop()

    hedge, intercept = calculate_hedge_ratio(df[ticker_y], df[ticker_x])
    spread = compute_spread(df[ticker_y], df[ticker_x], hedge, intercept)
    adf = perform_adf_test(spread)
    z = compute_zscore(spread, ROLLING_WINDOW)
    corr = df[ticker_x].corr(df[ticker_y])

    st.session_state.update(
        analysis_done=True,
        df=df, hedge=hedge, intercept=intercept, spread=spread,
        adf=adf, z=z, corr=corr,
        ticker_x=ticker_x, ticker_y=ticker_y,
        signal_log=[], live_running=False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.analysis_done:
    s = st.session_state
    tx, ty = s.ticker_x, s.ticker_y
    df, hedge, intercept, spread = s.df, s.hedge, s.intercept, s.spread
    adf, z, corr = s.adf, s.z, s.corr

    latest_z = z.dropna().iloc[-1] if not z.dropna().empty else 0
    z_signal = generate_signal(latest_z, Z_ENTRY, Z_EXIT, Z_STOP)
    is_valid = adf["is_stationary"]

    # ── Statistics Summary (top) ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Statistics Summary")

    s1, s2, s3, s4, s5, s6, s7 = st.columns(7)
    with s1:
        st.markdown(
            card("Observations", f"{len(df)}",
                 f"{df.index[0].strftime('%b %Y')} → {df.index[-1].strftime('%b %Y')}"),
            unsafe_allow_html=True,
        )
    with s2:
        corr_cls = "status-valid" if abs(corr) > 0.8 else "status-invalid"
        st.markdown(
            card("Correlation", f"{corr:.4f}", "",  corr_cls),
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            card("Hedge Ratio (β)", f"{hedge:.4f}"),
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            card("Intercept (α)", f"{intercept:.4f}"),
            unsafe_allow_html=True,
        )
    with s5:
        pval_cls = "status-valid" if adf["p_value"] < 0.05 else "status-invalid"
        st.markdown(
            card("ADF p-value", f"{adf['p_value']:.4f}", "", pval_cls),
            unsafe_allow_html=True,
        )
    with s6:
        st.markdown(
            card("Z-Score", f"{latest_z:+.4f}"),
            unsafe_allow_html=True,
        )
    with s7:
        sig_cls = SIGNAL_CSS.get(z_signal, "signal-hold")
        pair_cls = "status-valid" if is_valid else "status-invalid"
        pair_txt = "VALID" if is_valid else "WEAK"
        st.markdown(
            card("Pair Status", f'<span class="{pair_cls}">{pair_txt}</span>'),
            unsafe_allow_html=True,
        )

    # ── Recommendation ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Recommendation")

    latest_price_x = df[tx].iloc[-1]
    latest_price_y = df[ty].iloc[-1]
    rec_x, rec_y, confidence, reasons = build_recommendation(
        latest_z, z_signal, tx, ty,
        latest_price_x, latest_price_y,
        corr, is_valid, adf["p_value"],
    )

    rc1, rc2, rc3 = st.columns([2, 2, 1])
    with rc1:
        st.markdown(
            rec_card(tx, latest_price_x, rec_x[0], rec_x[1], rec_x[2]),
            unsafe_allow_html=True,
        )
    with rc2:
        st.markdown(
            rec_card(ty, latest_price_y, rec_y[0], rec_y[1], rec_y[2]),
            unsafe_allow_html=True,
        )
    with rc3:
        conf_score, conf_label, conf_cls = confidence
        st.markdown(
            f'<div class="confidence-bar">'
            f'<div class="conf-label">Confidence</div>'
            f'<div class="conf-value {conf_cls}">{conf_label}</div>'
            f'<div class="conf-label" style="margin-top:6px;">{conf_score}/100</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Why this recommendation?"):
        for r in reasons:
            st.write(f"- {r}")

    # ── 1. Correlation ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 1. Correlation")
    explain(
        "<b>Correlation</b> measures how closely two stock prices move together. "
        "A value near <b>+1</b> means they move in the same direction; "
        "near <b>−1</b> means opposite directions; near <b>0</b> means no relationship. "
        "For pairs trading we look for high positive correlation (> 0.80)."
    )
    st.plotly_chart(chart_prices(df, tx, ty), use_container_width=True)

    # ── 2. Cointegration (ADF) ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 2. Cointegration  (Augmented Dickey-Fuller Test)")
    explain(
        "<b>Cointegration</b> tells us whether the spread between two stocks tends to "
        "revert to a mean over time — this is the foundation of pairs trading. "
        "We test this with the <b>Augmented Dickey-Fuller (ADF)</b> test.<br><br>"
        "<b>Key rule:</b> if the <b>p-value &lt; 0.05</b> → the spread is <em>stationary</em> → the pair is valid.<br>"
        "The <b>ADF Statistic</b> should be more negative than the critical values to reject the null hypothesis "
        "of a unit root (non-stationarity)."
    )

    status_cls = "status-valid" if is_valid else "status-invalid"
    status_txt = "COINTEGRATED" if is_valid else "NOT COINTEGRATED"

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown(
            card("ADF Statistic", f"{adf['adf_statistic']:.4f}",
                 "More negative = stronger evidence"),
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            card("p-value", f"{adf['p_value']:.6f}",
                 "Must be < 0.05 for validity",
                 "status-valid" if adf["p_value"] < 0.05 else "status-invalid"),
            unsafe_allow_html=True,
        )
    with a3:
        cv_1 = adf["critical_values"].get("1%", 0)
        cv_5 = adf["critical_values"].get("5%", 0)
        st.markdown(
            card("Critical Values", f"{cv_1:.2f} / {cv_5:.2f}",
                 "1% / 5% significance levels"),
            unsafe_allow_html=True,
        )
    with a4:
        st.markdown(
            card("Verdict", status_txt, "", status_cls),
            unsafe_allow_html=True,
        )

    if is_valid:
        st.success(
            f"The spread between **{ty}** and **{tx}** is **stationary** "
            f"(p = {adf['p_value']:.6f}). This pair is suitable for mean-reversion trading."
        )
    else:
        st.warning(
            f"The spread is **non-stationary** (p = {adf['p_value']:.6f}). "
            f"This pair may not reliably revert to the mean. Signals should be treated with caution."
        )

    # ── 3. Hedge Ratio & Spread ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 3. Hedge Ratio & Spread")
    explain(
        "The <b>Hedge Ratio (β)</b> and <b>Intercept (α)</b> come from regressing "
        "stock Y on stock X using Ordinary Least Squares (OLS). "
        "β tells us how many units of X to trade for each unit of Y.<br><br>"
        "<b>Spread = Y − (β · X + α)</b><br><br>"
        "If the pair is cointegrated, this spread will fluctuate around zero, "
        "creating trading opportunities when it deviates too far."
    )

    h1, h2 = st.columns([1, 2])
    with h1:
        st.markdown(
            card("Hedge Ratio (β)", f"{hedge:.4f}",
                 f"For every 1 share of {ty}, hedge with {hedge:.2f} shares of {tx}"),
            unsafe_allow_html=True,
        )
        st.markdown(
            card("Intercept (α)", f"{intercept:.4f}",
                 "Constant offset from OLS regression"),
            unsafe_allow_html=True,
        )
        st.markdown(
            card("Spread Std Dev", f"{spread.std():.4f}"),
            unsafe_allow_html=True,
        )
    with h2:
        st.plotly_chart(chart_scatter(df, tx, ty, hedge), use_container_width=True)

    st.plotly_chart(chart_spread(spread), use_container_width=True)

    # ── 4. Z-Score ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 4. Z-Score")
    explain(
        "The <b>Z-Score</b> measures how far the current spread is from its recent average, "
        "expressed in standard deviations.<br><br>"
        "<b>Formula:</b><br>"
        "<code style='color:#06b6d4;'>"
        "Spread = Price Y − (β × Price X + α)<br>"
        "Z-Score = (Spread − Rolling Mean) / Rolling Std Dev"
        "</code><br><br>"
        "<b>Z &gt; +1</b> → Spread is unusually <em>high</em> → expect it to fall → <b>Short</b> the spread<br>"
        "<b>Z &lt; −1</b> → Spread is unusually <em>low</em> → expect it to rise → <b>Long</b> the spread<br>"
        "<b>|Z| &lt; 0.5</b> → Spread is near normal → <b>Exit</b> / no trade"
    )

    z1, z2 = st.columns(2)
    with z1:
        st.markdown(
            card("Current Z-Score", f"{latest_z:+.4f}"),
            unsafe_allow_html=True,
        )
    with z2:
        st.markdown(
            card("Current Signal", f'<span class="{sig_cls}">{z_signal}</span>'),
            unsafe_allow_html=True,
        )

    st.plotly_chart(chart_zscore(z), use_container_width=True)

    # ── 5. Live Monitor ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 5. Live Signal Monitor")
    explain(
        "Start the live feed to fetch the latest prices every 60 seconds. "
        "The system recalculates the spread and z-score in real time and "
        "shows you the current trading signal."
    )

    btn1, btn2, _ = st.columns([1, 1, 4])
    with btn1:
        start_live = st.button("Start Live Feed", use_container_width=True, type="primary")
    with btn2:
        stop_live = st.button("Stop", use_container_width=True)

    if stop_live:
        st.session_state.live_running = False
    if start_live:
        st.session_state.live_running = True
        st.session_state.signal_log = []

    if st.session_state.live_running:
        signal_ph = st.empty()
        chart_ph = st.empty()
        status_ph = st.empty()

        tracker = PnLTracker()
        local_df = df.copy()
        local_hedge = hedge
        local_intercept = intercept
        tick = 0

        status_ph.info(f"Refreshing every **{REFRESH_SEC}s** — press **Stop** to end.")

        while st.session_state.live_running:
            result = fetch_live_data(tx, ty)
            if result is None:
                status_ph.warning("Fetch failed — retrying next interval...")
                time.sleep(REFRESH_SEC)
                continue

            price_x, price_y, ts = result
            local_df = update_dataset(local_df, tx, ty, price_x, price_y, ts)

            tick += 1
            live_spread = compute_spread(local_df[ty], local_df[tx], local_hedge, local_intercept)
            live_z = compute_zscore(live_spread, ROLLING_WINDOW)
            cur_z = live_z.iloc[-1]

            if np.isnan(cur_z):
                status_ph.warning("Z-score is NaN — waiting for sufficient data...")
                time.sleep(REFRESH_SEC)
                continue

            cur_spread = live_spread.iloc[-1]
            signal = generate_signal(cur_z, Z_ENTRY, Z_EXIT, Z_STOP)
            pnl = tracker.update(signal, cur_spread)

            sig_cls = SIGNAL_CSS.get(signal, "signal-hold")
            st.session_state.signal_log.append({
                "Time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                tx: f"{price_x:,.2f}",
                ty: f"{price_y:,.2f}",
                "Spread": f"{cur_spread:.4f}",
                "Z-Score": f"{cur_z:+.4f}",
                "Signal": signal,
            })

            with signal_ph.container():
                lc1, lc2, lc3, lc4 = st.columns(4)
                with lc1:
                    st.markdown(card(tx, f"{price_x:,.2f}"), unsafe_allow_html=True)
                with lc2:
                    st.markdown(card(ty, f"{price_y:,.2f}"), unsafe_allow_html=True)
                with lc3:
                    st.markdown(card("Z-Score", f"{cur_z:+.4f}"), unsafe_allow_html=True)
                with lc4:
                    st.markdown(
                        card("Signal", f'<span class="{sig_cls}">{signal}</span>'),
                        unsafe_allow_html=True,
                    )

            with chart_ph.container():
                st.plotly_chart(
                    chart_zscore(live_z.tail(200)),
                    use_container_width=True,
                    key=f"live_{tick}",
                )

            time.sleep(REFRESH_SEC)

    # ── Signal log ──────────────────────────────────────────────────────
    if st.session_state.signal_log:
        st.markdown("#### Signal Log")
        log_df = pd.DataFrame(st.session_state.signal_log)
        st.dataframe(log_df.iloc[::-1], use_container_width=True, hide_index=True, height=300)

        st.download_button(
            "Download CSV",
            data=log_df.to_csv(index=False),
            file_name=f"signals_{tx}_{ty}_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
        )

else:
    st.markdown("")
    st.info("Enter two stock ticker symbols in the sidebar and click **Run Analysis** to begin.")
