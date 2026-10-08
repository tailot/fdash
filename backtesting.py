# Copyright (c) 2026 Vincenzo Tilotta
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import numpy as np
import pandas as pd

from quant_engine import (HORIZON, SOGLIA_TREND, MODELLO_FALLBACK, _trend_da_previsione,
                          naive_returns, tfm3_returns)


def _map_signal(signal):
    if pd.isna(signal):
        return 0
    mapping = {
        "BUY": 1,
        "SELL": -1,
        "BULLISH": 1,
        "BEARISH": -1,
        "EQUILIBRIO": 0,
        "HOLD": 0,
        "NEUTRAL": 0,
    }
    return mapping.get(str(signal).upper(), 0)


def walk_forward_backtest(df: pd.DataFrame, signal_col: str, target_col: str,
                         train_window: int = 60, test_window: int = 20) -> pd.DataFrame:
    """
    Walk-forward backtest: train on window, test on next window.
    Returns DataFrame with accuracy, win counts per fold.
    """
    if signal_col not in df.columns:
        raise ValueError(f"Missing signal column: {signal_col}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    ordered = df.copy().reset_index(drop=True)
    results = []
    n = len(ordered)

    if n < train_window + test_window:
        return pd.DataFrame(columns=[
            "train_start", "train_end", "test_start", "test_end",
            "accuracy", "signal_count_buy", "signal_count_sell", "signal_count_hold"
        ])

    step = max(1, test_window)
    for i in range(0, n - train_window - test_window + 1, step):
        train = ordered.iloc[i:i + train_window]
        test = ordered.iloc[i + train_window:i + train_window + test_window]
        if train.empty or test.empty:
            continue

        train_signals = train[signal_col].map(_map_signal)
        test_targets = pd.to_numeric(test[target_col], errors="coerce").fillna(0.0)
        if test_targets.empty:
            continue

        match = np.sign(train_signals.iloc[: len(test_targets)].to_numpy()) * np.sign(test_targets.to_numpy())
        accuracy = float((match > 0).mean()) if len(match) > 0 else 0.0

        results.append({
            "train_start": i,
            "train_end": i + train_window - 1,
            "test_start": i + train_window,
            "test_end": i + train_window + len(test_targets) - 1,
            "accuracy": accuracy,
            "signal_count_buy": int((train_signals > 0).sum()),
            "signal_count_sell": int((train_signals < 0).sum()),
            "signal_count_hold": int((train_signals == 0).sum()),
        })

    return pd.DataFrame(results)


def compute_strategy_metrics(df: pd.DataFrame, target_col: str = "future_return") -> dict:
    """Compute aggregate performance metrics."""
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    signal_col = "signal" if "signal" in df.columns else "Trend_TimesFM"
    if signal_col not in df.columns:
        raise ValueError(f"Missing signal column: {signal_col}")

    work = df.copy()
    work["signal_score"] = work[signal_col].map(_map_signal).fillna(0)
    returns = pd.to_numeric(work[target_col], errors="coerce").fillna(0.0)
    pnl = returns * work["signal_score"].astype(float)

    return {
        "mean_return": float(pnl.mean()),
        "median_return": float(np.median(pnl)),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "avg_pnl": float(pnl.mean()),
        "max_drawdown": float(np.max(np.maximum.accumulate(-pnl))) if len(pnl) else 0.0,
    }


# ----------------------------------------------------------------------------------------------
# Temporal walk-forward backtest (point-in-time forecasts vs realized returns)
# ----------------------------------------------------------------------------------------------
def backtest_temporale(prices: pd.DataFrame, horizon: int = HORIZON, forecaster=None,
                       soglia_trend: float = SOGLIA_TREND, n_date: int = 30,
                       passo: int | None = None, min_context: int = 252, progress=None) -> pd.DataFrame:
    """
    For each past date t (spaced `passo` days apart, default = horizon -> non-overlapping windows)
    the forecast is computed using ONLY prices up to t, the signal (BUY/SELL/EQUILIBRIO) is derived
    with the same rule as the dashboard, and compared with the REALIZED return from t to t+horizon.
    Returns a long DataFrame: Date, Ticker, signal, mu_cum, sigma_cum, realized_return.
    """
    passo = passo or horizon
    n = len(prices)
    last = n - 1 - horizon
    if last < min_context:
        raise ValueError(f"Not enough history: need > {min_context + horizon} days, got {n}.")
    idx = sorted(range(last, min_context - 1, -passo))[-n_date:]

    tickers = list(prices.columns)
    values = {t: prices[t].to_numpy(dtype=float) for t in tickers}
    rows = []
    for k, i in enumerate(idx):
        arrays = [values[t][: i + 1] for t in tickers]          # no look-ahead
        if forecaster is not None:
            cum_mu, cum_sig = tfm3_returns(forecaster, arrays, horizon)
        else:
            cum_mu, cum_sig = naive_returns(arrays, horizon)
        for j, t in enumerate(tickers):
            mu, sg = float(cum_mu[j, -1]), float(cum_sig[j, -1])
            rows.append({
                "Date": prices.index[i],
                "Ticker": t,
                "signal": _trend_da_previsione(mu, sg, soglia_trend),
                "mu_cum": mu,
                "sigma_cum": sg,
                "realized_return": float(values[t][i + horizon] / values[t][i] - 1.0),
            })
        if progress is not None:
            progress((k + 1) / len(idx))
    return pd.DataFrame(rows)


def riepilogo_backtest_temporale(bt: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Aggregate metrics + per-signal table from the output of backtest_temporale."""
    if bt.empty:
        return {}, pd.DataFrame()
    score = bt["signal"].map(_map_signal).astype(float)
    pnl = score * bt["realized_return"]
    active = score != 0
    summary = {
        "n_obs": int(len(bt)),
        "n_active": int(active.sum()),
        "hit_rate": float((pnl[active] > 0).mean()) if active.any() else float("nan"),
        "avg_pnl_active": float(pnl[active].mean()) if active.any() else float("nan"),
        "avg_return_all": float(bt["realized_return"].mean()),   # buy&hold benchmark
    }
    by_signal = (bt.groupby("signal")["realized_return"]
                 .agg(osservazioni="count", rendimento_medio="mean", rendimento_mediano="median",
                      pct_positivi=lambda x: float((x > 0).mean()))
                 .reset_index())
    return summary, by_signal
