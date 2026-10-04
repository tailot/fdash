"""
Behavioral Finance Signals Module.

Detects behavioral market anomalies and institutional activity:
1. Unusual volume spikes (> 2x average or > 2 std dev)
2. Earnings surprises (EPS actual vs estimate)
3. Insider trade signals (net buying vs selling)
"""

import math
import logging
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def detect_volume_spike(volumes: pd.Series, prices: pd.Series = None, lookback: int = 20) -> dict:
    """
    Detects unusual volume spikes in recent price/volume data.
    """
    if volumes is None or len(volumes) < lookback:
        return {
            "is_spike": False,
            "volume_ratio": 1.0,
            "z_score": 0.0,
            "signal": "NORMAL"
        }

    clean_v = volumes.dropna()
    if len(clean_v) < 5:
        return {"is_spike": False, "volume_ratio": 1.0, "z_score": 0.0, "signal": "NORMAL"}

    latest_vol = clean_v.iloc[-1]
    hist_vol = clean_v.iloc[-lookback:-1] if len(clean_v) >= lookback else clean_v.iloc[:-1]

    mean_vol = hist_vol.mean()
    std_vol = hist_vol.std(ddof=1)

    if mean_vol <= 0 or math.isnan(mean_vol):
        return {"is_spike": False, "volume_ratio": 1.0, "z_score": 0.0, "signal": "NORMAL"}

    volume_ratio = latest_vol / mean_vol
    z_score = (latest_vol - mean_vol) / (std_vol + 1e-9)

    is_spike = (volume_ratio >= 2.0) or (z_score >= 2.0)

    # Determine direction if price history is supplied
    direction = "NEUTRAL"
    if prices is not None and len(prices) >= 2:
        price_change = prices.iloc[-1] - prices.iloc[-2]
        if price_change > 0:
            direction = "BUY"
        elif price_change < 0:
            direction = "SELL"

    if is_spike:
        signal = f"UNUSUAL_{direction}_SPIKE" if direction != "NEUTRAL" else "UNUSUAL_VOLUME_SPIKE"
    else:
        signal = "NORMAL"

    return {
        "is_spike": bool(is_spike),
        "volume_ratio": round(float(volume_ratio), 2),
        "z_score": round(float(z_score), 2),
        "signal": signal
    }


def detect_earnings_surprise(ticker_symbol: str) -> dict:
    """
    Fetches recent quarterly earnings surprises for the given ticker.
    """
    try:
        t_obj = yf.Ticker(ticker_symbol)
        # Try getting earnings dates / history
        dates = t_obj.earnings_dates
        if dates is not None and not dates.empty and "Surprise(%)" in dates.columns:
            clean_surprises = dates["Surprise(%)"].dropna()
            if not clean_surprises.empty:
                last_surprise = float(clean_surprises.iloc[0]) * 100.0  # Percentage
                if last_surprise > 2.0:
                    sig = "POSITIVE_SURPRISE"
                elif last_surprise < -2.0:
                    sig = "NEGATIVE_SURPRISE"
                else:
                    sig = "NEUTRAL"
                return {
                    "last_surprise_pct": round(last_surprise, 2),
                    "signal": sig
                }
    except Exception as e:
        logger.debug(f"Earnings surprise fetch error for {ticker_symbol}: {e}")

    return {
        "last_surprise_pct": None,
        "signal": "NEUTRAL"
    }


def detect_insider_activity(ticker_symbol: str) -> dict:
    """
    Analyzes recent insider trades for the ticker to identify net insider sentiment.
    """
    try:
        t_obj = yf.Ticker(ticker_symbol)
        insiders = t_obj.insider_transactions
        if insiders is not None and not insiders.empty:
            # Check text or transaction shares
            buying_shares = 0
            selling_shares = 0
            for _, row in insiders.head(15).iterrows():
                text = str(row.get("Text", "")).lower()
                shares = row.get("Shares", 0)
                try:
                    shares = float(shares) if not math.isnan(float(shares)) else 0
                except (ValueError, TypeError):
                    shares = 0

                if "buy" in text or "purchase" in text:
                    buying_shares += shares
                elif "sale" in text or "sell" in text:
                    selling_shares += shares

            if buying_shares > selling_shares * 1.5 and buying_shares > 0:
                return {"action": "NET_BUYING", "signal": "BULLISH"}
            elif selling_shares > buying_shares * 1.5 and selling_shares > 0:
                return {"action": "NET_SELLING", "signal": "BEARISH"}
    except Exception as e:
        logger.debug(f"Insider activity fetch error for {ticker_symbol}: {e}")

    return {"action": "NEUTRAL", "signal": "NEUTRAL"}


def get_behavioral_signals(ticker_symbol: str, prices: pd.Series = None, volumes: pd.Series = None) -> dict:
    """
    Consolidates behavioral finance signals (Volume Spikes, Earnings Surprises, Insider Trades).
    """
    vol_spike = detect_volume_spike(volumes, prices)
    earn_surp = detect_earnings_surprise(ticker_symbol)
    insider = detect_insider_activity(ticker_symbol)

    # Construct synthesis label
    signals_list = []
    if vol_spike["is_spike"]:
        signals_list.append(f"Volume Spike ({vol_spike['volume_ratio']}x)")
    if earn_surp["signal"] == "POSITIVE_SURPRISE":
        signals_list.append(f"Earnings Beat (+{earn_surp['last_surprise_pct']}%)")
    elif earn_surp["signal"] == "NEGATIVE_SURPRISE":
        signals_list.append(f"Earnings Miss ({earn_surp['last_surprise_pct']}%)")
    if insider["signal"] == "BULLISH":
        signals_list.append("Net Insider Buying")
    elif insider["signal"] == "BEARISH":
        signals_list.append("Net Insider Selling")

    synthesis = ", ".join(signals_list) if signals_list else "No Unusual Behavioral Anomaly"

    return {
        "ticker": ticker_symbol,
        "volume_spike": vol_spike,
        "earnings_surprise": earn_surp,
        "insider_activity": insider,
        "synthesis": synthesis
    }
