"""
Price Alerts Engine.

Pure logic (no Streamlit): rule normalization, price download, rule evaluation,
persistence and alarm-sound generation. The UI lives in `alerts_ui.py`.

Configuration (via code or environment variables):
  POLL_INTERVAL_SECONDS -> seconds between two calls to the finance service (default 60)
  CONFIG_PATH           -> JSON file where the alert rules are persisted
"""

import io
import json
import os
import wave
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------------- configuration
POLL_INTERVAL_SECONDS = int(os.environ.get("FDASH_ALERT_POLL_SECONDS", "60"))
CONFIG_PATH = os.environ.get("FDASH_ALERTS_FILE", "alerts_config.json")
MAX_LOG_ENTRIES = 200

CONDITIONS = (">=", "<=")  # price rises to/above target | falls to/below target
RULE_COLUMNS = ["symbol", "condition", "target", "active"]


# ----------------------------------------------------------------------------- rules
def empty_rules() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": pd.Series(dtype="object"),
        "condition": pd.Series(dtype="object"),
        "target": pd.Series(dtype="float64"),
        "active": pd.Series(dtype="bool"),
    })


def normalize_rules(df: pd.DataFrame) -> list:
    """Cleans the table coming from the editor: drops incomplete/invalid rows."""
    rules = []
    if df is None or len(df) == 0:
        return rules
    for _, row in df.iterrows():
        symbol = str(row.get("symbol") or "").strip().upper()
        condition = row.get("condition")
        target = pd.to_numeric(row.get("target"), errors="coerce")
        if not symbol or condition not in CONDITIONS or pd.isna(target) or target <= 0:
            continue
        active = row.get("active")
        active = True if pd.isna(active) else bool(active)
        rules.append({"symbol": symbol, "condition": condition, "target": float(target), "active": active})
    return rules


def rule_key(rule: dict) -> str:
    return f"{rule['symbol']}|{rule['condition']}|{rule['target']:.6f}"


def is_triggered(rule: dict, price: float) -> bool:
    if rule["condition"] == ">=":
        return price >= rule["target"]
    return price <= rule["target"]


def evaluate_rules(rules: list, prices: dict, fired: set):
    """
    Returns (events, new_fired).
    A rule fires once when its condition becomes true; it re-arms as soon as the
    condition becomes false again (so the alarm is not repeated every minute).
    """
    events, new_fired = [], set()
    now = datetime.now().strftime("%H:%M:%S")
    for rule in rules:
        if not rule["active"]:
            continue
        price = prices.get(rule["symbol"])
        if price is None:
            if rule_key(rule) in fired:  # no data: keep current state
                new_fired.add(rule_key(rule))
            continue
        if is_triggered(rule, price):
            new_fired.add(rule_key(rule))
            if rule_key(rule) not in fired:
                events.append({"time": now, "symbol": rule["symbol"], "condition": rule["condition"],
                               "target": rule["target"], "price": float(price)})
    return events, new_fired


# ----------------------------------------------------------------------------- finance service
def fetch_last_prices(symbols) -> dict:
    """ONE batched call to the finance service for all the symbols -> {symbol: last 1-minute close}."""
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
    if not symbols:
        return {}
    raw = yf.download(symbols, period="1d", interval="1m", prepost=True, auto_adjust=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        return {}
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])
    prices = {}
    for s in symbols:
        if s in close.columns:
            serie = close[s].dropna()
            if not serie.empty:
                prices[s] = float(serie.iloc[-1])
    return prices


# ----------------------------------------------------------------------------- persistence
def load_rules(path: str = None) -> pd.DataFrame:
    path = path or CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data.get("rules", []), columns=RULE_COLUMNS)
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        df["active"] = df["active"].fillna(True).astype(bool)
        return df
    except (OSError, ValueError):
        return empty_rules()


def save_rules(df: pd.DataFrame, path: str = None) -> None:
    path = path or CONFIG_PATH
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rules": normalize_rules(df)}, f, indent=2)
    except OSError:
        pass  # read-only filesystem: rules simply stay in the session


# ----------------------------------------------------------------------------- alarm sound
def make_beep_wav(freq: float = 880.0, beep_s: float = 0.22, gap_s: float = 0.12,
                  repeats: int = 3, rate: int = 22050) -> bytes:
    """Short 'beep-beep-beep' as WAV bytes (no audio asset needed)."""
    t = np.linspace(0, beep_s, int(rate * beep_s), endpoint=False)
    tone = 0.6 * np.sin(2 * np.pi * freq * t) * np.minimum(1, np.minimum(t, beep_s - t) * 80)
    gap = np.zeros(int(rate * gap_s))
    signal = np.concatenate([np.concatenate([tone, gap]) for _ in range(repeats)])
    pcm = (signal * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()
