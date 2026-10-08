"""
Discover Engine: Fetches and filters company data based on Volume, Market Cap, and Birth Year.
Supports S&P 500 (~500 stocks) and large NASDAQ universes (~4,000 stocks).
"""

import io
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

# Unit Multipliers Mapping
UNIT_MULTIPLIERS = {
    "1": 1.0,
    "units": 1.0,
    "k": 1_000.0,
    "thousands": 1_000.0,
    "mln": 1_000_000.0,
    "m": 1_000_000.0,
    "millions": 1_000_000.0,
    "mld": 1_000_000_000.0,
    "b": 1_000_000_000.0,
    "billions": 1_000_000_000.0,
}


def universe_sp500() -> list[str]:
    """Retrieves list of S&P 500 stock tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        syms = (
            tables[0]["Symbol"]
            .astype(str)
            .str.replace(".", "-", regex=False)
            .str.strip()
            .str.upper()
            .tolist()
        )
        return [s for s in syms if s and s.isalnum() or "-" in s]
    except Exception:
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA",
            "UNH", "JNJ", "JPM", "V", "PG", "XOM", "HD", "MA", "COST", "ABBV"
        ]


def get_unit_multiplier(unit_str: str) -> float:
    """Returns the numeric multiplier for a given unit string."""
    if not unit_str:
        return 1.0
    key = str(unit_str).strip().lower()
    return UNIT_MULTIPLIERS.get(key, 1.0)


def fetch_single_stock_info(symbol: str) -> dict:
    """Fetches company metadata, market cap, daily volume, and birth year for a single ticker."""
    clean_sym = symbol.strip().upper()
    try:
        t = yf.Ticker(clean_sym)
        fi = getattr(t, "fast_info", None)

        # Fast Info extraction
        fast_price = getattr(fi, "last_price", None) if fi else None
        fast_mc = getattr(fi, "market_cap", None) if fi else None
        fast_vol = (
            getattr(fi, "last_volume", None) or getattr(fi, "three_month_average_volume", None)
            if fi else None
        )

        info = t.info or {}

        # Name & Sector
        name = info.get("shortName") or info.get("longName") or clean_sym
        sector = info.get("sector") or "N/A"

        # Price
        price = (
            fast_price
            or info.get("currentPrice")
            or info.get("regularMarketPreviousClose")
            or info.get("previousClose")
            or info.get("navPrice")
        )
        if price is not None:
            try:
                price = round(float(price), 2)
            except (ValueError, TypeError):
                price = None

        # Market Cap & Volume
        mc = fast_mc or info.get("marketCap")
        if mc is not None:
            try:
                mc = float(mc)
            except (ValueError, TypeError):
                mc = None

        vol = fast_vol or info.get("volume") or info.get("regularMarketVolume") or info.get("averageVolume")
        if vol is not None:
            try:
                vol = float(vol)
            except (ValueError, TypeError):
                vol = None

        # Birth Year (Quotation / First Trade Year)
        birth_year = None
        ms = info.get("firstTradeDateMilliseconds")
        if ms:
            try:
                birth_year = datetime.fromtimestamp(ms / 1000.0).year
            except Exception:
                birth_year = None

        if not birth_year:
            try:
                hist = t.history(period="max")
                if not hist.empty:
                    birth_year = int(hist.index[0].year)
            except Exception:
                birth_year = None

        return {
            "Ticker": clean_sym,
            "Name": name,
            "Sector": sector,
            "Price": price,
            "MarketCap": mc,
            "Volume": vol,
            "BirthYear": birth_year,
            "Error": None,
        }
    except Exception as e:
        return {
            "Ticker": clean_sym,
            "Name": clean_sym,
            "Sector": "N/A",
            "Price": None,
            "MarketCap": None,
            "Volume": None,
            "BirthYear": None,
            "Error": str(e),
        }


def fetch_stocks_batch(tickers: list[str], max_workers: int = 20) -> list[dict]:
    """Fetches stock info concurrently for a list of tickers."""
    unique_tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t and t.strip()]))
    if not unique_tickers:
        return []

    workers = min(max_workers, len(unique_tickers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(fetch_single_stock_info, unique_tickers))
    return results


def evaluate_condition(value: float, target: float, operator: str) -> bool:
    """Evaluates a numeric condition given a target value and comparison operator."""
    if value is None or pd.isna(value) or target is None or pd.isna(target):
        return False

    op = str(operator).strip()
    if op in (">=", "≥", "min", "greater_equal"):
        return value >= target
    elif op in ("<=", "≤", "max", "less_equal", "inferiore"):
        return value <= target
    elif op in (">", "greater"):
        return value > target
    elif op in ("<", "less"):
        return value < target
    elif op in ("==", "=", "equal", "uguale"):
        return value == target
    return True


def filter_discovered_stocks(
    stocks: list[dict],
    vol_target: float = None,
    vol_unit: str = "1",
    vol_op: str = ">=",
    mc_target: float = None,
    mc_unit: str = "mln",
    mc_op: str = ">=",
    birth_year_target: int = None,
    birth_year_op: str = "<=",
) -> list[dict]:
    """Filters stocks by daily volume, market cap, and birth year."""
    filtered = []

    # Effective targets
    raw_vol_target = (vol_target * get_unit_multiplier(vol_unit)) if vol_target is not None else None
    raw_mc_target = (mc_target * get_unit_multiplier(mc_unit)) if mc_target is not None else None

    for stock in stocks:
        if stock.get("Error") and stock.get("Price") is None:
            continue

        # Check Volume
        if raw_vol_target is not None:
            if not evaluate_condition(stock.get("Volume"), raw_vol_target, vol_op):
                continue

        # Check Market Cap
        if raw_mc_target is not None:
            if not evaluate_condition(stock.get("MarketCap"), raw_mc_target, mc_op):
                continue

        # Check Birth Year
        if birth_year_target is not None:
            if not evaluate_condition(stock.get("BirthYear"), birth_year_target, birth_year_op):
                continue

        filtered.append(stock)

    return filtered


def format_market_cap(mc: float) -> str:
    """Formats market cap float into human-readable string (e.g. $1.50B or $250.00M)."""
    if mc is None or pd.isna(mc):
        return "N/A"
    if mc >= 1_000_000_000:
        return f"${mc / 1_000_000_000:.2f}B"
    elif mc >= 1_000_000:
        return f"${mc / 1_000_000:.2f}M"
    elif mc >= 1_000:
        return f"${mc / 1_000:.2f}K"
    else:
        return f"${mc:.2f}"


def format_volume(vol: float) -> str:
    """Formats volume float into human-readable string (e.g. 10.50M or 500.00K)."""
    if vol is None or pd.isna(vol):
        return "N/A"
    if vol >= 1_000_000_000:
        return f"{vol / 1_000_000_000:.2f}B"
    elif vol >= 1_000_000:
        return f"{vol / 1_000_000:.2f}M"
    elif vol >= 1_000:
        return f"{vol / 1_000:.2f}K"
    else:
        return f"{int(vol):,}"
