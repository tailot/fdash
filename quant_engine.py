"""
Motore quantitativo: pipeline per la previsione quantilica dei prezzi.

Logica di calcolo:
  * prezzi aggiustati (auto_adjust=True), storico 5y, calendario comune, buchi isolati riempiti
  * TimesFM-3 applicato ai RENDIMENTI LOG (ultimi CONTEXT_LEN giorni), non ai prezzi
  * decili del modello -> mediana (decile 0.5), P10 (0.1), P90 (0.9)
  * sigma per passo = (P90 - P10) / (2 * Z80); sigma cumulata = sqrt(cumsum(sigma^2))
  * prezzo previsto = ultimo * exp(mu_cum -/+ Z80 * sigma_cum)
  * universo = titoli NASDAQ per market cap decrescente (screener Nasdaq, fallback Wikipedia)

Se `timesfm3` non e' installato il motore NON inventa un trend: usa la baseline
"Naive (0%)" (mediana 0) con sigma storica, e lo dichiara nella colonna `Modello`.
"""
import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

Z80 = 1.2815515655446004          # quantile 90% della normale standard
CONTEXT_LEN = 1024                # giorni di rendimenti in input al modello
HISTORY = "5y"
HORIZON = 5
SOGLIA_TREND = 0.10               # NOTA: specifica della dashboard, vedi _trend_da_previsione

MODELLO_TFM3 = "TimesFM-3"
MODELLO_FALLBACK = "Naive (0%) - TimesFM-3 non installato"

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
      "Accept": "application/json, text/plain, */*"}
SHARE_CLASS_DUPES = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA"}


# ----------------------------------------------------------------------------------------------
# Universo
# ----------------------------------------------------------------------------------------------
def _clean_universe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Ticker"] = (df["Ticker"].astype(str).str.strip().str.upper()
                    .str.replace("/", "-", regex=False).str.replace(".", "-", regex=False))
    df = df[df["Ticker"].str.fullmatch(r"[A-Z]{1,5}(-[A-Z])?")]
    df = df.dropna(subset=["MarketCap"]).sort_values("MarketCap", ascending=False).drop_duplicates("Ticker")
    present = set(df["Ticker"])
    df = df[~df["Ticker"].map(lambda t: SHARE_CLASS_DUPES.get(t) in present)]
    return df.reset_index(drop=True)


def universe_nasdaq_api() -> pd.DataFrame:
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=NASDAQ&download=true"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"]["rows"])
    mc = pd.to_numeric(df["marketCap"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    return _clean_universe(pd.DataFrame({"Ticker": df["symbol"], "MarketCap": mc}))


def universe_wikipedia() -> pd.DataFrame:
    r = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers=UA, timeout=30)
    r.raise_for_status()
    tab = next(t for t in pd.read_html(io.StringIO(r.text)) if "Ticker" in t.columns)
    syms = tab["Ticker"].astype(str).tolist()
    caps = []
    for s in syms:
        try:
            caps.append(float(getattr(yf.Ticker(s.replace(".", "-")).fast_info, "market_cap")))
        except Exception:
            caps.append(np.nan)
    return _clean_universe(pd.DataFrame({"Ticker": syms, "MarketCap": caps}))


def ricava_universo(n_tickers: int = 100, buffer: int = 20, tickers_manuali=None):
    """Ritorna (DataFrame Ticker/MarketCap, fonte, lista candidati)."""
    if tickers_manuali:
        uni = pd.DataFrame({"Ticker": [t.strip().upper() for t in tickers_manuali], "MarketCap": np.nan})
        return uni, "lista manuale", uni["Ticker"].tolist()
    uni, fonte = None, None
    for nome, fn in [("screener Nasdaq", universe_nasdaq_api), ("Wikipedia Nasdaq-100", universe_wikipedia)]:
        try:
            uni = fn()
            fonte = nome
            if len(uni) >= 30:
                break
        except Exception as e:  # noqa: BLE001 - si passa alla fonte successiva
            warnings.warn(f"{nome}: non disponibile ({type(e).__name__}: {e})")
    if uni is None or len(uni) < 30:
        raise RuntimeError("Universo non ricavato: usa una lista manuale di ticker.")
    return uni, fonte, uni.head(n_tickers + buffer)["Ticker"].tolist()


# ----------------------------------------------------------------------------------------------
# Prezzi (calendario comune)
# ----------------------------------------------------------------------------------------------
def prepara_prezzi(raw: pd.DataFrame, cand: list, n_tickers: int):
    """Allineamento prezzi: tiene titoli con storico ~completo, ffill, dropna. Ritorna (prices, scartati)."""
    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw = raw.dropna(axis=1, thresh=int(0.99 * len(raw)))
    prices = raw.ffill().dropna()
    tickers = [t for t in cand if t in prices.columns][:n_tickers]
    scartati = [t for t in cand if t not in tickers]
    return prices[tickers], scartati


def scarica_prezzi(cand: list, n_tickers: int, history: str = HISTORY):
    raw = yf.download(cand, period=history, auto_adjust=True, progress=False, threads=True)["Close"]
    if isinstance(raw, pd.Series):          # un solo ticker
        raw = raw.to_frame(cand[0])
    return prepara_prezzi(raw, cand, n_tickers)


# ----------------------------------------------------------------------------------------------
# Modello
# ----------------------------------------------------------------------------------------------
def _cum_stats(med, q10, q90):
    sig = (q90 - q10) / (2 * Z80)
    return np.cumsum(med, axis=1), np.sqrt(np.cumsum(sig ** 2, axis=1))


def _log_returns(price_arrays, context_len=CONTEXT_LEN):
    return [np.diff(np.log(np.asarray(p, dtype=float)))[-context_len:].astype(np.float32) for p in price_arrays]


def carica_timesfm3(per_core_batch_size: int = 32):
    """Carica TimesFM-3. Ritorna il forecaster oppure None se il pacchetto non c'e'."""
    try:
        import torch
        from timesfm3 import TimesFM3Evaluator, ModelConfig
    except ImportError:
        return None
    torch.set_float32_matmul_precision("high")
    config = ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch",
                         per_core_batch_size=per_core_batch_size,
                         device="cuda" if torch.cuda.is_available() else "cpu")
    return TimesFM3Evaluator(config)


def tfm3_returns(forecaster, price_arrays, horizon, use_symmetric_averaging=False, context_len=CONTEXT_LEN):
    """TimesFM-3: rendimento log cumulato (mediana) e sigma cumulata per i passi 1..horizon."""
    outs = list(forecaster.predict_batch(_log_returns(price_arrays, context_len), horizon=horizon,
                                         return_quantiles=True,
                                         use_symmetric_averaging=use_symmetric_averaging))
    q = np.stack([np.asarray(o.quantiles) for o in outs])          # (n, horizon, 9): decili 0.1 ... 0.9
    assert q.shape[1:] == (horizon, 9), f"forma inattesa dei quantili: {q.shape}"
    return _cum_stats(q[:, :, 4], q[:, :, 0], q[:, :, 8])          # mediana = decile 0.5, P10, P90


def naive_returns(price_arrays, horizon, context_len=CONTEXT_LEN):
    """Baseline 'Naive (0%)' con sigma storica: mediana 0, intervallo simmetrico."""
    mu = np.zeros((len(price_arrays), horizon))
    sg_step = np.array([np.std(r[-252:], ddof=1) if len(r) > 2 else 0.02
                        for r in _log_returns(price_arrays, context_len)])
    sg_step = np.where(np.isfinite(sg_step) & (sg_step > 0), sg_step, 0.02)
    sig = np.repeat(sg_step[:, None], horizon, axis=1)
    return np.cumsum(mu, axis=1), np.sqrt(np.cumsum(sig ** 2, axis=1))


def _trend_da_previsione(mu: float, sg: float, soglia: float = SOGLIA_TREND) -> str:
    """Trend dalla previsione: BUY/SELL se la mediana si sposta di almeno `soglia` x sigma cumulata."""
    if sg <= 0:
        return "EQUILIBRIO"
    z = mu / sg
    if z > soglia:
        return "BUY"
    if z < -soglia:
        return "SELL"
    return "EQUILIBRIO"


# ----------------------------------------------------------------------------------------------
# Previsione finale (una riga per titolo)
# ----------------------------------------------------------------------------------------------
def tabella_previsioni(prices: pd.DataFrame, horizon: int = HORIZON, forecaster=None, mc_map=None,
                       use_symmetric_averaging: bool = False, soglia_trend: float = SOGLIA_TREND) -> pd.DataFrame:
    tickers = list(prices.columns)
    arrays = [prices[t].values for t in tickers]
    if forecaster is not None:
        cum_mu, cum_sig = tfm3_returns(forecaster, arrays, horizon, use_symmetric_averaging)
        modello = MODELLO_TFM3
    else:
        cum_mu, cum_sig = naive_returns(arrays, horizon)
        modello = MODELLO_FALLBACK

    end_date = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=horizon)[-1]  # non considera le festivita'
    mc_map = mc_map or {}
    rows = []
    for i, t in enumerate(tickers):
        last = float(prices[t].iloc[-1])
        mu, sg = float(cum_mu[i, -1]), float(cum_sig[i, -1])
        r = np.diff(np.log(prices[t].values))[-252:]
        sigma_ann = float(np.std(r, ddof=1) * np.sqrt(252) * 100) if len(r) > 2 else np.nan
        mc = mc_map.get(t, np.nan)
        rows.append({
            "Ticker": t,
            "Ultimo_Prezzo": round(last, 2),
            "Trend_TimesFM": _trend_da_previsione(mu, sg, soglia_trend),
            "Sigma_%": round(sigma_ann, 2),
            "Data_Previsione": end_date.strftime("%Y-%m-%d"),
            "P10": round(last * np.exp(mu - Z80 * sg), 2),
            "Mediana": round(last * np.exp(mu), 2),
            "P90": round(last * np.exp(mu + Z80 * sg), 2),
            "Rend_Mediano_%": round((np.exp(mu) - 1) * 100, 2),
            "Incertezza_Sigma_%": round(sg * 100, 2),
            "Modello": modello,
        })
    return (pd.DataFrame(rows).sort_values("Rend_Mediano_%", ascending=False, kind="stable")
            .reset_index(drop=True))


def calcola_previsioni(tickers=None, n_tickers: int = 100, buffer: int = 20, history: str = HISTORY,
                       horizon: int = HORIZON, forecaster=None, use_symmetric_averaging: bool = False,
                       soglia_trend: float = SOGLIA_TREND, min_tickers: int = 20):
    """
    Pipeline completa di previsione. `tickers=None` -> universo automatico (top n_tickers NASDAQ per market cap).
    Ritorna (df_previsioni, info) con info = {fonte, scartati, giorni, dal, al, run_ts}.
    """
    uni, fonte, cand = ricava_universo(n_tickers, buffer, tickers)
    n_target = len(cand) if tickers else n_tickers
    prices, scartati = scarica_prezzi(cand, n_target, history)
    if tickers is None and len(prices.columns) < min_tickers:
        raise RuntimeError("Troppi pochi titoli con dati completi.")
    if prices.shape[1] == 0:
        raise ValueError("Nessun titolo con dati sufficienti.")
    mc_map = dict(zip(uni["Ticker"], uni["MarketCap"]))
    df = tabella_previsioni(prices, horizon, forecaster, mc_map, use_symmetric_averaging, soglia_trend)
    info = {"fonte": fonte, "scartati": scartati, "giorni": len(prices),
            "dal": prices.index[0].date(), "al": prices.index[-1].date(),
            "run_ts": datetime.now().strftime("%Y-%m-%d %H:%M")}
    return df, info


def calcola_quant_trend_ticker(ticker: str, period: str = HISTORY, horizon: int = HORIZON, forecaster=None) -> dict:
    """Compatibilita' con la vecchia API: previsione per un solo ticker."""
    df, _ = calcola_previsioni([ticker], history=period, horizon=horizon, forecaster=forecaster)
    return df.iloc[0].to_dict()
