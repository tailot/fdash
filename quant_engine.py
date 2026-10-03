import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def calcola_quant_trend_ticker(ticker: str, period: str = "1mo", horizon: int = 5) -> dict:
    """
    Scarica i dati storici giornalieri di yfinance per il ticker fornito e calcola:
      - Ultimo_Prezzo
      - Trend_TimesFM (BUY / SELL / EQUILIBRIO basato su medie mobili e rendimento recente)
      - Sigma_% / Incertezza_Sigma_% (Volatilità percentualizzata)
      - Proiezioni probabilistiche multi-giorno (TimesFM/quantiliche):
        * Data_Previsione: data futura a +horizon giorni lavorativi
        * P10: prezzo quantile 10% (scenario ribassista)
        * Mediana: prezzo medio previsto
        * P90: prezzo quantile 90% (scenario rialzista)
        * Rend_Mediano_%: rendimento mediano atteso percentualizzato
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty or len(df) < 5:
        df = yf.download(ticker, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 2:
        raise ValueError(f"Dati storici insufficienti per il ticker {ticker}.")

    df = df.dropna()
    close_prices = df["Close"].astype(float)
    ultimo_prezzo = float(close_prices.iloc[-1])

    # Calcolo dei rendimenti giornalieri e della volatilità (Sigma %)
    returns = close_prices.pct_change().dropna()
    daily_sigma = float(returns.std()) if len(returns) > 1 else 0.02
    if np.isnan(daily_sigma) or daily_sigma == 0:
        daily_sigma = 0.02

    sigma_pct = float(daily_sigma * np.sqrt(252) * 100)

    # Definizione del Trend
    short_ma = float(close_prices.tail(5).mean())
    long_ma = float(close_prices.tail(20).mean()) if len(close_prices) >= 20 else float(close_prices.mean())

    ret_totale = float((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100)

    if short_ma > long_ma and ret_totale > 1.0:
        trend = "BUY"
        drift_giornaliero = 0.0015
    elif short_ma < long_ma and ret_totale < -1.0:
        trend = "SELL"
        drift_giornaliero = -0.0015
    else:
        trend = "EQUILIBRIO"
        drift_giornaliero = 0.0

    # Calcolo Proiezioni Quantiliche a +horizon giorni
    # Data previsione (+horizon giorni lavorativi)
    data_rif = df.index[-1]
    if isinstance(data_rif, pd.Timestamp):
        data_previsione = (data_rif + pd.tseries.offsets.BDay(horizon)).strftime("%Y-%m-%d")
    else:
        data_previsione = (datetime.now() + timedelta(days=horizon*7//5)).strftime("%Y-%m-%d")

    mu_cumulato = drift_giornaliero * horizon
    sigma_cumulato = daily_sigma * np.sqrt(horizon)

    # Quantili
    z_80 = 1.28155  # 80% CI (P10 e P90)
    mediana = ultimo_prezzo * np.exp(mu_cumulato)
    p10 = ultimo_prezzo * np.exp(mu_cumulato - z_80 * sigma_cumulato)
    p90 = ultimo_prezzo * np.exp(mu_cumulato + z_80 * sigma_cumulato)
    rend_mediano_pct = (np.exp(mu_cumulato) - 1) * 100

    return {
        "Ticker": ticker.upper(),
        "Ultimo_Prezzo": round(ultimo_prezzo, 2),
        "Trend_TimesFM": trend,
        "Sigma_%": round(sigma_pct, 2),
        "Incertezza_Sigma_%": round(sigma_pct, 2),
        "Data_Previsione": data_previsione,
        "P10": round(p10, 2),
        "Mediana": round(mediana, 2),
        "P90": round(p90, 2),
        "Rend_Mediano_%": round(rend_mediano_pct, 2)
    }
