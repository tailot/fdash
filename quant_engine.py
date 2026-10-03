import numpy as np
import pandas as pd
import yfinance as yf

def calcola_quant_trend_ticker(ticker: str, period: str = "1mo") -> dict:
    """
    Scarica i dati storici giornalieri di yfinance per il ticker fornito e calcola:
      - Ultimo_Prezzo
      - Trend_TimesFM (BUY / SELL / EQUILIBRIO basato sul rendimento recente e le medie mobili)
      - Sigma_% (Volatilità percentualizzata)
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty or len(df) < 5:
        # Fallback con download generico se history ritorna vuoto
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
    sigma_pct = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 1 else 0.0
    if np.isnan(sigma_pct) or sigma_pct == 0:
        sigma_pct = float(abs(close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100)

    # Definizione del Trend (Simulazione quantitativa)
    short_ma = float(close_prices.tail(5).mean())
    long_ma = float(close_prices.tail(20).mean()) if len(close_prices) >= 20 else float(close_prices.mean())

    ret_totale = float((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100)

    if short_ma > long_ma and ret_totale > 1.0:
        trend = "BUY"
    elif short_ma < long_ma and ret_totale < -1.0:
        trend = "SELL"
    else:
        trend = "EQUILIBRIO"

    return {
        "Ticker": ticker.upper(),
        "Ultimo_Prezzo": round(ultimo_prezzo, 2),
        "Trend_TimesFM": trend,
        "Sigma_%": round(sigma_pct, 2)
    }
