import numpy as np
import pandas as pd
import yfinance as yf

def stima_buy_sell(df: pd.DataFrame, metodo: str = "clv") -> pd.DataFrame:
    """
    Stima il volume di Buy e Sell minute-by-minute in base al metodo scelto.
    Metodi disponibili:
      - 'candela': buy se close > open, sell se close < open, 0.5 se pari.
      - 'clv': Close Location Value -> ripartizione proporzionale nel range high-low.
      - 'tick': confronto con il close precedente (tick rule).
    """
    d = df.copy()
    v = d["Volume"].astype(float)

    if metodo == "candela":
        buy_share = np.where(d["Close"] > d["Open"], 1.0,
                     np.where(d["Close"] < d["Open"], 0.0, 0.5))
    elif metodo == "clv":
        rng = (d["High"] - d["Low"]).replace(0, np.nan)
        clv = (((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / rng).fillna(0)
        buy_share = ((1 + clv) / 2).to_numpy()
    elif metodo == "tick":
        segno = np.sign(d["Close"].diff()).replace(0, np.nan).ffill().fillna(0)
        buy_share = np.where(segno > 0, 1.0, np.where(segno < 0, 0.0, 0.5))
    else:
        raise ValueError(f"Metodo non valido: {metodo}")

    d["buy"] = v * buy_share
    d["sell"] = v * (1 - buy_share)
    d["delta"] = d["buy"] - d["sell"]
    return d

def verdetto_delta(delta_pct: float, soglia: float = 2.0) -> str:
    if delta_pct > soglia:
        return "BUY"
    elif delta_pct < -soglia:
        return "SELL"
    return "EQUILIBRIO"

def calcola_microstruttura_ticker(ticker: str, giorni: int = 1, data_fine: str = "", metodo: str = "clv",
                                  solo_orari_regolari: bool = True, soglia_equilibrio: float = 2.0,
                                  n_bin: int = 30, prezzo_riferimento: float = 0.0, area_valore_pct: float = 70.0) -> dict:
    """
    Scarica dati intraday a 1m da yfinance per il ticker fornito e calcola:
      - Ultimo Prezzo
      - Delta Volumi Intra %
      - Verdetto Volumi (BUY / SELL / EQUILIBRIO)
      - % Trader in Perdita (Volume Profile)
      - Dettagli aggiuntivi (POC, VWAP, Area Valore, DataFrame per giorno)
    """
    raw = yf.download(ticker, period="8d", interval="1m",
                      prepost=not solo_orari_regolari,
                      auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"Nessun dato scaricato per il ticker '{ticker}'. Verifica il simbolo.")

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[df["Volume"] > 0]
    df["giorno"] = df.index.date

    giorni_disponibili = sorted(df["giorno"].unique())
    if data_fine:
        limite = pd.Timestamp(data_fine).date()
        giorni_disponibili = [g for g in giorni_disponibili if g <= limite]

    giorni_scelti = giorni_disponibili[-giorni:]
    if not giorni_scelti:
        raise ValueError(f"Nessun giorno disponibile per {ticker} con la data richiesta.")

    df = df[df["giorno"].isin(giorni_scelti)].copy()

    # 1. Buy/Sell Delta Estimation
    df = stima_buy_sell(df, metodo)
    tot_vol = df["Volume"].sum()
    buy_vol = df["buy"].sum()
    sell_vol = df["sell"].sum()
    delta_vol = buy_vol - sell_vol
    delta_pct = (delta_vol / tot_vol) * 100 if tot_vol > 0 else 0.0
    verdetto = verdetto_delta(delta_pct, soglia_equilibrio)

    ultimo_prezzo = float(df["Close"].iloc[-1])
    prezzo_ref = prezzo_riferimento if prezzo_riferimento > 0 else ultimo_prezzo

    # 2. Volume Profile & % Trader In Perdita
    df["prezzo_tipico"] = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp = df["prezzo_tipico"].to_numpy()
    vol = df["Volume"].to_numpy(dtype=float)

    vol_loss = vol[tp > prezzo_ref].sum()
    vol_gain = vol[tp < prezzo_ref].sum()
    pct_perdita = (vol_loss / tot_vol * 100) if tot_vol > 0 else 0.0
    pct_guadagno = (vol_gain / tot_vol * 100) if tot_vol > 0 else 0.0

    vwap = (tp * vol).sum() / tot_vol if tot_vol > 0 else ultimo_prezzo

    bins = np.linspace(tp.min(), tp.max(), n_bin + 1)
    idx = np.clip(np.digitize(tp, bins) - 1, 0, n_bin - 1)
    profilo = np.bincount(idx, weights=vol, minlength=n_bin)
    centri = (bins[:-1] + bins[1:]) / 2.0
    poc = centri[profilo.argmax()]

    return {
        "Ticker": ticker.upper(),
        "Ultimo_Prezzo": round(ultimo_prezzo, 2),
        "Delta_Volumi_Intra": round(delta_pct, 2),
        "Verdetto_Volumi": verdetto,
        "%_Trader_In_Perdita": round(pct_perdita, 2),
        "%_Trader_In_Guadagno": round(pct_guadagno, 2),
        "POC": round(poc, 2),
        "VWAP": round(vwap, 2),
        "Volume_Totale": int(tot_vol),
        "df_minuti": df
    }
