"""
Motore volumi: replica il Colab "analisi_volumi_buy_sell".

Stessa logica del notebook:
  * dati a 1 minuto (Yahoo: ~8 giorni), solo orari regolari, volume > 0
  * stima buy/sell con 3 metodi (candela / clv / tick) e verdetto con soglia di equilibrio
  * Volume Profile sul prezzo tipico (H+L+C)/3: % in perdita / guadagno, POC, VWAP, area di valore (VAL/VAH)
  * zone piu' affollate sopra/sotto il prezzo, top 5 zone, dettaglio per giorno, conclusioni testuali
"""
import numpy as np
import pandas as pd
import yfinance as yf


def stima_buy_sell(df: pd.DataFrame, metodo: str = "clv") -> pd.DataFrame:
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
    if delta_pct < -soglia:
        return "SELL"
    return "EQUILIBRIO"


def scarica_minuti(ticker: str, solo_orari_regolari: bool = True) -> pd.DataFrame:
    raw = yf.download(ticker, period="8d", interval="1m", prepost=not solo_orari_regolari,
                      auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"Nessun dato scaricato per il ticker '{ticker}'. Verifica il simbolo.")
    return raw


def prepara_minuti(raw: pd.DataFrame, giorni: int = 1, data_fine: str = ""):
    """Pulizia e selezione degli ultimi `giorni` (cella 3 del Colab). Ritorna (df, lista_giorni, avviso)."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[df["Volume"] > 0]
    df["giorno"] = df.index.date

    disponibili = sorted(df["giorno"].unique())
    if data_fine:
        limite = pd.Timestamp(data_fine).date()
        disponibili = [g for g in disponibili if g <= limite]
    scelti = disponibili[-giorni:]
    if not scelti:
        raise ValueError("Nessun giorno disponibile per la data scelta (i dati a 1 minuto coprono solo ~8 giorni).")
    avviso = ""
    if len(scelti) < giorni:
        avviso = f"Attenzione: disponibili solo {len(scelti)} giorni su {giorni} richiesti."
    return df[df["giorno"].isin(scelti)].copy(), scelti, avviso


def volume_profile(df: pd.DataFrame, prezzo_att: float, n_bin: int = 30, area_valore_pct: float = 70.0) -> dict:
    """Cella 8 del Colab: quote in perdita/guadagno, POC, VWAP, area di valore, zone affollate."""
    tp = ((df["High"] + df["Low"] + df["Close"]) / 3).to_numpy()
    vol = df["Volume"].to_numpy(dtype=float)
    vol_tot = vol.sum()

    perdita = vol[tp > prezzo_att].sum()
    guadagno = vol[tp < prezzo_att].sum()
    pari = vol_tot - perdita - guadagno

    lo, hi = tp.min(), tp.max()
    if hi == lo:                                   # titolo fermo: evita bin degeneri
        lo, hi = lo - 1e-6, hi + 1e-6
    bins = np.linspace(lo, hi, n_bin + 1)
    idx = np.clip(np.digitize(tp, bins) - 1, 0, n_bin - 1)
    profilo = np.bincount(idx, weights=vol, minlength=n_bin)
    centri = (bins[:-1] + bins[1:]) / 2

    poc = centri[profilo.argmax()]
    vwap = (tp * vol).sum() / vol_tot

    ordine = np.argsort(profilo)[::-1]
    cum = np.cumsum(profilo[ordine])
    n_area = np.searchsorted(cum, vol_tot * area_valore_pct / 100) + 1
    bin_area = ordine[:n_area]
    val, vah = centri[bin_area].min(), centri[bin_area].max()

    sopra = [i for i in range(n_bin) if centri[i] > prezzo_att]
    sotto = [i for i in range(n_bin) if centri[i] < prezzo_att]
    zona_sopra = max(sopra, key=lambda i: profilo[i]) if sopra else None
    zona_sotto = max(sotto, key=lambda i: profilo[i]) if sotto else None

    def fmt_zona(i):
        return f"{bins[i]:.2f} - {bins[i+1]:.2f} ({profilo[i]/vol_tot*100:.1f}% del volume)"

    top = sorted(np.argsort(profilo)[::-1][:5], key=lambda k: -profilo[k])
    zone = pd.DataFrame([{
        "Zona di prezzo": f"{bins[i]:.2f} - {bins[i+1]:.2f}",
        "% del volume": round(profilo[i] / vol_tot * 100, 1),
        "Posizione": "sopra il prezzo (in perdita)" if centri[i] > prezzo_att else "sotto il prezzo (in guadagno)",
        "Distanza dal prezzo %": round((centri[i] / prezzo_att - 1) * 100, 2),
    } for i in top])

    return {
        "pct_perdita": perdita / vol_tot * 100, "pct_guadagno": guadagno / vol_tot * 100,
        "pct_pari": pari / vol_tot * 100, "poc": poc, "vwap": vwap, "val": val, "vah": vah,
        "bins": bins, "centri": centri, "profilo": profilo,
        "zona_sopra": None if zona_sopra is None else fmt_zona(zona_sopra),
        "zona_sotto": None if zona_sotto is None else fmt_zona(zona_sotto),
        "zone_top5": zone,
    }


def conclusioni(pct_perdita: float, prezzo_att: float, val: float, vah: float):
    """Soglie identiche al Colab: >=60% perdita, <=40% guadagno, altrimenti misto."""
    if pct_perdita >= 60:
        maggioranza = "PERDITA"
    elif pct_perdita <= 40:
        maggioranza = "GUADAGNO"
    else:
        maggioranza = "MISTA"
    if prezzo_att > vah:
        posizione = "SOPRA"
    elif prezzo_att < val:
        posizione = "SOTTO"
    else:
        posizione = "DENTRO"
    return maggioranza, posizione


def dettaglio_per_giorno(df: pd.DataFrame, prezzo_att: float, soglia: float = 2.0) -> pd.DataFrame:
    """Tabelle per giorno delle celle 4 e 8 del Colab, unite."""
    tp_all = (df["High"] + df["Low"] + df["Close"]) / 3
    righe = []
    for g, d in df.groupby("giorno"):
        tot = d["Volume"].sum()
        b, s = d["buy"].sum(), d["sell"].sum()
        dp = (b - s) / tot * 100
        v = d["Volume"].to_numpy(dtype=float)
        p = tp_all.loc[d.index].to_numpy()
        righe.append({
            "Giorno": g, "Volume totale": int(tot), "Buy": int(b), "Sell": int(s), "Delta": int(b - s),
            "Delta %": round(dp, 2),
            "Minuti buy": int((d["delta"] > 0).sum()), "Minuti sell": int((d["delta"] < 0).sum()),
            "Vincitore": verdetto_delta(dp, soglia),
            "% in perdita": round(v[p > prezzo_att].sum() / v.sum() * 100, 1),
            "% in guadagno": round(v[p < prezzo_att].sum() / v.sum() * 100, 1),
            "VWAP giorno": round((p * v).sum() / v.sum(), 2),
        })
    return pd.DataFrame(righe).set_index("Giorno")


def calcola_microstruttura_ticker(ticker: str, giorni: int = 1, data_fine: str = "", metodo: str = "clv",
                                  solo_orari_regolari: bool = True, soglia_equilibrio: float = 2.0,
                                  n_bin: int = 30, prezzo_riferimento: float = 0.0, area_valore_pct: float = 70.0,
                                  raw: pd.DataFrame = None) -> dict:
    """
    Analisi completa di un ticker, equivalente a eseguire il Colab. `raw` permette di passare i minuti gia'
    scaricati (utile per i test); altrimenti vengono scaricati da yfinance.
    """
    if raw is None:
        raw = scarica_minuti(ticker, solo_orari_regolari)
    df, giorni_scelti, avviso = prepara_minuti(raw, giorni, data_fine)

    df = stima_buy_sell(df, metodo)
    tot_vol = df["Volume"].sum()
    buy_vol, sell_vol = df["buy"].sum(), df["sell"].sum()
    delta_pct = (buy_vol - sell_vol) / tot_vol * 100 if tot_vol > 0 else 0.0
    verdetto = verdetto_delta(delta_pct, soglia_equilibrio)

    ultimo_prezzo = float(df["Close"].iloc[-1])
    prezzo_att = prezzo_riferimento if prezzo_riferimento > 0 else ultimo_prezzo
    df["prezzo_tipico"] = (df["High"] + df["Low"] + df["Close"]) / 3

    vp = volume_profile(df, prezzo_att, n_bin, area_valore_pct)
    maggioranza, posizione = conclusioni(vp["pct_perdita"], prezzo_att, vp["val"], vp["vah"])

    return {
        "Ticker": ticker.upper(),
        "Ultimo_Prezzo": round(ultimo_prezzo, 2),
        "Prezzo_Riferimento": round(prezzo_att, 2),
        "Delta_Volumi_Intra": round(delta_pct, 2),
        "Verdetto_Volumi": verdetto,
        "Buy_%": round(buy_vol / tot_vol * 100, 1),
        "Sell_%": round(sell_vol / tot_vol * 100, 1),
        "%_Trader_In_Perdita": round(vp["pct_perdita"], 2),
        "%_Trader_In_Guadagno": round(vp["pct_guadagno"], 2),
        "Maggioranza_Acquirenti": maggioranza,          # PERDITA / GUADAGNO / MISTA
        "Posizione_Area_Valore": posizione,             # SOPRA / SOTTO / DENTRO
        "POC": round(vp["poc"], 2),
        "VWAP": round(vp["vwap"], 2),
        "VAL": round(vp["val"], 2),
        "VAH": round(vp["vah"], 2),
        "Volume_Totale": int(tot_vol),
        "Giorni_Analizzati": len(giorni_scelti),
        "Avviso": avviso,
        "zona_sopra": vp["zona_sopra"],
        "zona_sotto": vp["zona_sotto"],
        "zone_top5": vp["zone_top5"],
        "dettaglio_giorni": dettaglio_per_giorno(df, prezzo_att, soglia_equilibrio),
        "df_minuti": df,
        "volume_profile": vp,
    }
