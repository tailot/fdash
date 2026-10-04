import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from volume_engine import calcola_microstruttura_ticker
from quant_engine import calcola_previsioni, carica_timesfm3, HORIZON, SOGLIA_TREND, MODELLO_FALLBACK
from gemini_enrichment import genera_analisi_gemini

st.set_page_config(page_title="Dashboard Master di Analisi Finanziaria", page_icon="📈", layout="wide")

st.title("📊 Dashboard Master di Analisi Finanziaria Integrata")
st.markdown("""
Unisce le due analisi — **previsione TimesFM-3** (rendimenti log, quantili P10/P90) e
**microstruttura dei volumi** (buy/sell stimato + Volume Profile) — e le arricchisce con **Gemini 2.5 Flash**.
""")

@st.cache_resource(show_spinner="Caricamento TimesFM-3...")
def get_forecaster():
    return carica_timesfm3()


# ---------------------------------------------------------------------------------------------- sidebar
st.sidebar.header("⚙️ Opzioni")
api_key = st.sidebar.text_input("Chiave API Gemini (gemini-2.5-flash):", type="password",
                                value=os.environ.get("GEMINI_API_KEY", ""),
                                help="Senza chiave viene usata l'analisi euristica locale.")

st.sidebar.subheader("Universo")
universo = st.sidebar.radio("Titoli:", ["Lista manuale", "Top N Nasdaq per market cap"], index=0)
if universo == "Lista manuale":
    input_tickers = st.sidebar.text_input("Ticker (separati da virgola):", "MSTR, AAPL, NVDA, TSLA, MSFT")
    n_top = 100
else:
    input_tickers = ""
    n_top = st.sidebar.slider("N titoli:", 20, 100, 100, step=10)

st.sidebar.subheader("Previsione (TimesFM-3)")
horizon = st.sidebar.slider("Orizzonte (giorni di borsa):", 1, 20, HORIZON)
soglia_trend = st.sidebar.number_input("Soglia trend (x sigma cumulata)", 0.0, 1.0, SOGLIA_TREND, 0.05,
                                       help="Specifica della dashboard: BUY/SELL se |mediana| > soglia × sigma.")
usa_tfm = st.sidebar.checkbox("Usa TimesFM-3 (se installato)", value=True)

st.sidebar.subheader("Volumi (volumi al minuto)")
metodo = st.sidebar.selectbox("Metodo buy/sell:", ["clv", "candela", "tick"], index=0)
giorni_intra = st.sidebar.slider("Giorni analizzati:", 1, 5, 1)
data_fine = st.sidebar.text_input("Data fine (AAAA-MM-GG, vuoto = ultimo giorno):", "")
solo_regolari = st.sidebar.checkbox("Solo orari regolari", value=True)
soglia_eq = st.sidebar.number_input("Soglia equilibrio %", 0.0, 20.0, 2.0, 0.5)
n_bin = st.sidebar.slider("Fasce Volume Profile:", 10, 100, 30, step=5)
area_valore = st.sidebar.slider("Area di valore %:", 50, 90, 70, step=5)
prezzo_rif = st.sidebar.number_input("Prezzo di riferimento (0 = ultimo close):", 0.0, value=0.0)

if st.sidebar.button("Esegui Analisi Nativa", type="primary"):
    manuali = [t.strip().upper() for t in input_tickers.split(",") if t.strip()] or None
    forecaster = get_forecaster() if usa_tfm else None
    with st.spinner("Download prezzi e previsione..."):
        try:
            dfq, info = calcola_previsioni(manuali, n_tickers=n_top, horizon=horizon,
                                           forecaster=forecaster, soglia_trend=soglia_trend)
        except Exception as e:
            st.error(f"Previsione non riuscita: {e}")
            st.stop()

    righe_vol, errori = [], []
    barra = st.progress(0.0)
    stato = st.empty()
    for i, t in enumerate(dfq["Ticker"]):
        stato.text(f"Volumi al minuto: {t} ({i + 1}/{len(dfq)})")
        try:
            r = calcola_microstruttura_ticker(t, giorni=giorni_intra, data_fine=data_fine, metodo=metodo,
                                              solo_orari_regolari=solo_regolari, soglia_equilibrio=soglia_eq,
                                              n_bin=n_bin, prezzo_riferimento=prezzo_rif,
                                              area_valore_pct=area_valore)
            righe_vol.append(r)
        except Exception as e:
            errori.append(f"{t}: {e}")
        barra.progress((i + 1) / len(dfq))
    stato.empty()
    st.session_state["nativo"] = {"quant": dfq, "info": info, "vol": righe_vol, "errori": errori,
                                  "metodo": metodo}

nat = st.session_state.get("nativo")

# ---------------------------------------------------------------------------------------------- report
if not nat:
    st.info("👈 Imposta i parametri nella barra laterale e premi «Esegui Analisi Nativa».")
    st.stop()

df_quant = nat["quant"]
df_vol = pd.DataFrame([{k: v for k, v in r.items()
                        if not isinstance(v, (pd.DataFrame, dict)) and k != "zone_top5"}
                       for r in nat["vol"]]) if nat["vol"] else pd.DataFrame(
                           columns=["Ticker", "Delta_Volumi_Intra", "%_Trader_In_Perdita"])
if True:
    info = nat["info"]
    modello = df_quant["Modello"].iloc[0] if "Modello" in df_quant else ""
    st.caption(f"Universo: {info['fonte']} · {len(df_quant)} titoli · {info['giorni']} giorni "
               f"({info['dal']} → {info['al']}) · run {info['run_ts']} · modello: {modello}")
    if modello == MODELLO_FALLBACK:
        st.warning("TimesFM-3 non è installato: la previsione usa la baseline **Naive (0%)** "
                   "(mediana = ultimo prezzo, intervallo dalla volatilità storica), quindi il trend risulta "
                   "quasi sempre EQUILIBRIO. Installa `requirements-timesfm.txt` per utilizzare TimesFM-3.")
    if info["scartati"]:
        st.caption("Scartati (storico incompleto): " + ", ".join(info["scartati"][:40]))
    for e in nat["errori"]:
        st.warning(f"Volumi non disponibili — {e}")

# outer join: un ticker senza dati volumi resta nel report con valori mancanti (nessun dato inventato)
merged = pd.merge(df_quant, df_vol, on="Ticker", how="outer", suffixes=("", "_vol"))
if "Ultimo_Prezzo_vol" in merged.columns:
    merged["Ultimo_Prezzo"] = merged["Ultimo_Prezzo"].fillna(merged["Ultimo_Prezzo_vol"])
if merged.empty:
    st.error("Nessun dato disponibile.")
    st.stop()

for col in ["Ultimo_Prezzo", "Trend_TimesFM", "Sigma_%", "Data_Previsione", "P10", "Mediana", "P90",
            "Rend_Mediano_%", "Incertezza_Sigma_%", "Delta_Volumi_Intra", "%_Trader_In_Perdita"]:
    if col not in merged.columns:
        merged[col] = np.nan

st.subheader("1. Arricchimento (Gemini 2.5 Flash / euristica)")


@st.cache_data(show_spinner=False)
def _arricchisci(righe: tuple, key: str):
    out = []
    for r in righe:
        out.append(genera_analisi_gemini(*r[:6], api_key=key, rend_mediano_pct=r[6], incertezza_pct=r[7]))
    return out


chiavi = ["Ticker", "Ultimo_Prezzo", "Trend_TimesFM", "Sigma_%", "Delta_Volumi_Intra", "%_Trader_In_Perdita",
          "Rend_Mediano_%", "Incertezza_Sigma_%"]
righe_in = tuple(tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in row)
                 for row in merged[chiavi].itertuples(index=False, name=None))
with st.spinner("Generazione sentiment e sintesi..."):
    arr = _arricchisci(righe_in, api_key)
merged["Sentiment News"] = [a["Sentiment News"] for a in arr]
merged["Sintesi / Verdetto"] = [a["Sintesi / Verdetto"] for a in arr]

renamed = merged.rename(columns={
    "Ultimo_Prezzo": "Ultimo Prezzo", "Trend_TimesFM": "Trend TimesFM", "Sigma_%": "Sigma %",
    "Delta_Volumi_Intra": "Delta Volumi Intra", "%_Trader_In_Perdita": "% Trader in Perdita",
    "Data_Previsione": "Data previsione", "Rend_Mediano_%": "Rend. mediano %",
    "Incertezza_Sigma_%": "Incertezza (sigma) %",
    "Verdetto_Volumi": "Verdetto Volumi", "Maggioranza_Acquirenti": "Maggioranza acquirenti",
    "Posizione_Area_Valore": "Posizione vs area valore",
})

colonne = ["Ticker", "Ultimo Prezzo", "Trend TimesFM", "Sigma %", "Data previsione",
           "P10", "Mediana", "P90", "Rend. mediano %", "Incertezza (sigma) %",
           "Delta Volumi Intra", "Verdetto Volumi", "% Trader in Perdita", "Maggioranza acquirenti",
           "Posizione vs area valore", "POC", "VWAP", "VAL", "VAH",
           "Sentiment News", "Sintesi / Verdetto"]
final_df = renamed[[c for c in colonne if c in renamed.columns]]
if "Rend. mediano %" in final_df:
    final_df = final_df.sort_values("Rend. mediano %", ascending=False, na_position="last").reset_index(drop=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ticker analizzati", len(final_df))
c2.metric("Trend BUY", int((final_df["Trend TimesFM"] == "BUY").sum()))
c3.metric("Delta volumi medio %", f"{final_df['Delta Volumi Intra'].mean():.2f}%")
c4.metric("% media volume in perdita", f"{final_df['% Trader in Perdita'].mean():.2f}%")

st.subheader("2. Tabella Report Finale Integrato")
st.dataframe(final_df, width="stretch")
st.download_button("📥 Scarica report_finale_integrato.csv", final_df.to_csv(index=False).encode("utf-8"),
                   file_name="report_finale_integrato.csv", mime="text/csv", type="primary")

st.subheader("3. Grafici comparativi")
g1, g2 = st.columns(2)
with g1:
    st.markdown("#### Delta volumi intraday % (buy − sell)")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    d = final_df["Delta Volumi Intra"].fillna(0)
    ax1.bar(final_df["Ticker"], d, color=["#2e9e5b" if v >= 0 else "#d6453d" for v in d])
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.set_ylabel("Delta volumi %")
    plt.setp(ax1.get_xticklabels(), rotation=90, fontsize=7)
    st.pyplot(fig1)
with g2:
    st.markdown("#### % volume in perdita (Volume Profile)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    p = final_df["% Trader in Perdita"].fillna(0)
    ax2.bar(final_df["Ticker"], p, color=["#d6453d" if v >= 60 else ("#2e9e5b" if v <= 40 else "#e08a00") for v in p])
    ax2.axhline(60, color="#d6453d", ls=":", label="≥60% maggioranza in perdita")
    ax2.axhline(40, color="#2e9e5b", ls=":", label="≤40% maggioranza in guadagno")
    ax2.set_ylabel("% volume in perdita")
    ax2.legend(fontsize=7)
    plt.setp(ax2.get_xticklabels(), rotation=90, fontsize=7)
    st.pyplot(fig2)

# Dettaglio per ticker
if nat and nat["vol"]:
    st.subheader("4. Dettaglio ticker (Volume Profile e per giorno)")
    scelto = st.selectbox("Ticker:", [r["Ticker"] for r in nat["vol"]])
    r = next(x for x in nat["vol"] if x["Ticker"] == scelto)
    if r["Avviso"]:
        st.warning(r["Avviso"])
    st.markdown(
        f"**{scelto}** · prezzo di riferimento {r['Prezzo_Riferimento']:.2f} · metodo `{nat['metodo']}` — "
        f"Buy {r['Buy_%']}% | Sell {r['Sell_%']}% | Delta {r['Delta_Volumi_Intra']:+.2f}% → **{r['Verdetto_Volumi']}**")
    st.markdown(
        f"POC **{r['POC']}** · VWAP **{r['VWAP']}** · area di valore **{r['VAL']} – {r['VAH']}** · "
        f"in perdita **{r['%_Trader_In_Perdita']}%** / in guadagno **{r['%_Trader_In_Guadagno']}%** · "
        f"acquirenti: **{r['Maggioranza_Acquirenti']}** · prezzo **{r['Posizione_Area_Valore']}** l'area di valore")
    if r["zona_sopra"]:
        st.caption(f"Zona più affollata SOPRA il prezzo (possibile resistenza): {r['zona_sopra']}")
    if r["zona_sotto"]:
        st.caption(f"Zona più affollata SOTTO il prezzo (possibile supporto): {r['zona_sotto']}")

    vp = r["volume_profile"]
    fig, ax = plt.subplots(figsize=(8, 5))
    pa = r["Prezzo_Riferimento"]
    ax.barh(vp["centri"], vp["profilo"], height=(vp["bins"][1] - vp["bins"][0]) * 0.95,
            color=["#d6453d" if c > pa else "#2e9e5b" for c in vp["centri"]], edgecolor="white", linewidth=0.5)
    ax.axhline(pa, color="black", lw=2, label=f"Prezzo {pa:.2f}")
    ax.axhline(vp["poc"], color="#3b6fd6", ls="--", lw=1.5, label=f"POC {vp['poc']:.2f}")
    ax.axhline(vp["vwap"], color="#e08a00", ls=":", lw=2, label=f"VWAP {vp['vwap']:.2f}")
    ax.axhspan(vp["val"], vp["vah"], color="gray", alpha=0.15, label="Area di valore")
    ax.set_title("Volume Profile — rosso: acquirenti in PERDITA | verde: in GUADAGNO")
    ax.legend(loc="lower right", fontsize=8)
    st.pyplot(fig)

    t1, t2 = st.columns(2)
    t1.markdown("**Le 5 zone con più volume**")
    t1.dataframe(r["zone_top5"].set_index("Zona di prezzo"), width="stretch")
    t2.markdown("**Dettaglio per giorno**")
    t2.dataframe(r["dettaglio_giorni"], width="stretch")

st.caption("Buy/sell è una stima dalle candele a 1 minuto (non vero order flow); 'in perdita/guadagno' considera solo "
           "i volumi del periodo. Analisi statistica, non un consiglio di investimento. I pesi di TimesFM-3 sono "
           "sotto licenza non commerciale.")
