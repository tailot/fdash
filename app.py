import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from volume_engine import calcola_microstruttura_ticker
from quant_engine import calcola_quant_trend_ticker
from gemini_enrichment import genera_analisi_gemini

# Configurazione pagina Streamlit
st.set_page_config(
    page_title="Dashboard Master di Analisi Finanziaria",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Dashboard Master di Analisi Finanziaria Integrata")
st.markdown("""
Questa applicazione raccoglie, unisce e arricchisce i dati quantitativi (modello **TimesFM-3**)
e di microstruttura dei volumi intraday, arricchendoli con analisi qualitative generate tramite **Gemini 2.5 Flash**.
""")

# Sidebar - Impostazioni e Ingestion Dati
st.sidebar.header("⚙️ Opzioni & Ingestion Dati")

api_key = st.sidebar.text_input(
    "Chiave API Gemini (gemini-2.5-flash):",
    type="password",
    value=os.environ.get("GEMINI_API_KEY", ""),
    help="Inserisci la tua chiave API Google Gemini per abilitare l'arricchimento AI."
)

modalita_ingestion = st.sidebar.radio(
    "Seleziona Modalità Ingestion Dati:",
    ["Carica File CSV", "Leggi da Cartella Locale", "Calcolo Nativo (Motore Python)"],
    index=0
)

df_colab1 = None
df_colab2 = None

if modalita_ingestion == "Carica File CSV":
    st.sidebar.subheader("Carica i file CSV")
    file1 = st.sidebar.file_uploader("Output Colab 1 (TimesFM-3)", type=["csv"], key="file1")
    file2 = st.sidebar.file_uploader("Output Colab 2 (Volumi)", type=["csv"], key="file2")

    if file1 and file2:
        try:
            df_colab1 = pd.read_csv(file1)
            df_colab2 = pd.read_csv(file2)
        except Exception as e:
            st.error(f"Errore nella lettura dei file caricati: {e}")

elif modalita_ingestion == "Leggi da Cartella Locale":
    st.sidebar.subheader("Cartella Locale: data/")
    path_c1 = os.path.join("data", "colab1_timesfm.csv")
    path_c2 = os.path.join("data", "colab2_volumi.csv")

    if os.path.exists(path_c1) and os.path.exists(path_c2):
        try:
            df_colab1 = pd.read_csv(path_c1)
            df_colab2 = pd.read_csv(path_c2)
            st.sidebar.success("File trovati e caricati da 'data/'!")
        except Exception as e:
            st.sidebar.error(f"Errore durante il caricamento da data/: {e}")
    else:
        st.sidebar.warning("File CSV non trovati nella cartella 'data/'.")

elif modalita_ingestion == "Calcolo Nativo (Motore Python)":
    st.sidebar.subheader("Calcolo Nativo")
    input_tickers = st.sidebar.text_input("Ticker (separati da virgola):", "MSTR, AAPL, NVDA, TSLA, MSFT")
    metodo_volumi = st.sidebar.selectbox("Metodo Stima Buy/Sell Volumi:", ["clv", "candela", "tick"], index=0)
    giorni_intra = st.sidebar.slider("Giorni Intraday Analizzati:", min_value=1, max_value=5, value=1)

    if st.sidebar.button("Esegui Analisi Nativa"):
        tickers_list = [t.strip().upper() for t in input_tickers.split(",") if t.strip()]

        c1_data = []
        c2_data = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, t in enumerate(tickers_list):
            status_text.text(f"Elaborazione in corso per {t}...")
            try:
                # Calcolo Quant
                q_res = calcola_quant_trend_ticker(t)
                c1_data.append(q_res)

                # Calcolo Volumi Microstruttura
                v_res = calcola_microstruttura_ticker(t, giorni=giorni_intra, metodo=metodo_volumi)
                c2_data.append(v_res)
            except Exception as e:
                st.warning(f"Impossibile elaborare il ticker {t}: {e}")

            progress_bar.progress((idx + 1) / len(tickers_list))

        status_text.text("Elaborazione completata!")

        if c1_data and c2_data:
            df_colab1 = pd.DataFrame(c1_data)[["Ticker", "Ultimo_Prezzo", "Trend_TimesFM", "Sigma_%"]]
            df_colab2 = pd.DataFrame(c2_data)[["Ticker", "Delta_Volumi_Intra", "%_Trader_In_Perdita"]]

# Processamento e Join dei dati
if df_colab1 is not None and df_colab2 is not None:
    # Normalizzazione nomi colonna Ticker
    df_colab1["Ticker"] = df_colab1["Ticker"].astype(str).str.strip().str.upper()
    df_colab2["Ticker"] = df_colab2["Ticker"].astype(str).str.strip().str.upper()

    # Structural Join basato su Ticker
    merged_df = pd.merge(df_colab1, df_colab2, on="Ticker", how="inner")

    if merged_df.empty:
        st.error("Nessuna corrispondenza trovata tra i ticker nei due dataset.")
    else:
        st.subheader("1. Arricchimento tramite Gemini API (gemini-2.5-flash)")

        with st.spinner("Generazione sentiment e sintesi/verdetto tramite Gemini..."):
            sentiment_list = []
            verdetto_list = []

            for _, row in merged_df.iterrows():
                enrichment = genera_analisi_gemini(
                    ticker=row["Ticker"],
                    ultimo_prezzo=float(row.get("Ultimo_Prezzo", 0.0)),
                    trend_timesfm=str(row.get("Trend_TimesFM", "EQUILIBRIO")),
                    sigma_pct=float(row.get("Sigma_%", 0.0)),
                    delta_volumi_intra=float(row.get("Delta_Volumi_Intra", 0.0)),
                    pct_trader_in_perdita=float(row.get("%_Trader_In_Perdita", 0.0)),
                    api_key=api_key
                )
                sentiment_list.append(enrichment["Sentiment News"])
                verdetto_list.append(enrichment["Sintesi / Verdetto"])

            merged_df["Sentiment News"] = sentiment_list
            merged_df["Sintesi / Verdetto"] = verdetto_list

        # Rinominazione colonne per corrispondenza esatta
        renamed_df = merged_df.rename(columns={
            "Ultimo_Prezzo": "Ultimo Prezzo",
            "Trend_TimesFM": "Trend TimesFM",
            "Sigma_%": "Sigma %",
            "Delta_Volumi_Intra": "Delta Volumi Intra",
            "%_Trader_In_Perdita": "% Trader in Perdita"
        })

        exact_columns = [
            "Ticker", "Ultimo Prezzo", "Trend TimesFM", "Sigma %",
            "Delta Volumi Intra", "% Trader in Perdita", "Sentiment News", "Sintesi / Verdetto"
        ]

        # Filtra e ordina colonne esatte
        final_df = renamed_df[exact_columns]

        # Dashboard KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Totale Ticker Analizzati", len(final_df))
        col2.metric("Trend Positivi (BUY)", (final_df["Trend TimesFM"] == "BUY").sum())
        col3.metric("Delta Volumi Medio %", f"{final_df['Delta Volumi Intra'].mean():.2f}%")
        col4.metric("% Media Trader in Perdita", f"{final_df['% Trader in Perdita'].mean():.2f}%")

        st.subheader("2. Tabella Report Finale Integrato")
        st.dataframe(final_df, use_container_width=True)

        # Export CSV
        csv_data = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Scarica report_finale_integrato.csv",
            data=csv_data,
            file_name="report_finale_integrato.csv",
            mime="text/csv",
            type="primary"
        )

        st.subheader("3. Visualizzazione Grafica comparativa")
        g1, g2 = st.columns(2)

        with g1:
            st.markdown("#### Delta Volumi Intraday % per Ticker")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            colors1 = ["#2e9e5b" if v >= 0 else "#d6453d" for v in final_df["Delta Volumi Intra"]]
            ax1.bar(final_df["Ticker"], final_df["Delta Volumi Intra"], color=colors1)
            ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax1.set_ylabel("Delta Volumi %")
            st.pyplot(fig1)

        with g2:
            st.markdown("#### % Trader in Perdita per Ticker")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            colors2 = ["#d6453d" if v >= 50 else "#2e9e5b" for v in final_df["% Trader in Perdita"]]
            ax2.bar(final_df["Ticker"], final_df["% Trader in Perdita"], color=colors2)
            ax2.axhline(50, color="orange", linestyle=":", label="Soglia 50%")
            ax2.set_ylabel("% Trader in Perdita")
            ax2.legend()
            st.pyplot(fig2)

else:
    st.info("👈 Seleziona una modalità di ingestion dalla barra laterale per iniziare l'analisi.")
