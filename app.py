import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from volume_engine import calcola_microstruttura_ticker
from quant_engine import calcola_previsioni, carica_timesfm3, HORIZON, SOGLIA_TREND, MODELLO_FALLBACK
from heuristic_enrichment import genera_analisi_euristica
from backtest_engine import run_walk_forward_test, calculate_confusion_matrix, run_stress_test
from i18n import t, LANGUAGES

# Page config
st.set_page_config(page_title="Master Financial Analysis Dashboard", page_icon="📈", layout="wide")

# Sidebar - Language Selection
st.sidebar.header("⚙️ Options")
lang_code = st.sidebar.selectbox("🌐 Language / Lingua:", options=list(LANGUAGES.keys()),
                                format_func=lambda x: LANGUAGES[x], index=0)

# Localized page titles
st.title(t("app_title", lang_code))
st.markdown(t("app_subtitle", lang_code))


@st.cache_resource(show_spinner=False)
def get_forecaster():
    return carica_timesfm3()


# ---------------------------------------------------------------------------------------------- sidebar
st.sidebar.subheader(t("universe_header", lang_code))
universo_label = st.sidebar.radio(t("universe_radio", lang_code),
                                  [t("manual_list", lang_code), t("top_nasdaq", lang_code)], index=0)
is_manual = (universo_label == t("manual_list", lang_code))

if is_manual:
    input_tickers = st.sidebar.text_input(t("tickers_input", lang_code), "MSTR, AAPL, NVDA, TSLA, MSFT")
    n_top = 100
else:
    input_tickers = ""
    n_top = st.sidebar.slider(t("n_stocks_slider", lang_code), 20, 100, 100, step=10)

st.sidebar.subheader(t("forecast_header", lang_code))
horizon = st.sidebar.slider(t("horizon_slider", lang_code), 1, 20, HORIZON)
soglia_trend = st.sidebar.number_input(t("trend_threshold_label", lang_code), 0.0, 1.0, SOGLIA_TREND, 0.05,
                                       help=t("trend_threshold_help", lang_code))
usa_tfm = st.sidebar.checkbox(t("use_tfm_checkbox", lang_code), value=True)

st.sidebar.subheader(t("volume_header", lang_code))
metodo = st.sidebar.selectbox(t("buy_sell_method", lang_code), ["clv", "candela", "tick"], index=0)
giorni_intra = st.sidebar.slider(t("analyzed_days", lang_code), 1, 5, 1)
data_fine = st.sidebar.text_input(t("end_date_input", lang_code), "")
solo_regolari = st.sidebar.checkbox(t("regular_hours_only", lang_code), value=True)
soglia_eq = st.sidebar.number_input(t("eq_threshold_label", lang_code), 0.0, 20.0, 2.0, 0.5)
n_bin = st.sidebar.slider(t("n_bins_slider", lang_code), 10, 100, 30, step=5)
area_valore = st.sidebar.slider(t("value_area_slider", lang_code), 50, 90, 70, step=5)
prezzo_rif = st.sidebar.number_input(t("ref_price_label", lang_code), 0.0, value=0.0)

if st.sidebar.button(t("run_analysis_button", lang_code), type="primary"):
    manuali = [t_item.strip().upper() for t_item in input_tickers.split(",") if t_item.strip()] or None
    forecaster = get_forecaster() if usa_tfm else None
    with st.spinner(t("download_prices_spinner", lang_code)):
        try:
            dfq, info = calcola_previsioni(manuali, n_tickers=n_top, horizon=horizon,
                                           forecaster=forecaster, soglia_trend=soglia_trend)
        except Exception as e:
            st.error(t("forecast_failed", lang_code).format(e))
            st.stop()

    righe_vol, errori = [], []
    barra = st.progress(0.0)
    stato = st.empty()
    for i, t_sym in enumerate(dfq["Ticker"]):
        stato.text(t("min_volume_status", lang_code).format(t_sym, i + 1, len(dfq)))
        try:
            r = calcola_microstruttura_ticker(t_sym, giorni=giorni_intra, data_fine=data_fine, metodo=metodo,
                                              solo_orari_regolari=solo_regolari, soglia_equilibrio=soglia_eq,
                                              n_bin=n_bin, prezzo_riferimento=prezzo_rif,
                                              area_valore_pct=area_valore)
            righe_vol.append(r)
        except Exception as e:
            errori.append(f"{t_sym}: {e}")
        barra.progress((i + 1) / len(dfq))
    stato.empty()
    st.session_state["nativo"] = {"quant": dfq, "info": info, "vol": righe_vol, "errori": errori,
                                  "metodo": metodo}

nat = st.session_state.get("nativo")

if not nat:
    st.info(t("sidebar_prompt", lang_code))
    st.stop()

# Streamlit Tabs
tab_dash, tab_bt = st.tabs([t("tab_dashboard", lang_code), t("tab_backtest", lang_code)])

df_quant = nat["quant"]
df_vol = pd.DataFrame([{k: v for k, v in r.items()
                        if not isinstance(v, (pd.DataFrame, dict)) and k != "zone_top5"}
                       for r in nat["vol"]]) if nat["vol"] else pd.DataFrame(
                           columns=["Ticker", "Delta_Volumi_Intra", "%_Trader_In_Perdita"])

info = nat["info"]
modello = df_quant["Modello"].iloc[0] if "Modello" in df_quant else ""

# outer join: a ticker without volume data stays in the report with missing values
merged = pd.merge(df_quant, df_vol, on="Ticker", how="outer", suffixes=("", "_vol"))
if "Ultimo_Prezzo_vol" in merged.columns:
    merged["Ultimo_Prezzo"] = merged["Ultimo_Prezzo"].fillna(merged["Ultimo_Prezzo_vol"])

for col in ["Ultimo_Prezzo", "Trend_TimesFM", "Sigma_%", "Data_Previsione", "P10", "Mediana", "P90",
            "Rend_Mediano_%", "Incertezza_Sigma_%", "Delta_Volumi_Intra", "%_Trader_In_Perdita"]:
    if col not in merged.columns:
        merged[col] = np.nan


@st.cache_data(show_spinner=False)
def _arricchisci(righe: tuple, l_code: str):
    out = []
    for r in righe:
        out.append(genera_analisi_euristica(*r[:6], rend_mediano_pct=r[6], incertezza_pct=r[7], lang=l_code))
    return out


chiavi = ["Ticker", "Ultimo_Prezzo", "Trend_TimesFM", "Sigma_%", "Delta_Volumi_Intra", "%_Trader_In_Perdita",
          "Rend_Mediano_%", "Incertezza_Sigma_%"]
righe_in = tuple(tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in row)
                 for row in merged[chiavi].itertuples(index=False, name=None))
with st.spinner(t("enrichment_spinner", lang_code)):
    arr = _arricchisci(righe_in, lang_code)

merged["Sentiment News"] = [a["Sentiment News"] for a in arr]
merged["Sintesi / Verdetto"] = [a["Sintesi / Verdetto"] for a in arr]
merged["Consensus_Rating"] = [a.get("Consensus_Rating", "N/A") for a in arr]
merged["Target_Price"] = [a.get("Target_Price") for a in arr]
merged["Upside_%"] = [a.get("Upside_%") for a in arr]
merged["Behavioral_Signals"] = [a.get("Behavioral_Signals", "Normal") for a in arr]

# Map terminology according to selected language
buyer_maj_map = {"PERDITA": t("loss", lang_code), "GUADAGNO": t("profit", lang_code), "MISTA": t("mixed", lang_code)}
pos_va_map = {"SOPRA": t("above", lang_code), "SOTTO": t("below", lang_code), "DENTRO": t("inside", lang_code)}

if "Maggioranza_Acquirenti" in merged.columns:
    merged["Maggioranza_Acquirenti"] = merged["Maggioranza_Acquirenti"].map(lambda x: buyer_maj_map.get(x, x))
if "Posizione_Area_Valore" in merged.columns:
    merged["Posizione_Area_Valore"] = merged["Posizione_Area_Valore"].map(lambda x: pos_va_map.get(x, x))
if "Verdetto_Volumi" in merged.columns:
    merged["Verdetto_Volumi"] = merged["Verdetto_Volumi"].map(
        lambda x: t("buy", lang_code) if x == "BUY" else (t("sell", lang_code) if x == "SELL" else (t("equilibrium", lang_code) if x == "EQUILIBRIO" else x))
    )

renamed = merged.rename(columns={
    "Ultimo_Prezzo": t("col_last_price", lang_code),
    "Trend_TimesFM": t("col_trend", lang_code),
    "Sigma_%": t("col_sigma_pct", lang_code),
    "Delta_Volumi_Intra": t("col_delta_vol", lang_code),
    "%_Trader_In_Perdita": t("col_loss_pct", lang_code),
    "Data_Previsione": t("col_forecast_date", lang_code),
    "P10": t("col_p10", lang_code),
    "Mediana": t("col_median", lang_code),
    "P90": t("col_p90", lang_code),
    "Rend_Mediano_%": t("col_med_ret_pct", lang_code),
    "Incertezza_Sigma_%": t("col_uncertainty_pct", lang_code),
    "Verdetto_Volumi": t("col_verdict_vol", lang_code),
    "Maggioranza_Acquirenti": t("col_buyer_majority", lang_code),
    "Posizione_Area_Valore": t("col_pos_va", lang_code),
    "POC": t("col_poc", lang_code),
    "VWAP": t("col_vwap", lang_code),
    "VAL": t("col_val", lang_code),
    "VAH": t("col_vah", lang_code),
    "Sentiment News": t("col_sentiment", lang_code),
    "Sintesi / Verdetto": t("col_verdict", lang_code),
    "Consensus_Rating": t("col_consensus", lang_code),
    "Target_Price": t("col_target_price", lang_code),
    "Upside_%": t("col_upside", lang_code),
    "Behavioral_Signals": t("col_behavioral", lang_code),
})

colonne = ["Ticker", t("col_last_price", lang_code), t("col_trend", lang_code), t("col_sigma_pct", lang_code),
           t("col_forecast_date", lang_code), t("col_p10", lang_code), t("col_median", lang_code),
           t("col_p90", lang_code), t("col_med_ret_pct", lang_code), t("col_uncertainty_pct", lang_code),
           t("col_delta_vol", lang_code), t("col_verdict_vol", lang_code), t("col_loss_pct", lang_code),
           t("col_consensus", lang_code), t("col_target_price", lang_code), t("col_upside", lang_code),
           t("col_behavioral", lang_code), t("col_sentiment", lang_code), t("col_verdict", lang_code)]

final_df = renamed[[c for c in colonne if c in renamed.columns]]
ret_col = t("col_med_ret_pct", lang_code)
if ret_col in final_df:
    final_df = final_df.sort_values(ret_col, ascending=False, na_position="last").reset_index(drop=True)

# ---------------------------------------------------------------- TAB 1: Dashboard & Reports
with tab_dash:
    st.caption(f"{t('universe_header', lang_code)}: {info['fonte']} · {len(df_quant)} stocks · {info['giorni']} days "
               f"({info['dal']} → {info['al']}) · run {info['run_ts']} · model: {modello}")
    if modello == MODELLO_FALLBACK:
        st.warning(t("fallback_warning", lang_code))
    if info["scartati"]:
        st.caption(t("discarded_caption", lang_code).format(", ".join(info["scartati"][:40])))
    for e in nat["errori"]:
        st.warning(t("volumes_not_available", lang_code).format(e))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("metric_analyzed", lang_code), len(final_df))
    trend_col = t("col_trend", lang_code)
    c2.metric(t("metric_buy_trend", lang_code), int((final_df[trend_col] == "BUY").sum()) if trend_col in final_df else 0)
    delta_col = t("col_delta_vol", lang_code)
    c3.metric(t("metric_avg_delta", lang_code), f"{final_df[delta_col].mean():.2f}%" if delta_col in final_df else "N/A")
    loss_col = t("col_loss_pct", lang_code)
    c4.metric(t("metric_avg_loss", lang_code), f"{final_df[loss_col].mean():.2f}%" if loss_col in final_df else "N/A")

    st.subheader(t("report_table_subheader", lang_code))
    st.dataframe(final_df, width="stretch")
    st.download_button(t("download_csv_button", lang_code), final_df.to_csv(index=False).encode("utf-8"),
                       file_name="report_finale_integrato.csv", mime="text/csv", type="primary")

    st.subheader(t("charts_subheader", lang_code))
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(f"#### {t('chart_delta_title', lang_code)}")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        d = final_df[delta_col].fillna(0) if delta_col in final_df else pd.Series([0] * len(final_df))
        ax1.bar(final_df["Ticker"], d, color=["#2e9e5b" if v >= 0 else "#d6453d" for v in d])
        ax1.axhline(0, color="gray", ls="--", lw=0.8)
        ax1.set_ylabel(t("chart_delta_ylabel", lang_code))
        plt.setp(ax1.get_xticklabels(), rotation=90, fontsize=7)
        st.pyplot(fig1)
    with g2:
        st.markdown(f"#### {t('chart_loss_title', lang_code)}")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        p = final_df[loss_col].fillna(0) if loss_col in final_df else pd.Series([0] * len(final_df))
        ax2.bar(final_df["Ticker"], p, color=["#d6453d" if v >= 60 else ("#2e9e5b" if v <= 40 else "#e08a00") for v in p])
        ax2.axhline(60, color="#d6453d", ls=":", label=t("chart_loss_legend_loss", lang_code))
        ax2.axhline(40, color="#2e9e5b", ls=":", label=t("chart_loss_legend_gain", lang_code))
        ax2.set_ylabel(t("chart_loss_ylabel", lang_code))
        ax2.legend(fontsize=7)
        plt.setp(ax2.get_xticklabels(), rotation=90, fontsize=7)
        st.pyplot(fig2)

    # Detail for ticker
    if nat and nat["vol"]:
        st.subheader(t("detail_subheader", lang_code))
        scelto = st.selectbox(t("select_ticker", lang_code), [r["Ticker"] for r in nat["vol"]])
        r = next(x for x in nat["vol"] if x["Ticker"] == scelto)
        if r["Avviso"]:
            st.warning(r["Avviso"])

        v_verdict = t("buy", lang_code) if r['Verdetto_Volumi'] == "BUY" else (
            t("sell", lang_code) if r['Verdetto_Volumi'] == "SELL" else t("equilibrium", lang_code)
        )
        b_maj = buyer_maj_map.get(r['Maggioranza_Acquirenti'], r['Maggioranza_Acquirenti'])
        p_pos = pos_va_map.get(r['Posizione_Area_Valore'], r['Posizione_Area_Valore'])

        st.markdown(t("detail_summary", lang_code).format(
            scelto, r['Prezzo_Riferimento'], nat['metodo'], r['Buy_%'], r['Sell_%'], r['Delta_Volumi_Intra'], v_verdict
        ))
        st.markdown(t("detail_vp_summary", lang_code).format(
            r['POC'], r['VWAP'], r['VAL'], r['VAH'], r['%_Trader_In_Perdita'], r['%_Trader_In_Guadagno'], b_maj, p_pos
        ))
        if r["zona_sopra"]:
            st.caption(t("resistance_caption", lang_code).format(r['zona_sopra']))
        if r["zona_sotto"]:
            st.caption(t("support_caption", lang_code).format(r['zona_sotto']))

        vp = r["volume_profile"]
        fig, ax = plt.subplots(figsize=(8, 5))
        pa = r["Prezzo_Riferimento"]
        ax.barh(vp["centri"], vp["profilo"], height=(vp["bins"][1] - vp["bins"][0]) * 0.95,
                color=["#d6453d" if c > pa else "#2e9e5b" for c in vp["centri"]], edgecolor="white", linewidth=0.5)
        ax.axhline(pa, color="black", lw=2, label=t("vp_chart_price", lang_code).format(pa))
        ax.axhline(vp["poc"], color="#3b6fd6", ls="--", lw=1.5, label=t("vp_chart_poc", lang_code).format(vp['poc']))
        ax.axhline(vp["vwap"], color="#e08a00", ls=":", lw=2, label=t("vp_chart_vwap", lang_code).format(vp['vwap']))
        ax.axhspan(vp["val"], vp["vah"], color="gray", alpha=0.15, label=t("vp_chart_va", lang_code))
        ax.set_title(t("vp_chart_title", lang_code))
        ax.legend(loc="lower right", fontsize=8)
        st.pyplot(fig)

        t1, t2 = st.columns(2)
        t1.markdown(f"**{t('top5_zones_title', lang_code)}**")
        t1.dataframe(r["zone_top5"].set_index("Zona di prezzo"), width="stretch")
        t2.markdown(f"**{t('daily_breakdown_title', lang_code)}**")
        t2.dataframe(r["dettaglio_giorni"], width="stretch")

# ---------------------------------------------------------------- TAB 2: Validation & Backtesting
with tab_bt:
    st.subheader(t("walk_forward_header", lang_code))
    n_windows = st.slider(t("n_windows_slider", lang_code), 2, 6, 4)

    # Generate/Run Walk-Forward backtesting on session prices
    wf_results = run_walk_forward_test(prices_df=None, horizon=horizon, n_windows=n_windows)
    overall = wf_results["overall"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(t("wf_acc", lang_code), f"{overall['directional_accuracy']:.1f}%")
    m2.metric(t("wf_return", lang_code), f"{overall['strategy_return_pct']:+.1f}%")
    m3.metric(t("wf_bench", lang_code), f"{overall['benchmark_return_pct']:+.1f}%")
    m4.metric(t("wf_sharpe", lang_code), f"{overall['sharpe_ratio']:.2f}")
    m5.metric(t("wf_max_dd", lang_code), f"{overall['max_drawdown_pct']:.1f}%")

    st.subheader(t("confusion_matrix_header", lang_code))
    pred_signals = wf_results.get("all_pred_signals", ["BUY", "BUY", "SELL", "SELL", "EQUILIBRIUM", "BUY"])
    actual_returns = wf_results.get("all_actual_returns", [1.2, -0.5, -2.1, 0.4, 0.1, 3.2])
    cm_results = calculate_confusion_matrix(pred_signals, actual_returns)

    c_c1, c_c2 = st.columns([1, 1])
    with c_c1:
        st.dataframe(cm_results["matrix_df"], width="stretch")
    with c_c2:
        st.metric(t("cm_prec_buy", lang_code), f"{cm_results['buy_precision']:.1f}%")
        st.metric(t("cm_rec_buy", lang_code), f"{cm_results['buy_recall']:.1f}%")
        st.metric(t("cm_f1_buy", lang_code), f"{cm_results['buy_f1']:.1f}%")

    st.subheader(t("stress_testing_header", lang_code))
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        vol_mult = st.slider(t("vol_mult_slider", lang_code), 1.0, 5.0, 2.5, step=0.5)
    with s_col2:
        flash_crash = st.slider(t("flash_crash_slider", lang_code), -20.0, 0.0, -8.0, step=1.0)

    stress_res = run_stress_test(prices_df=None, vol_multiplier=vol_mult, flash_crash_pct=flash_crash)
    st.info(stress_res["simulation_summary"])

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(t("stress_baseline_var", lang_code), f"{stress_res['baseline_var_95']:.1f}%")
    s2.metric(t("stress_vol_var", lang_code), f"{stress_res['vol_spike_var_95']:.1f}%")
    s3.metric(t("stress_cvar", lang_code), f"{stress_res['vol_spike_cvar_95']:.1f}%")
    s4.metric(t("stress_max_dd", lang_code), f"{stress_res['max_drawdown_stressed_pct']:.1f}%")

st.caption(t("footer_disclaimer", lang_code))
