import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from volume_engine import calcola_microstruttura_ticker
from quant_engine import calcola_previsioni, carica_timesfm3, HORIZON, SOGLIA_TREND, MODELLO_FALLBACK
from heuristic_enrichment import genera_analisi_euristica
from backtesting import backtest_temporale, riepilogo_backtest_temporale
from i18n import t, LANGUAGES
from db_engine import save_run, list_runs, get_run, delete_run
from alerts_ui import init_alerts_state, render_alert_monitor, render_alerts_section

import streamlit_authenticator as stauth

# Page config
st.set_page_config(page_title="Master Financial Analysis Dashboard", page_icon="📈", layout="wide")

# Authentication Guard
def _to_plain_dict(obj):
    if hasattr(obj, "items"):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_plain_dict(x) for x in obj]
    return obj


def _has_auth_config():
    try:
        return "credentials" in st.secrets and "cookie" in st.secrets
    except Exception:
        return False


if _has_auth_config():
    credentials_dict = _to_plain_dict(st.secrets["credentials"])

    authenticator = stauth.Authenticate(
        credentials_dict,
        st.secrets["cookie"]["name"],
        st.secrets["cookie"]["key"],
        st.secrets["cookie"]["expiration_days"]
    )

    authenticator.login(location="main")

    if st.session_state.get("authentication_status") is False:
        st.error("Username o password non validi.")
        st.stop()
    elif st.session_state.get("authentication_status") is None:
        st.warning("Inserisci le credenziali per accedere.")
        st.stop()

    # Authenticated session
    authenticator.logout(location="sidebar")
else:
    st.session_state.setdefault("name", "User")

# Sidebar - Language Selection
st.sidebar.header("⚙️ Options")
lang_code = st.sidebar.selectbox("🌐 Language / Lingua:", options=list(LANGUAGES.keys()),
                                format_func=lambda x: LANGUAGES[x], index=0)

st.sidebar.write(t("welcome", lang_code).format(st.session_state.get('name', 'User')))

# Section navigation (switchable at any time; analysis state is kept in st.session_state)
SECTIONS = ["analysis", "alerts"]
st.sidebar.radio(t("nav_label", lang_code), SECTIONS, key="section",
                 format_func=lambda k: t(f"nav_{k}", lang_code))
section = st.session_state.get("section", "analysis")

# Alerts monitor: rendered in every section, so price checks continue while the user is in "Analysis"
init_alerts_state()
with st.sidebar:
    render_alert_monitor(lang_code)
st.sidebar.divider()

# Localized page titles
st.title(t("app_title", lang_code))


@st.cache_resource(show_spinner=False)
def get_forecaster():
    return carica_timesfm3()


def render_market_snapshot(final_df: pd.DataFrame, trend_col: str, delta_col: str,
                           loss_col: str, ret_col: str, lang_code: str = "en"):
    """Market Snapshot: KPIs + leader boards for immediate market state visibility."""
    snapshot_df = final_df.copy()

    def compute_action(row):
        buy_signals = 0
        sell_signals = 0

        if trend_col in row and row[trend_col] == "BUY":
            buy_signals += 1
        elif trend_col in row and row[trend_col] == "SELL":
            sell_signals += 1

        if delta_col in row and pd.notna(row[delta_col]):
            if row[delta_col] > 0:
                buy_signals += 1
            elif row[delta_col] < 0:
                sell_signals += 1

        if loss_col in row and pd.notna(row[loss_col]):
            if row[loss_col] <= 40:
                buy_signals += 1
            elif row[loss_col] >= 60:
                sell_signals += 1

        if ret_col in row and pd.notna(row[ret_col]):
            if row[ret_col] > 0:
                buy_signals += 1
            elif row[ret_col] < 0:
                sell_signals += 1

        if buy_signals >= 3:
            return "BUY"
        elif sell_signals >= 3:
            return "SELL"
        elif max(buy_signals, sell_signals) >= 2:
            return "WATCH"
        else:
            return "WAIT"

    snapshot_df["action"] = snapshot_df.apply(compute_action, axis=1)

    st.subheader(t("market_snapshot_header", lang_code))

    k1, k2, k3, k4 = st.columns(4)
    total = len(snapshot_df)
    buy_cnt = (snapshot_df["action"] == "BUY").sum()
    sell_cnt = (snapshot_df["action"] == "SELL").sum()
    avg_delta = snapshot_df[delta_col].mean() if delta_col in snapshot_df and pd.notna(snapshot_df[delta_col].mean()) else 0.0

    k1.metric(t("snapshot_total_stocks", lang_code), total)
    k2.metric(t("snapshot_buy_signals", lang_code), buy_cnt, f"{(buy_cnt/total)*100:.1f}%" if total > 0 else "0%")
    k3.metric(t("snapshot_sell_signals", lang_code), sell_cnt, f"{(sell_cnt/total)*100:.1f}%" if total > 0 else "0%")
    k4.metric(t("snapshot_avg_delta", lang_code), f"{avg_delta:.1f}%")

    st.divider()

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown(f"#### {t('snapshot_top_buy', lang_code)}")
        if delta_col in snapshot_df.columns:
            top_buy = snapshot_df.nlargest(5, delta_col)[["Ticker", delta_col, trend_col, "action"]]
        else:
            top_buy = snapshot_df.head(5)[["Ticker", trend_col, "action"]].copy()
            top_buy[delta_col] = np.nan
        top_buy = top_buy.copy()
        top_buy.columns = ["Ticker", t("col_action_vol_delta", lang_code), t("col_action_trend", lang_code), t("col_action_action", lang_code)]
        top_buy[t("col_action_vol_delta", lang_code)] = top_buy[t("col_action_vol_delta", lang_code)].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
        st.dataframe(top_buy, width="stretch", hide_index=True)

    with right_col:
        st.markdown(f"#### {t('snapshot_top_sell', lang_code)}")
        if delta_col in snapshot_df.columns:
            top_sell = snapshot_df.nsmallest(5, delta_col)[["Ticker", delta_col, trend_col, "action"]]
        else:
            top_sell = snapshot_df.head(5)[["Ticker", trend_col, "action"]].copy()
            top_sell[delta_col] = np.nan
        top_sell = top_sell.copy()
        top_sell.columns = ["Ticker", t("col_action_vol_delta", lang_code), t("col_action_trend", lang_code), t("col_action_action", lang_code)]
        top_sell[t("col_action_vol_delta", lang_code)] = top_sell[t("col_action_vol_delta", lang_code)].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
        st.dataframe(top_sell, width="stretch", hide_index=True)

    st.divider()
    return snapshot_df


def render_action_table(final_df: pd.DataFrame, trend_col: str, delta_col: str,
                        loss_col: str, ret_col: str, price_col: str = "Ultimo_Prezzo",
                        lang_code: str = "en", key_suffix: str = "live"):
    """Actionable Signals: convergence-based decision matrix."""
    action_df = final_df.copy()
    action_df["buy_signals"] = 0
    action_df["sell_signals"] = 0

    if trend_col in action_df.columns:
        action_df.loc[action_df[trend_col] == "BUY", "buy_signals"] += 1
        action_df.loc[action_df[trend_col] == "SELL", "sell_signals"] += 1

    if delta_col in action_df.columns:
        action_df.loc[action_df[delta_col] > 0, "buy_signals"] += 1
        action_df.loc[action_df[delta_col] < 0, "sell_signals"] += 1

    if loss_col in action_df.columns:
        action_df.loc[action_df[loss_col] <= 40, "buy_signals"] += 1
        action_df.loc[action_df[loss_col] >= 60, "sell_signals"] += 1

    if ret_col in action_df.columns:
        action_df.loc[action_df[ret_col] > 0, "buy_signals"] += 1
        action_df.loc[action_df[ret_col] < 0, "sell_signals"] += 1

    def compute_action(row):
        b = row["buy_signals"]
        s = row["sell_signals"]

        if b >= 3 and s <= 1:
            return "BUY"
        elif s >= 3 and b <= 1:
            return "SELL"
        elif max(b, s) >= 2:
            return "WATCH"
        else:
            return "WAIT"

    action_df["action"] = action_df.apply(compute_action, axis=1)
    action_df["confidence"] = (action_df["buy_signals"] + action_df["sell_signals"]) / 4.0

    action_priority = {"BUY": 0, "SELL": 1, "WATCH": 2, "WAIT": 3}
    action_df["action_priority"] = action_df["action"].map(action_priority)
    action_df = action_df.sort_values(["action_priority", "confidence"], ascending=[True, False])
    action_df = action_df.drop("action_priority", axis=1)

    st.subheader(t("action_table_header", lang_code))

    filter_col, sort_col, _ = st.columns([2, 2, 1])
    with filter_col:
        filter_action = st.selectbox(
            t("filter_by_action", lang_code),
            [t("all_filter", lang_code), "BUY", "SELL", "WATCH", "WAIT"],
            key=f"filter_action_{key_suffix}"
        )
    with sort_col:
        sort_by = st.selectbox(
            t("sort_by", lang_code),
            [
                t("sort_action_confidence", lang_code),
                t("sort_confidence", lang_code),
                t("sort_vol_delta", lang_code)
            ],
            key=f"sort_by_{key_suffix}"
        )

    filtered_df = action_df.copy()
    if filter_action != t("all_filter", lang_code):
        filtered_df = filtered_df[filtered_df["action"] == filter_action]

    if sort_by == t("sort_confidence", lang_code):
        filtered_df = filtered_df.sort_values("confidence", ascending=False)
    elif sort_by == t("sort_vol_delta", lang_code) and delta_col in filtered_df.columns:
        filtered_df = filtered_df.sort_values(delta_col, key=abs, ascending=False)

    cols_to_use = ["Ticker", price_col, trend_col, delta_col, loss_col, ret_col, "confidence", "action"]
    display_cols = [c for c in cols_to_use if c in filtered_df.columns]
    display_df = filtered_df[display_cols].copy()

    col_name_mapping = {
        "Ticker": "Ticker",
        price_col: t("col_action_price", lang_code),
        trend_col: t("col_action_trend", lang_code),
        delta_col: t("col_action_vol_delta", lang_code),
        loss_col: t("col_action_loss_pct", lang_code),
        ret_col: t("col_action_return_pct", lang_code),
        "confidence": t("col_action_confidence", lang_code),
        "action": t("col_action_action", lang_code)
    }
    display_df = display_df.rename(columns=col_name_mapping)

    price_label = t("col_action_price", lang_code)
    vol_delta_label = t("col_action_vol_delta", lang_code)
    loss_label = t("col_action_loss_pct", lang_code)
    ret_label = t("col_action_return_pct", lang_code)
    conf_label = t("col_action_confidence", lang_code)

    if price_label in display_df.columns:
        display_df[price_label] = display_df[price_label].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    if vol_delta_label in display_df.columns:
        display_df[vol_delta_label] = display_df[vol_delta_label].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
    if loss_label in display_df.columns:
        display_df[loss_label] = display_df[loss_label].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A")
    if ret_label in display_df.columns:
        display_df[ret_label] = display_df[ret_label].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
    if conf_label in display_df.columns:
        display_df[conf_label] = display_df[conf_label].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    st.dataframe(display_df, width="stretch", hide_index=True)

    st.markdown(t("signal_legend_markdown", lang_code))

    return filtered_df


def render_report_view(nat: dict, params: dict, is_historical: bool = False, run_id: int = None):
    """Renders the analysis report view (metrics, heuristics, table, backtesting, charts, ticker details)."""
    df_quant = nat["quant"]
    df_vol = pd.DataFrame([{k: v for k, v in r.items()
                            if not isinstance(v, (pd.DataFrame, dict)) and k != "zone_top5"}
                           for r in nat["vol"]]) if nat["vol"] else pd.DataFrame(
                                columns=["Ticker", "Delta_Volumi_Intra", "%_Trader_In_Perdita"])

    info = nat["info"]
    modello = df_quant["Modello"].iloc[0] if "Modello" in df_quant else ""
    st.caption(f"{t('universe_header', lang_code)}: {info.get('fonte', '')} · {len(df_quant)} stocks · "
               f"run {info.get('run_ts', '')} · model: {modello}")
    if modello == MODELLO_FALLBACK:
        st.warning(t("fallback_warning", lang_code))
    if info.get("scartati"):
        st.caption(t("discarded_caption", lang_code).format(", ".join(info["scartati"][:40])))
    for e in nat.get("errori", []):
        st.warning(t("volumes_not_available", lang_code).format(e))

    # outer join: a ticker without volume data stays in the report with missing values
    merged = pd.merge(df_quant, df_vol, on="Ticker", how="outer", suffixes=("", "_vol"))
    if "Ultimo_Prezzo_vol" in merged.columns:
        merged["Ultimo_Prezzo"] = merged["Ultimo_Prezzo"].fillna(merged["Ultimo_Prezzo_vol"])
    if merged.empty:
        st.error(t("outer_join_no_data", lang_code))
        return

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

    # Map terminology according to selected language
    buyer_maj_map = {
        "PERDITA": t("loss", lang_code),
        "GUADAGNO": t("profit", lang_code),
        "MISTA": t("mixed", lang_code)
    }
    pos_va_map = {
        "SOPRA": t("above", lang_code),
        "SOTTO": t("below", lang_code),
        "DENTRO": t("inside", lang_code)
    }
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
    })

    colonne = ["Ticker", t("col_last_price", lang_code), t("col_trend", lang_code), t("col_sigma_pct", lang_code),
               t("col_forecast_date", lang_code), t("col_p10", lang_code), t("col_median", lang_code),
               t("col_p90", lang_code), t("col_med_ret_pct", lang_code), t("col_uncertainty_pct", lang_code),
               t("col_delta_vol", lang_code), t("col_verdict_vol", lang_code), t("col_loss_pct", lang_code),
               t("col_buyer_majority", lang_code), t("col_pos_va", lang_code), t("col_poc", lang_code),
               t("col_vwap", lang_code), t("col_val", lang_code), t("col_vah", lang_code),
               t("col_sentiment", lang_code), t("col_verdict", lang_code)]
    final_df = renamed[[c for c in colonne if c in renamed.columns]]
    ret_col = t("col_med_ret_pct", lang_code)
    trend_col = t("col_trend", lang_code)
    delta_col = t("col_delta_vol", lang_code)
    loss_col = t("col_loss_pct", lang_code)
    price_col = t("col_last_price", lang_code)

    if ret_col in final_df:
        final_df = final_df.sort_values(ret_col, ascending=False, na_position="last").reset_index(drop=True)

    # Priority 1.1: Market Snapshot
    st.divider()
    snapshot_df = render_market_snapshot(
        final_df=final_df,
        trend_col=trend_col,
        delta_col=delta_col,
        loss_col=loss_col,
        ret_col=ret_col,
        lang_code=lang_code
    )

    # Priority 3: Actionable Signals Table
    st.divider()
    action_df = render_action_table(
        final_df=final_df,
        trend_col=trend_col,
        delta_col=delta_col,
        loss_col=loss_col,
        ret_col=ret_col,
        price_col=price_col,
        lang_code=lang_code,
        key_suffix=f"{run_id or 'live'}"
    )

    # Backtesting / Walk-forward validation (temporal: point-in-time forecast vs realized return)
    st.subheader("🧪 Backtesting / Walk-forward")
    prezzi_bt = info.get("prezzi")
    if prezzi_bt is None:
        st.info("Backtesting not available for historical archive without live price history cached in memory.")
    elif modello == MODELLO_FALLBACK:
        st.info("Backtesting requires TimesFM-3: the Naive baseline always returns EQUILIBRIO.")
    else:
        horizon = params.get("horizon", HORIZON)
        soglia_trend = params.get("soglia_trend", SOGLIA_TREND)
        st.caption(f"Forecasts are recomputed on past dates using only data available at that date, then compared "
                   f"with the realized return over the next {horizon} days (non-overlapping windows).")
        n_date_bt = st.number_input("Number of past dates", 5, 100, 30, 5, key=f"bt_n_date_{run_id or 'live'}")
        if st.button("Run temporal backtest", key=f"bt_run_{run_id or 'live'}"):
            barra_bt = st.progress(0.0)
            try:
                bt_df = backtest_temporale(prezzi_bt, horizon=horizon, forecaster=get_forecaster(),
                                           soglia_trend=soglia_trend, n_date=int(n_date_bt),
                                           progress=barra_bt.progress)
                st.session_state["bt_result"] = (info.get("run_ts"), bt_df)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
            barra_bt.empty()
        saved = st.session_state.get("bt_result")
        if saved and saved[0] == info.get("run_ts"):
            riepilogo, per_segnale = riepilogo_backtest_temporale(saved[1])
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Observations", riepilogo["n_obs"])
            b2.metric("Directional signals", riepilogo["n_active"])
            b3.metric("Hit rate", f"{riepilogo['hit_rate']:.1%}" if riepilogo["n_active"] else "N/A")
            b4.metric("Avg P&L / signal", f"{riepilogo['avg_pnl_active']:.2%}" if riepilogo["n_active"] else "N/A",
                      delta=f"{riepilogo['avg_pnl_active'] - riepilogo['avg_return_all']:.2%} vs buy&hold"
                      if riepilogo["n_active"] else None)
            st.dataframe(per_segnale, width="stretch")
            st.caption("Few observations = noisy metrics. Statistical analysis, not investment advice.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("metric_analyzed", lang_code), len(final_df))
    c2.metric(t("metric_buy_trend", lang_code), int((final_df[trend_col] == "BUY").sum()) if trend_col in final_df else 0)
    delta_col = t("col_delta_vol", lang_code)
    c3.metric(t("metric_avg_delta", lang_code), f"{final_df[delta_col].mean():.2f}%" if delta_col in final_df else "N/A")
    loss_col = t("col_loss_pct", lang_code)
    c4.metric(t("metric_avg_loss", lang_code), f"{final_df[loss_col].mean():.2f}%" if loss_col in final_df else "N/A")

    st.subheader(t("report_table_subheader", lang_code))
    st.dataframe(final_df, width="stretch")
    st.download_button(t("download_csv_button", lang_code), final_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"report_finale_integrato_{run_id or 'live'}.csv", mime="text/csv", type="primary")

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
    if nat and nat.get("vol"):
        st.subheader(t("detail_subheader", lang_code))
        scelto = st.selectbox(t("select_ticker", lang_code), [r["Ticker"] for r in nat["vol"]], key=f"select_ticker_{run_id or 'live'}")
        r = next(x for x in nat["vol"] if x["Ticker"] == scelto)
        if r.get("Avviso"):
            st.warning(r["Avviso"])

        v_verdict = t("buy", lang_code) if r['Verdetto_Volumi'] == "BUY" else (
            t("sell", lang_code) if r['Verdetto_Volumi'] == "SELL" else t("equilibrium", lang_code)
        )
        b_maj = buyer_maj_map.get(r['Maggioranza_Acquirenti'], r['Maggioranza_Acquirenti'])
        p_pos = pos_va_map.get(r['Posizione_Area_Valore'], r['Posizione_Area_Valore'])

        st.markdown(t("detail_summary", lang_code).format(
            scelto, r['Prezzo_Riferimento'], nat.get('metodo', 'clv'), r['Buy_%'], r['Sell_%'], r['Delta_Volumi_Intra'], v_verdict
        ))
        st.markdown(t("detail_vp_summary", lang_code).format(
            r['POC'], r['VWAP'], r['VAL'], r['VAH'], r['%_Trader_In_Perdita'], r['%_Trader_In_Guadagno'], b_maj, p_pos
        ))
        if r.get("zona_sopra"):
            st.caption(t("resistance_caption", lang_code).format(r['zona_sopra']))
        if r.get("zona_sotto"):
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
        zone_df = r["zone_top5"]
        if isinstance(zone_df, list):
            zone_df = pd.DataFrame(zone_df)
        t1.dataframe(zone_df.set_index("Zona di prezzo") if "Zona di prezzo" in zone_df.columns else zone_df, width="stretch")

        t2.markdown(f"**{t('daily_breakdown_title', lang_code)}**")
        dett_df = r["dettaglio_giorni"]
        if isinstance(dett_df, list):
            dett_df = pd.DataFrame(dett_df)
        t2.dataframe(dett_df, width="stretch")

    st.caption(t("footer_disclaimer", lang_code))


def render_analysis_section():
    """Analysis section: live analysis + history archive."""
    st.markdown(t("app_subtitle", lang_code))

    # Top Navigation Tabs
    tab_live, tab_archive = st.tabs([t("tab_live_analysis", lang_code), t("tab_history_archive", lang_code)])

    with tab_live:
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

            nat_data = {"quant": dfq, "info": info, "vol": righe_vol, "errori": errori, "metodo": metodo}
            params_data = {
                "is_manual": is_manual, "input_tickers": input_tickers, "n_top": n_top,
                "horizon": horizon, "soglia_trend": soglia_trend, "usa_tfm": usa_tfm,
                "metodo": metodo, "giorni_intra": giorni_intra, "data_fine": data_fine,
                "solo_regolari": solo_regolari, "soglia_eq": soglia_eq, "n_bin": n_bin,
                "area_valore": area_valore, "prezzo_rif": prezzo_rif
            }
            st.session_state["nativo"] = nat_data
            st.session_state["params"] = params_data

            # Automatically persist run to SQLite database
            run_id = save_run(nat_data, params_data)
            st.session_state["last_saved_run_id"] = run_id

        nat = st.session_state.get("nativo")
        params = st.session_state.get("params", {})

        if not nat:
            st.info(t("sidebar_prompt", lang_code))
        else:
            if st.session_state.get("last_saved_run_id"):
                st.success(t("saved_to_db_notice", lang_code).format(st.session_state["last_saved_run_id"]))
            render_report_view(nat, params, is_historical=False)


    with tab_archive:
        st.header(t("archive_title", lang_code))
        st.markdown(t("archive_subtitle", lang_code))

        historical_runs = list_runs()
        if not historical_runs:
            st.info(t("no_history_msg", lang_code))
        else:
            run_options = {r["id"]: f"#{r['id']} - {r['run_label']}" for r in historical_runs}
            selected_id = st.selectbox(
                t("select_run_label", lang_code),
                options=list(run_options.keys()),
                format_func=lambda x: run_options[x]
            )

            if selected_id:
                run_data = get_run(selected_id)
                if run_data:
                    col_info, col_del = st.columns([3, 1])
                    with col_info:
                        st.caption(f"**ID:** {selected_id} | **Timestamp:** {run_data['metadata']['timestamp']} | "
                                   f"**Universe:** {run_data['metadata']['universe_type']} | "
                                   f"**Model:** {run_data['metadata']['model_used']}")
                    with col_del:
                        if st.button(t("delete_run_button", lang_code), key=f"del_{selected_id}"):
                            delete_run(selected_id)
                            st.success(t("run_deleted_success", lang_code).format(selected_id))
                            st.rerun()

                    st.divider()
                    render_report_view(run_data["nativo"], run_data["params"], is_historical=True, run_id=selected_id)


# Section routing
if section == "alerts":
    render_alerts_section(lang_code)
else:
    st.session_state["alerts_editor_base"] = None  # editor is rebuilt from saved rules when re-entering "Alerts"
    render_analysis_section()
