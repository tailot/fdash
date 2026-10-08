"""
Discover UI: Streamlit user interface for discovering stocks based on Volume, Market Cap, and Birth Year.
"""

import pandas as pd
import streamlit as st
from i18n import t
from quant_engine import ricava_universo
from discover_engine import (
    fetch_stocks_batch,
    filter_discovered_stocks,
    format_market_cap,
    format_volume,
    universe_sp500,
)


def render_discover_section(lang_code: str = "en"):
    """Renders the Discover section in Streamlit."""
    st.header(t("discover_title", lang_code))
    st.markdown(t("discover_subtitle", lang_code))

    st.divider()

    # Layout for parameters
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(t("discover_volume_header", lang_code))
        vol_op = st.selectbox(
            t("discover_op_label", lang_code),
            options=[">=", "<="],
            format_func=lambda x: f"{x} ({t('op_greater_equal', lang_code) if x == '>=' else t('op_less_equal', lang_code)})",
            key="discover_vol_op",
        )
        vol_val = st.number_input(
            t("discover_value_label", lang_code),
            min_value=0.0,
            value=1.0,
            step=1.0,
            key="discover_vol_val",
        )
        vol_unit = st.selectbox(
            t("discover_unit_label", lang_code),
            options=["mln", "k", "mld", "1"],
            format_func=lambda x: t(f"unit_{x}", lang_code),
            key="discover_vol_unit",
        )

    with c2:
        st.subheader(t("discover_mc_header", lang_code))
        mc_op = st.selectbox(
            t("discover_op_label", lang_code),
            options=[">=", "<="],
            format_func=lambda x: f"{x} ({t('op_greater_equal', lang_code) if x == '>=' else t('op_less_equal', lang_code)})",
            key="discover_mc_op",
        )
        mc_val = st.number_input(
            t("discover_value_label", lang_code),
            min_value=0.0,
            value=1.0,
            step=1.0,
            key="discover_mc_val",
        )
        mc_unit = st.selectbox(
            t("discover_unit_label", lang_code),
            options=["mld", "mln", "k"],
            format_func=lambda x: t(f"unit_{x}", lang_code),
            key="discover_mc_unit",
        )

    with c3:
        st.subheader(t("discover_birth_header", lang_code))
        birth_op = st.selectbox(
            t("discover_op_label", lang_code),
            options=["<=", ">=", "=="],
            format_func=lambda x: f"{x} ({t('op_birth_inferiore', lang_code) if x == '<=' else (t('op_birth_superiore', lang_code) if x == '>=' else t('op_birth_uguale', lang_code))})",
            key="discover_birth_op",
        )
        birth_year = st.number_input(
            t("discover_birth_year_label", lang_code),
            min_value=1800,
            max_value=2100,
            value=2010,
            step=1,
            key="discover_birth_year",
        )

    st.divider()

    # Universe Selection
    st.subheader(t("discover_universe_header", lang_code))
    uni_col1, uni_col2 = st.columns([1, 2])
    with uni_col1:
        universo_type = st.radio(
            t("universe_radio", lang_code),
            options=["sp500", "top_nasdaq", "all_nasdaq", "manual_list"],
            format_func=lambda x: t(f"discover_universe_{x}", lang_code),
            key="discover_universe_type",
        )

    with uni_col2:
        if universo_type == "top_nasdaq":
            n_top = st.slider(t("n_stocks_slider", lang_code), 50, 3000, 500, step=50, key="discover_n_top")
            tickers_input = ""
        elif universo_type == "manual_list":
            n_top = 100
            tickers_input = st.text_input(
                t("tickers_input", lang_code),
                "AAPL, MSFT, NVDA, TSLA, PLTR, AMZN, GOOGL, META, MSTR, AMD, NFLX, DIS, INTC, CSCO",
                key="discover_tickers_input",
            )
        else:
            n_top = 100
            tickers_input = ""

    run_discover = st.button(t("discover_button", lang_code), type="primary")

    if run_discover:
        cand = []
        if universo_type == "manual_list":
            cand = [t_item.strip().upper() for t_item in tickers_input.split(",") if t_item.strip()]
        elif universo_type == "sp500":
            with st.spinner(t("discover_fetching_universe", lang_code)):
                cand = universe_sp500()
        elif universo_type == "all_nasdaq":
            with st.spinner(t("discover_fetching_universe", lang_code)):
                try:
                    _, _, cand = ricava_universo(n_tickers=4500, buffer=0, tickers_manuali=None)
                except Exception as e:
                    st.error(f"Error retrieving universe: {e}")
                    cand = []
        else:  # top_nasdaq
            with st.spinner(t("discover_fetching_universe", lang_code)):
                try:
                    _, _, cand = ricava_universo(n_tickers=n_top, buffer=0, tickers_manuali=None)
                except Exception as e:
                    st.error(f"Error retrieving universe: {e}")
                    cand = []

        if not cand:
            st.warning(t("discover_no_tickers", lang_code))
            return

        with st.spinner(t("discover_fetching_info", lang_code).format(len(cand))):
            stocks_data = fetch_stocks_batch(cand, max_workers=20)

        filtered = filter_discovered_stocks(
            stocks=stocks_data,
            vol_target=vol_val,
            vol_unit=vol_unit,
            vol_op=vol_op,
            mc_target=mc_val,
            mc_unit=mc_unit,
            mc_op=mc_op,
            birth_year_target=int(birth_year),
            birth_year_op=birth_op,
        )

        st.session_state["discover_results"] = {
            "scanned": len(stocks_data),
            "filtered": filtered,
        }

    # Display saved results if any
    res = st.session_state.get("discover_results")
    if res:
        filtered_list = res.get("filtered", [])
        scanned_cnt = res.get("scanned", 0)
        found_cnt = len(filtered_list)

        m1, m2, m3 = st.columns(3)
        m1.metric(t("discover_metric_scanned", lang_code), scanned_cnt)
        m2.metric(t("discover_metric_found", lang_code), found_cnt)
        m3.metric(
            t("discover_metric_rate", lang_code),
            f"{(found_cnt / scanned_cnt)*100:.1f}%" if scanned_cnt > 0 else "0%",
        )

        st.divider()

        if not filtered_list:
            st.warning(t("discover_no_match", lang_code))
        else:
            df_out = pd.DataFrame(filtered_list)

            # Create formatted columns for clean Streamlit display
            df_display = pd.DataFrame()
            df_display["Ticker"] = df_out["Ticker"]
            df_display[t("col_company_name", lang_code)] = df_out["Name"]
            df_display[t("col_sector", lang_code)] = df_out["Sector"]
            df_display[t("col_action_price", lang_code)] = df_out["Price"].apply(
                lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
            )
            df_display[t("col_market_cap", lang_code)] = df_out["MarketCap"].apply(format_market_cap)
            df_display[t("col_volume", lang_code)] = df_out["Volume"].apply(format_volume)
            df_display[t("col_birth_year", lang_code)] = df_out["BirthYear"].apply(
                lambda x: str(int(x)) if pd.notna(x) else "N/A"
            )

            st.dataframe(df_display, width="stretch", hide_index=True)

            csv_data = df_out[["Ticker", "Name", "Sector", "Price", "MarketCap", "Volume", "BirthYear"]].to_csv(
                index=False
            )
            st.download_button(
                t("discover_download_csv", lang_code),
                data=csv_data.encode("utf-8"),
                file_name="discovered_stocks.csv",
                mime="text/csv",
                type="primary",
            )
