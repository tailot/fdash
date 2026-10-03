import os
import pytest
import pandas as pd
from volume_engine import stima_buy_sell, verdetto_delta
from quant_engine import calcola_quant_trend_ticker
from gemini_enrichment import genera_analisi_gemini

def test_sample_csv_files_exist():
    assert os.path.exists("data/colab1_timesfm.csv")
    assert os.path.exists("data/colab2_volumi.csv")

def test_colab_data_merge():
    df1 = pd.read_csv("data/colab1_timesfm.csv")
    df2 = pd.read_csv("data/colab2_volumi.csv")

    df1["Ticker"] = df1["Ticker"].astype(str).str.strip().str.upper()
    df2["Ticker"] = df2["Ticker"].astype(str).str.strip().str.upper()

    merged = pd.merge(df1, df2, on="Ticker", how="inner")
    assert not merged.empty
    assert "Ultimo_Prezzo" in merged.columns
    assert "Delta_Volumi_Intra" in merged.columns

def test_stima_buy_sell():
    data = {
        "Open": [100.0, 102.0, 101.0],
        "High": [103.0, 104.0, 102.0],
        "Low": [99.0, 100.0, 100.0],
        "Close": [102.0, 101.0, 101.0],
        "Volume": [1000, 2000, 1500]
    }
    df = pd.DataFrame(data)

    res_clv = stima_buy_sell(df, metodo="clv")
    assert "buy" in res_clv.columns
    assert "sell" in res_clv.columns
    assert "delta" in res_clv.columns
    assert len(res_clv) == 3

def test_verdetto_delta():
    assert verdetto_delta(5.0) == "BUY"
    assert verdetto_delta(-3.0) == "SELL"
    assert verdetto_delta(1.0) == "EQUILIBRIO"

def test_gemini_enrichment_fallback():
    enrichment = genera_analisi_gemini(
        ticker="MSTR",
        ultimo_prezzo=160.01,
        trend_timesfm="BUY",
        sigma_pct=10.44,
        delta_volumi_intra=9.79,
        pct_trader_in_perdita=59.9,
        api_key=""
    )
    assert "Sentiment News" in enrichment
    assert "Sintesi / Verdetto" in enrichment
    assert "MSTR" in enrichment["Sentiment News"]
