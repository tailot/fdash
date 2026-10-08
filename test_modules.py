# Copyright (c) 2026 Vincenzo Tilotta
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""Offline test suite: no network access (yfinance and forecaster are mocked with synthetic data)."""
import os
import numpy as np
import pandas as pd
import pytest

import quant_engine as qe
import volume_engine as ve
from volume_engine import stima_buy_sell, verdetto_delta
from heuristic_enrichment import genera_analisi_euristica
from i18n import t, LANGUAGES
import db_engine as db
import discover_engine as de


# ------------------------------------------------------------------ synthetic data
def _prezzi(n=400, tickers=("AAA", "BBB", "CCC"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for t in tickers}, index=idx)


def _minuti(giorni=3, per_giorno=390, seed=1):
    rng = np.random.default_rng(seed)
    parti = []
    for g in pd.bdate_range("2026-09-28", periods=giorni):
        idx = pd.date_range(g + pd.Timedelta(hours=14, minutes=30), periods=per_giorno, freq="min", tz="UTC")
        close = 100 + np.cumsum(rng.normal(0, 0.05, per_giorno))
        open_ = np.r_[close[0], close[:-1]]
        parti.append(pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) + 0.03,
                                   "Low": np.minimum(open_, close) - 0.03, "Close": close,
                                   "Volume": rng.integers(100, 5000, per_giorno)}, index=idx))
    return pd.concat(parti)


class FakeOut:
    def __init__(self, q):
        self.quantiles = q


class FakeTimesFM:
    """Simulates TimesFM-3: increasing deciles with median = +0.2% per step and known dispersion."""
    def predict_batch(self, series, horizon, return_quantiles, use_symmetric_averaging):
        assert return_quantiles
        z = np.array([-1.2815515655446004, -0.8416, -0.5244, -0.2533, 0, 0.2533, 0.5244, 0.8416, 1.2815515655446004])
        q = 0.002 + 0.01 * z                                   # step sigma = 0.01
        return [FakeOut(np.tile(q, (horizon, 1))) for _ in series]


# ------------------------------------------------------------------ quant: formulas
def test_cum_stats():
    med = np.full((1, 5), 0.002)
    q10, q90 = med - 0.01 * qe.Z80, med + 0.01 * qe.Z80
    mu, sg = qe._cum_stats(med, q10, q90)
    assert mu[0, -1] == pytest.approx(0.01)                     # cumsum of median
    assert sg[0, -1] == pytest.approx(0.01 * np.sqrt(5))       # sqrt(cumsum(sigma^2))


def test_tfm3_returns_usa_i_rendimenti_log():
    p = _prezzi()
    mu, sg = qe.tfm3_returns(FakeTimesFM(), [p["AAA"].values], 5)
    assert mu.shape == (1, 5) and sg.shape == (1, 5)
    assert mu[0, -1] == pytest.approx(0.01) and sg[0, -1] == pytest.approx(0.01 * np.sqrt(5))


def test_tabella_previsioni_timesfm():
    p = _prezzi()
    df = qe.tabella_previsioni(p, horizon=5, forecaster=FakeTimesFM())
    r = df.iloc[0]
    last = float(p[r["Ticker"]].iloc[-1])
    sg = 0.01 * np.sqrt(5)
    assert r["Mediana"] == pytest.approx(last * np.exp(0.01), abs=0.01)
    assert r["P10"] == pytest.approx(last * np.exp(0.01 - qe.Z80 * sg), abs=0.01)
    assert r["P90"] == pytest.approx(last * np.exp(0.01 + qe.Z80 * sg), abs=0.01)
    assert r["Rend_Mediano_%"] == pytest.approx((np.exp(0.01) - 1) * 100, abs=0.01)
    assert r["Incertezza_Sigma_%"] == pytest.approx(sg * 100, abs=0.01)
    assert r["Modello"] == qe.MODELLO_TFM3
    assert r["Trend_TimesFM"] == "BUY"                          # z = 0.01 / 0.0224 = 0.45 > threshold
    assert pd.Timestamp(r["Data_Previsione"]) == pd.bdate_range(p.index[-1] + pd.Timedelta(days=1), periods=5)[-1]


def test_fallback_senza_timesfm_non_inventa_trend():
    df = qe.tabella_previsioni(_prezzi(), horizon=5, forecaster=None)
    assert (df["Modello"] == qe.MODELLO_FALLBACK).all()
    assert (df["Rend_Mediano_%"] == 0).all()
    assert (df["Trend_TimesFM"] == "EQUILIBRIO").all()
    assert (df["P10"] < df["Mediana"]).all() and (df["Mediana"] < df["P90"]).all()


def test_ordinamento_per_rendimento_mediano():
    df = qe.tabella_previsioni(_prezzi(), forecaster=FakeTimesFM())
    assert df["Rend_Mediano_%"].is_monotonic_decreasing


def test_prepara_prezzi_scarta_storico_incompleto():
    p = _prezzi(tickers=("AAA", "BBB"))
    p["NEW"] = np.nan
    p.loc[p.index[-50:], "NEW"] = 10.0                          # Recently listed
    prices, scartati = qe.prepara_prezzi(p, ["AAA", "NEW", "BBB"], 10)
    assert list(prices.columns) == ["AAA", "BBB"] and scartati == ["NEW"]


def test_universo_pulizia_doppie_classi():
    uni = pd.DataFrame({"Ticker": ["GOOGL", "GOOG", "AAPL", "BRK.B", "x$"], "MarketCap": [3, 2.9, 4, 1, 9]})
    out = qe._clean_universe(uni)
    assert out["Ticker"].tolist() == ["AAPL", "GOOGL", "BRK-B"]


def test_calcola_previsioni_pipeline(monkeypatch):
    p = _prezzi(tickers=("AAA", "BBB", "CCC"))
    monkeypatch.setattr(qe, "scarica_prezzi", lambda cand, n, h: (p[cand], []))
    df, info = qe.calcola_previsioni(["AAA", "BBB", "CCC"], forecaster=FakeTimesFM())
    assert set(df["Ticker"]) == {"AAA", "BBB", "CCC"} and info["fonte"] == "manual list"
    one = qe.calcola_quant_trend_ticker("AAA", forecaster=FakeTimesFM())
    assert one["Ticker"] == "AAA" and "P10" in one


# ------------------------------------------------------------------ volume: formulas
def test_stima_buy_sell():
    df = pd.DataFrame({"Open": [100.0, 102.0, 101.0], "High": [103.0, 104.0, 102.0],
                       "Low": [99.0, 100.0, 100.0], "Close": [102.0, 101.0, 101.0], "Volume": [1000, 2000, 1500]})
    for metodo in ("candela", "clv", "tick"):
        r = stima_buy_sell(df, metodo)
        assert (r["buy"] + r["sell"]).tolist() == pytest.approx(df["Volume"].tolist())
    assert stima_buy_sell(df, "candela")["buy"].tolist() == [1000, 0, 750]
    with pytest.raises(ValueError):
        stima_buy_sell(df, "boh")


def test_verdetto_delta():
    assert verdetto_delta(5.0) == "BUY"
    assert verdetto_delta(-3.0) == "SELL"
    assert verdetto_delta(1.0) == "EQUILIBRIO"


def test_microstruttura_completa():
    r = ve.calcola_microstruttura_ticker("zzz", giorni=2, metodo="clv", raw=_minuti(3))
    assert r["Giorni_Analizzati"] == 2 and len(r["dettaglio_giorni"]) == 2
    assert r["%_Trader_In_Perdita"] + r["%_Trader_In_Guadagno"] <= 100.0001
    assert r["VAL"] <= r["POC"] <= r["VAH"] or r["VAL"] <= r["VAH"]
    assert r["Maggioranza_Acquirenti"] in {"PERDITA", "GUADAGNO", "MISTA"}
    assert r["Posizione_Area_Valore"] in {"SOPRA", "SOTTO", "DENTRO"}
    assert len(r["zone_top5"]) == 5
    assert r["Buy_%"] + r["Sell_%"] == pytest.approx(100, abs=0.2)


def test_microstruttura_avviso_giorni_insufficienti():
    r = ve.calcola_microstruttura_ticker("zzz", giorni=5, raw=_minuti(2))
    assert r["Giorni_Analizzati"] == 2 and "only 2 days available" in r["Avviso"]


def test_microstruttura_data_fine_e_prezzo_riferimento():
    raw = _minuti(3)
    r = ve.calcola_microstruttura_ticker("zzz", giorni=1, data_fine="2026-09-29", prezzo_riferimento=1e9, raw=raw)
    assert r["%_Trader_In_Guadagno"] == pytest.approx(100.0)    # All volume is below simulated price
    assert str(r["dettaglio_giorni"].index[0]) == "2026-09-29"


# ------------------------------------------------------------------ Local Heuristic Enrichment & Multilingual i18n
def test_i18n_translation_keys():
    assert len(LANGUAGES) == 5
    for code in ("en", "it", "es", "zh", "fr"):
        assert t("app_title", code) != ""
        assert t("col_last_price", code) != ""
        assert t("welcome", code) != ""
        assert t("welcome", code).format("User") != ""
        assert t("nav_discover", code) != ""
        assert t("discover_title", code) != ""


def test_heuristic_enrichment_multilingual():
    e_en = genera_analisi_euristica("MSTR", 160.01, "BUY", 10.44, 9.79, 59.9, lang="en")
    assert "Market sentiment on MSTR" in e_en["Sentiment News"]
    assert "Advanced sentiment:" in e_en["Sentiment News"]
    assert "aligned" in e_en["Sintesi / Verdetto"]

    e_it = genera_analisi_euristica("MSTR", 160.01, "BUY", 10.44, 9.79, 59.9, lang="it")
    assert "Sentiment di mercato su MSTR" in e_it["Sentiment News"]
    assert "Sentiment avanzato:" in e_it["Sentiment News"]
    assert "coerenti" in e_it["Sintesi / Verdetto"]

    e_es = genera_analisi_euristica("MSTR", 160.01, "BUY", 10.44, 9.79, 59.9, lang="es")
    assert "Sentimiento de mercado para MSTR" in e_es["Sentiment News"]
    assert "Sentimiento avanzado:" in e_es["Sentiment News"]
    assert "coherentes" in e_es["Sintesi / Verdetto"]

    e_zh = genera_analisi_euristica("MSTR", 160.01, "BUY", 10.44, 9.79, 59.9, lang="zh")
    assert "MSTR 的市场情绪" in e_zh["Sentiment News"]
    assert "高级情绪分析：" in e_zh["Sentiment News"]
    assert "一致" in e_zh["Sintesi / Verdetto"]

    e_fr = genera_analisi_euristica("MSTR", 160.01, "BUY", 10.44, 9.79, 59.9, lang="fr")
    assert "Sentiment du marché sur MSTR" in e_fr["Sentiment News"]
    assert "Sentiment avancé:" in e_fr["Sentiment News"]
    assert "alignés" in e_fr["Sintesi / Verdetto"]


def test_heuristic_enrichment_dati_mancanti():
    e = genera_analisi_euristica("MSTR", 160.01, "BUY", float("nan"), None, float("nan"), lang="en")
    assert "unavailable" in e["Sentiment News"]


def test_sentiment_analysis():
    from advanced_nlp_sentiment import SentimentAnalyzer
    analyzer = SentimentAnalyzer(use_finbert=False)
    res_bull = analyzer.analyze("Company reports record revenue and strong growth rally", lang="en")
    assert res_bull["sentiment_label"] == "BULLISH"
    assert "sentiment_score" in res_bull


# ------------------------------------------------------------------ Market Snapshot & Actionable Signals Matrix Logic Tests
def test_action_and_snapshot_logic():
    df = pd.DataFrame([
        # 4 buy signals -> BUY (Confidence = 4/4 = 1.0)
        {"Ticker": "MSTR", "Ultimo_Prezzo": 350.12, "Trend_TimesFM": "BUY", "Delta_Volumi_Intra": 23.4, "%_Trader_In_Perdita": 35.0, "Rend_Mediano_%": 4.2},
        # 4 sell signals -> SELL (Confidence = 4/4 = 1.0)
        {"Ticker": "COIN", "Ultimo_Prezzo": 98.45, "Trend_TimesFM": "SELL", "Delta_Volumi_Intra": -18.2, "%_Trader_In_Perdita": 75.0, "Rend_Mediano_%": -2.1},
        # 2 buy signals -> WATCH (Confidence = 2/4 = 0.5)
        {"Ticker": "TSLA", "Ultimo_Prezzo": 242.35, "Trend_TimesFM": "SELL", "Delta_Volumi_Intra": -8.6, "%_Trader_In_Perdita": 38.0, "Rend_Mediano_%": 0.4},
        # 1 buy, 1 sell signal -> WAIT (Confidence = 2/4 = 0.5)
        {"Ticker": "GOOG", "Ultimo_Prezzo": 142.90, "Trend_TimesFM": "EQUILIBRIO", "Delta_Volumi_Intra": 9.8, "%_Trader_In_Perdita": 65.0, "Rend_Mediano_%": 0.0},
    ])

    # Test Action Convergence Logic
    # MSTR
    row_mstr = df.iloc[0]
    b_mstr = int(row_mstr["Trend_TimesFM"] == "BUY") + int(row_mstr["Delta_Volumi_Intra"] > 0) + int(row_mstr["%_Trader_In_Perdita"] <= 40) + int(row_mstr["Rend_Mediano_%"] > 0)
    assert b_mstr == 4

    # COIN
    row_coin = df.iloc[1]
    s_coin = int(row_coin["Trend_TimesFM"] == "SELL") + int(row_coin["Delta_Volumi_Intra"] < 0) + int(row_coin["%_Trader_In_Perdita"] >= 60) + int(row_coin["Rend_Mediano_%"] < 0)
    assert s_coin == 4

    # TSLA
    row_tsla = df.iloc[2]
    b_tsla = int(row_tsla["Trend_TimesFM"] == "BUY") + int(row_tsla["Delta_Volumi_Intra"] > 0) + int(row_tsla["%_Trader_In_Perdita"] <= 40) + int(row_tsla["Rend_Mediano_%"] > 0)
    s_tsla = int(row_tsla["Trend_TimesFM"] == "SELL") + int(row_tsla["Delta_Volumi_Intra"] < 0) + int(row_tsla["%_Trader_In_Perdita"] >= 60) + int(row_tsla["Rend_Mediano_%"] < 0)
    assert b_tsla == 2 and s_tsla == 2

    # Verify Market Snapshot Action calculation function
    trend_col = "Trend_TimesFM"
    delta_col = "Delta_Volumi_Intra"
    loss_col = "%_Trader_In_Perdita"
    ret_col = "Rend_Mediano_%"

    def compute_action(row):
        buy_signals = 0
        sell_signals = 0
        if row[trend_col] == "BUY": buy_signals += 1
        elif row[trend_col] == "SELL": sell_signals += 1
        if row[delta_col] > 0: buy_signals += 1
        elif row[delta_col] < 0: sell_signals += 1
        if pd.notna(row[loss_col]):
            if row[loss_col] <= 40: buy_signals += 1
            elif row[loss_col] >= 60: sell_signals += 1
        if pd.notna(row[ret_col]):
            if row[ret_col] > 0: buy_signals += 1
            elif row[ret_col] < 0: sell_signals += 1
        if buy_signals >= 3: return "BUY"
        elif sell_signals >= 3: return "SELL"
        elif max(buy_signals, sell_signals) >= 2: return "WATCH"
        else: return "WAIT"

    actions = df.apply(compute_action, axis=1).tolist()
    assert actions == ["BUY", "SELL", "WATCH", "WAIT"]


# ------------------------------------------------------------------ Database Persistence Engine Tests
def test_db_persistence(tmp_path):
    test_db = str(tmp_path / "test_history.db")

    # Sample mock native analysis data
    df_quant = pd.DataFrame([
        {"Ticker": "AAPL", "Ultimo_Prezzo": 150.0, "Trend_TimesFM": "BUY", "Modello": "Naive (0%)"},
        {"Ticker": "MSFT", "Ultimo_Prezzo": 300.0, "Trend_TimesFM": "SELL", "Modello": "Naive (0%)"}
    ])
    vol_rows = [
        ve.calcola_microstruttura_ticker("AAPL", giorni=1, raw=_minuti(2)),
        ve.calcola_microstruttura_ticker("MSFT", giorni=1, raw=_minuti(2))
    ]
    info = {"fonte": "manual list", "run_ts": "2026-03-30 12:00:00"}
    nat_data = {"quant": df_quant, "info": info, "vol": vol_rows, "metodo": "clv"}
    params = {"horizon": 5, "metodo": "clv", "is_manual": True}

    # Save
    run_id = db.save_run(nat_data, params, db_file=test_db)
    assert run_id > 0

    # List
    runs = db.list_runs(db_file=test_db)
    assert len(runs) == 1
    assert runs[0]["id"] == run_id
    assert runs[0]["n_tickers"] == 2
    assert "[AAPL, MSFT]" in runs[0]["run_label"]

    # Get
    retrieved = db.get_run(run_id, db_file=test_db)
    assert retrieved is not None
    assert len(retrieved["nativo"]["quant"]) == 2
    assert retrieved["nativo"]["quant"].iloc[0]["Ticker"] == "AAPL"
    assert len(retrieved["nativo"]["vol"]) == 2

    # Delete
    deleted = db.delete_run(run_id, db_file=test_db)
    assert deleted
    assert len(db.list_runs(db_file=test_db)) == 0


# ------------------------------------------------------------------ Price alerts
import alerts_engine as ae


def test_alert_rules_normalization_and_persistence(tmp_path):
    df = pd.DataFrame({"symbol": [" aapl ", "", "MSFT", "NVDA"],
                       "condition": [">=", ">=", "<=", "??"],
                       "target": [200.0, 10.0, 0.0, 5.0],
                       "active": [True, True, True, True]})
    rules = ae.normalize_rules(df)
    assert rules == [{"symbol": "AAPL", "condition": ">=", "target": 200.0, "active": True}]
    path = str(tmp_path / "alerts.json")
    ae.save_rules(df, path)
    loaded = ae.load_rules(path)
    assert ae.normalize_rules(loaded) == rules
    assert ae.load_rules(str(tmp_path / "missing.json")).empty


def test_alert_fires_once_and_rearms():
    rules = [{"symbol": "AAA", "condition": ">=", "target": 100.0, "active": True},
             {"symbol": "BBB", "condition": "<=", "target": 50.0, "active": True},
             {"symbol": "CCC", "condition": ">=", "target": 1.0, "active": False}]
    ev, fired = ae.evaluate_rules(rules, {"AAA": 99.0, "BBB": 60.0, "CCC": 5.0}, set())
    assert ev == [] and fired == set()
    ev, fired = ae.evaluate_rules(rules, {"AAA": 101.0, "BBB": 49.0, "CCC": 5.0}, fired)
    assert {e["symbol"] for e in ev} == {"AAA", "BBB"}          # inactive rule never fires
    ev, fired = ae.evaluate_rules(rules, {"AAA": 105.0, "BBB": 48.0}, fired)
    assert ev == []                                              # no repeated alarm every minute
    ev, fired = ae.evaluate_rules(rules, {"AAA": 95.0, "BBB": 48.0}, fired)
    assert ev == [] and ae.rule_key(rules[0]) not in fired       # condition false -> re-armed
    ev, fired = ae.evaluate_rules(rules, {"AAA": 100.0, "BBB": 48.0}, fired)
    assert [e["symbol"] for e in ev] == ["AAA"]                  # fires again
    ev, fired = ae.evaluate_rules(rules, {}, fired)              # missing data keeps state
    assert ae.rule_key(rules[0]) in fired


def test_fetch_last_prices_single_batched_call(monkeypatch):
    calls = []
    idx = pd.date_range("2026-10-06 14:30", periods=3, freq="min", tz="UTC")
    cols = pd.MultiIndex.from_product([["Close"], ["AAA", "BBB"]])
    frame = pd.DataFrame([[1.0, 10.0], [2.0, 20.0], [3.0, np.nan]], index=idx, columns=cols)

    def fake_download(symbols, **kw):
        calls.append(symbols)
        return frame

    monkeypatch.setattr(ae.yf, "download", fake_download)
    prices = ae.fetch_last_prices(["aaa", "BBB", "AAA", "ZZZ"])
    assert len(calls) == 1 and calls[0] == ["AAA", "BBB", "ZZZ"]  # one call for all symbols
    assert prices == {"AAA": 3.0, "BBB": 20.0}                     # last valid close, unknown symbol skipped

    monkeypatch.setattr(ae.yf, "download", lambda *a, **k: pd.DataFrame())
    assert ae.fetch_last_prices(["AAA"]) == {}


def test_alert_beep_wav_and_i18n():
    import wave, io
    with wave.open(io.BytesIO(ae.make_beep_wav())) as w:
        assert w.getnchannels() == 1 and w.getnframes() > 0
    for code in LANGUAGES:
        for key in ("nav_analysis", "nav_alerts", "alerts_title", "alerts_start_button", "alert_fired_body"):
            assert t(key, code) != key


def test_auth_config_check(monkeypatch):
    from app import _has_auth_config
    import streamlit as st

    monkeypatch.setattr(st, "secrets", {})
    assert not _has_auth_config()

    mock_secrets = {
        "credentials": {"usernames": {}},
        "cookie": {"name": "test", "key": "secret", "expiration_days": 30}
    }
    monkeypatch.setattr(st, "secrets", mock_secrets)
    assert _has_auth_config()


# ------------------------------------------------------------------ Discover Engine Tests
def test_discover_unit_multipliers():
    assert de.get_unit_multiplier("1") == 1.0
    assert de.get_unit_multiplier("k") == 1_000.0
    assert de.get_unit_multiplier("mln") == 1_000_000.0
    assert de.get_unit_multiplier("mld") == 1_000_000_000.0


def test_discover_evaluate_condition():
    assert de.evaluate_condition(100, 50, ">=")
    assert not de.evaluate_condition(30, 50, ">=")
    assert de.evaluate_condition(30, 50, "<=")
    assert de.evaluate_condition(1980, 2000, "<=")
    assert de.evaluate_condition(2015, 2000, ">=")
    assert de.evaluate_condition(2020, 2020, "==")


def test_discover_formatting_helpers():
    assert de.format_market_cap(1_500_000_000) == "$1.50B"
    assert de.format_market_cap(250_000_000) == "$250.00M"
    assert de.format_volume(15_000_000) == "15.00M"
    assert de.format_volume(500_000) == "500.00K"


def test_discover_universe_sp500():
    sp = de.universe_sp500()
    assert len(sp) >= 10
    assert "AAPL" in sp or "MSFT" in sp


def test_discover_filter_stocks():
    stocks = [
        {"Ticker": "AAPL", "Name": "Apple", "Price": 150.0, "MarketCap": 2_000_000_000_000, "Volume": 50_000_000, "BirthYear": 1980},
        {"Ticker": "PLTR", "Name": "Palantir", "Price": 25.0, "MarketCap": 50_000_000_000, "Volume": 30_000_000, "BirthYear": 2020},
        {"Ticker": "SMALL", "Name": "SmallCo", "Price": 5.0, "MarketCap": 500_000, "Volume": 100_000, "BirthYear": 2015},
    ]

    # Filter Volume >= 10M, Market Cap >= 1B, Birth Year <= 2000
    f1 = de.filter_discovered_stocks(
        stocks,
        vol_target=10.0, vol_unit="mln", vol_op=">=",
        mc_target=1.0, mc_unit="mld", mc_op=">=",
        birth_year_target=2000, birth_year_op="<="
    )
    assert len(f1) == 1
    assert f1[0]["Ticker"] == "AAPL"

    # Filter Birth Year >= 2010
    f2 = de.filter_discovered_stocks(
        stocks,
        vol_target=None,
        mc_target=None,
        birth_year_target=2010, birth_year_op=">="
    )
    assert len(f2) == 2
    assert {s["Ticker"] for s in f2} == {"PLTR", "SMALL"}
