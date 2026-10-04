"""Offline test suite: no network access (yfinance and forecaster are mocked with synthetic data)."""
import numpy as np
import pandas as pd
import pytest

import quant_engine as qe
import volume_engine as ve
from volume_engine import stima_buy_sell, verdetto_delta
from heuristic_enrichment import genera_analisi_euristica
from i18n import t, LANGUAGES


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


def test_heuristic_enrichment_multilingual():
    e_en = genera_analisi_euristica("MSTR", 160.01, "BUY", 10.44, 9.79, 59.9, lang="en")
    assert "Market sentiment on MSTR" in e_en["Sentiment News"]
    assert "aligned" in e_en["Sintesi / Verdetto"]

    e_it = genera_analisi_euristica("MSTR", 160.01, "BUY", 10.44, 9.79, 59.9, lang="it")
    assert "Sentiment di mercato su MSTR" in e_it["Sentiment News"]
    assert "coerenti" in e_it["Sintesi / Verdetto"]

    e_es = genera_analisi_euristica("MSTR", 160.01, "BUY", 10.44, 9.79, 59.9, lang="es")
    assert "Sentimiento de mercado para MSTR" in e_es["Sentiment News"]
    assert "coherentes" in e_es["Sintesi / Verdetto"]

    e_zh = genera_analisi_euristica("MSTR", 160.01, "BUY", 10.44, 9.79, 59.9, lang="zh")
    assert "MSTR 的市场情绪" in e_zh["Sentiment News"]
    assert "一致" in e_zh["Sintesi / Verdetto"]

    e_fr = genera_analisi_euristica("MSTR", 160.01, "BUY", 10.44, 9.79, 59.9, lang="fr")
    assert "Sentiment du marché sur MSTR" in e_fr["Sentiment News"]
    assert "alignés" in e_fr["Sintesi / Verdetto"]


def test_heuristic_enrichment_dati_mancanti():
    e = genera_analisi_euristica("MSTR", 160.01, "BUY", float("nan"), None, float("nan"), lang="en")
    assert "unavailable" in e["Sentiment News"]
