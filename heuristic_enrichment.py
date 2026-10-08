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

import math

from advanced_nlp_sentiment import analyze_sentiment

LANG_NAMES = {
    "en": "English",
    "it": "Italian",
    "es": "Spanish",
    "zh": "Chinese",
    "fr": "French",
}


def _num(x):
    try:
        x = float(x)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _fmt(x, suffix=""):
    return "N/A" if x is None else f"{x}{suffix}"


def _coerenza(trend: str, delta: float | None, lang: str = "en") -> str:
    """Compares model trend signal with intraday volume delta."""
    if delta is None or trend not in ("BUY", "SELL"):
        return ""
    if lang == "it":
        if (trend == "BUY" and delta > 2.0) or (trend == "SELL" and delta < -2.0):
            return " Volumi intraday coerenti con il segnale."
        if (trend == "BUY" and delta < -2.0) or (trend == "SELL" and delta > 2.0):
            return " Divergenza: i volumi intraday vanno nella direzione opposta al segnale."
        return " Volumi intraday in equilibrio, conferma debole."
    if lang == "es":
        if (trend == "BUY" and delta > 2.0) or (trend == "SELL" and delta < -2.0):
            return " Volúmenes intradía coherentes con la señal."
        if (trend == "BUY" and delta < -2.0) or (trend == "SELL" and delta > 2.0):
            return " Divergencia: los volúmenes intradía van en dirección opuesta a la señal."
        return " Volúmenes intradía en equilibrio, confirmación débil."
    if lang == "zh":
        if (trend == "BUY" and delta > 2.0) or (trend == "SELL" and delta < -2.0):
            return " 日内成交量与信号方向一致。"
        if (trend == "BUY" and delta < -2.0) or (trend == "SELL" and delta > 2.0):
            return " 背离：日内成交量方向与信号相反。"
        return " 日内成交量处于平衡状态，确认力度较弱。"
    if lang == "fr":
        if (trend == "BUY" and delta > 2.0) or (trend == "SELL" and delta < -2.0):
            return " Volumes intraday alignés avec le signal."
        if (trend == "BUY" and delta < -2.0) or (trend == "SELL" and delta > 2.0):
            return " Divergence: les volumes intraday vont dans la direction opposée."
        return " Volumes intraday en équilibre, confirmation faible."
    if (trend == "BUY" and delta > 2.0) or (trend == "SELL" and delta < -2.0):
        return " Intraday volume is aligned with signal."
    if (trend == "BUY" and delta < -2.0) or (trend == "SELL" and delta > 2.0):
        return " Divergence: intraday volume goes against signal."
    return " Intraday volume in equilibrium, weak confirmation."


def _build_advanced_sentiment(ticker: str, trend_timesfm: str, sigma_pct, delta_volumi_intra,
                             lang: str = "en") -> str:
    try:
        sentiment_data = analyze_sentiment(
            f"{ticker} {trend_timesfm} volatility {sigma_pct} intraday volume {delta_volumi_intra}",
            lang=lang,
            use_finbert=True,
        )
        label = sentiment_data["sentiment_label"]
        score = sentiment_data["sentiment_score"]
        confidence = sentiment_data["confidence"]
        method = sentiment_data["method"]

        if lang == "it":
            return (
                f"Sentiment avanzato: {label} ({score:.2f}, confidenza={confidence:.2f}) tramite {method}."
            )
        if lang == "es":
            return (
                f"Sentimiento avanzado: {label} ({score:.2f}, confianza={confidence:.2f}) vía {method}."
            )
        if lang == "zh":
            return (
                f"高级情绪分析：{label} (得分: {score:.2f}, 置信度: {confidence:.2f})，方法: {method}。"
            )
        if lang == "fr":
            return (
                f"Sentiment avancé: {label} ({score:.2f}, confiance={confidence:.2f}) via {method}."
            )
        return (
            f"Advanced sentiment: {label} ({score:.2f}, confidence={confidence:.2f}) via {method}."
        )
    except Exception as e:
        if lang == "it":
            return f"Sentiment avanzato: non disponibile ({type(e).__name__})."
        if lang == "es":
            return f"Sentimiento avanzado: no disponible ({type(e).__name__})."
        if lang == "zh":
            return f"高级情绪分析：不可用 ({type(e).__name__})。"
        if lang == "fr":
            return f"Sentiment avancé: non disponible ({type(e).__name__})."
        return f"Advanced sentiment: unavailable ({type(e).__name__})."


def genera_analisi_euristica(ticker: str, ultimo_prezzo: float, trend_timesfm: str, sigma_pct: float,
                              delta_volumi_intra, pct_trader_in_perdita,
                              rend_mediano_pct=None, incertezza_pct=None, lang: str = "en") -> dict:
    """
    Generates localized heuristic qualitative analysis with ADVANCED NLP sentiment.
    """
    delta = _num(delta_volumi_intra)
    perd = _num(pct_trader_in_perdita)
    rend = _num(rend_mediano_pct)
    inc = _num(incertezza_pct)
    advanced_sentiment = _build_advanced_sentiment(
        ticker, trend_timesfm, sigma_pct, delta_volumi_intra, lang=lang
    )

    if lang == "it":
        sentiment = f"Sentiment di mercato su {ticker}: volatilità annualizzata {_fmt(_num(sigma_pct), '%')}"
        if delta is None:
            sentiment += ". Volumi intraday non disponibili."
        elif delta > 2.0:
            sentiment += ". Pressione rialzista sui volumi intraday."
        elif delta < -2.0:
            sentiment += ". Prevalenza dei volumi in vendita nell'intraday."
        else:
            sentiment += ". Volumi intraday in fase di equilibrio."
        sentiment += f" {advanced_sentiment}"

        verdetto = f"Segnale del modello: {trend_timesfm}."
        if rend is not None:
            verdetto += f" Rendimento mediano previsto {rend:+.2f}%."
        verdetto += _coerenza(trend_timesfm, delta, lang)
        if perd is None:
            verdetto += " Volume Profile non disponibile."
        elif perd >= 60:
            verdetto += f" La maggior parte degli acquirenti ({perd:.1f}% del volume) è IN PERDITA."
        elif perd <= 40:
            verdetto += f" La maggior parte degli acquirenti è IN GUADAGNO ({100 - perd:.1f}%)."
        else:
            verdetto += " Situazione mista tra acquirenti in perdita e in guadagno."
        verdetto += f" {advanced_sentiment}"

    elif lang == "es":
        sentiment = f"Sentimiento de mercado para {ticker}: volatilidad anualizada {_fmt(_num(sigma_pct), '%')}"
        if delta is None:
            sentiment += ". Volúmenes intradía no disponibles."
        elif delta > 2.0:
            sentiment += ". Presión alcista en los volúmenes intradía."
        elif delta < -2.0:
            sentiment += ". Predominio de volúmenes de venta en el intradía."
        else:
            sentiment += ". Volúmenes intradía en equilibrio."
        sentiment += f" {advanced_sentiment}"

        verdetto = f"Señal del modelo: {trend_timesfm}."
        if rend is not None:
            verdetto += f" Rendimiento mediano previsto {rend:+.2f}%."
        verdetto += _coerenza(trend_timesfm, delta, lang)
        if perd is None:
            verdetto += " Volume Profile no disponible."
        elif perd >= 60:
            verdetto += f" La mayoría de los compradores ({perd:.1f}% del volumen) está EN PÉRDIDA."
        elif perd <= 40:
            verdetto += f" La mayoría de los compradores está EN GANANCIA ({100 - perd:.1f}%)."
        else:
            verdetto += " Situación mixta entre compradores en pérdida y en ganancia."
        verdetto += f" {advanced_sentiment}"

    elif lang == "zh":
        sentiment = f"{ticker} 的市场情绪：年化波动率 {_fmt(_num(sigma_pct), '%')}"
        if delta is None:
            sentiment += "。日内成交量不可用。"
        elif delta > 2.0:
            sentiment += "。日内买盘成交量存在上涨压力。"
        elif delta < -2.0:
            sentiment += "。日内卖盘成交量占主导。"
        else:
            sentiment += "。日内成交量处于平衡状态。"
        sentiment += f" {advanced_sentiment}"

        verdetto = f"模型信号：{trend_timesfm}。"
        if rend is not None:
            verdetto += f" 预测中位数收益率 {rend:+.2f}%。"
        verdetto += _coerenza(trend_timesfm, delta, lang)
        if perd is None:
            verdetto += " Volume Profile 不可用。"
        elif perd >= 60:
            verdetto += f" 大多数买家 ({perd:.1f}% 的成交量) 处于亏损状态。"
        elif perd <= 40:
            verdetto += f" 大多数买家处于盈利状态 ({100 - perd:.1f}%)。"
        else:
            verdetto += " 处于亏损与盈利买家混合分布状态。"
        verdetto += f" {advanced_sentiment}"

    elif lang == "fr":
        sentiment = f"Sentiment du marché sur {ticker}: volatilité annualisée {_fmt(_num(sigma_pct), '%')}"
        if delta is None:
            sentiment += ". Volumes intraday non disponibles."
        elif delta > 2.0:
            sentiment += ". Pression haussière sur les volumes intraday."
        elif delta < -2.0:
            sentiment += ". Prévalence des volumes vendeurs en intraday."
        else:
            sentiment += ". Volumes intraday en équilibre."
        sentiment += f" {advanced_sentiment}"

        verdetto = f"Signal du modèle: {trend_timesfm}."
        if rend is not None:
            verdetto += f" Rendement médian prévu {rend:+.2f}%."
        verdetto += _coerenza(trend_timesfm, delta, lang)
        if perd is None:
            verdetto += " Volume Profile non disponible."
        elif perd >= 60:
            verdetto += f" La majorité des acheteurs ({perd:.1f}% du volume) est EN PERTE."
        elif perd <= 40:
            verdetto += f" La majorité des acheteurs est EN GAIN ({100 - perd:.1f}%)."
        else:
            verdetto += " Situation mixte entre acheteurs en perte et en gain."
        verdetto += f" {advanced_sentiment}"

    else:  # en
        sentiment = f"Market sentiment on {ticker}: annualized volatility {_fmt(_num(sigma_pct), '%')}"
        if delta is None:
            sentiment += ". Intraday volume unavailable."
        elif delta > 2.0:
            sentiment += ". Bullish intraday volume pressure."
        elif delta < -2.0:
            sentiment += ". Bearish intraday volume pressure."
        else:
            sentiment += ". Intraday volume in equilibrium."
        sentiment += f" {advanced_sentiment}"

        verdetto = f"Model signal: {trend_timesfm}."
        if rend is not None:
            verdetto += f" Predicted median return {rend:+.2f}%."
        verdetto += _coerenza(trend_timesfm, delta, lang)
        if perd is None:
            verdetto += " Volume Profile unavailable."
        elif perd >= 60:
            verdetto += f" Most buyers ({perd:.1f}% of volume) are IN LOSS."
        elif perd <= 40:
            verdetto += f" Most buyers are IN PROFIT ({100 - perd:.1f}%)."
        else:
            verdetto += " Mixed distribution between buyers in loss and profit."
        verdetto += f" {advanced_sentiment}"

    return {"Sentiment News": sentiment, "Sintesi / Verdetto": verdetto}
