import math
from advanced_nlp_sentiment import analyze_ticker_news
from consensus_analytics import get_consensus_analytics
from behavioral_finance import get_behavioral_signals

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
    elif lang == "es":
        if (trend == "BUY" and delta > 2.0) or (trend == "SELL" and delta < -2.0):
            return " Volúmenes intradía coherentes con la señal."
        if (trend == "BUY" and delta < -2.0) or (trend == "SELL" and delta > 2.0):
            return " Divergencia: los volúmenes intradía van en dirección opuesta a la señal."
        return " Volúmenes intradía en equilibrio, confirmación débil."
    elif lang == "zh":
        if (trend == "BUY" and delta > 2.0) or (trend == "SELL" and delta < -2.0):
            return " 日内成交量与信号方向一致。"
        if (trend == "BUY" and delta < -2.0) or (trend == "SELL" and delta > 2.0):
            return " 背离：日内成交量方向与信号相反。"
        return " 日内成交量处于平衡状态，确认力度较弱。"
    elif lang == "fr":
        if (trend == "BUY" and delta > 2.0) or (trend == "SELL" and delta < -2.0):
            return " Volumes intraday alignés avec le signal."
        if (trend == "BUY" and delta < -2.0) or (trend == "SELL" and delta > 2.0):
            return " Divergence: les volumes intraday vont dans la direction opposée."
        return " Volumes intraday en équilibre, confirmation faible."
    else:  # "en" default
        if (trend == "BUY" and delta > 2.0) or (trend == "SELL" and delta < -2.0):
            return " Intraday volume is aligned with signal."
        if (trend == "BUY" and delta < -2.0) or (trend == "SELL" and delta > 2.0):
            return " Divergence: intraday volume goes against signal."
        return " Intraday volume in equilibrium, weak confirmation."


def genera_analisi_euristica(ticker: str, ultimo_prezzo: float, trend_timesfm: str, sigma_pct: float,
                             delta_volumi_intra, pct_trader_in_perdita,
                             rend_mediano_pct=None, incertezza_pct=None, lang: str = "en",
                             consensus_data: dict = None, behavioral_data: dict = None,
                             nlp_sentiment_data: dict = None) -> dict:
    """
    Generates localized heuristic qualitative analysis incorporating NLP news sentiment,
    consensus analytics, and behavioral finance signals.
    """
    delta = _num(delta_volumi_intra)
    perd = _num(pct_trader_in_perdita)
    rend = _num(rend_mediano_pct)
    inc = _num(incertezza_pct)

    # Fetch NLP sentiment if not provided
    if nlp_sentiment_data is None:
        nlp_sentiment_data = analyze_ticker_news(ticker, lang=lang)

    # Fetch consensus data if not provided
    if consensus_data is None:
        consensus_data = get_consensus_analytics(ticker, current_price=_num(ultimo_prezzo))

    # Fetch behavioral signals if not provided
    if behavioral_data is None:
        behavioral_data = get_behavioral_signals(ticker)

    nlp_label = nlp_sentiment_data.get("sentiment_label", "NEUTRAL")
    nlp_score = nlp_sentiment_data.get("avg_score", 0.0)
    nlp_method = nlp_sentiment_data.get("method", "rules")
    rec = consensus_data.get("recommendation", "N/A")
    target_m = consensus_data.get("target_mean")
    upside = consensus_data.get("upside_pct")
    beh_synth = behavioral_data.get("synthesis", "Normal")

    if lang == "it":
        sentiment = f"Sentiment NLP ({nlp_method.upper()}): {nlp_label} (punteggio: {nlp_score:+.2f}). "
        sentiment += f"Volatilità annualizzata {_fmt(_num(sigma_pct), '%')}."
        if delta is None:
            sentiment += " Volumi intraday non disponibili."
        elif delta > 2.0:
            sentiment += " Pressione rialzista sui volumi intraday."
        elif delta < -2.0:
            sentiment += " Prevalenza dei volumi in vendita nell'intraday."
        else:
            sentiment += " Volumi intraday in fase di equilibrio."

        verdetto = f"Segnale modello: {trend_timesfm}."
        if rend is not None:
            verdetto += f" Rendimento mediano previsto {rend:+.2f}%."
        verdetto += _coerenza(trend_timesfm, delta, lang)
        if target_m is not None and upside is not None:
            verdetto += f" Consensus analisti: {rec} (Target: ${target_m:.2f}, upside {upside:+.1f}%)."
        if beh_synth and beh_synth != "No Unusual Behavioral Anomaly":
            verdetto += f" Segnali comportamentali: {beh_synth}."
        if perd is None:
            verdetto += " Volume Profile non disponibile."
        elif perd >= 60:
            verdetto += f" Maggioranza acquirenti ({perd:.1f}% volume) IN PERDITA."
        elif perd <= 40:
            verdetto += f" Maggioranza acquirenti IN GUADAGNO ({100 - perd:.1f}%)."
        else:
            verdetto += " Situazione mista acquirenti in perdita e guadagno."

    elif lang == "es":
        sentiment = f"Sentimiento NLP ({nlp_method.upper()}): {nlp_label} (puntuación: {nlp_score:+.2f}). "
        sentiment += f"Volatilidad anualizada {_fmt(_num(sigma_pct), '%')}."
        if delta is None:
            sentiment += " Volúmenes intradía no disponibles."
        elif delta > 2.0:
            sentiment += " Presión alcista en volúmenes intradía."
        elif delta < -2.0:
            sentiment += " Predominio de ventas intradía."
        else:
            sentiment += " Volúmenes intradía en equilibrio."

        verdetto = f"Señal del modelo: {trend_timesfm}."
        if rend is not None:
            verdetto += f" Rendimiento mediano previsto {rend:+.2f}%."
        verdetto += _coerenza(trend_timesfm, delta, lang)
        if target_m is not None and upside is not None:
            verdetto += f" Consenso analistas: {rec} (Objetivo: ${target_m:.2f}, upside {upside:+.1f}%)."
        if beh_synth and beh_synth != "No Unusual Behavioral Anomaly":
            verdetto += f" Señales de finanzas conductuales: {beh_synth}."
        if perd is None:
            verdetto += " Volume Profile no disponible."
        elif perd >= 60:
            verdetto += f" La mayoría ({perd:.1f}% del volumen) está EN PÉRDIDA."
        elif perd <= 40:
            verdetto += f" La mayoría está EN GANANCIA ({100 - perd:.1f}%)."
        else:
            verdetto += " Situación mixta entre compradores."

    elif lang == "zh":
        sentiment = f"NLP 新闻情绪 ({nlp_method.upper()}): {nlp_label} (得分: {nlp_score:+.2f})。"
        sentiment += f"年化波动率 {_fmt(_num(sigma_pct), '%')}。"
        if delta is None:
            sentiment += " 日内成交量不可用。"
        elif delta > 2.0:
            sentiment += " 日内买盘存在上涨压力。"
        elif delta < -2.0:
            sentiment += " 日内卖盘占主导。"
        else:
            sentiment += " 日内成交量处于平衡状态。"

        verdetto = f"模型信号：{trend_timesfm}。"
        if rend is not None:
            verdetto += f" 预测中位数收益率 {rend:+.2f}%。"
        verdetto += _coerenza(trend_timesfm, delta, lang)
        if target_m is not None and upside is not None:
            verdetto += f" 分析师共识: {rec} (目标价: ${target_m:.2f}, 空间 {upside:+.1f}%)。"
        if beh_synth and beh_synth != "No Unusual Behavioral Anomaly":
            verdetto += f" 行为金融信号: {beh_synth}。"
        if perd is None:
            verdetto += " Volume Profile 不可用。"
        elif perd >= 60:
            verdetto += f" 大多数买家 ({perd:.1f}% 成交量) 处于亏损状态。"
        elif perd <= 40:
            verdetto += f" 大多数买家处于盈利状态 ({100 - perd:.1f}%)。"
        else:
            verdetto += " 处于亏损与盈利买家混合分布状态。"

    elif lang == "fr":
        sentiment = f"Sentiment NLP ({nlp_method.upper()}): {nlp_label} (score: {nlp_score:+.2f}). "
        sentiment += f"Volatilité annualisée {_fmt(_num(sigma_pct), '%')}."
        if delta is None:
            sentiment += " Volumes intraday non disponibles."
        elif delta > 2.0:
            sentiment += " Pression haussière sur les volumes intraday."
        elif delta < -2.0:
            sentiment += " Prévalence des volumes vendeurs en intraday."
        else:
            sentiment += " Volumes intraday en équilibre."

        verdetto = f"Signal du modèle: {trend_timesfm}."
        if rend is not None:
            verdetto += f" Rendement médian prévu {rend:+.2f}%."
        verdetto += _coerenza(trend_timesfm, delta, lang)
        if target_m is not None and upside is not None:
            verdetto += f" Consensus analystes: {rec} (Cible: ${target_m:.2f}, upside {upside:+.1f}%)."
        if beh_synth and beh_synth != "No Unusual Behavioral Anomaly":
            verdetto += f" Signaux de finance comportementale: {beh_synth}."
        if perd is None:
            verdetto += " Volume Profile non disponible."
        elif perd >= 60:
            verdetto += f" La majorité ({perd:.1f}% du volume) est EN PERTE."
        elif perd <= 40:
            verdetto += f" La majorité est EN GAIN ({100 - perd:.1f}%)."
        else:
            verdetto += " Situation mixte entre acheteurs."

    else:  # "en" default
        sentiment = f"NLP Sentiment ({nlp_method.upper()}): {nlp_label} (score: {nlp_score:+.2f}). "
        sentiment += f"Annualized volatility {_fmt(_num(sigma_pct), '%')}."
        if delta is None:
            sentiment += " Intraday volume unavailable."
        elif delta > 2.0:
            sentiment += " Bullish intraday volume pressure."
        elif delta < -2.0:
            sentiment += " Bearish intraday volume pressure."
        else:
            sentiment += " Intraday volume in equilibrium."

        verdetto = f"Model signal: {trend_timesfm}."
        if rend is not None:
            verdetto += f" Predicted median return {rend:+.2f}%."
        verdetto += _coerenza(trend_timesfm, delta, lang)
        if target_m is not None and upside is not None:
            verdetto += f" Analyst consensus: {rec} (Target: ${target_m:.2f}, upside {upside:+.1f}%)."
        if beh_synth and beh_synth != "No Unusual Behavioral Anomaly":
            verdetto += f" Behavioral finance signals: {beh_synth}."
        if perd is None:
            verdetto += " Volume Profile unavailable."
        elif perd >= 60:
            verdetto += f" Most buyers ({perd:.1f}% of volume) are IN LOSS."
        elif perd <= 40:
            verdetto += f" Most buyers are IN PROFIT ({100 - perd:.1f}%)."
        else:
            verdetto += " Mixed distribution between buyers in loss and profit."

    return {
        "Sentiment News": sentiment,
        "Sintesi / Verdetto": verdetto,
        "Consensus_Rating": rec,
        "Target_Price": target_m,
        "Upside_%": upside,
        "Behavioral_Signals": beh_synth,
        "NLP_Sentiment_Label": nlp_label,
        "NLP_Sentiment_Score": nlp_score
    }
