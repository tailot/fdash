import os
import json
import math

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


def genera_analisi_gemini(ticker: str, ultimo_prezzo: float, trend_timesfm: str, sigma_pct: float,
                          delta_volumi_intra, pct_trader_in_perdita, api_key: str = None,
                          rend_mediano_pct=None, incertezza_pct=None, lang: str = "en") -> dict:
    """
    Generates qualitative analysis for 'Sentiment News' and 'Sintesi / Verdetto'.
    Uses Gemini API if available, otherwise falls back to localized heuristic analysis.
    """
    delta = _num(delta_volumi_intra)
    perd = _num(pct_trader_in_perdita)
    rend = _num(rend_mediano_pct)
    inc = _num(incertezza_pct)
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    target_lang = LANG_NAMES.get(lang, "English")

    if key and key.strip():
        try:
            from google import genai
            client = genai.Client(api_key=key.strip())
            prompt = f"""
You are a senior quantitative financial analyst.
Analyze the following ticker based on its metrics (N/A = missing data, do not invent it):

Ticker: {ticker}
Last Price: ${_fmt(_num(ultimo_prezzo))}
Model Trend (5-day TimesFM-3 forecast): {trend_timesfm}
Predicted Median Return: {_fmt(rend, '%')}
Uncertainty (cumulative sigma): {_fmt(inc, '%')}
Annualized Volatility: {_fmt(_num(sigma_pct), '%')}
Intraday Volume Delta % (buy - sell): {_fmt(delta, '%')}
% Volume in loss (Volume Profile): {_fmt(perd, '%')}

Respond EXACTLY in JSON with keys "Sentiment_News" and "Sintesi_Verdetto":
{{
  "Sentiment_News": "<summary of sentiment>",
  "Sintesi_Verdetto": "<concise verdict on consistency between forecast, intraday volume, and Volume Profile>"
}}
Respond in {target_lang} language in a professional, concise, and direct tone.
"""
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            res = json.loads(text)
            return {
                "Sentiment News": res.get("Sentiment_News", "Sentiment unavailable."),
                "Sintesi / Verdetto": res.get("Sintesi_Verdetto", "Summary unavailable."),
            }
        except Exception:
            pass  # Fallback to local heuristic

    # Local Heuristic Fallback
    if lang == "it":
        sentiment = f"Sentiment di mercato su {ticker}: volatilità annualizzata {_fmt(_num(sigma_pct), '%')}."
        if delta is None:
            sentiment += " Volumi intraday non disponibili."
        elif delta > 2.0:
            sentiment += " Pressione rialzista sui volumi intraday."
        elif delta < -2.0:
            sentiment += " Prevalenza dei volumi in vendita nell'intraday."
        else:
            sentiment += " Volumi intraday in fase di equilibrio."

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

    elif lang == "es":
        sentiment = f"Sentimiento de mercado para {ticker}: volatilidad anualizada {_fmt(_num(sigma_pct), '%')}."
        if delta is None:
            sentiment += " Volúmenes intradía no disponibles."
        elif delta > 2.0:
            sentiment += " Presión alcista en los volúmenes intradía."
        elif delta < -2.0:
            sentiment += " Predominio de volúmenes de venta en el intradía."
        else:
            sentiment += " Volúmenes intradía en equilibrio."

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

    elif lang == "zh":
        sentiment = f"{ticker} 的市场情绪：年化波动率 {_fmt(_num(sigma_pct), '%')}。"
        if delta is None:
            sentiment += " 日内成交量不可用。"
        elif delta > 2.0:
            sentiment += " 日内买盘成交量存在上涨压力。"
        elif delta < -2.0:
            sentiment += " 日内卖盘成交量占主导。"
        else:
            sentiment += " 日内成交量处于平衡状态。"

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

    elif lang == "fr":
        sentiment = f"Sentiment du marché sur {ticker}: volatilité annualisée {_fmt(_num(sigma_pct), '%')}."
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
        if perd is None:
            verdetto += " Volume Profile non disponible."
        elif perd >= 60:
            verdetto += f" La majorité des acheteurs ({perd:.1f}% du volume) est EN PERTE."
        elif perd <= 40:
            verdetto += f" La majorité des acheteurs est EN GAIN ({100 - perd:.1f}%)."
        else:
            verdetto += " Situation mixte entre acheteurs en perte et en gain."

    else:  # "en" default
        sentiment = f"Market sentiment on {ticker}: annualized volatility {_fmt(_num(sigma_pct), '%')}."
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
        if perd is None:
            verdetto += " Volume Profile unavailable."
        elif perd >= 60:
            verdetto += f" Most buyers ({perd:.1f}% of volume) are IN LOSS."
        elif perd <= 40:
            verdetto += f" Most buyers are IN PROFIT ({100 - perd:.1f}%)."
        else:
            verdetto += " Mixed distribution between buyers in loss and profit."

    return {"Sentiment News": sentiment, "Sintesi / Verdetto": verdetto}
