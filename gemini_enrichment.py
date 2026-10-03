import os
import json
import math


def _num(x):
    try:
        x = float(x)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _fmt(x, suffix=""):
    return "n/d" if x is None else f"{x}{suffix}"


def _coerenza(trend: str, delta: float | None) -> str:
    """Confronta il segno del trend del modello con il delta volumi (soglia 2%, come il Colab volumi)."""
    if delta is None or trend not in ("BUY", "SELL"):
        return ""
    if (trend == "BUY" and delta > 2.0) or (trend == "SELL" and delta < -2.0):
        return " Volumi intraday coerenti con il segnale."
    if (trend == "BUY" and delta < -2.0) or (trend == "SELL" and delta > 2.0):
        return " Divergenza: i volumi intraday vanno nella direzione opposta al segnale."
    return " Volumi intraday in equilibrio, conferma debole."


def genera_analisi_gemini(ticker: str, ultimo_prezzo: float, trend_timesfm: str, sigma_pct: float,
                          delta_volumi_intra, pct_trader_in_perdita, api_key: str = None,
                          rend_mediano_pct=None, incertezza_pct=None) -> dict:
    """
    Genera le colonne qualitative 'Sentiment News' e 'Sintesi / Verdetto'.
    Con chiave Gemini: gemini-2.5-flash. Senza chiave (o in caso di errore): analisi euristica locale che
    usa le stesse soglie dei Colab (perdita >= 60% / guadagno <= 40%, equilibrio volumi +-2%).
    I valori mancanti (None/NaN) vengono dichiarati "n/d", mai inventati.
    """
    delta = _num(delta_volumi_intra)
    perd = _num(pct_trader_in_perdita)
    rend = _num(rend_mediano_pct)
    inc = _num(incertezza_pct)
    key = api_key or os.environ.get("GEMINI_API_KEY", "")

    if key and key.strip():
        try:
            from google import genai
            client = genai.Client(api_key=key.strip())
            prompt = f"""
Sei un analista finanziario senior e quantitativo.
Analizza il seguente ticker in base alle sue metriche (n/d = dato non disponibile, non inventarlo):

Ticker: {ticker}
Ultimo Prezzo: ${_fmt(_num(ultimo_prezzo))}
Trend dal modello (previsione TimesFM-3 a 5 giorni): {trend_timesfm}
Rendimento mediano previsto: {_fmt(rend, '%')}
Incertezza (sigma cumulata): {_fmt(inc, '%')}
Volatilita' annualizzata: {_fmt(_num(sigma_pct), '%')}
Delta Volumi Intra % (buy - sell): {_fmt(delta, '%')}
% Volume in perdita (Volume Profile): {_fmt(perd, '%')}

Rispondi ESATTAMENTE in JSON:
{{
  "Sentiment_News": "<sintesi del sentiment; non hai accesso alle notizie in tempo reale: se non le conosci dillo>",
  "Sintesi_Verdetto": "<giudizio sintetico sulla coerenza tra previsione, volumi intraday e Volume Profile>"
}}
Rispondi in italiano in modo professionale, conciso e diretto.
"""
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            res = json.loads(text)
            return {
                "Sentiment News": res.get("Sentiment_News", "Sentiment non disponibile."),
                "Sintesi / Verdetto": res.get("Sintesi_Verdetto", "Sintesi non disponibile."),
            }
        except Exception:
            pass  # fallback euristico

    sentiment = f"Sentiment di mercato su {ticker}: volatilita' annualizzata {_fmt(_num(sigma_pct), '%')}."
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
    verdetto += _coerenza(trend_timesfm, delta)
    if perd is None:
        verdetto += " Volume Profile non disponibile."
    elif perd >= 60:
        verdetto += f" La maggior parte degli acquirenti ({perd:.1f}% del volume) e' IN PERDITA: possibile pressione di vendita verso le zone sopra."
    elif perd <= 40:
        verdetto += f" La maggior parte degli acquirenti e' IN GUADAGNO ({100 - perd:.1f}%): rischio di presa di profitto."
    else:
        verdetto += " Situazione mista tra acquirenti in perdita e in guadagno."
    return {"Sentiment News": sentiment, "Sintesi / Verdetto": verdetto}
