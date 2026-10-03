import os
import json
from google import genai

def genera_analisi_gemini(ticker: str, ultimo_prezzo: float, trend_timesfm: str, sigma_pct: float,
                           delta_volumi_intra: float, pct_trader_in_perdita: float,
                           api_key: str = None) -> dict:
    """
    Genera per ogni ticker due colonne qualitative via Gemini API (gemini-2.5-flash):
      - Sentiment News
      - Sintesi / Verdetto
    Se la chiave API non è fornita o la chiamata fallisce, restituisce un’analisi euristica sintetica locale.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "")

    if key and key.strip():
        try:
            client = genai.Client(api_key=key.strip())
            prompt = f"""
Sei un analista finanziario senior e quantitativo.
Analizza il seguente ticker finanziario in base alle sue metriche quantitative e volumetriche:

Ticker: {ticker}
Ultimo Prezzo: ${ultimo_prezzo}
Trend TimesFM (modello quantitativo): {trend_timesfm}
Sigma % (Volatilità %): {sigma_pct}%
Delta Volumi Intra %: {delta_volumi_intra}%
% Trader in Perdita: {pct_trader_in_perdita}%

Fornisci la tua risposta ESATTAMENTE in formato JSON con la seguente struttura:
{{
  "Sentiment_News": "<Sintetica analisi del sentiment di mercato, valutazioni correnti e contesto notizie>",
  "Sintesi_Verdetto": "<Giudizio finale sintetico sulla coerenza tra modello quantitativo, volumi intraday e contesto di mercato>"
}}
Rispondi in italiano in modo professionale, conciso e diretto.
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            text_response = response.text.strip()

            # Pulizia markdown del json se presente
            if "```json" in text_response:
                text_response = text_response.split("```json")[1].split("```")[0].strip()
            elif "```" in text_response:
                text_response = text_response.split("```")[1].split("```")[0].strip()

            res_json = json.loads(text_response)
            return {
                "Sentiment News": res_json.get("Sentiment_News", "Sentiment neutrale con volatilità moderata."),
                "Sintesi / Verdetto": res_json.get("Sintesi_Verdetto", "Segnale moderatamente coerente tra trend e volumi.")
            }
        except Exception as e:
            # Fallback euristico se l'API fallisce o la chiave non è valida
            pass

    # Generazione euristica sintetica locale (Fallback)
    sentiment = f"Sentiment di mercato su {ticker} caratterizzato da volatilità del {sigma_pct}%."
    if delta_volumi_intra > 2.0:
        sentiment += " Pressione rialzista sui volumi intraday."
    elif delta_volumi_intra < -2.0:
        sentiment += " Prevalenza dei volumi in vendita nell'intraday."
    else:
        sentiment += " Volumi intraday in fase di equilibrio."

    verdetto = f"Trend {trend_timesfm} confermato."
    if pct_trader_in_perdita > 55.0:
        verdetto += f" Attenzione: il {pct_trader_in_perdita}% dei trader intraday è in perdita (potenziale resistenza)."
    elif pct_trader_in_perdita < 40.0:
        verdetto += f" Maggioranza dei trader in guadagno ({100-pct_trader_in_perdita:.1f}%), possibile estensione."
    else:
        verdetto += " Coerenza moderata tra segnale quantitativo e volumi."

    return {
        "Sentiment News": sentiment,
        "Sintesi / Verdetto": verdetto
    }
