"""
Advanced NLP Sentiment Analysis Module.

Capabilities:
  1. FinBERT (financial sentiment model) with caching
  2. Rule-based fallback for keywords (bullish/bearish/neutral)
  3. Confidence scores and sentiment intensity
  4. Multi-language support
  5. Graceful degradation if transformers not available
  6. Direct yfinance news headline scraping and sentiment scoring
"""
import warnings
import logging
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# Bullish / Bearish keyword lexicon (domain-specific)
SENTIMENT_LEXICON = {
    "en": {
        "bullish": [
            "surge", "rally", "breakout", "bullish", "strength", "momentum", "outperform",
            "upgrade", "beat", "exceed", "growth", "strong", "gain", "recovery",
            "profit", "revenue", "earnings", "innovation", "expansion", "acquisition"
        ],
        "bearish": [
            "plunge", "crash", "breakdown", "bearish", "weakness", "downtrend", "underperform",
            "downgrade", "miss", "decline", "contraction", "weak", "loss", "selloff", "pressure",
            "cut", "suspend", "delay", "concern", "risk", "threat", "scandal"
        ]
    },
    "it": {
        "bullish": [
            "rialzo", "balzo", "rimbalzo", "rialzista", "forza", "momentum", "sovraperformer",
            "upgrade", "superare", "crescita", "forte", "guadagno", "rallentamento", "recupero",
            "profitto", "ricavi", "utili", "innovazione", "espansione", "acquisizione"
        ],
        "bearish": [
            "crollo", "calo", "ribasso", "ribassista", "debolezza", "tendenza negativa", "underperformer",
            "downgrade", "fallimento", "declino", "contrazione", "debole", "perdita", "vendita",
            "pressione", "taglio", "sospensione", "ritardo", "preoccupazione", "rischio"
        ]
    }
}

# Rule thresholds (tunable)
RULE_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for rule-based signal
NEUTRAL_ZONE = 0.1  # Neutrality threshold ([-0.1, 0.1] = NEUTRAL)


class SentimentAnalyzer:
    """
    Unified sentiment analyzer with FinBERT + rule-based fallback.
    """
    def __init__(self, use_finbert=True, cache_size=500):
        self.use_finbert = use_finbert
        self.cache_size = cache_size
        self.cache = {}  # Text -> (sentiment_score, confidence)
        self.finbert_model = None
        self.finbert_tokenizer = None
        
        if use_finbert:
            self._load_finbert()
    
    def _load_finbert(self):
        """Load FinBERT model (financial BERT)."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            model_name = "ProsusAI/finbert"
            logger.info(f"Loading {model_name}...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.finbert_model.to(self.device)
            self.finbert_model.eval()
            logger.info(f"FinBERT loaded on {self.device}")
        except ImportError:
            logger.warning("transformers/torch not available; falling back to rule-based sentiment")
            self.use_finbert = False
        except Exception as e:
            logger.warning(f"FinBERT load failed ({e}); using rule-based fallback")
            self.use_finbert = False
    
    def _finbert_predict(self, text: str) -> tuple:
        """
        FinBERT inference: returns (sentiment_score, confidence).
        sentiment_score: -1 (negative) to +1 (positive), 0 = neutral
        confidence: 0-1
        """
        if not self.use_finbert or self.finbert_model is None:
            return None, None
        
        try:
            import torch
            
            # Truncate to max length
            inputs = self.finbert_tokenizer(
                text[:512],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # FinBERT labels: 0=negative, 1=neutral, 2=positive
            probs_np = probs[0].cpu().numpy()
            sentiment_label = np.argmax(probs_np)
            confidence = float(np.max(probs_np))
            
            # Convert to [-1, 0, 1] scale
            sentiment_score = float((sentiment_label - 1))  # -1, 0, 1
            
            return sentiment_score, confidence
        except Exception as e:
            logger.warning(f"FinBERT inference failed: {e}")
            return None, None
    
    def _rule_based_predict(self, text: str, lang: str = "en") -> tuple:
        """
        Rule-based sentiment using keyword matching.
        Returns (sentiment_score, confidence).
        """
        text_lower = text.lower()
        lexicon = SENTIMENT_LEXICON.get(lang, SENTIMENT_LEXICON["en"])
        
        bullish_count = sum(1 for kw in lexicon["bullish"] if kw in text_lower)
        bearish_count = sum(1 for kw in lexicon["bearish"] if kw in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, 0.0  # Neutral, low confidence
        
        # Sentiment score: (bullish - bearish) / total, normalized to [-1, 1]
        sentiment_score = (bullish_count - bearish_count) / total
        confidence = min(total / 10.0, 1.0)  # Confidence grows with keyword density
        
        return sentiment_score, confidence
    
    def analyze(self, text: str, lang: str = "en") -> dict:
        """
        Unified analysis: FinBERT first, fallback to rules.
        
        Returns:
        {
            "sentiment_label": "BULLISH" | "NEUTRAL" | "BEARISH",
            "sentiment_score": float (-1 to 1),
            "confidence": float (0 to 1),
            "method": "finbert" | "rules",
            "raw_scores": (finbert_score, rule_score, finbert_conf, rule_conf)
        }
        """
        # Check cache
        cache_key = (text[:100], lang)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try FinBERT
        finbert_score, finbert_conf = None, None
        if self.use_finbert:
            finbert_score, finbert_conf = self._finbert_predict(text)
        
        # Rule-based fallback
        rule_score, rule_conf = self._rule_based_predict(text, lang)
        
        # Choose best result
        if finbert_score is not None and finbert_conf is not None and finbert_conf >= 0.6:
            sentiment_score = finbert_score
            confidence = finbert_conf
            method = "finbert"
        elif rule_conf > RULE_CONFIDENCE_THRESHOLD:
            sentiment_score = rule_score
            confidence = rule_conf
            method = "rules"
        elif finbert_score is not None:
            sentiment_score = finbert_score
            confidence = finbert_conf
            method = "finbert"
        else:
            sentiment_score = rule_score
            confidence = rule_conf
            method = "rules"
        
        # Classify
        if abs(sentiment_score) <= NEUTRAL_ZONE:
            sentiment_label = "NEUTRAL"
        elif sentiment_score > 0:
            sentiment_label = "BULLISH"
        else:
            sentiment_label = "BEARISH"
        
        result = {
            "sentiment_label": sentiment_label,
            "sentiment_score": float(sentiment_score),
            "confidence": float(confidence),
            "method": method,
            "raw_scores": (finbert_score, rule_score, finbert_conf, rule_conf)
        }
        
        # Cache
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = result
        
        return result


# Singleton instance
_analyzer = None

def get_sentiment_analyzer(use_finbert=True):
    """Get or create singleton analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer(use_finbert=use_finbert)
    return _analyzer


def analyze_sentiment(text: str, lang: str = "en", use_finbert=True) -> dict:
    """Convenience function for one-off sentiment analysis."""
    analyzer = get_sentiment_analyzer(use_finbert)
    return analyzer.analyze(text, lang)


def analyze_ticker_news(ticker_symbol: str, lang: str = "en", use_finbert=True) -> dict:
    """
    Fetches ticker news from yfinance and computes aggregate NLP news sentiment.
    """
    headlines = []
    top_headline = None
    try:
        t_obj = yf.Ticker(ticker_symbol)
        news_items = t_obj.news or []
        for item in news_items:
            # Handle dictionary formats
            title = None
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, dict):
                    title = content.get("title")
                if not title:
                    title = item.get("title")
            if title:
                headlines.append(title)

        if headlines:
            top_headline = headlines[0]
    except Exception as e:
        logger.debug(f"News fetch failed for {ticker_symbol}: {e}")

    if not headlines:
        return {
            "ticker": ticker_symbol,
            "sentiment_label": "NEUTRAL",
            "avg_score": 0.0,
            "confidence": 0.0,
            "method": "rules",
            "headlines_count": 0,
            "top_headline": None
        }

    analyzer = get_sentiment_analyzer(use_finbert)
    scores = []
    confidences = []
    methods = []

    for headline in headlines[:10]:  # Top 10 headlines
        res = analyzer.analyze(headline, lang=lang)
        scores.append(res["sentiment_score"])
        confidences.append(res["confidence"])
        methods.append(res["method"])

    avg_score = float(np.mean(scores)) if scores else 0.0
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    primary_method = "finbert" if "finbert" in methods else "rules"

    if abs(avg_score) <= NEUTRAL_ZONE:
        label = "NEUTRAL"
    elif avg_score > 0:
        label = "BULLISH"
    else:
        label = "BEARISH"

    return {
        "ticker": ticker_symbol,
        "sentiment_label": label,
        "avg_score": round(avg_score, 2),
        "confidence": round(avg_conf, 2),
        "method": primary_method,
        "headlines_count": len(headlines),
        "top_headline": top_headline
    }
