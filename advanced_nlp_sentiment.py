import math
import logging
import numpy as np

logger = logging.getLogger(__name__)

SENTIMENT_LEXICON = {
    "en": {
        "bullish": [
            "surge", "rally", "breakout", "bullish", "strength", "momentum",
            "outperform", "upgrade", "beat", "exceed", "growth", "strong",
            "gain", "recovery", "profit", "revenue", "earnings", "innovation",
            "expansion", "acquisition"
        ],
        "bearish": [
            "plunge", "crash", "breakdown", "bearish", "weakness", "downtrend",
            "underperform", "downgrade", "miss", "decline", "contraction",
            "weak", "loss", "selloff", "pressure", "cut", "suspend", "delay",
            "concern", "risk", "threat", "scandal"
        ]
    },
    "it": {
        "bullish": [
            "rialzo", "balzo", "rimbalzo", "rialzista", "forza", "momentum",
            "sovraperformer", "superare", "crescita", "forte", "guadagno",
            "recupero", "profitto", "ricavi", "utili", "innovazione",
            "espansione", "acquisizione"
        ],
        "bearish": [
            "crollo", "calo", "ribasso", "ribassista", "debolezza",
            "tendenza negativa", "fallimento", "declino", "contrazione",
            "debole", "perdita", "vendita", "pressione", "taglio",
            "sospensione", "ritardo", "preoccupazione", "rischio"
        ]
    },
    "es": {
        "bullish": [
            "repunte", "subida", "ruptura", "alcista", "fuerza", "momento",
            "mejorar", "mejora", "superar", "crecimiento", "fuerte", "ganancia",
            "recuperación", "beneficio", "ingresos", "beneficios", "innovación",
            "expansión", "adquisición"
        ],
        "bearish": [
            "caída", "colapso", "quebrantamiento", "bajista", "debilidad", "caída",
            "bajo rendimiento", "degradación", "fracaso", "declive", "contracción",
            "débil", "pérdida", "venta", "presión", "corte", "suspensión",
            "retardo", "preocupación", "riesgo", "amenaza"
        ]
    },
    "fr": {
        "bullish": [
            "hausse", "rebond", "rupture", "haussier", "force", "momentum",
            "performance", "amélioration", "surperformer", "croissance", "fort",
            "gain", "récupération", "profit", "revenus", "bénéfices", "innovation",
            "expansion", "acquisition"
        ],
        "bearish": [
            "chute", "effondrement", "vente", "baissier", "faiblesse", "baisse",
            "sous-performance", "dégradation", "échec", "déclin", "contraction",
            "faible", "perte", "vendre", "pression", "coupe", "suspension",
            "retard", "préoccupation", "risque", "menace"
        ]
    },
    "zh": {
        "bullish": [
            "上涨", "反弹", "突破", "看涨", "强势", "动量", "跑赢",
            "上调", "超预期", "增长", "强", "收益", "恢复", "利润",
            "收入", "盈利", "创新", "扩张", "收购"
        ],
        "bearish": [
            "下跌", "崩盘", "突破失败", "看跌", "疲软", "下行趋势",
            "跑输", "下调", "失误", "下滑", "收缩", "弱", "亏损",
            "抛售", "压力", "削减", "暂停", "延迟", "担忧", "风险", "威胁"
        ]
    },
}

RULE_CONFIDENCE_THRESHOLD = 0.5
NEUTRAL_ZONE = 0.1


def _patch_transformers_dummy_modules():
    import sys
    for name, m in list(sys.modules.items()):
        if name.startswith("transformers.") and hasattr(m, "__getattr__"):
            orig_getattr = m.__getattr__
            if getattr(orig_getattr, "_is_safe", False):
                continue
            def make_safe_getattr(og, mod_name):
                def safe_getattr(attr):
                    if attr.startswith("__"):
                        raise AttributeError(f"module {mod_name} has no attribute {attr}")
                    return og(attr)
                safe_getattr._is_safe = True
                return safe_getattr
            m.__getattr__ = make_safe_getattr(orig_getattr, name)


class SentimentAnalyzer:
    def __init__(self, use_finbert=True, cache_size=500):
        self.use_finbert = use_finbert
        self.cache_size = cache_size
        self.cache = {}
        self.finbert_model = None
        self.finbert_tokenizer = None

        if use_finbert:
            self._load_finbert()

    def _load_finbert(self):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            _patch_transformers_dummy_modules()

            model_name = "ProsusAI/finbert"
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.finbert_model.to(self.device)
            self.finbert_model.eval()
        except Exception:
            self.use_finbert = False

    def _finbert_predict(self, text):
        if not self.use_finbert or self.finbert_model is None:
            return None, None
        try:
            import torch

            inputs = self.finbert_tokenizer(
                text[:512],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            probs_np = probs[0].cpu().numpy()
            label = int(np.argmax(probs_np))
            confidence = float(np.max(probs_np))
            score = float(label - 1)
            return score, confidence
        except Exception as e:
            logger.warning("FinBERT failed: %s", e)
            return None, None

    def _rule_based_predict(self, text, lang="en"):
        tx = text.lower()
        lex = SENTIMENT_LEXICON.get(lang, SENTIMENT_LEXICON["en"])
        bullish = sum(1 for token in lex["bullish"] if token in tx)
        bearish = sum(1 for token in lex["bearish"] if token in tx)
        total = bullish + bearish
        if total == 0:
            return 0.0, 0.0
        score = (bullish - bearish) / total
        confidence = min(total / 10.0, 1.0)
        return score, confidence

    def analyze(self, text, lang="en"):
        cache_key = (text[:100], lang)
        if cache_key in self.cache:
            return self.cache[cache_key]

        finbert_score, finbert_conf = self._finbert_predict(text)
        rule_score, rule_conf = self._rule_based_predict(text, lang)

        if finbert_score is not None and finbert_conf is not None and finbert_conf >= 0.6:
            score = finbert_score
            confidence = finbert_conf
            method = "finbert"
        elif rule_conf > RULE_CONFIDENCE_THRESHOLD:
            score = rule_score
            confidence = rule_conf
            method = "rules"
        elif finbert_score is not None:
            score = finbert_score
            confidence = finbert_conf
            method = "finbert"
        else:
            score = rule_score
            confidence = rule_conf
            method = "rules"

        if abs(score) <= NEUTRAL_ZONE:
            label = "NEUTRAL"
        elif score > 0:
            label = "BULLISH"
        else:
            label = "BEARISH"

        result = {
            "sentiment_label": label,
            "sentiment_score": float(score),
            "confidence": float(confidence),
            "method": method,
            "raw_scores": (finbert_score, rule_score, finbert_conf, rule_conf),
        }

        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = result
        return result


_analyzer = None


def get_sentiment_analyzer(use_finbert=True):
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer(use_finbert=use_finbert)
    return _analyzer


def analyze_sentiment(text, lang="en", use_finbert=True):
    return get_sentiment_analyzer(use_finbert).analyze(text, lang)
