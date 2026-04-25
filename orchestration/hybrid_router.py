"""
Hybrid Router
=============
Dependency-free first-pass text classifier for obvious domain-routing cases.

The classifier uses small curated bag-of-words/domain phrase profiles. It is
intentionally conservative: it only accepts a route when both the top
confidence and top-vs-second margin are high. Otherwise the caller should fall
back to the LLM supervisor, which handles ambiguity, clarification, and
cross-domain synthesis metadata.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


DOMAIN_PROFILES: Dict[str, Dict[str, float]] = {
    "industrial": {
        "plc": 3.0,
        "scada": 3.0,
        "hmi": 2.5,
        "drive": 2.5,
        "drives": 2.5,
        "powerflex": 4.0,
        "vfd": 3.5,
        "fault": 2.8,
        "alarm": 2.0,
        "motor": 2.0,
        "voltage": 1.8,
        "undervoltage": 3.0,
        "overcurrent": 3.0,
        "ladder": 2.4,
        "rslogix": 3.0,
        "studio5000": 3.0,
        "contactor": 2.5,
        "breaker": 2.0,
        "maintenance": 1.6,
        "troubleshoot": 1.8,
        "f002": 4.0,
        "f003": 4.0,
        "f004": 4.0,
        "f005": 4.0,
        "f007": 4.0,
    },
    "recipe": {
        "recipe": 3.5,
        "cook": 2.3,
        "cooking": 2.3,
        "bake": 2.3,
        "baking": 2.3,
        "ingredient": 2.5,
        "ingredients": 2.5,
        "substitute": 3.0,
        "substitution": 3.0,
        "egg": 2.2,
        "eggs": 2.2,
        "flour": 2.0,
        "pasta": 2.4,
        "pancake": 2.5,
        "pancakes": 2.5,
        "vegetarian": 2.8,
        "vegan": 2.8,
        "keto": 1.8,
        "meal": 2.0,
        "dinner": 2.0,
        "garlic": 1.8,
        "oven": 1.5,
        "thermometer": 1.2,
        "calorie": 2.0,
        "nutrition": 2.0,
    },
    "scientific": {
        "paper": 3.0,
        "papers": 3.0,
        "research": 3.0,
        "arxiv": 4.0,
        "study": 2.4,
        "studies": 2.4,
        "abstract": 2.5,
        "citation": 2.5,
        "citations": 2.5,
        "literature": 3.0,
        "survey": 2.8,
        "summarize": 1.8,
        "transformer": 2.4,
        "transformers": 2.4,
        "embedding": 2.3,
        "embeddings": 2.3,
        "neural": 2.0,
        "model": 1.5,
        "models": 1.5,
        "dataset": 2.0,
        "experiment": 2.0,
        "hypothesis": 2.4,
        "clinical": 2.0,
        "evidence": 1.8,
    },
}


PHRASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "industrial": {
        "fault code": 3.5,
        "powerflex 525": 5.0,
        "powerflex 755": 5.0,
        "allen bradley": 3.5,
        "variable frequency drive": 4.0,
        "control panel": 2.5,
        "machine fault": 3.0,
        "motor drive": 3.0,
    },
    "recipe": {
        "what can i substitute": 4.0,
        "how do i cook": 3.0,
        "quick dinner": 3.0,
        "weeknight meal": 3.0,
        "recipe for": 3.5,
        "gluten free": 3.0,
        "meal prep": 2.5,
    },
    "scientific": {
        "recent research": 3.5,
        "research paper": 4.0,
        "arxiv paper": 4.5,
        "literature review": 4.0,
        "scientific evidence": 3.5,
        "summarize papers": 3.5,
        "state of the art": 3.0,
    },
}


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]*|f\d{3,4}")


@dataclass(frozen=True)
class HybridRoutePrediction:
    """Prediction from the cheap first-pass router."""

    domain: str
    confidence: float
    second_domain: str
    second_confidence: float
    margin: float
    scores: Dict[str, float]
    accepted: bool
    reason: str


class LexicalDomainClassifier:
    """
    Lightweight bag-of-words domain classifier.

    This is a deliberately conservative routing shortcut. It should handle
    obvious domain-specific vocabulary and leave ambiguous cases to the LLM.
    """

    def __init__(
        self,
        profiles: Optional[Dict[str, Dict[str, float]]] = None,
        phrase_weights: Optional[Dict[str, Dict[str, float]]] = None,
        confidence_threshold: float = 0.85,
        margin_threshold: float = 0.20,
    ) -> None:
        self.profiles = profiles or DOMAIN_PROFILES
        self.phrase_weights = phrase_weights or PHRASE_WEIGHTS
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

    def predict(self, query: str) -> HybridRoutePrediction:
        """Return top domain prediction and whether it is safe to accept."""
        scores = self._score(query)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_domain, top_score = ranked[0]
        second_domain, second_score = ranked[1]

        if top_score <= 0:
            return HybridRoutePrediction(
                domain="fallback",
                confidence=0.0,
                second_domain="",
                second_confidence=0.0,
                margin=0.0,
                scores=scores,
                accepted=False,
                reason="No strong lexical evidence for any specialist domain.",
            )

        probabilities = self._softmax(scores)
        confidence = probabilities[top_domain]
        second_confidence = probabilities[second_domain]
        margin = confidence - second_confidence
        cross_domain_evidence = second_score >= 1.5 and (top_score - second_score) < 6.0
        accepted = (
            confidence >= self.confidence_threshold
            and margin >= self.margin_threshold
            and not cross_domain_evidence
        )

        reason = (
            f"Hybrid lexical router predicted {top_domain} "
            f"(confidence={confidence:.2f}, margin={margin:.2f}). "
        )
        if accepted:
            reason += "High-confidence route accepted without LLM supervisor call."
        elif cross_domain_evidence:
            reason += "Multiple domain profiles matched; escalating to LLM supervisor."
        else:
            reason += "Low confidence or low margin; escalating to LLM supervisor."

        return HybridRoutePrediction(
            domain=top_domain,
            confidence=confidence,
            second_domain=second_domain if second_score > 0 else "",
            second_confidence=second_confidence if second_score > 0 else 0.0,
            margin=margin,
            scores=scores,
            accepted=accepted,
            reason=reason,
        )

    def _score(self, query: str) -> Dict[str, float]:
        query_norm = query.lower()
        tokens = self._tokens(query_norm)
        scores = {domain: 0.0 for domain in self.profiles}

        for domain, weights in self.profiles.items():
            scores[domain] += sum(weights.get(token, 0.0) for token in tokens)

        for domain, phrases in self.phrase_weights.items():
            for phrase, weight in phrases.items():
                if phrase in query_norm:
                    scores[domain] += weight

        return scores

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return TOKEN_RE.findall(text.lower())

    @staticmethod
    def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
        # Scale scores so one clear domain can pass a high confidence threshold
        # while ambiguous multi-domain queries still keep a visible margin.
        values = list(scores.values())
        max_score = max(values)
        exps = {
            domain: math.exp((score - max_score) / 2.0)
            for domain, score in scores.items()
        }
        denom = sum(exps.values())
        if denom <= 0:
            return {domain: 0.0 for domain in scores}
        return {domain: value / denom for domain, value in exps.items()}
