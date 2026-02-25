"""
Recipe Assistant Agent
======================
Specialized for cooking guidance, ingredient substitutions,
and recipe recommendations. Adds ingredient-aware query preprocessing
and nutrition highlighting.
"""

from __future__ import annotations

import logging
import re

from agents.base_agent import BaseAgent
from rag_core.crag_pipeline import CRAGPipeline, CRAGResult

logger = logging.getLogger(__name__)

# Common substitution triggers
SUBSTITUTION_KEYWORDS = [
    "substitute", "replace", "instead of", "alternative",
    "swap", "without", "allergy", "intolerant", "vegan",
    "dairy-free", "gluten-free", "no eggs",
]

# Cooking technique synonyms for query expansion
TECHNIQUE_SYNONYMS = {
    "fry": ["sauté", "pan-fry", "deep-fry", "stir-fry"],
    "bake": ["roast", "oven"],
    "boil": ["simmer", "poach", "blanch"],
    "grill": ["broil", "char-grill", "barbecue"],
}


class RecipeAgent(BaseAgent):
    """
    Recipe Assistant Specialist.

    Enhancements over base agent:
    - Detects substitution queries and adds context
    - Expands cooking technique terms for broader retrieval
    - Tags dietary information in responses
    """

    @property
    def domain(self) -> str:
        return "recipe"

    @property
    def description(self) -> str:
        return (
            "Recipe assistant for cooking guidance, ingredient substitutions, "
            "recipe search, nutritional information, and food preparation techniques."
        )

    def preprocess_query(self, query: str) -> str:
        """
        Recipe-specific query preprocessing:
        1. Detect substitution intent and enrich query
        2. Expand cooking technique terms
        """
        processed = query
        query_lower = query.lower()

        # Check for substitution queries
        is_substitution = any(kw in query_lower for kw in SUBSTITUTION_KEYWORDS)
        if is_substitution:
            # Add "substitution" explicitly for retrieval boost
            if "substitut" not in query_lower:
                processed += " substitution alternative"
            logger.debug("Detected substitution query")

        # Expand cooking techniques
        for technique, synonyms in TECHNIQUE_SYNONYMS.items():
            if technique in query_lower:
                expansion = " ".join(synonyms[:2])  # Add top 2 synonyms
                processed += f" {expansion}"
                logger.debug(f"Expanded technique '{technique}' with: {expansion}")
                break

        return processed

    def postprocess_response(self, result: CRAGResult) -> CRAGResult:
        """
        Add dietary tags if relevant dietary terms are detected
        in the response.
        """
        dietary_tags = {
            "vegan": ["vegan", "plant-based"],
            "vegetarian": ["vegetarian", "no meat"],
            "gluten-free": ["gluten-free", "celiac"],
            "dairy-free": ["dairy-free", "lactose"],
            "nut-free": ["nut-free", "nut allergy"],
        }

        response_lower = result.response.lower()
        detected_tags = [
            tag for tag, keywords in dietary_tags.items()
            if any(kw in response_lower for kw in keywords)
        ]

        if detected_tags:
            tags_str = " | ".join(f"🏷️ {tag}" for tag in detected_tags)
            result.response += f"\n\n{tags_str}"

        return result
