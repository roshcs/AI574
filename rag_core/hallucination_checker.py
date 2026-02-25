"""
Hallucination Checker
=====================
Validates that a generated response is grounded in its source documents.
Flags unsupported claims and provides a confidence score.

Usage:
    from rag_core.hallucination_checker import HallucinationChecker
    checker = HallucinationChecker(llm=llm)
    result = checker.check(response_text, source_docs)
    if result["grounded"]:
        # Safe to return to user
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import HALLUCINATION_CHECK_SYSTEM, HALLUCINATION_CHECK_USER
from config.settings import CONFIG

logger = logging.getLogger(__name__)


class HallucinationChecker:
    """Validates response grounding against source documents."""

    def __init__(self, llm):
        self.llm = llm
        self.confidence_threshold = CONFIG.rag.confidence_threshold

    def check(
        self,
        response: str,
        source_documents: List[Document],
    ) -> dict:
        """
        Check if the response is grounded in the source documents.

        Returns:
            {
                "grounded": bool,
                "confidence": float,
                "issues": list[str],
                "should_escalate": bool,
            }
        """
        sources_text = "\n\n---\n\n".join(
            f"[Source {i+1}]\n{doc.page_content[:1500]}"
            for i, doc in enumerate(source_documents)
        )

        user_content = HALLUCINATION_CHECK_USER.format(
            sources=sources_text,
            response=response[:3000],
        )

        try:
            result = self.llm.invoke_and_parse_json([
                SystemMessage(content=HALLUCINATION_CHECK_SYSTEM),
                HumanMessage(content=user_content),
            ])
            if "error" in result:
                logger.warning("Hallucination check parse failed")
                return self._default_result()

            grounded = result.get("grounded", False)
            confidence = float(result.get("confidence", 0.5))
            issues = result.get("issues", [])

            return {
                "grounded": grounded,
                "confidence": confidence,
                "issues": issues,
                "should_escalate": not grounded or confidence < self.confidence_threshold,
            }

        except Exception as e:
            logger.error(f"Hallucination check failed: {e}")
            return self._default_result()

    @staticmethod
    def _default_result() -> dict:
        """Fail-safe: assume not fully grounded when check fails."""
        return {
            "grounded": False,
            "confidence": 0.5,
            "issues": ["Hallucination check could not be completed"],
            "should_escalate": True,
        }
