"""
Query Rewriter
==============
Rewrites queries that failed to retrieve relevant documents.
Uses the LLM to add domain-specific terms, expand abbreviations,
or decompose compound questions.

Usage:
    from rag_core.query_rewriter import QueryRewriter
    rewriter = QueryRewriter(llm=llm)
    new_query = rewriter.rewrite(query, domain="industrial", failure_context="...")
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from config.prompts import REWRITER_PROMPT

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrites queries to improve retrieval on retry."""

    def __init__(self, llm):
        self.llm = llm

    def rewrite(
        self,
        query: str,
        domain: str = "general",
        failure_context: str = "No relevant documents found.",
    ) -> str:
        """
        Rewrite a query for better retrieval.

        Args:
            query: The original (or previously rewritten) query.
            domain: Target domain for terminology hints.
            failure_context: Why the previous retrieval failed.

        Returns:
            The rewritten query string.
        """
        prompt = REWRITER_PROMPT.format(
            query=query,
            domain=domain,
            failure_context=failure_context,
        )

        try:
            result = self.llm.invoke([HumanMessage(content=prompt)])
            rewritten = result.content.strip()
            # Sanity check — don't return empty or overly long rewrites
            if not rewritten or len(rewritten) > len(query) * 5:
                logger.warning("Rewrite was empty or too long, keeping original")
                return query
            logger.info(f"Query rewritten: '{query[:60]}' → '{rewritten[:60]}'")
            return rewritten
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return query
