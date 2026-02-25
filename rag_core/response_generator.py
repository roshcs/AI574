"""
Response Generator
==================
Produces grounded responses from query + relevant documents.
Injects domain-specific system prompts and formats source context.

Usage:
    from rag_core.response_generator import ResponseGenerator
    generator = ResponseGenerator(llm=llm)
    response = generator.generate(query, relevant_docs, domain="industrial")
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

import config.prompts as prompt_module
from config.settings import CONFIG

logger = logging.getLogger(__name__)


def _build_domain_prompts() -> dict:
    """Build domain→prompt map from the config registry, falling back to legacy names."""
    prompts = {}
    for spec in CONFIG.supervisor.domain_registry:
        template = getattr(prompt_module, spec.prompt_key, None)
        if template:
            prompts[spec.name] = template
        else:
            logger.warning("No prompt template '%s' for domain '%s'", spec.prompt_key, spec.name)
    return prompts


_DOMAIN_PROMPTS = _build_domain_prompts()


class ResponseGenerator:
    """Generates grounded responses using retrieved context."""

    def __init__(self, llm):
        self.llm = llm

    def generate(
        self,
        query: str,
        documents: List[Document],
        domain: str,
    ) -> dict:
        """
        Generate a response grounded in the provided documents.

        Args:
            query: User's original query.
            documents: Relevant documents from retrieval.
            domain: Target domain for prompt selection.

        Returns:
            {
                "response": str,
                "sources": [{"source": str, "chunk_index": int}, ...],
            }
        """
        # Format context from documents
        context = self._format_context(documents)
        sources = self._extract_sources(documents)

        # Select domain prompt
        prompt_template = _DOMAIN_PROMPTS.get(domain)
        if not prompt_template:
            logger.warning(f"No prompt for domain '{domain}', using generic")
            prompt_template = (
                "Answer the user's query using ONLY the provided context.\n\n"
                "Retrieved Context:\n{context}\n\nUser Query: {query}"
            )

        prompt = prompt_template.format(context=context, query=query)

        try:
            result = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = result.content.strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            response_text = (
                "I encountered an error generating a response. "
                "Please try rephrasing your question."
            )

        return {
            "response": response_text,
            "sources": sources,
            "domain": domain,
            "num_context_docs": len(documents),
        }

    @staticmethod
    def _format_context(
        documents: List[Document],
        max_total_chars: int = 12_000,
    ) -> str:
        """Format documents into a numbered context string, truncating to budget."""
        parts = []
        total = 0
        per_doc_budget = max(500, max_total_chars // max(len(documents), 1))
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content[:per_doc_budget]
            part = f"[Document {i} | Source: {source}]\n{content}"
            total += len(part)
            parts.append(part)
            if total >= max_total_chars:
                logger.info(f"Context budget reached at document {i}/{len(documents)}")
                break
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _extract_sources(documents: List[Document]) -> list:
        """Extract source attribution from document metadata."""
        sources = []
        for doc in documents:
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "parent_id": doc.metadata.get("parent_id", ""),
                "similarity_score": doc.metadata.get("similarity_score", 0.0),
            })
        return sources
