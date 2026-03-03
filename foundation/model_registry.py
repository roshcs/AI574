"""
Model Registry
==============
Maps friendly model IDs to LangChain-compatible chat model constructors.
Provides ``get_llm()`` for runtime model selection and a thin
``HostedChatModelAdapter`` that adds ``invoke_and_parse_json()`` to any
hosted LangChain chat model (Gemini, Groq, etc.).

Usage:
    from foundation.model_registry import get_llm, list_models
    llm = get_llm("gemini_flash")
    result = llm.invoke_and_parse_json([HumanMessage(content="...")])
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON parse helper (shared with KerasHubChatModel)
# ---------------------------------------------------------------------------

def _parse_llm_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from raw LLM output."""
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{") or cleaned.startswith("["):
                text = cleaned
                break

    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("JSON parse failed, raw output: %s", text[:200])
    return {"error": "parse_failed", "raw": text}


# ---------------------------------------------------------------------------
# Adapter: gives any BaseChatModel the invoke_and_parse_json() interface
# ---------------------------------------------------------------------------

class HostedChatModelAdapter:
    """Wraps a LangChain ``BaseChatModel`` and adds ``invoke_and_parse_json``.

    This makes hosted models (Gemini, Groq, …) compatible with the
    supervisor, document grader, and hallucination checker which all
    call ``llm.invoke_and_parse_json(messages)``.

    Automatically translates ``max_new_tokens`` to the provider-specific
    parameter name so callers don't need to care about provider differences.
    """

    _TOKEN_KWARG_MAP: Dict[str, str] = {
        "chat-google-generative-ai": "max_output_tokens",
        "groq-chat": "max_tokens",
    }

    def __init__(self, model: BaseChatModel):
        self._model = model

    def _translate_kwargs(self, kwargs: dict) -> dict:
        """Map generic ``max_new_tokens`` to the provider's expected key."""
        if "max_new_tokens" not in kwargs:
            return kwargs
        budget = kwargs.pop("max_new_tokens")
        provider_type = getattr(self._model, "_llm_type", "")
        target_key = self._TOKEN_KWARG_MAP.get(provider_type, "max_tokens")
        kwargs[target_key] = budget
        return kwargs

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def invoke(self, messages, **kwargs):
        kwargs = self._translate_kwargs(kwargs)
        return self._model.invoke(messages, **kwargs)

    def invoke_and_parse_json(
        self, messages: List[BaseMessage], **kwargs
    ) -> dict:
        kwargs = self._translate_kwargs(kwargs)
        result = self._model.invoke(messages, **kwargs)
        return _parse_llm_json(result.content.strip())

    @property
    def _llm_type(self) -> str:
        return getattr(self._model, "_llm_type", "hosted-adapter")


# ---------------------------------------------------------------------------
# Registry: model_id → factory callable
# ---------------------------------------------------------------------------

def _make_gemma3(**kwargs) -> Any:
    from foundation.llm_wrapper import KerasHubChatModel
    from config.settings import CONFIG
    return KerasHubChatModel(config=CONFIG.llm)


def _make_gemini_flash(**kwargs) -> HostedChatModelAdapter:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai is required for Gemini models. "
            "Install with: pip install langchain-google-genai"
        )
    api_key = kwargs.get("api_key") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY env var (or api_key kwarg) is required for Gemini."
        )
    model = ChatGoogleGenerativeAI(
        model=kwargs.get("model_name", "gemini-2.5-flash"),
        google_api_key=api_key,
        temperature=kwargs.get("temperature", 0.7),
    )
    return HostedChatModelAdapter(model)


def _make_gemini_pro(**kwargs) -> HostedChatModelAdapter:
    return _make_gemini_flash(model_name="gemini-2.5-pro", **kwargs)


def _make_groq_llama(**kwargs) -> HostedChatModelAdapter:
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError(
            "langchain-groq is required for Groq models. "
            "Install with: pip install langchain-groq"
        )
    api_key = kwargs.get("api_key") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY env var (or api_key kwarg) is required for Groq."
        )
    model = ChatGroq(
        model=kwargs.get("model_name", "llama-3.1-8b-instant"),
        api_key=api_key,
        temperature=kwargs.get("temperature", 0.7),
    )
    return HostedChatModelAdapter(model)


MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "gemma3": _make_gemma3,
    "gemini_flash": _make_gemini_flash,
    "gemini_pro": _make_gemini_pro,
    "groq_llama": _make_groq_llama,
}


def get_llm(model_id: str, **kwargs) -> Any:
    """Resolve a model_id to a LangChain-compatible chat model.

    Args:
        model_id: Friendly identifier (see ``list_models()``).
        **kwargs: Forwarded to the model factory (e.g. ``api_key``).

    Returns:
        A chat model with ``invoke()`` and ``invoke_and_parse_json()``.
    """
    factory = MODEL_REGISTRY.get(model_id)
    if factory is None:
        raise ValueError(
            f"Unknown model_id '{model_id}'. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    logger.info("Creating LLM for model_id='%s'", model_id)
    return factory(**kwargs)


def list_models() -> List[str]:
    """Return all registered model IDs."""
    return sorted(MODEL_REGISTRY)


def register_model(model_id: str, factory: Callable[..., Any]) -> None:
    """Add a custom model factory at runtime."""
    MODEL_REGISTRY[model_id] = factory
    logger.info("Registered custom model_id='%s'", model_id)
