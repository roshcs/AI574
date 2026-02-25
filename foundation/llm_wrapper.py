"""
KerasHub ↔ LangChain Bridge
============================
Custom wrapper that makes KerasHub models (Gemma 3, Gemma 2, Llama 3.1)
compatible with LangChain's BaseChatModel interface, enabling use inside
LangGraph agents.

This is one of the novel contributions of the project — no existing library
provides this integration.

Usage:
    from foundation.llm_wrapper import KerasHubChatModel
    llm = KerasHubChatModel(config=CONFIG.llm)
    response = llm.invoke([HumanMessage(content="Hello")])
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

import sys
sys.path.append("..")
from config.settings import LLMConfig, CONFIG

logger = logging.getLogger(__name__)


# ── Prompt Formatters ─────────────────────────────────────────────────────────
# Each model family has its own control token format.

def _format_gemma(messages: List[BaseMessage]) -> str:
    """Format messages using Gemma 2 control tokens."""
    parts = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            # Gemma 2 doesn't have a system role — prepend to first user turn
            parts.append(f"<start_of_turn>user\n[System Instructions]\n{msg.content}\n")
        elif isinstance(msg, HumanMessage):
            # Avoid double user tag if system was prepended
            if parts and parts[-1].startswith("<start_of_turn>user\n[System"):
                parts[-1] += f"\n{msg.content}<end_of_turn>\n"
            else:
                parts.append(f"<start_of_turn>user\n{msg.content}<end_of_turn>\n")
        elif isinstance(msg, AIMessage):
            parts.append(f"<start_of_turn>model\n{msg.content}<end_of_turn>\n")
    # Open the model turn for generation
    parts.append("<start_of_turn>model\n")
    return "".join(parts)


def _format_llama(messages: List[BaseMessage]) -> str:
    """Format messages using Llama 3.1 control tokens."""
    parts = ["<|begin_of_text|>"]
    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{msg.content}<|eot_id|>"
            )
        elif isinstance(msg, HumanMessage):
            parts.append(
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{msg.content}<|eot_id|>"
            )
        elif isinstance(msg, AIMessage):
            parts.append(
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{msg.content}<|eot_id|>"
            )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


# Map preset names to formatters
_FORMATTERS = {
    "gemma": _format_gemma,
    "llama": _format_llama,
}

def _get_formatter(preset: str):
    """Select the right prompt formatter based on model preset name."""
    preset_lower = preset.lower()
    for key, fn in _FORMATTERS.items():
        if key in preset_lower:
            return fn
    raise ValueError(
        f"No prompt formatter registered for preset '{preset}'. "
        f"Supported families: {list(_FORMATTERS.keys())}"
    )


# ── KerasHub Chat Model ──────────────────────────────────────────────────────

class KerasHubChatModel(BaseChatModel):
    """
    LangChain-compatible chat model backed by a KerasHub CausalLM.

    Handles:
    1. Prompt Construction — converts LangChain messages to model-specific
       control token format.
    2. Inference — triggers KerasHub generate() with configured sampling.
    3. Output Parsing — strips the prompt echo, extracts generated text.
    """

    # Pydantic fields (LangChain requires these to be class-level)
    config: LLMConfig = None
    model: Any = None           # keras_hub.models.CausalLM (set at init)
    _formatter: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or CONFIG.llm
        self._formatter = _get_formatter(self.config.preset)
        self._load_model()

    @staticmethod
    def _load_preset(keras_hub, preset: str, dtype: str):
        """Load a KerasHub model, using Gemma3CausalLM for gemma3 presets."""
        if "gemma3" in preset.lower():
            return keras_hub.models.Gemma3CausalLM.from_preset(preset, dtype=dtype)
        return keras_hub.models.CausalLM.from_preset(preset, dtype=dtype)

    def _load_model(self):
        """Load the KerasHub model with configured precision."""
        try:
            import os
            os.environ["KERAS_BACKEND"] = self.config.backend

            import keras_hub

            logger.info(f"Loading KerasHub model: {self.config.preset} "
                        f"(dtype={self.config.dtype})")
            self.model = self._load_preset(
                keras_hub, self.config.preset, self.config.dtype
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load primary model: {e}")
            if self.config.fallback_preset:
                logger.info(f"Attempting fallback: {self.config.fallback_preset}")
                self._formatter = _get_formatter(self.config.fallback_preset)
                import keras_hub
                self.model = self._load_preset(
                    keras_hub, self.config.fallback_preset, self.config.dtype
                )
            else:
                raise

    # ── LangChain Interface ───────────────────────────────────────────────

    @property
    def _llm_type(self) -> str:
        return "kerashub-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Core generation method called by LangChain."""

        # 1. Format prompt
        prompt = self._formatter(messages)

        # Allow callers to override token budget (e.g. short tasks)
        token_budget = kwargs.get("max_new_tokens", self.config.max_new_tokens)

        # 2. Run inference (with optional timing for throughput diagnosis)
        t0 = time.perf_counter()
        raw_output = self.model.generate(
            prompt,
            max_length=len(prompt) + token_budget,
        )
        elapsed = time.perf_counter() - t0

        # 3. Strip prompt echo — KerasHub returns prompt + generation
        generated_text = raw_output
        if isinstance(raw_output, str) and raw_output.startswith(prompt):
            generated_text = raw_output[len(prompt):]
        elif isinstance(raw_output, str):
            # Fallback for Gemma 3: exact startswith may fail due to
            # whitespace/token differences.  Find the last model-turn
            # marker and take everything after it.
            marker = "<start_of_turn>model\n"
            last_model = raw_output.rfind(marker)
            if last_model != -1:
                generated_text = raw_output[last_model + len(marker):]

        # 4. Clean up trailing control tokens
        for token in ["<end_of_turn>", "<|eot_id|>", "<|end_of_text|>"]:
            generated_text = generated_text.split(token)[0]

        generated_text = generated_text.strip()

        # 5. Log throughput (approx: ~4 chars/token for English)
        est_tokens = max(1, len(generated_text) // 4)
        tps = est_tokens / elapsed if elapsed > 0 else 0
        logger.info(
            f"LLM call: {elapsed:.1f}s | ~{est_tokens} output tokens | "
            f"~{tps:.1f} tok/s"
        )

        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=generated_text))
            ]
        )

    # ── Convenience ───────────────────────────────────────────────────────

    def invoke_and_parse_json(
        self, messages: List[BaseMessage], **kwargs
    ) -> dict:
        """Generate and parse a JSON response. Useful for grader/router.

        Uses the short token budget by default since structured JSON
        outputs (routing, grading, validation) are compact.
        """
        kwargs.setdefault("max_new_tokens", self.config.short_max_new_tokens)
        result = self.invoke(messages, **kwargs)
        text = result.content.strip()

        # Strip markdown code fences (handles ```json ... ``` and ``` ... ```)
        if "```" in text:
            parts = text.split("```")
            for part in parts[1::2]:  # odd-indexed parts are inside fences
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                if cleaned.startswith("{") or cleaned.startswith("["):
                    text = cleaned
                    break

        text = text.strip()

        # Try direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                return parsed[0]
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: extract first { ... } substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("JSON parse failed, raw output: %s", text[:200])
        return {"error": "parse_failed", "raw": text}
