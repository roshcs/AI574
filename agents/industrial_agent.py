"""
Industrial Troubleshooting Agent
================================
Specialized for PLC/SCADA diagnostics, fault code interpretation,
and maintenance procedures. Adds industrial-specific query preprocessing
and safety warnings to responses.
"""

from __future__ import annotations

import logging
import re

from agents.base_agent import BaseAgent
from rag_core.crag_pipeline import CRAGPipeline, CRAGResult

logger = logging.getLogger(__name__)

# Known fault code patterns by vendor
FAULT_CODE_PATTERNS = {
    "siemens": re.compile(r'\b[Ff]\d{4,5}\b'),           # F0003, F07900
    "allen_bradley": re.compile(r'\bF\d{1,3}\b'),         # F2, F33
    "generic": re.compile(r'\b(?:fault|error|alarm)\s*(?:code\s*)?[:#]?\s*\d+', re.I),
}

# Equipment model patterns
EQUIPMENT_PATTERNS = re.compile(
    r'\b(S7-\d{3,4}|1[257]\d{2}|6ES\d|PanelView|PowerFlex\s*\d+)\b', re.I
)


class IndustrialAgent(BaseAgent):
    """
    Industrial Equipment Troubleshooting Specialist.

    Enhancements over base agent:
    - Extracts and normalizes fault codes from queries
    - Expands industrial abbreviations
    - Adds safety warnings to responses involving electrical/mechanical work
    """

    @property
    def domain(self) -> str:
        return "industrial"

    @property
    def description(self) -> str:
        return (
            "Industrial equipment troubleshooting specialist. Handles PLC/SCADA "
            "diagnostics, fault code interpretation, motor drives, industrial "
            "networking, and maintenance procedures."
        )

    def preprocess_query(self, query: str) -> str:
        """
        Industrial-specific query preprocessing:
        1. Extract and normalize fault codes
        2. Identify equipment models
        3. Expand common abbreviations for better retrieval
        """
        processed = query

        # Extract fault codes and make them explicit
        for vendor, pattern in FAULT_CODE_PATTERNS.items():
            matches = pattern.findall(processed)
            if matches:
                # Ensure fault code is prominent for retrieval
                codes = ", ".join(matches)
                if "fault" not in processed.lower():
                    processed += f" (fault code: {codes})"
                logger.debug(f"Extracted fault codes ({vendor}): {codes}")

        # Extract equipment models
        equip_matches = EQUIPMENT_PATTERNS.findall(processed)
        if equip_matches:
            logger.debug(f"Detected equipment: {equip_matches}")

        return processed

    def postprocess_response(self, result: CRAGResult) -> CRAGResult:
        """
        Add safety warnings for responses involving potentially
        dangerous procedures (electrical, mechanical, pressurized systems).
        """
        safety_keywords = [
            "high voltage", "lockout", "tagout", "loto",
            "energized", "live wire", "arc flash",
            "pressurized", "hydraulic", "pneumatic",
            "rotating", "pinch point", "confined space",
        ]

        response_lower = result.response.lower()
        triggered_warnings = [
            kw for kw in safety_keywords if kw in response_lower
        ]

        if triggered_warnings:
            safety_notice = (
                "\n\n⚠️ **SAFETY WARNING**: This procedure may involve "
                "hazardous conditions. Ensure all applicable safety protocols "
                "(lockout/tagout, PPE, permits) are followed. Consult your "
                "site safety officer before proceeding."
            )
            result.response += safety_notice
            logger.info(f"Safety warning added (triggers: {triggered_warnings})")

        return result
