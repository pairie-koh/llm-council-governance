"""Base classes for governance structures."""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from backend.governance.utils import (
    build_stage1_prompt,
    extract_final_answer,
    extract_final_answer_with_fallback,
)
from backend.openrouter import query_model, query_models_parallel

logger = logging.getLogger(__name__)

# Share Stage 1 responses across governance structures (same question, same
# models, temperature 0.0). All multi-model structures run the identical
# Stage 1, so re-querying per structure multiplies cost ~6x for no
# information gain - and sharing makes structure comparisons strictly fairer
# (every structure judges the same Stage 1 answers). Self-consistency is
# unaffected (it samples via its own path at temperature 0.7).
# Disable with STAGE1_SHARED_CACHE=false in .env.
STAGE1_SHARED_CACHE = os.getenv("STAGE1_SHARED_CACHE", "true").lower() == "true"

# Per-prompt locks so concurrent trials of the same question don't all fire
# Stage 1 on a cache miss; only the first does, the rest wait then read.
_stage1_locks: Dict[str, asyncio.Lock] = {}


@dataclass
class CouncilResult:
    """Result from running a governance structure."""

    final_answer: str
    stage1_responses: Dict[str, str]  # model -> response
    stage2_data: Optional[Any] = None  # Structure-specific
    stage3_data: Optional[Any] = None  # Structure-specific
    metadata: Optional[Dict[str, Any]] = None  # Timings, token counts, etc.


class GovernanceStructure(ABC):
    """Abstract base class for governance structures."""

    def __init__(self, council_models: List[str], chairman_model: str):
        self.council_models = council_models
        self.chairman_model = chairman_model

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this structure."""
        pass

    @abstractmethod
    async def run(self, query: str) -> CouncilResult:
        """Execute the governance process and return result."""
        pass

    async def _stage1_collect_responses(self, query: str) -> Dict[str, str]:
        """
        Stage 1: Query all council models in parallel with standardized prompt.

        This is the common first stage for all governance structures.

        When STAGE1_SHARED_CACHE is enabled (default), responses are cached
        on disk keyed by (model, prompt, temperature=0.0) and shared across
        structures, so each question pays for Stage 1 once instead of once
        per structure.

        Args:
            query: The question/prompt to ask the council

        Returns:
            Dictionary mapping model names to their responses
        """
        prompt = build_stage1_prompt(query)
        messages = [{"role": "user", "content": prompt}]

        if not STAGE1_SHARED_CACHE:
            results = await query_models_parallel(self.council_models, messages)
            return {
                model: result.get("content", result.get("error", ""))
                for model, result in results.items()
            }

        from backend.governance.stage1_cache import get_cache

        cache = get_cache()
        # The prompt embeds the full question text, so it uniquely keys the
        # question; benchmark/question_id are not needed for correctness.
        lock = _stage1_locks.setdefault(prompt, asyncio.Lock())
        async with lock:
            responses: Dict[str, str] = {}
            missing = []
            for model in self.council_models:
                cached = cache.get("", "", model, prompt)
                if cached is not None:
                    responses[model] = cached
                else:
                    missing.append(model)

            if missing:
                results = await query_models_parallel(missing, messages)
                for model, result in results.items():
                    content = result.get("content")
                    if content:
                        # Only cache successful responses; errors stay
                        # uncached so a retry can succeed.
                        cache.set("", "", model, prompt, content)
                        responses[model] = content
                    else:
                        responses[model] = result.get("error", "")

            return responses

    async def _get_chairman_answer(self, query: str) -> str:
        """
        Get chairman's answer for tiebreaker.

        This is used by voting structures (B, C, E) to break ties.

        Args:
            query: The question/prompt to ask the chairman

        Returns:
            The extracted final answer from the chairman's response
        """
        prompt = build_stage1_prompt(query)
        messages = [{"role": "user", "content": prompt}]
        result = await query_model(self.chairman_model, messages)

        content = result.get("content", "")
        answer = extract_final_answer(content)

        if answer is None:
            logger.warning(
                "Chairman tiebreaker extraction failed for model %s, using fallback",
                self.chairman_model,
            )
            return extract_final_answer_with_fallback(content)

        return answer
