"""Provider routing for phase-2 runners.

anthropic/* models go direct to the Anthropic API when ANTHROPIC_API_KEY is
set (research credits, $0 marginal cost to the study budget); everything else
goes through OpenRouter. Both paths return the same record shape so runners
and analysis code never care which provider served a call.

Anthropic specifics handled here:
- Fable 5 / Opus 4.8 use ADAPTIVE thinking (``{"type": "adaptive"}``);
  the old ``enabled``/``budget_tokens`` form returns a 400 on these models.
  Effort is left at the API default (``high``) for parity with the scout's
  OpenRouter calls.
- content arrives as blocks; the answer is in ``text`` blocks, not block 0
  (block 0 is usually ``thinking``)
- ``stop_reason: "refusal"`` is Fable's native refusal signal. NO fallback
  models are configured on purpose: refusals are a measured outcome in this
  study, and a silent fallback would contaminate the data.
- temperature must be omitted (sampling params rejected on these models)
"""

import asyncio
from typing import Optional

import httpx

from backend.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_API_URL,
    ANTHROPIC_MODEL_IDS,
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
)
from backend.openrouter import NO_SAMPLING_PARAM_MODELS

MAX_TOKENS = 24000
# Adaptive thinking on the direct API can spend more tokens than OpenRouter's
# default reasoning did in the scout (smoke: Opus hit 24k inside thinking and
# never emitted text). Tokens on the research key are free, so give direct
# Anthropic calls more headroom to avoid no_answer truncations.
ANTHROPIC_MAX_TOKENS = 48000
TIMEOUT_S = 600.0
RETRIES = 5


def is_anthropic_direct(model: str) -> bool:
    """True when this model will be served by the Anthropic API directly."""
    return bool(ANTHROPIC_API_KEY) and model in ANTHROPIC_MODEL_IDS


def _empty_record(error: Optional[str], provider: str) -> dict:
    return {
        "content": "",
        "finish_reason": None,
        "native_finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "cost": None,
        "provider": provider,
        "error": error,
    }


async def _query_anthropic(
    client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int, timeout_s: float
) -> dict:
    payload = {
        "model": ANTHROPIC_MODEL_IDS[model],
        "max_tokens": max_tokens,
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": prompt}],
    }
    last_err = None
    for attempt in range(RETRIES):
        try:
            resp = await client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=httpx.Timeout(connect=30.0, read=timeout_s, write=30.0, pool=120.0),
            )
            if resp.status_code in (429, 500, 502, 503, 529):
                retry_after = float(resp.headers.get("retry-after") or 0)
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                await asyncio.sleep(max(retry_after, min(2 * 2**attempt, 60)))
                continue
            resp.raise_for_status()
            data = resp.json()
            text = "".join(
                b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
            )
            usage = data.get("usage", {}) or {}
            stop_reason = data.get("stop_reason")
            return {
                "content": text,
                "finish_reason": stop_reason,
                "native_finish_reason": stop_reason,
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
                # Anthropic folds thinking tokens into output_tokens
                "reasoning_tokens": None,
                # Research credits: $0 against the study's OpenRouter budget
                "cost": 0.0,
                "provider": "anthropic-direct",
                "error": None,
            }
        # MemoryError is retryable here: the host box runs under memory
        # pressure (home-dir git scans ballooning to GBs) and allocation
        # failures inside a single HTTP read shouldn't kill the whole run.
        except (
            httpx.TransportError,
            httpx.HTTPStatusError,
            RuntimeError,
            KeyError,
            ValueError,
            MemoryError,
        ) as e:
            last_err = e
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 400:
                break  # non-retryable
            await asyncio.sleep(min(2 * 2**attempt, 60))
    return _empty_record(str(last_err), "anthropic-direct")


async def _query_openrouter(
    client: httpx.AsyncClient, model: str, prompt: str, max_tokens: int, timeout_s: float
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if not any(m in model for m in NO_SAMPLING_PARAM_MODELS):
        payload["temperature"] = 0.0

    last_err = None
    for attempt in range(RETRIES):
        try:
            resp = await client.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=httpx.Timeout(connect=30.0, read=timeout_s, write=30.0, pool=120.0),
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"API error: {data['error']}")
            choice = data["choices"][0]
            usage = data.get("usage", {}) or {}
            return {
                "content": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "native_finish_reason": choice.get("native_finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                ),
                "cost": usage.get("cost"),
                "provider": "openrouter",
                "error": None,
            }
        # MemoryError is retryable here: the host box runs under memory
        # pressure (home-dir git scans ballooning to GBs) and allocation
        # failures inside a single HTTP read shouldn't kill the whole run.
        except (
            httpx.TransportError,
            httpx.HTTPStatusError,
            RuntimeError,
            KeyError,
            ValueError,
            MemoryError,
        ) as e:
            last_err = e
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 400:
                break  # non-retryable
            await asyncio.sleep(min(2 * 2**attempt, 45))
    return _empty_record(str(last_err), "openrouter")


async def query_model_routed(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_tokens: int = MAX_TOKENS,
    timeout_s: float = TIMEOUT_S,
) -> dict:
    """One call, routed by provider; same record shape either way."""
    if is_anthropic_direct(model):
        return await _query_anthropic(
            client, model, prompt, max(max_tokens, ANTHROPIC_MAX_TOKENS), timeout_s
        )
    return await _query_openrouter(client, model, prompt, max_tokens, timeout_s)
