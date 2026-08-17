from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping


MILLION_TOKENS = 1_000_000
LONG_CONTEXT_THRESHOLD = 272_000


@dataclass(frozen=True)
class TokenRates:
    input: float
    cached_input: float | None
    output: float
    cache_write: float | None = None


@dataclass(frozen=True)
class ModelPricing:
    standard: TokenRates
    long_context: TokenRates | None = None
    long_context_threshold: int = LONG_CONTEXT_THRESHOLD


# Standard processing rates in USD per million text tokens. Keep this catalog in
# one place so model aliases, snapshots, and pricing updates do not leak into the
# runtime. Source: https://developers.openai.com/api/docs/pricing
OPENAI_MODEL_PRICING: Mapping[str, ModelPricing] = {
    "gpt-5.6-sol": ModelPricing(
        TokenRates(2.50, 0.25, 15.00, 3.125),
        TokenRates(5.00, 0.50, 22.50, 6.25),
    ),
    "gpt-5.6-terra": ModelPricing(
        TokenRates(1.00, 0.10, 6.00, 1.25),
        TokenRates(2.00, 0.20, 9.00, 2.50),
    ),
    "gpt-5.6-luna": ModelPricing(
        TokenRates(0.10, 0.01, 0.60, 0.125),
        TokenRates(0.20, 0.02, 0.90, 0.25),
    ),
    "gpt-5.5-pro": ModelPricing(
        TokenRates(30.00, None, 180.00),
    ),
    "gpt-5.5": ModelPricing(
        TokenRates(5.00, 0.50, 30.00),
        TokenRates(10.00, 1.00, 45.00),
    ),
    "gpt-5.4-pro": ModelPricing(
        TokenRates(30.00, None, 180.00),
        TokenRates(60.00, None, 270.00),
    ),
    "gpt-5.4-mini": ModelPricing(TokenRates(0.75, 0.075, 4.50)),
    "gpt-5.4-nano": ModelPricing(TokenRates(0.20, 0.02, 1.25)),
    "gpt-5.4": ModelPricing(
        TokenRates(2.50, 0.25, 15.00),
        TokenRates(5.00, 0.50, 22.50),
    ),
    "gpt-4.1-mini": ModelPricing(TokenRates(0.40, 0.10, 1.60)),
    "gpt-4.1-nano": ModelPricing(TokenRates(0.10, 0.025, 0.40)),
    "gpt-4.1": ModelPricing(TokenRates(2.00, 0.50, 8.00)),
    "gpt-4o-mini": ModelPricing(TokenRates(0.15, 0.075, 0.60)),
    "gpt-4o": ModelPricing(TokenRates(2.50, 1.25, 10.00)),
    "o4-mini": ModelPricing(TokenRates(1.10, 0.275, 4.40)),
    "o3-mini": ModelPricing(TokenRates(1.10, 0.55, 4.40)),
    "o3": ModelPricing(TokenRates(2.00, 0.50, 8.00)),
}


def calculate_token_cost_usd(
    *,
    provider: str | None,
    model: str | None,
    usage: Mapping[str, int],
) -> float:
    pricing = _pricing_for(provider, model)
    rates = _rates_with_overrides(pricing, usage)
    if rates is None:
        return 0.0

    input_tokens = _token_count(usage, "input")
    cached_tokens = min(input_tokens, _token_count(usage, "cache_read"))
    cache_write_tokens = min(
        input_tokens - cached_tokens,
        _token_count(usage, "cache_write"),
    )
    uncached_tokens = input_tokens - cached_tokens - cache_write_tokens
    output_tokens = _token_count(usage, "output")

    cached_rate = rates.cached_input if rates.cached_input is not None else rates.input
    cache_write_rate = rates.cache_write if rates.cache_write is not None else rates.input
    return (
        uncached_tokens * rates.input
        + cached_tokens * cached_rate
        + cache_write_tokens * cache_write_rate
        + output_tokens * rates.output
    ) / MILLION_TOKENS


def _pricing_for(provider: str | None, model: str | None) -> ModelPricing | None:
    if (provider or "").strip().casefold() != "openai" or not model:
        return None
    normalized = model.strip().casefold()
    for model_id in sorted(OPENAI_MODEL_PRICING, key=len, reverse=True):
        if normalized == model_id or normalized.startswith(f"{model_id}-"):
            return OPENAI_MODEL_PRICING[model_id]
    return None


def _rates_with_overrides(
    pricing: ModelPricing | None,
    usage: Mapping[str, int],
) -> TokenRates | None:
    rates = None
    if pricing is not None:
        rates = pricing.standard
        if (
            pricing.long_context is not None
            and _token_count(usage, "input") > pricing.long_context_threshold
        ):
            rates = pricing.long_context

    input_override = _nonnegative_float_env("AGENT_INPUT_COST_USD_PER_MILLION_TOKENS")
    cached_override = _nonnegative_float_env("AGENT_CACHED_INPUT_COST_USD_PER_MILLION_TOKENS")
    cache_write_override = _nonnegative_float_env("AGENT_CACHE_WRITE_COST_USD_PER_MILLION_TOKENS")
    output_override = _nonnegative_float_env("AGENT_OUTPUT_COST_USD_PER_MILLION_TOKENS")
    if rates is None and all(
        value is None
        for value in (input_override, cached_override, cache_write_override, output_override)
    ):
        return None

    baseline = rates or TokenRates(0.0, None, 0.0)
    return TokenRates(
        input=baseline.input if input_override is None else input_override,
        cached_input=baseline.cached_input if cached_override is None else cached_override,
        cache_write=baseline.cache_write if cache_write_override is None else cache_write_override,
        output=baseline.output if output_override is None else output_override,
    )


def _token_count(usage: Mapping[str, int], name: str) -> int:
    value = usage.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _nonnegative_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value
