import unittest
from unittest.mock import patch

from agent.pricing import calculate_token_cost_usd


class TokenPricingTests(unittest.TestCase):
    def test_openai_pricing_discounts_cached_input(self) -> None:
        cost = calculate_token_cost_usd(
            provider="openai",
            model="gpt-5.4-mini",
            usage={"input": 1_000_000, "cache_read": 250_000, "output": 100_000},
        )

        self.assertAlmostEqual(cost, 1.03125)

    def test_snapshot_uses_base_model_pricing(self) -> None:
        cost = calculate_token_cost_usd(
            provider="openai",
            model="gpt-5.5-2026-04-23",
            usage={"input": 1_000, "output": 100},
        )

        self.assertAlmostEqual(cost, 0.008)

    def test_long_context_rates_apply_per_request(self) -> None:
        cost = calculate_token_cost_usd(
            provider="openai",
            model="gpt-5.5",
            usage={"input": 300_000, "output": 10_000},
        )

        self.assertAlmostEqual(cost, 3.45)

    def test_environment_rates_support_unlisted_providers(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "AGENT_INPUT_COST_USD_PER_MILLION_TOKENS": "2",
                "AGENT_CACHED_INPUT_COST_USD_PER_MILLION_TOKENS": "0.5",
                "AGENT_OUTPUT_COST_USD_PER_MILLION_TOKENS": "8",
            },
            clear=True,
        ):
            cost = calculate_token_cost_usd(
                provider="anthropic",
                model="custom-model",
                usage={"input": 1_000_000, "cache_read": 200_000, "output": 100_000},
            )

        self.assertAlmostEqual(cost, 2.5)

    def test_unknown_model_without_overrides_has_zero_estimated_cost(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            cost = calculate_token_cost_usd(
                provider="openai",
                model="unlisted-model",
                usage={"input": 1_000_000, "output": 1_000_000},
            )

        self.assertEqual(cost, 0.0)

    def test_invalid_environment_rates_are_ignored(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "AGENT_INPUT_COST_USD_PER_MILLION_TOKENS": "nan",
                "AGENT_OUTPUT_COST_USD_PER_MILLION_TOKENS": "-1",
            },
            clear=True,
        ):
            cost = calculate_token_cost_usd(
                provider="openai",
                model="unlisted-model",
                usage={"input": 1_000_000, "output": 1_000_000},
            )

        self.assertEqual(cost, 0.0)


if __name__ == "__main__":
    unittest.main()
