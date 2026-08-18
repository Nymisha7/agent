import unittest
from unittest.mock import patch

from agent.pricing import calculate_token_cost, calculate_token_cost_usd


class TokenPricingTests(unittest.TestCase):
    def test_openai_pricing_discounts_cached_input(self) -> None:
        cost = calculate_token_cost(
            provider="openai",
            model="gpt-5.4-mini",
            usage={"input": 1_000_000, "cache_read": 250_000, "output": 100_000},
        )

        self.assertAlmostEqual(cost.input, 0.5625)
        self.assertAlmostEqual(cost.cached_input, 0.01875)
        self.assertAlmostEqual(cost.output, 0.45)
        self.assertAlmostEqual(cost.total, 1.03125)

    def test_scalar_cost_api_remains_compatible(self) -> None:
        cost = calculate_token_cost_usd(
            provider="openai",
            model="gpt-5.4-mini",
            usage={"input": 1_000, "output": 100},
        )

        self.assertAlmostEqual(cost, 0.0012)

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

    def test_gpt_5_6_family_uses_current_model_tier_rates(self) -> None:
        expected = {
            "gpt-5.6": 3.5,
            "gpt-5.6-sol": 3.5,
            "gpt-5.6-terra": 1.75,
            "gpt-5.6-luna": 0.7,
        }

        for model, total in expected.items():
            with self.subTest(model=model):
                cost = calculate_token_cost_usd(
                    provider="openai",
                    model=model,
                    usage={"input": 100_000, "output": 100_000},
                )
                self.assertAlmostEqual(cost, total)

    def test_gpt_5_6_cache_write_uses_documented_multiplier(self) -> None:
        cost = calculate_token_cost(
            provider="openai",
            model="gpt-5.6-terra",
            usage={"input": 100_000, "cache_write": 100_000},
        )

        self.assertAlmostEqual(cost.cache_write, 0.3125)
        self.assertAlmostEqual(cost.total, 0.3125)

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
