import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.config import ExperimentConfig, InterventionConfig
from moe_attention_guided_steering.datasets import load_experiment
from moe_attention_guided_steering.pipeline import run_pipeline


class PipelineTestCase(unittest.TestCase):
    """Exercise the toy pipeline end to end so refactors stay honest."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_experiment(str(ROOT / "examples" / "toy_experiment.json"))

    def test_pipeline_selects_expected_experts(self) -> None:
        config = ExperimentConfig(
            intervention=InterventionConfig(
                top_k_experts=1,
                activation_threshold=0.2,
                deactivation_threshold=-0.2,
            )
        )
        artifacts = run_pipeline(self.dataset, config)

        layer0 = artifacts.steering_plan.layers[0]
        layer1 = artifacts.steering_plan.layers[1]

        self.assertEqual(layer0.experts_to_activate, [1])
        self.assertEqual(layer0.experts_to_deactivate, [0])
        self.assertEqual(layer1.experts_to_activate, [2])
        self.assertEqual(layer1.experts_to_deactivate, [0])

    def test_pipeline_selects_highest_attention_tokens(self) -> None:
        artifacts = run_pipeline(self.dataset, ExperimentConfig())
        layer0_positive_tokens = artifacts.selected_tokens_by_layer[0]["positive"]
        layer1_negative_tokens = artifacts.selected_tokens_by_layer[1]["negative"]

        self.assertEqual([token.token_text for token in layer0_positive_tokens], ["evidence", "honestly"])
        self.assertEqual([token.token_text for token in layer1_negative_tokens], ["confident", "over"])


if __name__ == "__main__":
    unittest.main()
