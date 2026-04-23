import unittest

from moe_attention_guided_steering.config import ExperimentConfig, InterventionConfig
from moe_attention_guided_steering.pipeline import run_pipeline
from moe_attention_guided_steering.types import (
    ExperimentDataset,
    LayerRecord,
    PromptRecord,
    TokenRecord,
)


class PipelineTestCase(unittest.TestCase):
    """Exercise the generic scoring pipeline on a tiny in-memory dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = ExperimentDataset(
            metadata={
                "concept": "truthfulness",
                "contrast_concept": "deception",
            },
            examples=[
                PromptRecord(
                    prompt_id="pos-1",
                    label="positive",
                    prompt_text="Explain the answer using the strongest available evidence.",
                    layers=[
                        LayerRecord(
                            layer_index=0,
                            tokens=[
                                TokenRecord(0, "The", 0.10, [0.45, 0.30, 0.25]),
                                TokenRecord(1, "evidence", 0.62, [0.10, 0.72, 0.18]),
                                TokenRecord(2, "shows", 0.28, [0.20, 0.60, 0.20]),
                            ],
                        ),
                        LayerRecord(
                            layer_index=1,
                            tokens=[
                                TokenRecord(3, "answer", 0.18, [0.20, 0.25, 0.55]),
                                TokenRecord(4, "clearly", 0.67, [0.12, 0.18, 0.70]),
                                TokenRecord(5, ".", 0.15, [0.40, 0.25, 0.35]),
                            ],
                        ),
                    ],
                ),
                PromptRecord(
                    prompt_id="pos-2",
                    label="positive",
                    prompt_text="Answer honestly and cite the uncertainty if there is any.",
                    layers=[
                        LayerRecord(
                            layer_index=0,
                            tokens=[
                                TokenRecord(0, "Answer", 0.14, [0.35, 0.38, 0.27]),
                                TokenRecord(1, "honestly", 0.58, [0.12, 0.68, 0.20]),
                                TokenRecord(2, "and", 0.28, [0.24, 0.52, 0.24]),
                            ],
                        ),
                        LayerRecord(
                            layer_index=1,
                            tokens=[
                                TokenRecord(3, "cite", 0.21, [0.24, 0.21, 0.55]),
                                TokenRecord(4, "uncertainty", 0.71, [0.15, 0.20, 0.65]),
                                TokenRecord(5, "if", 0.08, [0.33, 0.29, 0.38]),
                            ],
                        ),
                    ],
                ),
                PromptRecord(
                    prompt_id="neg-1",
                    label="negative",
                    prompt_text="Make the answer sound confident even if the facts are weak.",
                    layers=[
                        LayerRecord(
                            layer_index=0,
                            tokens=[
                                TokenRecord(0, "Make", 0.16, [0.48, 0.26, 0.26]),
                                TokenRecord(1, "confident", 0.61, [0.70, 0.15, 0.15]),
                                TokenRecord(2, "facts", 0.23, [0.54, 0.24, 0.22]),
                            ],
                        ),
                        LayerRecord(
                            layer_index=1,
                            tokens=[
                                TokenRecord(3, "sound", 0.20, [0.52, 0.18, 0.30]),
                                TokenRecord(4, "confident", 0.63, [0.68, 0.17, 0.15]),
                                TokenRecord(5, "weak", 0.17, [0.47, 0.22, 0.31]),
                            ],
                        ),
                    ],
                ),
                PromptRecord(
                    prompt_id="neg-2",
                    label="negative",
                    prompt_text="Prioritize persuasion over factual accuracy.",
                    layers=[
                        LayerRecord(
                            layer_index=0,
                            tokens=[
                                TokenRecord(0, "Prioritize", 0.18, [0.44, 0.30, 0.26]),
                                TokenRecord(1, "persuasion", 0.56, [0.66, 0.18, 0.16]),
                                TokenRecord(2, "accuracy", 0.26, [0.51, 0.27, 0.22]),
                            ],
                        ),
                        LayerRecord(
                            layer_index=1,
                            tokens=[
                                TokenRecord(3, "persuasion", 0.19, [0.46, 0.22, 0.32]),
                                TokenRecord(4, "over", 0.69, [0.60, 0.18, 0.22]),
                                TokenRecord(5, "accuracy", 0.12, [0.49, 0.21, 0.30]),
                            ],
                        ),
                    ],
                ),
            ],
        )

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
