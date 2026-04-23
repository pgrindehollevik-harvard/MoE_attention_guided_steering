import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.config import ExperimentConfig, InterventionConfig
from moe_attention_guided_steering.olmoe_backend import (
    OLMoERouterTrace,
    _reshape_router_logits_for_prompt,
    build_experiment_dataset_from_router_traces,
    build_fixed_layer_to_token_index,
    build_selection_scores_for_candidates,
    run_olmoe_steering_pipeline,
    steering_plan_to_router_bias_by_layer,
)


class DummyRouterTensor:
    """Tiny tensor stand-in for testing the flatten-to-(B,S,E) reshape logic."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def view(self, batch_size, sequence_length, expert_count):
        rebuilt = []
        index = 0
        for _ in range(batch_size):
            prompt_rows = []
            for _ in range(sequence_length):
                prompt_rows.append(self.rows[index])
                index += 1
            rebuilt.append(prompt_rows)
        return rebuilt


class OLMoEBackendTestCase(unittest.TestCase):
    """Verify the first OLMoE backend without requiring a real model."""

    def test_build_fixed_layer_to_token_index_repeats_shared_choice(self) -> None:
        self.assertEqual(build_fixed_layer_to_token_index(3, fixed_token_index=-1), {0: -1, 1: -1, 2: -1})

    def test_build_selection_scores_for_candidates_is_one_hot(self) -> None:
        self.assertEqual(build_selection_scores_for_candidates([-3, -2, -1], -2), [0.0, 1.0, 0.0])

    def test_reshape_router_logits_for_prompt_accepts_flattened_olmoe_output(self) -> None:
        tensor = DummyRouterTensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        self.assertEqual(
            _reshape_router_logits_for_prompt(tensor, batch_size=1, sequence_length=3),
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        )

    def test_build_experiment_dataset_from_router_traces_preserves_expert_vectors(self) -> None:
        positive = OLMoERouterTrace(
            prompt_id="fears:Bugs:0:positive",
            label="positive",
            prompt_text="prompt+prefix",
            candidate_token_texts=["tok_a", "tok_b"],
            candidate_token_relative_indices=[-2, -1],
            layer_router_probabilities=[
                [[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]],
                [[0.6, 0.2, 0.2], [0.2, 0.3, 0.5]],
            ],
        )
        negative = OLMoERouterTrace(
            prompt_id="fears:Bugs:0:negative",
            label="negative",
            prompt_text="prompt",
            candidate_token_texts=["tok_a", "tok_b"],
            candidate_token_relative_indices=[-2, -1],
            layer_router_probabilities=[
                [[0.6, 0.2, 0.2], [0.2, 0.5, 0.3]],
                [[0.4, 0.4, 0.2], [0.3, 0.4, 0.3]],
            ],
        )

        dataset = build_experiment_dataset_from_router_traces(
            prompt_traces=[positive, negative],
            concept="Bugs",
            contrast_concept="generic_statement_control",
            concept_type="fears",
            model_id="allenai/OLMoE-1B-7B-0125-Instruct",
            model_tag="olmoe",
            selection_layer_to_token_index={0: -1, 1: -2},
            selection_source="attention_guided",
        )

        self.assertEqual(dataset.metadata["selection_source"], "attention_guided")
        self.assertEqual(dataset.examples[0].layers[0].tokens[1].attention_weight, 1.0)
        self.assertEqual(dataset.examples[0].layers[1].tokens[0].attention_weight, 1.0)
        self.assertEqual(dataset.examples[0].layers[0].tokens[1].expert_loads, [0.9, 0.05, 0.05])

    def test_run_olmoe_steering_pipeline_builds_sparse_expert_plan(self) -> None:
        positive = OLMoERouterTrace(
            prompt_id="p",
            label="positive",
            prompt_text="prompt+prefix",
            candidate_token_texts=["tok_a", "tok_b"],
            candidate_token_relative_indices=[-2, -1],
            layer_router_probabilities=[
                [[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]],
            ],
        )
        negative = OLMoERouterTrace(
            prompt_id="n",
            label="negative",
            prompt_text="prompt",
            candidate_token_texts=["tok_a", "tok_b"],
            candidate_token_relative_indices=[-2, -1],
            layer_router_probabilities=[
                [[0.6, 0.2, 0.2], [0.2, 0.5, 0.3]],
            ],
        )
        dataset = build_experiment_dataset_from_router_traces(
            prompt_traces=[positive, negative],
            concept="Bugs",
            contrast_concept="generic_statement_control",
            concept_type="fears",
            model_id="allenai/OLMoE-1B-7B-0125-Instruct",
            model_tag="olmoe",
            selection_layer_to_token_index={0: -1},
            selection_source="fixed_token_-1",
        )

        artifacts = run_olmoe_steering_pipeline(
            dataset,
            config=ExperimentConfig(
                intervention=InterventionConfig(
                    top_k_experts=1,
                    activation_threshold=0.2,
                    deactivation_threshold=-0.2,
                )
            ),
        )

        self.assertEqual(artifacts.steering_plan.layers[0].experts_to_activate, [0])
        self.assertEqual(artifacts.steering_plan.layers[0].experts_to_deactivate, [1])

    def test_steering_plan_to_router_bias_by_layer_creates_dense_vectors(self) -> None:
        positive = OLMoERouterTrace(
            prompt_id="p",
            label="positive",
            prompt_text="prompt+prefix",
            candidate_token_texts=["tok_a", "tok_b"],
            candidate_token_relative_indices=[-2, -1],
            layer_router_probabilities=[
                [[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]],
            ],
        )
        negative = OLMoERouterTrace(
            prompt_id="n",
            label="negative",
            prompt_text="prompt",
            candidate_token_texts=["tok_a", "tok_b"],
            candidate_token_relative_indices=[-2, -1],
            layer_router_probabilities=[
                [[0.6, 0.2, 0.2], [0.2, 0.5, 0.3]],
            ],
        )
        dataset = build_experiment_dataset_from_router_traces(
            prompt_traces=[positive, negative],
            concept="Bugs",
            contrast_concept="generic_statement_control",
            concept_type="fears",
            model_id="allenai/OLMoE-1B-7B-0125-Instruct",
            model_tag="olmoe",
            selection_layer_to_token_index={0: -1},
            selection_source="fixed_token_-1",
        )
        artifacts = run_olmoe_steering_pipeline(
            dataset,
            config=ExperimentConfig(
                intervention=InterventionConfig(
                    top_k_experts=1,
                    activation_threshold=0.2,
                    deactivation_threshold=-0.2,
                )
            ),
        )

        bias = steering_plan_to_router_bias_by_layer(artifacts.steering_plan, coefficient=3.5)
        self.assertEqual(bias, {0: [3.5, -3.5, 0.0]})


if __name__ == "__main__":
    unittest.main()
