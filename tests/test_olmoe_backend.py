import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.config import ExperimentConfig, InterventionConfig
from moe_attention_guided_steering.olmoe_backend import (
    OLMoESpanActivationTrace,
    _apply_bias_to_gate_output,
    _reshape_router_logits_for_prompt,
    build_experiment_dataset_from_span_traces,
    run_olmoe_steermoe_pipeline,
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
    """Verify the stage-1 OLMoE backend without requiring a real model."""

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

    def test_build_experiment_dataset_from_span_traces_preserves_activation_vectors(self) -> None:
        positive = OLMoESpanActivationTrace(
            prompt_id="fears:Bugs:0:positive",
            label="positive",
            prompt_text="prompt+prefix",
            span_name="user_content",
            span_token_start=4,
            span_token_end=8,
            span_token_texts=["tok_a", "tok_b", "tok_c", "tok_d"],
            top_k_experts_per_token=2,
            layer_expert_activation_rates=[
                [0.5, 0.25, 0.25],
                [0.75, 0.25, 0.0],
            ],
        )
        negative = OLMoESpanActivationTrace(
            prompt_id="fears:Bugs:0:negative",
            label="negative",
            prompt_text="prompt",
            span_name="user_content",
            span_token_start=4,
            span_token_end=6,
            span_token_texts=["tok_a", "tok_b"],
            top_k_experts_per_token=2,
            layer_expert_activation_rates=[
                [0.1, 0.4, 0.5],
                [0.25, 0.5, 0.25],
            ],
        )

        dataset = build_experiment_dataset_from_span_traces(
            prompt_traces=[positive, negative],
            concept="Bugs",
            contrast_concept="generic_statement_control",
            concept_type="fears",
            model_id="allenai/OLMoE-1B-7B-0125-Instruct",
            model_tag="olmoe",
            readout_span="user_content",
        )

        self.assertEqual(dataset.metadata["readout_statistic"], "topk_activation_rate")
        self.assertEqual(dataset.examples[0].layers[0].tokens[0].attention_weight, 1.0)
        self.assertEqual(dataset.examples[0].layers[0].tokens[0].expert_loads, [0.5, 0.25, 0.25])
        self.assertIn("user_content_activation_rate", dataset.examples[0].layers[0].tokens[0].token_text)

    def test_run_olmoe_steermoe_pipeline_builds_sparse_expert_plan(self) -> None:
        positive = OLMoESpanActivationTrace(
            prompt_id="p",
            label="positive",
            prompt_text="prompt+prefix",
            span_name="user_content",
            span_token_start=2,
            span_token_end=4,
            span_token_texts=["tok_a", "tok_b"],
            top_k_experts_per_token=2,
            layer_expert_activation_rates=[
                [0.8, 0.1, 0.1],
            ],
        )
        negative = OLMoESpanActivationTrace(
            prompt_id="n",
            label="negative",
            prompt_text="prompt",
            span_name="user_content",
            span_token_start=2,
            span_token_end=4,
            span_token_texts=["tok_a", "tok_b"],
            top_k_experts_per_token=2,
            layer_expert_activation_rates=[
                [0.2, 0.6, 0.2],
            ],
        )
        dataset = build_experiment_dataset_from_span_traces(
            prompt_traces=[positive, negative],
            concept="Bugs",
            contrast_concept="generic_statement_control",
            concept_type="fears",
            model_id="allenai/OLMoE-1B-7B-0125-Instruct",
            model_tag="olmoe",
            readout_span="user_content",
        )

        artifacts = run_olmoe_steermoe_pipeline(
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

    def test_steering_plan_to_router_bias_by_layer_scales_by_selected_delta(self) -> None:
        positive = OLMoESpanActivationTrace(
            prompt_id="p",
            label="positive",
            prompt_text="prompt+prefix",
            span_name="user_content",
            span_token_start=2,
            span_token_end=4,
            span_token_texts=["tok_a", "tok_b"],
            top_k_experts_per_token=2,
            layer_expert_activation_rates=[
                [0.8, 0.1, 0.1],
            ],
        )
        negative = OLMoESpanActivationTrace(
            prompt_id="n",
            label="negative",
            prompt_text="prompt",
            span_name="user_content",
            span_token_start=2,
            span_token_end=4,
            span_token_texts=["tok_a", "tok_b"],
            top_k_experts_per_token=2,
            layer_expert_activation_rates=[
                [0.2, 0.6, 0.2],
            ],
        )
        dataset = build_experiment_dataset_from_span_traces(
            prompt_traces=[positive, negative],
            concept="Bugs",
            contrast_concept="generic_statement_control",
            concept_type="fears",
            model_id="allenai/OLMoE-1B-7B-0125-Instruct",
            model_tag="olmoe",
            readout_span="user_content",
        )
        artifacts = run_olmoe_steermoe_pipeline(
            dataset,
            config=ExperimentConfig(
                intervention=InterventionConfig(
                    top_k_experts=1,
                    activation_threshold=0.2,
                    deactivation_threshold=-0.2,
                )
            ),
        )

        bias = steering_plan_to_router_bias_by_layer(artifacts.steering_plan, coefficient=1.5)
        self.assertEqual(bias, {0: [1.5, -1.25, 0.0]})

    def test_apply_bias_to_gate_output_supports_tuple_style_gate_outputs(self) -> None:
        import torch

        router_logits = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.2, 0.3],
            ],
            dtype=torch.float32,
        )
        top_k_weights = torch.tensor(
            [
                [0.5, 0.5],
                [0.5250, 0.4750],
            ],
            dtype=torch.float32,
        )
        top_k_indices = torch.tensor(
            [
                [0, 1],
                [2, 1],
            ],
            dtype=torch.long,
        )

        biased_output = _apply_bias_to_gate_output(
            output=(router_logits, top_k_weights, top_k_indices),
            bias_vector=torch.tensor([0.0, 0.8, 0.0], dtype=torch.float32),
        )

        biased_router_logits, biased_top_k_weights, biased_top_k_indices = biased_output[:3]
        self.assertTrue(torch.allclose(biased_router_logits[-1], torch.tensor([0.1, 1.0, 0.3])))
        self.assertEqual(tuple(biased_top_k_indices[-1].tolist()), (1, 2))
        self.assertAlmostEqual(float(biased_top_k_weights[-1].sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
