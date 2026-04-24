import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.attention_collection import HFModelResources
from moe_attention_guided_steering.olmoe_backend import (
    OLMoEPairedRoutingTrace,
    OLMoETargetRoutingTrace,
    SteerMoERiskDifferenceScore,
    _apply_bias_to_gate_output,
    _reshape_router_logits_for_prompt,
    build_steermoe_activation_table_from_paired_traces,
    build_steermoe_replication_plan,
    collect_olmoe_target_routing_trace_for_messages,
    find_chat_target_token_span,
    steermoe_plan_to_router_bias_by_layer,
)
from moe_attention_guided_steering.upstream_prompt_datasets import (
    StatementPromptPair,
    build_custom_steering_examples_from_statement_prompt_pairs,
)


class DummyRouterTensor:
    """Tiny tensor stand-in for testing flattened `(B*S, E)` router outputs."""

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
    """Verify the stage-1 OLMoE SteerMoE replication backend."""

    def test_build_custom_steering_examples_preserves_shared_body_target(self) -> None:
        examples = build_custom_steering_examples_from_statement_prompt_pairs(
            [
                StatementPromptPair(
                    prompt_id="fears:Bugs:0",
                    concept_type="fears",
                    concept_value="Bugs",
                    statement_index=0,
                    statement_text="Life is what you make of it.",
                    positive_prefix_text="Personify someone who is terrified of Bugs.  ",
                    body_text="What are your thoughts? Statement: Life is what you make of it.",
                    positive_full_prompt="Personify someone who is terrified of Bugs.  What are your thoughts? Statement: Life is what you make of it.",
                    negative_full_prompt="What are your thoughts? Statement: Life is what you make of it.",
                )
            ]
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].messages_0_target, examples[0].messages_1_target)
        self.assertIn("Bugs", examples[0].messages_0[0]["content"])
        self.assertNotIn("Bugs", examples[0].messages_1[0]["content"])

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

    def test_find_chat_target_token_span_uses_removed_target_comparison(self) -> None:
        class DummyTokenizer:
            def apply_chat_template(
                self,
                messages,
                tokenize: bool,
                add_generation_prompt: bool,
                return_tensors: str,
            ):
                text = messages[0]["content"]
                mapping = {
                    "prefixbody": [[10, 20, 21, 30, 31, 99]],
                    "prefix": [[10, 20, 21, 99]],
                }
                return mapping[text]

        self.assertEqual(
            find_chat_target_token_span(
                tokenizer=DummyTokenizer(),
                messages=[{"role": "user", "content": "prefixbody"}],
                target_text="body",
            ),
            (3, 5),
        )

    def test_collect_olmoe_target_routing_trace_summarizes_target_rows(self) -> None:
        import torch

        class DummyEncoding(dict):
            def to(self, device: str):
                return self

        class DummyTokenizer:
            def apply_chat_template(
                self,
                messages,
                tokenize: bool,
                add_generation_prompt: bool,
                return_tensors: str,
            ):
                text = messages[0]["content"]
                mapping = {
                    "prefixbody": DummyEncoding(
                        {"input_ids": torch.tensor([[10, 20, 21, 30, 31, 99]]), "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]])}
                    ),
                    "prefix": DummyEncoding(
                        {"input_ids": torch.tensor([[10, 20, 21, 99]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])}
                    ),
                }
                return mapping[text]

            def convert_ids_to_tokens(self, ids):
                return [f"tok_{token_id}" for token_id in ids]

        class DummyOutputs:
            def __init__(self):
                self.router_logits = [
                    torch.tensor(
                        [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.1, 0.7, 0.2],
                            [0.5, 0.1, 0.4],
                            [0.0, 0.0, 0.0],
                        ],
                        dtype=torch.float32,
                    )
                ]

        class DummyModel:
            device = "cpu"

            def __init__(self):
                self.config = type("Config", (), {"num_experts_per_tok": 1})()

            def __call__(self, **kwargs):
                self.last_kwargs = kwargs
                return DummyOutputs()

        trace = collect_olmoe_target_routing_trace_for_messages(
            trace_id="demo",
            subset_label="messages_0",
            messages=[{"role": "user", "content": "prefixbody"}],
            target_text="body",
            resources=HFModelResources(
                model=DummyModel(),
                tokenizer=DummyTokenizer(),
                model_id="demo",
                model_tag="demo",
                num_candidate_suffix_tokens=0,
            ),
            target_name="statement_body",
        )

        self.assertEqual((trace.target_token_start, trace.target_token_end), (3, 5))
        self.assertEqual(trace.target_token_texts, ["tok_30", "tok_31"])
        self.assertEqual(trace.layer_expert_activation_counts, [[1, 1, 0]])
        self.assertEqual(trace.layer_expert_activation_rates, [[0.5, 0.5, 0.0]])

    def test_build_steermoe_activation_table_computes_risk_difference(self) -> None:
        paired_trace = OLMoEPairedRoutingTrace(
            example_id="fears:Bugs:0",
            concept_type="fears",
            concept_value="Bugs",
            statement_index=0,
            statement_text="stmt",
            body_text="body",
            messages_0_trace=OLMoETargetRoutingTrace(
                trace_id="p",
                subset_label="messages_0",
                messages=[{"role": "user", "content": "pos"}],
                target_name="statement_body",
                target_text="body",
                target_token_start=3,
                target_token_end=5,
                target_token_texts=["tok_30", "tok_31"],
                top_k_experts_per_token=1,
                layer_expert_activation_counts=[[2, 0, 0], [0, 1, 1]],
                layer_expert_activation_rates=[[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]],
            ),
            messages_1_trace=OLMoETargetRoutingTrace(
                trace_id="n",
                subset_label="messages_1",
                messages=[{"role": "user", "content": "neg"}],
                target_name="statement_body",
                target_text="body",
                target_token_start=1,
                target_token_end=3,
                target_token_texts=["tok_30", "tok_31"],
                top_k_experts_per_token=1,
                layer_expert_activation_counts=[[0, 2, 0], [0, 0, 2]],
                layer_expert_activation_rates=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ),
        )

        scores = build_steermoe_activation_table_from_paired_traces([paired_trace])
        layer0_expert0 = next(score for score in scores if score.layer_index == 0 and score.expert_index == 0)
        layer0_expert1 = next(score for score in scores if score.layer_index == 0 and score.expert_index == 1)
        layer1_expert2 = next(score for score in scores if score.layer_index == 1 and score.expert_index == 2)

        self.assertEqual(layer0_expert0.messages_0_activation_rate, 1.0)
        self.assertEqual(layer0_expert0.messages_1_activation_rate, 0.0)
        self.assertEqual(layer0_expert0.risk_difference, 1.0)
        self.assertEqual(layer0_expert1.risk_difference, -1.0)
        self.assertEqual(layer1_expert2.risk_difference, -0.5)

    def test_build_steermoe_replication_plan_selects_global_experts(self) -> None:
        activation_table = [
            SteerMoERiskDifferenceScore(0, 0, 6, 1, 10, 10, 0.6, 0.1, 0.5, 0.5),
            SteerMoERiskDifferenceScore(0, 1, 1, 7, 10, 10, 0.1, 0.7, -0.6, 0.6),
            SteerMoERiskDifferenceScore(1, 3, 5, 1, 10, 10, 0.5, 0.1, 0.4, 0.4),
        ]

        plan = build_steermoe_replication_plan(
            concept="Bugs",
            concept_type="fears",
            model_id="allenai/OLMoE-1B-7B-0125-Instruct",
            model_tag="olmoe",
            target_name="statement_body",
            paired_traces=[object()],
            activation_table=activation_table,
            top_positive_experts=1,
            top_negative_experts=1,
            minimum_abs_risk_difference=0.2,
        )

        self.assertEqual(
            [(selected.score.layer_index, selected.score.expert_index) for selected in plan.selected_positive_experts],
            [(0, 0)],
        )
        self.assertEqual(
            [(selected.score.layer_index, selected.score.expert_index) for selected in plan.selected_negative_experts],
            [(0, 1)],
        )
        self.assertEqual(plan.layers[0].experts_to_activate, [0])
        self.assertEqual(plan.layers[0].experts_to_deactivate, [1])

    def test_steermoe_plan_to_router_bias_by_layer_scales_selected_experts(self) -> None:
        activation_table = [
            SteerMoERiskDifferenceScore(0, 0, 6, 1, 10, 10, 0.6, 0.1, 0.5, 0.5),
            SteerMoERiskDifferenceScore(0, 1, 1, 7, 10, 10, 0.1, 0.7, -0.6, 0.6),
        ]
        plan = build_steermoe_replication_plan(
            concept="Bugs",
            concept_type="fears",
            model_id="allenai/OLMoE-1B-7B-0125-Instruct",
            model_tag="olmoe",
            target_name="statement_body",
            paired_traces=[object()],
            activation_table=activation_table,
            top_positive_experts=1,
            top_negative_experts=1,
            minimum_abs_risk_difference=0.1,
        )

        bias = steermoe_plan_to_router_bias_by_layer(plan, coefficient=1.5)
        self.assertEqual(bias, {0: [1.25, -1.5]})

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
