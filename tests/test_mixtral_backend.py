import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.attention_collection import HFModelResources
from moe_attention_guided_steering.manual_review import ManualReviewCase, ManualReviewPlan
from moe_attention_guided_steering.mixtral_backend import (
    collect_mixtral_target_routing_trace_for_messages,
    fill_manual_review_plan_with_mixtral_generations,
    mixtral_router_bias_hooks,
)


class MixtralBackendTestCase(unittest.TestCase):
    """Verify the Mixtral-specific SteerMoE hooks without loading real weights."""

    def test_collect_mixtral_target_routing_trace_summarizes_router_logits(self) -> None:
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
                        {"input_ids": torch.tensor([[10, 20, 21, 30, 31, 99]])}
                    ),
                    "prefix": DummyEncoding(
                        {"input_ids": torch.tensor([[10, 20, 21, 99]])}
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
                self.config = type("Config", (), {"num_experts_per_tok": 2})()

            def __call__(self, **kwargs):
                self.last_kwargs = kwargs
                return DummyOutputs()

        trace = collect_mixtral_target_routing_trace_for_messages(
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
        )

        self.assertEqual((trace.target_token_start, trace.target_token_end), (3, 5))
        self.assertEqual(trace.top_k_experts_per_token, 2)
        self.assertEqual(trace.layer_expert_activation_counts, [[1, 1, 2]])
        self.assertEqual(trace.layer_expert_activation_rates, [[0.5, 0.5, 1.0]])

    def test_mixtral_router_bias_hooks_patch_block_sparse_moe_gate(self) -> None:
        import torch

        gate = torch.nn.Identity()
        block = type("BlockSparseMoe", (), {"gate": gate})()
        layer = type("Layer", (), {"block_sparse_moe": block})()
        inner = type("InnerModel", (), {"layers": [layer]})()
        outer = type("OuterModel", (), {"model": inner})()

        with mixtral_router_bias_hooks(outer, {0: [0.0, 0.8, 0.0]}):
            output = gate(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.1, 0.2, 0.3],
                    ],
                    dtype=torch.float32,
                )
            )

        self.assertTrue(torch.allclose(output[-1], torch.tensor([0.1, 1.0, 0.3])))

    def test_fill_manual_review_plan_uses_question_only_prompt(self) -> None:
        import moe_attention_guided_steering.mixtral_backend as backend

        plan = ManualReviewPlan(
            concept_type="fears",
            evaluation_family="phobia",
            seed=7,
            sampled_concepts=["Bugs"],
            sampled_evaluation_versions=[1],
            evaluation_questions_by_version={1: "What is scary?"},
            condition_order=["mixtral_baseline", "mixtral_steermoe"],
            condition_labels={
                "mixtral_baseline": "Mixtral baseline",
                "mixtral_steermoe": "Mixtral + SteerMoE",
            },
            cases=[
                ManualReviewCase(
                    concept="Bugs",
                    evaluation_version=1,
                    evaluation_question="What is scary?",
                    full_prompt_text="What is scary?",
                    prefix_conditioned_prompt_text="Personify someone who is terrified of Bugs.  What is scary?",
                    responses={"mixtral_baseline": "", "mixtral_steermoe": ""},
                )
            ],
        )
        prompts = []

        def fake_generate(prompt_text, resources, bias_by_layer, **kwargs):
            prompts.append(prompt_text)
            return "steered" if bias_by_layer else "baseline"

        old_generate = backend.generate_with_mixtral_steering
        old_bias = backend.steermoe_plan_to_router_bias_by_layer
        backend.generate_with_mixtral_steering = fake_generate
        backend.steermoe_plan_to_router_bias_by_layer = lambda steering_plan, coefficient: {0: [1.0]}
        try:
            filled = backend.fill_manual_review_plan_with_mixtral_generations(
                plan=plan,
                resources=HFModelResources(
                    model=object(),
                    tokenizer=object(),
                    model_id="demo",
                    model_tag="demo",
                    num_candidate_suffix_tokens=0,
                ),
                steermoe_plans_by_concept={"Bugs": object()},
            )
        finally:
            backend.generate_with_mixtral_steering = old_generate
            backend.steermoe_plan_to_router_bias_by_layer = old_bias

        self.assertEqual(prompts, ["What is scary?", "What is scary?"])
        self.assertEqual(filled.cases[0].responses["mixtral_baseline"], "baseline")
        self.assertEqual(filled.cases[0].responses["mixtral_steermoe"], "steered")


if __name__ == "__main__":
    unittest.main()
