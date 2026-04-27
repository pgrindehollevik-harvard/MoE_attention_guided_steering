import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.generation import render_plain_prompt
from moe_attention_guided_steering.manual_review import build_manual_review_plan
from moe_attention_guided_steering.model_comparison import (
    ModelComparisonSpec,
    build_empty_model_comparison_results,
    model_comparison_spec_from_dict,
    prompt_text_for_model_comparison_case,
    render_model_comparison_markdown,
)
from moe_attention_guided_steering.reference_data import load_reference_concept_suite


class ModelComparisonTestCase(unittest.TestCase):
    """Verify the full-prefix model comparison scaffolding without loading GPUs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_reference_concept_suite("fears", str(ROOT / "data"))

    def test_prompt_modes_are_explicit(self) -> None:
        plan = build_manual_review_plan(self.suite, concept_sample_size=1, question_sample_size=1, seed=7)
        case = plan.cases[0]

        prefixed = prompt_text_for_model_comparison_case(
            plan=plan,
            concept=case.concept,
            evaluation_question=case.evaluation_question,
            prompt_mode="prefix",
        )
        question_only = prompt_text_for_model_comparison_case(
            plan=plan,
            concept=case.concept,
            evaluation_question=case.evaluation_question,
            prompt_mode="question",
        )

        self.assertIn("Personify someone who is terrified of", prefixed)
        self.assertEqual(question_only, case.evaluation_question)

    def test_empty_results_store_model_columns_and_prefixed_prompts(self) -> None:
        plan = build_manual_review_plan(self.suite, concept_sample_size=1, question_sample_size=1, seed=7)
        specs = [
            ModelComparisonSpec(
                label="OLMoE",
                model_id="allenai/OLMoE-1B-7B-0125-Instruct",
                model_tag="olmoe",
            )
        ]

        results = build_empty_model_comparison_results(plan, specs, prompt_mode="prefix")
        markdown = render_model_comparison_markdown(results)

        self.assertEqual(results["cases"][0]["responses"], {"olmoe": ""})
        self.assertIn("Personify someone who is terrified of", results["cases"][0]["prompt_text"])
        self.assertIn("This diagnostic asks whether each base model", markdown)

    def test_plain_prompt_template_requires_placeholder(self) -> None:
        self.assertEqual(render_plain_prompt("Question?", "human: {prompt} gpt:"), "human: Question? gpt:")
        with self.assertRaises(ValueError):
            render_plain_prompt("Question?", "human: gpt:")

    def test_model_spec_supports_per_model_device_loading(self) -> None:
        spec = model_comparison_spec_from_dict(
            {
                "label": "LLaMA-MoE",
                "model_id": "llama-moe/LLaMA-MoE-v1-3_5B-2_8-sft",
                "model_tag": "llama_moe",
                "device_map": None,
                "post_load_device": "cuda",
            }
        )

        self.assertIsNone(spec.device_map)
        self.assertEqual(spec.post_load_device, "cuda")


if __name__ == "__main__":
    unittest.main()
