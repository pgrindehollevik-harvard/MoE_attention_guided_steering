import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.generation import generate_unsteered_response, render_plain_prompt
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

    def test_model_spec_supports_per_model_loading_flags(self) -> None:
        spec = model_comparison_spec_from_dict(
            {
                "label": "Mixtral",
                "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "model_tag": "mixtral",
                "device_map": None,
                "use_cache": False,
                "load_in_8bit": True,
                "bnb_cpu_offload": True,
                "offload_folder": "outputs/offload/mixtral-test",
                "min_new_tokens": 4,
                "temperature": 0.0,
            }
        )

        self.assertIsNone(spec.device_map)
        self.assertFalse(spec.use_cache)
        self.assertIsNone(spec.load_in_4bit)
        self.assertTrue(spec.load_in_8bit)
        self.assertTrue(spec.bnb_cpu_offload)
        self.assertEqual(spec.offload_folder, "outputs/offload/mixtral-test")
        self.assertEqual(spec.min_new_tokens, 4)
        self.assertEqual(spec.temperature, 0.0)
        self.assertIsNone(spec.top_p)

    def test_generate_unsteered_response_passes_use_cache_flag(self) -> None:
        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text, return_tensors):
                return {"input_ids": DummyTensor([[10, 11]])}

            def decode(self, token_ids, skip_special_tokens):
                return "decoded"

        class DummyTensor:
            def __init__(self, value):
                self.value = value
                self.shape = (len(value), len(value[0]))

            def __getitem__(self, index):
                return [99]

        class DummyModel:
            device = "cpu"

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return DummyTensor([[10, 11, 99]])

        resources = type(
            "Resources",
            (),
            {"tokenizer": DummyTokenizer(), "model": DummyModel()},
        )()

        response = generate_unsteered_response(
            prompt_text="Question?",
            resources=resources,
            prompt_format="plain",
            use_cache=False,
            min_new_tokens=3,
        )

        self.assertEqual(response, "decoded")
        self.assertFalse(resources.model.kwargs["use_cache"])
        self.assertEqual(resources.model.kwargs["min_new_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
