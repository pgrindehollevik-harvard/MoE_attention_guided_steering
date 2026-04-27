import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.manual_review import (
    build_manual_review_html,
    build_manual_review_markdown,
    build_manual_review_plan,
    extract_evaluation_question,
    manual_review_plan_from_dict,
    manual_review_plan_to_dict,
)
from moe_attention_guided_steering.reference_data import load_reference_concept_suite


class ManualReviewTestCase(unittest.TestCase):
    """Verify the manual review helpers produce a stable inspection scaffold."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = load_reference_concept_suite("fears", str(ROOT / "data"))

    def test_extract_evaluation_question_reads_phobia_prompt(self) -> None:
        question = extract_evaluation_question(self.suite.evaluation_prompts_by_version[1])
        self.assertEqual(question, "What is the scariest thing in the world? Answer in 20 words or less.")

    def test_build_manual_review_plan_creates_cross_product(self) -> None:
        plan = build_manual_review_plan(self.suite, concept_sample_size=5, question_sample_size=5, seed=7)

        self.assertEqual(len(plan.sampled_concepts), 5)
        self.assertEqual(len(set(plan.sampled_concepts)), 5)
        self.assertEqual(len(plan.sampled_evaluation_versions), 5)
        self.assertEqual(len(set(plan.sampled_evaluation_versions)), 5)
        self.assertEqual(len(plan.cases), 25)
        self.assertEqual({case.concept for case in plan.cases}, set(plan.sampled_concepts))
        self.assertEqual(
            {case.evaluation_version for case in plan.cases},
            set(plan.sampled_evaluation_versions),
        )
        self.assertTrue(all(case.full_prompt_text for case in plan.cases))
        self.assertEqual(plan.cases[0].full_prompt_text, plan.cases[0].evaluation_question)
        self.assertIn(
            "Personify someone who is terrified of",
            plan.cases[0].prefix_conditioned_prompt_text,
        )

    def test_manual_review_plan_roundtrip_and_html(self) -> None:
        plan = build_manual_review_plan(self.suite, concept_sample_size=2, question_sample_size=2, seed=7)
        plan.cases[0].responses["baseline"] = "Baseline answer."
        plan.cases[0].responses["steermoe"] = "SteerMoE answer."

        restored = manual_review_plan_from_dict(manual_review_plan_to_dict(plan))
        html = build_manual_review_html(
            restored,
            title="Demo Review",
            companion_report="steermoe_notes.html",
        )

        self.assertEqual(restored.cases[0].responses["baseline"], "Baseline answer.")
        self.assertIn("Demo Review", html)
        self.assertIn("OLMoE baseline (question only)", html)
        self.assertIn("SteerMoE answer.", html)
        self.assertIn("steermoe_notes.html", html)
        self.assertIn("Question-only test prompt:", html)
        self.assertIn("Prefix-conditioned diagnostic prompt", html)

    def test_legacy_prefixed_full_prompt_is_not_used_as_test_prompt(self) -> None:
        plan = build_manual_review_plan(self.suite, concept_sample_size=1, question_sample_size=1, seed=7)
        payload = manual_review_plan_to_dict(plan)
        payload["cases"][0]["full_prompt_text"] = "Personify someone who is terrified of Bugs.  Question?"
        payload["cases"][0].pop("prefix_conditioned_prompt_text")

        restored = manual_review_plan_from_dict(payload)

        self.assertEqual(restored.cases[0].full_prompt_text, restored.cases[0].evaluation_question)
        self.assertEqual(
            restored.cases[0].prefix_conditioned_prompt_text,
            "Personify someone who is terrified of Bugs.  Question?",
        )

    def test_reference_model_condition_is_labeled_as_context_not_control(self) -> None:
        plan = build_manual_review_plan(
            self.suite,
            concept_sample_size=1,
            question_sample_size=1,
            seed=7,
            condition_order=["baseline", "llama_3_1_8b_baseline", "steermoe"],
        )
        plan.cases[0].responses["llama_3_1_8b_baseline"] = "Question-only Llama answer."

        markdown = build_manual_review_markdown(plan)
        html = build_manual_review_html(plan)

        self.assertIn("Llama 3.1 8B baseline (question only)", markdown)
        self.assertIn("Question-only Llama answer.", html)
        self.assertIn("reference-model column is unsteered question-only context", markdown)
        self.assertIn("reference-model columns are unsteered context", html)


if __name__ == "__main__":
    unittest.main()
