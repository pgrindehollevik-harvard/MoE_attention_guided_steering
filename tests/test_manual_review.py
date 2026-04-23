import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.manual_review import (
    build_manual_review_html,
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

    def test_manual_review_plan_roundtrip_and_html(self) -> None:
        plan = build_manual_review_plan(self.suite, concept_sample_size=2, question_sample_size=2, seed=7)
        plan.cases[0].baseline_response = "Baseline answer."
        plan.cases[0].moesteer_response = "MoESteer answer."

        restored = manual_review_plan_from_dict(manual_review_plan_to_dict(plan))
        html = build_manual_review_html(
            restored,
            title="Demo Review",
            companion_attention_report="attention_report.html",
        )

        self.assertEqual(restored.cases[0].baseline_response, "Baseline answer.")
        self.assertIn("Demo Review", html)
        self.assertIn("Baseline (no steering)", html)
        self.assertIn("MoESteer answer.", html)
        self.assertIn("attention_report.html", html)


if __name__ == "__main__":
    unittest.main()
