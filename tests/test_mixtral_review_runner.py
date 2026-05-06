import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.manual_review import ManualReviewCase, ManualReviewPlan
from run_mixtral_steermoe_review import (
    LLAMA_REFERENCE_CONDITION,
    MIXTRAL_BASELINE_CONDITION,
    MIXTRAL_STEERMOE_CONDITION,
    _prepare_question_only_plan,
)


class MixtralReviewRunnerTestCase(unittest.TestCase):
    """Verify the Mixtral question-only review runner report shape."""

    def test_prepare_question_only_plan_adds_llama_reference_column_first(self) -> None:
        plan = ManualReviewPlan(
            concept_type="fears",
            evaluation_family="phobia",
            seed=7,
            sampled_concepts=["Bugs"],
            sampled_evaluation_versions=[1],
            evaluation_questions_by_version={1: "What is scary?"},
            condition_order=["baseline", "steermoe"],
            condition_labels={"baseline": "Old baseline", "steermoe": "Old SteerMoE"},
            cases=[
                ManualReviewCase(
                    concept="Bugs",
                    evaluation_version=1,
                    evaluation_question="What is scary?",
                    full_prompt_text="old prompt",
                    prefix_conditioned_prompt_text="Personify someone who is terrified of Bugs. What is scary?",
                    responses={"baseline": "old"},
                )
            ],
        )

        prepared = _prepare_question_only_plan(plan, include_llama_reference=True)

        self.assertEqual(
            prepared.condition_order,
            [
                LLAMA_REFERENCE_CONDITION,
                MIXTRAL_BASELINE_CONDITION,
                MIXTRAL_STEERMOE_CONDITION,
            ],
        )
        self.assertEqual(prepared.cases[0].full_prompt_text, "What is scary?")
        self.assertEqual(
            set(prepared.cases[0].responses),
            {
                LLAMA_REFERENCE_CONDITION,
                MIXTRAL_BASELINE_CONDITION,
                MIXTRAL_STEERMOE_CONDITION,
            },
        )
        self.assertIn("Llama 3.1 8B", prepared.condition_labels[LLAMA_REFERENCE_CONDITION])

    def test_prepare_question_only_plan_can_skip_llama_reference(self) -> None:
        plan = ManualReviewPlan(
            concept_type="fears",
            evaluation_family="phobia",
            seed=7,
            sampled_concepts=["Bugs"],
            sampled_evaluation_versions=[1],
            evaluation_questions_by_version={1: "What is scary?"},
            condition_order=[],
            condition_labels={},
            cases=[
                ManualReviewCase(
                    concept="Bugs",
                    evaluation_version=1,
                    evaluation_question="What is scary?",
                    responses={},
                )
            ],
        )

        prepared = _prepare_question_only_plan(plan, include_llama_reference=False)

        self.assertEqual(
            prepared.condition_order,
            [MIXTRAL_BASELINE_CONDITION, MIXTRAL_STEERMOE_CONDITION],
        )


if __name__ == "__main__":
    unittest.main()
