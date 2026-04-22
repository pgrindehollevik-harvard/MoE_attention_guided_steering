import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.reference_data import (
    load_reference_concept_suite,
    load_reference_data,
    summarize_reference_data,
)


class ReferenceDataTestCase(unittest.TestCase):
    """Verify the imported attention-guided text assets are readable and mapped."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = str(ROOT / "data")
        cls.bundle = load_reference_data(cls.data_dir)

    def test_summary_matches_expected_inventory(self) -> None:
        summary = summarize_reference_data(self.bundle)

        self.assertEqual(summary["num_concept_families"], 7)
        self.assertEqual(summary["num_total_concepts"], 514)
        self.assertEqual(summary["num_evaluation_prompt_families"], 5)
        self.assertEqual(summary["num_evaluation_prompt_versions"], 25)
        self.assertEqual(summary["num_general_statement_classes"], 2)
        self.assertEqual(summary["num_total_general_statements"], 400)

    def test_fears_suite_uses_phobia_prompts(self) -> None:
        suite = load_reference_concept_suite("fears", self.data_dir)

        self.assertEqual(suite.evaluation_family, "phobia")
        self.assertEqual(len(suite.concepts), 101)
        self.assertEqual(suite.concepts[0], "Bathing")
        self.assertEqual(sorted(suite.evaluation_prompts_by_version), [1, 2, 3, 4, 5])
        self.assertIn("Score (0-1)", suite.evaluation_prompts_by_version[1])
        self.assertEqual(len(suite.general_statements_by_class["class_0"]), 200)
        self.assertEqual(len(suite.general_statements_by_class["class_1"]), 200)

    def test_unmapped_concept_type_raises_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            load_reference_concept_suite("jailbreaking", self.data_dir)


if __name__ == "__main__":
    unittest.main()
