from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.io_utils import ensure_directory, write_json
from moe_attention_guided_steering.manual_review import (
    build_manual_review_markdown,
    build_manual_review_plan,
    manual_review_plan_to_dict,
)
from moe_attention_guided_steering.reference_data import load_reference_concept_suite


def main() -> None:
    """Sample fears and evaluation questions, then write a review bundle."""
    parser = argparse.ArgumentParser(
        description="Prepare a manual qualitative review sheet for the currently configured comparison conditions."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the imported attention-guided steering data tree.",
    )
    parser.add_argument(
        "--concept-type",
        default="fears",
        help="Concept family to sample from. Defaults to fears.",
    )
    parser.add_argument(
        "--num-concepts",
        type=int,
        default=5,
        help="Number of concepts to sample for manual review.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of evaluation questions to sample for manual review.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for reproducible concept/question sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/manual_fear_review",
        help="Directory where the manual review JSON and Markdown should be written.",
    )
    args = parser.parse_args()

    concept_suite = load_reference_concept_suite(args.concept_type, args.data_dir)
    review_plan = build_manual_review_plan(
        concept_suite=concept_suite,
        concept_sample_size=args.num_concepts,
        question_sample_size=args.num_questions,
        seed=args.seed,
    )

    output_dir = ensure_directory(args.output_dir)
    write_json(
        manual_review_plan_to_dict(review_plan),
        output_dir / "manual_review_plan.json",
    )
    (output_dir / "manual_review_sheet.md").write_text(
        build_manual_review_markdown(review_plan)
    )

    print(f"Wrote manual review bundle to {output_dir}")
    print("Sampled concepts:")
    for concept in review_plan.sampled_concepts:
        print(f"- {concept}")


if __name__ == "__main__":
    main()
