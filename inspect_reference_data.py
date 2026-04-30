from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from moe_attention_guided_steering.reference_data import (
    load_reference_concept_suite,
    load_reference_data,
    summarize_reference_data,
)


def main() -> None:
    """Print a readable summary of the imported attention-guided data files.

    The imported text assets are experiment inputs and evaluation resources, not
    already-collected attention or expert traces.
    """
    parser = argparse.ArgumentParser(
        description="Inspect the imported attention-guided steering text assets."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the vendored attention-guided steering data tree.",
    )
    parser.add_argument(
        "--concept-type",
        default=None,
        help="Optional concept family to inspect in detail, such as fears or personas.",
    )
    args = parser.parse_args()

    bundle = load_reference_data(args.data_dir)
    summary = summarize_reference_data(bundle)

    print("Imported reference data summary")
    for key, value in summary.items():
        print(f"- {key}: {value}")

    print("- concept_counts:")
    for concept_type, concepts in sorted(bundle.concept_values_by_type.items()):
        print(f"  - {concept_type}: {len(concepts)}")

    print("- evaluation_prompt_versions:")
    for family, versions in sorted(bundle.evaluation_prompts_by_family.items()):
        version_list = ", ".join(str(version) for version in sorted(versions))
        print(f"  - {family}: {version_list}")

    if args.concept_type:
        suite = load_reference_concept_suite(args.concept_type, args.data_dir)
        print(f"- selected_concept_type: {suite.concept_type}")
        print(f"- mapped_evaluation_family: {suite.evaluation_family}")
        print(f"- num_concepts_in_family: {len(suite.concepts)}")
        print(
            "- sample_concepts: "
            + ", ".join(suite.concepts[:5] if suite.concepts else ["<empty>"])
        )
        print(
            "- available_prompt_versions: "
            + ", ".join(
                str(version) for version in sorted(suite.evaluation_prompts_by_version)
            )
        )
        print(
            "- general_statement_counts: "
            + ", ".join(
                f"{label}={len(statements)}"
                for label, statements in sorted(suite.general_statements_by_class.items())
            )
        )


if __name__ == "__main__":
    main()
