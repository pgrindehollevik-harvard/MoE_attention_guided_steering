from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from args import build_parser
from moe_attention_guided_steering.datasets import load_experiment, summarize_dataset


def main() -> None:
    """Validate the dataset and print a concise summary.

    This script is intentionally lightweight. It exists to give us a fast sanity
    check before we run any downstream scoring code.
    """
    parser = build_parser("Validate an attention-guided MoE experiment dataset.")
    args = parser.parse_args()

    dataset = load_experiment(args.input)
    summary = summarize_dataset(dataset)

    print("Dataset validation succeeded.")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print(f"- concept: {dataset.metadata.get('concept', 'unknown')}")
    print(f"- contrast_concept: {dataset.metadata.get('contrast_concept', 'unknown')}")


if __name__ == "__main__":
    main()
