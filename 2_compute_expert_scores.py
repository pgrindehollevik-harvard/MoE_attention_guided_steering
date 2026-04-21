from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from args import build_parser
from moe_attention_guided_steering.config import ExperimentConfig, SelectionConfig
from moe_attention_guided_steering.datasets import load_experiment
from moe_attention_guided_steering.io_utils import ensure_directory, scores_to_dict, write_json
from moe_attention_guided_steering.pipeline import run_pipeline


def main() -> None:
    """Compute expert deltas and write them to JSON.

    Even though this script runs the full pipeline internally, it only exports the
    score stage because expert deltas are the main artifact we want to inspect when
    deciding whether the concept signal is clean enough to justify an intervention.
    """
    parser = build_parser("Compute positive-vs-negative expert deltas.")
    args = parser.parse_args()

    dataset = load_experiment(args.input)
    config = ExperimentConfig(
        selection=SelectionConfig(minimum_attention_weight=args.min_attention)
    )
    artifacts = run_pipeline(dataset, config)
    output_dir = ensure_directory(args.output_dir)
    output_path = output_dir / "expert_scores.json"
    write_json(scores_to_dict(artifacts), output_path)
    print(f"Wrote expert scores to {output_path}")


if __name__ == "__main__":
    main()
