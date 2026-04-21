from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from args import build_parser
from moe_attention_guided_steering.config import ExperimentConfig, InterventionConfig, SelectionConfig
from moe_attention_guided_steering.datasets import load_experiment
from moe_attention_guided_steering.io_utils import ensure_directory, write_json
from moe_attention_guided_steering.pipeline import run_pipeline


def main() -> None:
    """Build a sparse steering plan from expert deltas.

    This is the first script in the pipeline that produces an actionable research
    artifact: a concrete list of experts to up-weight or down-weight at each layer.
    """
    parser = build_parser(
        "Build a layerwise steering plan from attention-guided expert scores.",
        include_intervention_args=True,
    )
    args = parser.parse_args()

    dataset = load_experiment(args.input)
    config = ExperimentConfig(
        selection=SelectionConfig(minimum_attention_weight=args.min_attention),
        intervention=InterventionConfig(
            top_k_experts=args.top_k_experts,
            activation_threshold=args.activation_threshold,
            deactivation_threshold=args.deactivation_threshold,
        ),
    )
    artifacts = run_pipeline(dataset, config)
    output_dir = ensure_directory(args.output_dir)
    output_path = output_dir / "steering_plan.json"
    write_json(asdict(artifacts.steering_plan), output_path)
    print(f"Wrote steering plan to {output_path}")


if __name__ == "__main__":
    main()
