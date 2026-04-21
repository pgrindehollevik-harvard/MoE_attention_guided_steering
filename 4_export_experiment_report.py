from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from args import build_parser
from moe_attention_guided_steering.config import ExperimentConfig, InterventionConfig, SelectionConfig
from moe_attention_guided_steering.datasets import load_experiment
from moe_attention_guided_steering.io_utils import (
    build_markdown_report,
    ensure_directory,
    scores_to_dict,
    selected_tokens_to_dict,
    write_json,
)
from moe_attention_guided_steering.pipeline import run_pipeline


def main() -> None:
    """Run the full toy pipeline and export a collaborator-friendly report.

    The report is meant to be something you can hand to a teammate who has not yet
    read the code. It includes the selected tokens, the expert deltas, and the
    resulting intervention recommendation in one place.
    """
    parser = build_parser(
        "Export a Markdown report for an attention-guided MoE steering experiment.",
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

    write_json(selected_tokens_to_dict(artifacts), output_dir / "selected_tokens.json")
    write_json(scores_to_dict(artifacts), output_dir / "expert_scores.json")
    write_json(asdict(artifacts.steering_plan), output_dir / "steering_plan.json")
    report_path = output_dir / "experiment_report.md"
    report_path.write_text(build_markdown_report(dataset, artifacts))
    print(f"Wrote report bundle to {output_dir}")


if __name__ == "__main__":
    main()
