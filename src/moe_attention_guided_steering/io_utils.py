import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from .pipeline import PipelineArtifacts
from .types import ExperimentDataset


def ensure_directory(path: str) -> Path:
    """Create an output directory if it does not exist yet."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(data: Any, path: Path) -> None:
    """Write JSON with stable formatting for easy diffs and review."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def selected_tokens_to_dict(artifacts: PipelineArtifacts) -> Dict[str, Any]:
    """Serialize selected tokens using stringified layer keys for JSON output."""
    return {
        str(layer_index): {
            label: [asdict(token) for token in tokens]
            for label, tokens in label_map.items()
        }
        for layer_index, label_map in artifacts.selected_tokens_by_layer.items()
    }


def scores_to_dict(artifacts: PipelineArtifacts) -> Dict[str, Any]:
    """Serialize expert scores in a JSON-friendly format."""
    return {
        str(layer_index): [asdict(score) for score in scores]
        for layer_index, scores in artifacts.scores_by_layer.items()
    }


def build_markdown_report(dataset: ExperimentDataset, artifacts: PipelineArtifacts) -> str:
    """Create a collaborator-friendly Markdown report.

    The report is intentionally narrative rather than purely machine-readable. The
    audience for this file is a teammate who wants to understand what happened in a
    run without opening the Python code.
    """
    lines = [
        "# Experiment Report",
        "",
        f"- Concept: **{dataset.metadata.get('concept', 'unknown')}**",
        f"- Contrast concept: **{dataset.metadata.get('contrast_concept', 'unknown')}**",
        f"- Examples: **{len(dataset.examples)}**",
        "",
        "## Layerwise Steering Plan",
        "",
    ]

    for layer in artifacts.steering_plan.layers:
        lines.append(f"### Layer {layer.layer_index}")
        lines.append("")
        lines.append(f"- Activate: `{layer.experts_to_activate}`")
        lines.append(f"- Deactivate: `{layer.experts_to_deactivate}`")
        lines.append(f"- Rationale: {layer.rationale}")
        lines.append("- Expert deltas:")
        for score in sorted(layer.scores, key=lambda item: item.delta, reverse=True):
            lines.append(
                "  "
                f"- Expert {score.expert_index}: "
                f"positive_mean={score.positive_mean:.3f}, "
                f"negative_mean={score.negative_mean:.3f}, delta={score.delta:.3f}"
            )
        lines.append("")

    lines.extend(["## Representative Tokens", ""])
    for layer_index in sorted(artifacts.selected_tokens_by_layer):
        lines.append(f"### Layer {layer_index}")
        lines.append("")
        for label in ["positive", "negative"]:
            lines.append(f"- {label.title()} examples:")
            for token in artifacts.selected_tokens_by_layer[layer_index][label]:
                lines.append(
                    "  "
                    f"- {token.prompt_id}: token='{token.token_text}', "
                    f"index={token.token_index}, attention={token.attention_weight:.3f}, "
                    f"expert_loads={token.expert_loads}"
                )
        lines.append("")

    return "\n".join(lines)
