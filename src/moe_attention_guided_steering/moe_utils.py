from statistics import fmean
from typing import Dict, List

from .config import InterventionConfig
from .types import ExpertShiftScore, LayerIntervention, SelectedToken, SteeringPlan


def compute_expert_scores_for_layer(
    layer_index: int,
    selected_tokens_by_label: Dict[str, List[SelectedToken]],
) -> List[ExpertShiftScore]:
    """Compute expert deltas for one layer.

    For each expert, we compare its mean load on the selected tokens from positive
    examples against its mean load on the selected tokens from negative examples.
    A positive delta means the expert is more associated with the target concept;
    a negative delta means the expert is more associated with the contrast concept.
    """
    positive_tokens = selected_tokens_by_label.get("positive", [])
    negative_tokens = selected_tokens_by_label.get("negative", [])
    if not positive_tokens or not negative_tokens:
        raise ValueError(
            f"Layer {layer_index} needs both positive and negative selections to score experts."
        )

    expert_count = len(positive_tokens[0].expert_loads)
    scores: List[ExpertShiftScore] = []
    for expert_index in range(expert_count):
        positive_mean = fmean(token.expert_loads[expert_index] for token in positive_tokens)
        negative_mean = fmean(token.expert_loads[expert_index] for token in negative_tokens)
        scores.append(
            ExpertShiftScore(
                layer_index=layer_index,
                expert_index=expert_index,
                positive_mean=positive_mean,
                negative_mean=negative_mean,
                delta=positive_mean - negative_mean,
            )
        )

    return scores


def build_layer_intervention(
    layer_index: int,
    scores: List[ExpertShiftScore],
    config: InterventionConfig,
) -> LayerIntervention:
    """Turn raw expert deltas into a sparse, human-readable intervention rule.

    We rank positive deltas and negative deltas separately because the research
    action is asymmetric: some experts should be strengthened while others should
    be suppressed. Sparse top-k rules are also easier to explain to collaborators
    than dense coefficient vectors in the first project version.
    """
    sorted_desc = sorted(scores, key=lambda score: score.delta, reverse=True)
    sorted_asc = sorted(scores, key=lambda score: score.delta)

    experts_to_activate = [
        score.expert_index
        for score in sorted_desc
        if score.delta >= config.activation_threshold
    ][: config.top_k_experts]

    experts_to_deactivate = [
        score.expert_index
        for score in sorted_asc
        if score.delta <= config.deactivation_threshold
    ][: config.top_k_experts]

    rationale = (
        f"Activate experts with delta >= {config.activation_threshold:.3f} and "
        f"deactivate experts with delta <= {config.deactivation_threshold:.3f}."
    )

    return LayerIntervention(
        layer_index=layer_index,
        experts_to_activate=experts_to_activate,
        experts_to_deactivate=experts_to_deactivate,
        rationale=rationale,
        scores=scores,
    )


def build_steering_plan(
    scores_by_layer: Dict[int, List[ExpertShiftScore]],
    concept: str,
    contrast_concept: str,
    config: InterventionConfig,
) -> SteeringPlan:
    """Build the complete layerwise steering plan for the experiment."""
    layers = [
        build_layer_intervention(layer_index, scores_by_layer[layer_index], config)
        for layer_index in sorted(scores_by_layer)
    ]
    return SteeringPlan(
        concept=concept,
        contrast_concept=contrast_concept,
        layers=layers,
    )
