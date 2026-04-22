from dataclasses import dataclass
from typing import Dict, List

from .attention_utils import index_selected_tokens_by_layer_and_label
from .config import ExperimentConfig
from .moe_utils import build_steering_plan, compute_expert_scores_for_layer
from .types import ExpertShiftScore, ExperimentDataset, SelectedToken, SteeringPlan


@dataclass
class PipelineArtifacts:
    """Everything produced by the hybrid attention-guided MoE pipeline."""

    selected_tokens_by_layer: Dict[int, Dict[str, List[SelectedToken]]]
    scores_by_layer: Dict[int, List[ExpertShiftScore]]
    steering_plan: SteeringPlan


def run_pipeline(dataset: ExperimentDataset, config: ExperimentConfig) -> PipelineArtifacts:
    """Execute the full pipeline from selection through intervention planning.

    The orchestration function is intentionally thin. Each major step still lives in
    its own module so that collaborators can inspect selection, scoring, or output
    logic without wading through a monolithic script.

    Shape flow through the pipeline:
    1. Input dataset supplies, for each layer, an attention vector `(T_l,)` and
       expert-load matrix `(T_l, X_l)` per example.
    2. Selection reduces each layer to one expert-load row `(X_l,)` per example.
    3. Grouping by label yields implicit matrices `(N_positive, X_l)` and
       `(N_negative, X_l)`.
    4. Scoring reduces those matrices to dense delta vectors `(X_l,)`.
    5. Planning sparsifies each dense vector into small index lists of experts to
       activate or deactivate.

    Important: this function orchestrates downstream processing only. It does not
    calculate attention weights or expert loads from a live model; those values
    must already be present in `dataset`.

    Inputs:
    - `dataset`: precomputed experiment traces in typed form.
    - `config`: selection and intervention settings for the run.

    Returns:
    - `PipelineArtifacts`: selected token rows, dense per-layer score vectors,
      and the final sparse steering plan.
    """
    selected_tokens_by_layer = index_selected_tokens_by_layer_and_label(
        dataset.examples,
        minimum_attention_weight=config.selection.minimum_attention_weight,
    )

    scores_by_layer = {
        layer_index: compute_expert_scores_for_layer(layer_index, selected_tokens)
        for layer_index, selected_tokens in selected_tokens_by_layer.items()
    }

    steering_plan = build_steering_plan(
        scores_by_layer=scores_by_layer,
        concept=dataset.metadata.get("concept", "unknown"),
        contrast_concept=dataset.metadata.get("contrast_concept", "unknown"),
        config=config.intervention,
    )

    return PipelineArtifacts(
        selected_tokens_by_layer=selected_tokens_by_layer,
        scores_by_layer=scores_by_layer,
        steering_plan=steering_plan,
    )
