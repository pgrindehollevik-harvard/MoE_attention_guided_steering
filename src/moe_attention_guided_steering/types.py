from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TokenRecord:
    """One token candidate inside a specific layer.

    `attention_weight` is the signal we use to decide whether this token is the
    most representative position for the layer. `expert_loads` is a vector whose
    length equals the number of experts in the MoE block for that layer.
    """

    token_index: int
    token_text: str
    attention_weight: float
    expert_loads: List[float]


@dataclass
class LayerRecord:
    """All token candidates available for one layer in one example."""

    layer_index: int
    tokens: List[TokenRecord]


@dataclass
class PromptRecord:
    """A single positive or negative example in the contrastive dataset."""

    prompt_id: str
    label: str
    prompt_text: str
    layers: List[LayerRecord]


@dataclass
class ExperimentDataset:
    """Top-level container for one steering experiment specification."""

    metadata: Dict[str, str]
    examples: List[PromptRecord]


@dataclass
class SelectedToken:
    """The token chosen to represent a layer for one example.

    We keep a copy of `expert_loads` here because downstream scoring is based only
    on the selected token, not on every candidate token that was available before
    the attention-based selection step.
    """

    prompt_id: str
    label: str
    layer_index: int
    token_index: int
    token_text: str
    attention_weight: float
    expert_loads: List[float]


@dataclass
class ExpertShiftScore:
    """Difference in expert usage between positive and negative examples."""

    layer_index: int
    expert_index: int
    positive_mean: float
    negative_mean: float
    delta: float


@dataclass
class LayerIntervention:
    """Recommended expert interventions for one layer."""

    layer_index: int
    experts_to_activate: List[int]
    experts_to_deactivate: List[int]
    rationale: str
    scores: List[ExpertShiftScore] = field(default_factory=list)


@dataclass
class SteeringPlan:
    """Full steering plan covering every layer in the experiment."""

    concept: str
    contrast_concept: str
    layers: List[LayerIntervention]
