from dataclasses import dataclass, field


@dataclass
class SelectionConfig:
    """Configuration for representative-token selection.

    The first hybrid version of the project uses a very simple rule: choose the
    highest-attention token in each layer. Even though the rule is simple, we keep
    it in a config object because the next iteration may swap in richer policies
    such as attention-to-instruction, head-specific voting, or top-k averaging.
    """

    minimum_attention_weight: float = 0.0


@dataclass
class InterventionConfig:
    """Configuration for converting expert deltas into steering decisions.

    `top_k_experts` keeps the intervention sparse and interpretable.
    The thresholds make it explicit that small fluctuations should not be treated
    as evidence that an expert genuinely represents the target concept.
    """

    top_k_experts: int = 1
    activation_threshold: float = 0.2
    deactivation_threshold: float = -0.2


@dataclass
class ExperimentConfig:
    """Bundle all experiment controls into one object.

    This mirrors the usual research workflow: a single run configuration should be
    easy to log, serialize, and pass through the full pipeline without threading a
    long list of scalar arguments through every function call.
    """

    selection: SelectionConfig = field(default_factory=SelectionConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
