from collections import defaultdict
from typing import Dict, List

from .types import LayerRecord, PromptRecord, SelectedToken


def select_representative_token(
    layer: LayerRecord,
    minimum_attention_weight: float = 0.0,
) -> SelectedToken:
    """Choose the token that best represents a layer.

    The current selection rule is intentionally transparent: among the token
    candidates that survive the minimum-attention filter, pick the one with the
    highest attention weight. If two tokens tie, prefer the earlier token index so
    the result is deterministic.

    This is the exact place where a future research iteration can swap in a more
    sophisticated rule without touching the rest of the MoE scoring code.
    """
    eligible_tokens = [
        token for token in layer.tokens if token.attention_weight >= minimum_attention_weight
    ]
    if not eligible_tokens:
        raise ValueError(
            f"Layer {layer.layer_index} has no tokens above the attention threshold {minimum_attention_weight}."
        )

    best_token = max(
        eligible_tokens,
        key=lambda token: (token.attention_weight, -token.token_index),
    )

    return SelectedToken(
        prompt_id="",
        label="",
        layer_index=layer.layer_index,
        token_index=best_token.token_index,
        token_text=best_token.token_text,
        attention_weight=best_token.attention_weight,
        expert_loads=list(best_token.expert_loads),
    )


def select_tokens_for_example(
    example: PromptRecord,
    minimum_attention_weight: float = 0.0,
) -> List[SelectedToken]:
    """Run representative-token selection for every layer in one example.

    We attach the example metadata after selection so the returned objects are fully
    self-describing. That makes downstream aggregation code simpler because it can
    work with a flat stream of `SelectedToken` records instead of carrying both the
    selected token and the parent example around separately.
    """
    selections: List[SelectedToken] = []
    for layer in example.layers:
        selected = select_representative_token(layer, minimum_attention_weight)
        selected.prompt_id = example.prompt_id
        selected.label = example.label
        selections.append(selected)
    return selections


def index_selected_tokens_by_layer_and_label(
    examples: List[PromptRecord],
    minimum_attention_weight: float = 0.0,
) -> Dict[int, Dict[str, List[SelectedToken]]]:
    """Group selected tokens by `(layer_index, label)` for MoE scoring.

    The expert-delta calculation asks a specific question: for a given layer and
    expert, how different is usage on positive examples versus negative examples?
    A nested index keyed by layer and label makes that comparison explicit and easy
    to inspect during debugging.
    """
    index: Dict[int, Dict[str, List[SelectedToken]]] = defaultdict(
        lambda: {"positive": [], "negative": []}
    )

    for example in examples:
        for selected in select_tokens_for_example(example, minimum_attention_weight):
            index[selected.layer_index][selected.label].append(selected)

    return dict(index)
