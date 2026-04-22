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

    Numeric interpretation of the input:
    - `layer.tokens` contains `T` token rows for one layer.
    - Each token row carries:
      - one attention value, so together they form an attention vector `(T,)`
      - one expert-load vector of length `X`, so together they form an
        expert-load matrix `(T, X)`

    The operation performed here is:
    - filter rows using the attention vector,
    - choose the row with the maximum surviving attention weight,
    - return that single row's expert-load vector of shape `(X,)` together with
      its token metadata.

    Inputs:
    - `layer`: one layer worth of token candidates.
    - `minimum_attention_weight`: threshold applied to the attention vector
      before argmax selection.

    Returns:
    - `SelectedToken`: one selected token row for the layer, carrying a single
      expert-load vector `(X,)`.
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

    Numeric interpretation:
    - input example contains `L` layers,
    - each layer starts as an attention vector `(T_l,)` plus expert-load matrix
      `(T_l, X_l)`,
    - each layer is reduced to one selected expert-load vector `(X_l,)`.

    Inputs:
    - `example`: one prompt example with all of its layers and token rows.
    - `minimum_attention_weight`: shared threshold applied independently to each
      layer's attention vector before selection.

    Returns:
    - `List[SelectedToken]` of length `L`, with one selected token per layer.
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

    Numeric interpretation:
    - before grouping, each example contributes one selected expert-load vector
      `(X_l,)` per layer,
    - after grouping, each `index[layer]["positive"]` can be viewed as a matrix
      of shape `(N_positive, X_l)`,
    - similarly, each `index[layer]["negative"]` can be viewed as a matrix of
      shape `(N_negative, X_l)`.

    Inputs:
    - `examples`: all prompt examples in the dataset.
    - `minimum_attention_weight`: layerwise threshold used before each argmax
      selection step.

    Returns:
    - nested dictionary keyed by layer and label, where each value is a list of
      selected token rows that can be stacked into label-specific matrices for
      scoring.
    """
    index: Dict[int, Dict[str, List[SelectedToken]]] = defaultdict(
        lambda: {"positive": [], "negative": []}
    )

    for example in examples:
        for selected in select_tokens_for_example(example, minimum_attention_weight):
            index[selected.layer_index][selected.label].append(selected)

    return dict(index)
