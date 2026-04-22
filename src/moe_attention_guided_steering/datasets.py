import json
from pathlib import Path
from typing import Dict, Iterable, List

from .types import ExperimentDataset, LayerRecord, PromptRecord, TokenRecord


def load_experiment(path: str) -> ExperimentDataset:
    """Load an experiment JSON file into strongly-typed dataclasses.

    The main reason to parse into dataclasses instead of passing raw dictionaries
    around is that research code gets confusing very quickly when each function has
    to remember the exact nested shape of the input payload. A typed in-memory
    representation makes the later pipeline code much easier to audit.

    Expected numeric structure:
    - examples: `E`
    - layers per example: `L`
    - candidate tokens per layer: `T_l`
    - experts per token in layer `l`: `X_l`

    In matrix terms, each layer contributes:
    - an attention vector of shape `(T_l,)`
    - an expert-load matrix of shape `(T_l, X_l)`

    Important: this function does not calculate attention. It reads already
    computed `attention_weight` scalars and `expert_loads` vectors from the input
    JSON file and converts them into typed Python records.

    Inputs:
    - `path`: location of the JSON file containing precomputed traces.

    Returns:
    - `ExperimentDataset`: nested dataclasses representing the full
      `[example][layer][token][expert]` structure.
    """
    payload = json.loads(Path(path).read_text())
    examples: List[PromptRecord] = []

    for raw_example in payload.get("examples", []):
        layers: List[LayerRecord] = []
        for raw_layer in raw_example.get("layers", []):
            tokens = [
                TokenRecord(
                    token_index=raw_token["token_index"],
                    token_text=raw_token["token_text"],
                    attention_weight=raw_token["attention_weight"],
                    expert_loads=list(raw_token["expert_loads"]),
                )
                for raw_token in raw_layer.get("tokens", [])
            ]
            layers.append(LayerRecord(layer_index=raw_layer["layer_index"], tokens=tokens))

        examples.append(
            PromptRecord(
                prompt_id=raw_example["prompt_id"],
                label=raw_example["label"],
                prompt_text=raw_example["prompt_text"],
                layers=layers,
            )
        )

    dataset = ExperimentDataset(metadata=payload.get("metadata", {}), examples=examples)
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: ExperimentDataset) -> None:
    """Reject malformed experiment files early.

    This validation is intentionally opinionated because dataset bugs are among the
    most expensive research bugs to debug later. We check the assumptions that the
    scoring code relies on:

    - at least one positive and one negative example exist,
    - every example exposes the same layer indices,
    - every token in the same layer uses the same number of experts.

    Shape assumptions being checked:
    - every example follows the same layer axis `L`,
    - within one layer, the expert-load matrix has a consistent width `X`,
    - every layer has at least one token row, so the attention vector `(T,)` and
      expert-load matrix `(T, X)` are both non-empty.

    Inputs:
    - `dataset`: parsed experiment structure whose nested records imply the
      numeric shapes used later by selection and scoring.

    Returns:
    - `None`. The function raises `ValueError` if any shape or labeling
      assumption needed by the pipeline is violated.
    """
    if not dataset.examples:
        raise ValueError("Dataset must contain at least one example.")

    labels = {example.label for example in dataset.examples}
    if labels != {"negative", "positive"}:
        raise ValueError(
            "Dataset labels must be exactly {'positive', 'negative'} for contrastive scoring."
        )

    reference_layers = [layer.layer_index for layer in dataset.examples[0].layers]
    if not reference_layers:
        raise ValueError("Every example must contain at least one layer.")

    for example in dataset.examples:
        layer_indices = [layer.layer_index for layer in example.layers]
        if layer_indices != reference_layers:
            raise ValueError(
                f"Example {example.prompt_id} uses layer indices {layer_indices}, expected {reference_layers}."
            )

        for layer in example.layers:
            if not layer.tokens:
                raise ValueError(
                    f"Example {example.prompt_id}, layer {layer.layer_index} has no token candidates."
                )

            expert_count = len(layer.tokens[0].expert_loads)
            if expert_count == 0:
                raise ValueError(
                    f"Example {example.prompt_id}, layer {layer.layer_index} has empty expert loads."
                )

            for token in layer.tokens:
                if len(token.expert_loads) != expert_count:
                    raise ValueError(
                        f"Layer {layer.layer_index} in example {example.prompt_id} mixes expert vector sizes."
                    )


def summarize_dataset(dataset: ExperimentDataset) -> Dict[str, int]:
    """Return a compact summary for CLI printing and reports."""
    labels = [example.label for example in dataset.examples]
    return {
        "num_examples": len(dataset.examples),
        "num_positive_examples": labels.count("positive"),
        "num_negative_examples": labels.count("negative"),
        "num_layers": len(dataset.examples[0].layers),
        "num_experts_per_layer": len(dataset.examples[0].layers[0].tokens[0].expert_loads),
    }


def iter_examples_by_label(dataset: ExperimentDataset, label: str) -> Iterable[PromptRecord]:
    """Yield examples with a specific label.

    Keeping this helper tiny may feel unnecessary today, but it documents the
    expected labels and gives us a single place to evolve if we later add splits,
    concept groups, or prompt metadata filters.
    """
    for example in dataset.examples:
        if example.label == label:
            yield example
