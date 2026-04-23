from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .attention_collection import HFModelResources
from .config import ExperimentConfig, InterventionConfig
from .datasets import validate_dataset
from .manual_review import ManualReviewPlan
from .types import ExperimentDataset, LayerRecord, PromptRecord, SelectedToken, TokenRecord, SteeringPlan
from .upstream_prompt_datasets import StatementPromptPair


@dataclass
class OLMoERouterTrace:
    """Per-prompt OLMoE router probabilities on the shared suffix candidates.

    Shapes:
    - chat-formatted `input_ids`: `(1, S)`
    - Hugging Face `outputs.router_logits[layer]`: `(B * S, E)` for OLMoE
    - reshaped per-layer router logits: `(B, S, E)`
    - candidate suffix slice for one prompt: `(N, E)`
    - full `layer_router_probabilities`: `(L, N, E)`

    where:
    - `B` is the batch size. The first implementation assumes `B = 1`.
    - `S` is the tokenized prompt length after applying the chat template.
    - `L` is the number of decoder layers in OLMoE.
    - `N` is the number of shared candidate suffix tokens at the end of the
      prompt, matching the attention-guided token-selection stage.
    - `E` is the number of experts in each OLMoE sparse layer.

    Meaning:
    - Each `layer_router_probabilities[layer][token][expert]` value is the
      router probability assigned to one expert for one candidate suffix token.
    - These are the MoE-side features that later become the expert-load vectors
      in the generic steering pipeline.
    """

    prompt_id: str
    label: str
    prompt_text: str
    candidate_token_texts: List[str]
    candidate_token_relative_indices: List[int]
    layer_router_probabilities: List[List[List[float]]]


@dataclass
class SteeringConditionArtifacts:
    """All steering artifacts for one condition and one concept.

    Shapes:
    - `dataset.examples[e].layers[l].tokens[t].expert_loads`: `(E,)`
    - `selected_tokens_by_layer[layer][label]`: implicit matrices
      `(N_positive, E)` and `(N_negative, E)` after selection
    - `steering_plan.layers[layer].scores`: dense delta vector `(E,)`

    Meaning:
    - `dataset` stores the collected OLMoE router probabilities in the generic
      experiment schema.
    - `selected_tokens_by_layer` shows which candidate suffix token row was used
      in each positive/negative example.
    - `steering_plan` is the sparse intervention recommendation derived from the
      dense expert-delta vectors.
    """

    dataset: ExperimentDataset
    selected_tokens_by_layer: Dict[int, Dict[str, List[SelectedToken]]]
    steering_plan: SteeringPlan


def _flatten_token_ids(token_ids: Any) -> List[int]:
    """Convert tokenizer output ids into a flat Python list.

    Inputs:
    - `token_ids`: tokenizer output that may be:
      - a tensor shaped `(1, S)`,
      - a nested list shaped `(1, S)`,
      - a mapping containing `input_ids`.

    Returns:
    - `List[int]` of length `S`.
    """
    if isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    elif hasattr(token_ids, "keys") and "input_ids" in token_ids:
        token_ids = token_ids["input_ids"]
    elif hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _normalize_model_inputs(encoded_inputs: Any) -> Dict[str, Any]:
    """Convert tokenizer outputs into a plain mapping for Hugging Face models.

    Output shape meaning:
    - `input_ids`: `(1, S)`
    - optional `attention_mask`: `(1, S)`
    """
    if isinstance(encoded_inputs, dict):
        normalized = dict(encoded_inputs)
    elif hasattr(encoded_inputs, "keys"):
        normalized = {key: encoded_inputs[key] for key in encoded_inputs.keys()}
    else:
        normalized = {"input_ids": encoded_inputs}

    if "input_ids" not in normalized:
        raise KeyError("Tokenizer output must contain input_ids.")
    return normalized


def _reshape_router_logits_for_prompt(
    layer_router_logits: Any,
    batch_size: int,
    sequence_length: int,
) -> Any:
    """Recover `(B, S, E)` router tensors from OLMoE's flattened layer outputs.

    Inputs:
    - `layer_router_logits`: one per-layer router tensor from Hugging Face.
      OLMoE commonly returns this in flattened shape `(B * S, E)`.
    - `batch_size`: `B` from the tokenized model input.
    - `sequence_length`: `S` from the tokenized model input.

    Returns:
    - tensor-like object with shape `(B, S, E)`.

    Raises:
    - `ValueError` if the returned router tensor does not match the expected OLMoE
      layout.
    """
    shape = tuple(layer_router_logits.shape)
    if len(shape) == 2 and shape[0] == batch_size * sequence_length:
        return layer_router_logits.view(batch_size, sequence_length, shape[1])
    if len(shape) == 3 and shape[0] == batch_size and shape[1] == sequence_length:
        return layer_router_logits

    raise ValueError(
        "Expected router logits with shape (B * S, E) or (B, S, E), "
        f"got {shape} for batch_size={batch_size}, sequence_length={sequence_length}."
    )


def build_fixed_layer_to_token_index(
    num_layers: int,
    fixed_token_index: int = -1,
) -> Dict[int, int]:
    """Create the original MoESteer-style fixed token map.

    Inputs:
    - `num_layers`: number of decoder layers `L`.
    - `fixed_token_index`: shared negative token index used for every layer.
      `-1` means "the final shared suffix token", which is the blind fixed-token
      baseline that attention-guided steering is meant to improve upon.

    Returns:
    - `Dict[int, int]` with `L` entries, mapping every layer to the same selected
      candidate suffix token index.
    """
    return {layer_index: fixed_token_index for layer_index in range(num_layers)}


def load_layer_to_token_index(path: str) -> Dict[int, int]:
    """Load a JSON layer-to-token map and normalize string keys into ints."""
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text())
    return {int(layer_index): int(token_index) for layer_index, token_index in payload.items()}


def collect_olmoe_router_trace_for_prompt(
    prompt_id: str,
    label: str,
    prompt_text: str,
    resources: HFModelResources,
) -> OLMoERouterTrace:
    """Run one chat prompt through OLMoE and collect suffix-token router loads.

    Shape flow:
    1. Apply the model's chat template and tokenize the prompt:
       - `input_ids`: `(1, S)`
    2. Ask OLMoE to return per-layer router logits:
       - raw layer tensor: `(1 * S, E)` or `(1, S, E)`
    3. Reshape each layer to `(1, S, E)` and keep the last shared suffix tokens:
       - `(1, N, E)`
    4. Drop the singleton batch dimension and convert to probabilities:
       - `(N, E)`
    5. Stack over layers conceptually:
       - `(L, N, E)`

    Inputs:
    - `prompt_id`: stable identifier used for trace files and datasets.
    - `label`: `positive` or `negative`.
    - `prompt_text`: raw user-facing text inserted into the chat template.
    - `resources`: loaded OLMoE model, tokenizer, and shared token metadata.

    Returns:
    - `OLMoERouterTrace` containing one router-probability vector of length `E`
      for every `(layer, candidate suffix token)` pair.
    """
    import torch

    tokenizer = resources.tokenizer
    model = resources.model

    encoded_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(encoded_inputs, "to"):
        encoded_inputs = encoded_inputs.to(model.device)
    encoded_inputs = _normalize_model_inputs(encoded_inputs)

    batch_size, sequence_length = encoded_inputs["input_ids"].shape
    if batch_size != 1:
        raise ValueError("The first OLMoE collector only supports batch_size=1.")

    model_kwargs: Dict[str, Any] = {
        "input_ids": encoded_inputs["input_ids"],
        "output_router_logits": True,
        "return_dict": True,
        "use_cache": False,
    }
    if "attention_mask" in encoded_inputs:
        model_kwargs["attention_mask"] = encoded_inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(**model_kwargs)

    input_ids = _flatten_token_ids(encoded_inputs["input_ids"])
    candidate_token_ids = input_ids[-resources.num_candidate_suffix_tokens :]
    candidate_token_texts = tokenizer.convert_ids_to_tokens(candidate_token_ids)
    candidate_token_relative_indices = list(range(-resources.num_candidate_suffix_tokens, 0))

    layer_router_probabilities: List[List[List[float]]] = []
    for layer_router_logits in outputs.router_logits:
        reshaped = _reshape_router_logits_for_prompt(
            layer_router_logits=layer_router_logits,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        candidate_router_logits = reshaped[0, -resources.num_candidate_suffix_tokens :, :]
        candidate_router_probabilities = torch.softmax(candidate_router_logits, dim=-1)
        layer_router_probabilities.append(
            candidate_router_probabilities.detach().cpu().tolist()
        )

    return OLMoERouterTrace(
        prompt_id=prompt_id,
        label=label,
        prompt_text=prompt_text,
        candidate_token_texts=list(candidate_token_texts),
        candidate_token_relative_indices=candidate_token_relative_indices,
        layer_router_probabilities=layer_router_probabilities,
    )


def build_selection_scores_for_candidates(
    candidate_token_relative_indices: Sequence[int],
    selected_token_index: int,
) -> List[float]:
    """Encode one externally chosen token position as an argmax-ready score row.

    The generic pipeline currently expects each token row to have an
    `attention_weight` and then chooses the maximum-weight token in each layer.
    For live OLMoE traces, the selection policy is already known externally:

    - original MoESteer: use one fixed suffix token for every layer,
    - attention-guided MoESteer: use the per-layer token from `layer_to_token.json`.

    Rather than rewriting the downstream selection code, we encode that choice as
    a one-hot selector over the shared suffix candidate tokens.

    Inputs:
    - `candidate_token_relative_indices`: list of the candidate suffix token
      indices, usually `[-N, ..., -1]`.
    - `selected_token_index`: the chosen negative token index for this layer.

    Returns:
    - `List[float]` of length `N` with a single `1.0` at the chosen token and
      `0.0` elsewhere.
    """
    if selected_token_index not in candidate_token_relative_indices:
        available = ", ".join(str(index) for index in candidate_token_relative_indices)
        raise ValueError(
            f"Selected token index {selected_token_index} is not in the candidate set [{available}]."
        )
    return [
        1.0 if candidate_index == selected_token_index else 0.0
        for candidate_index in candidate_token_relative_indices
    ]


def build_experiment_dataset_from_router_traces(
    prompt_traces: Sequence[OLMoERouterTrace],
    concept: str,
    contrast_concept: str,
    concept_type: str,
    model_id: str,
    model_tag: str,
    selection_layer_to_token_index: Dict[int, int],
    selection_source: str,
) -> ExperimentDataset:
    """Convert OLMoE router traces into the repo's generic experiment schema.

    Shape mapping:
    - input `prompt_traces[p].layer_router_probabilities[layer]`: `(N, E)`
    - output `dataset.examples[p].layers[layer].tokens[token].expert_loads`: `(E,)`
    - output `attention_weight`: scalar one-hot selection score used only for the
      argmax token-selection step in the generic pipeline

    Inputs:
    - `prompt_traces`: collected positive and negative OLMoE router traces.
    - `concept`: target concept associated with positive prompts.
    - `contrast_concept`: label for the negative control prompts.
    - `concept_type`: family such as `fears`.
    - `model_id`: Hugging Face model identifier.
    - `model_tag`: filesystem-safe model short name.
    - `selection_layer_to_token_index`: one selected negative token index per
      layer, e.g. fixed `-1` or the attention-guided map.
    - `selection_source`: human-readable label such as `fixed_last_token` or
      `attention_guided`.

    Returns:
    - `ExperimentDataset` whose expert-load vectors can flow through the existing
      model-agnostic selection, scoring, and steering-plan code.
    """
    if not prompt_traces:
        raise ValueError("prompt_traces must be non-empty.")

    examples: List[PromptRecord] = []
    for trace in prompt_traces:
        layers: List[LayerRecord] = []
        for layer_index, token_expert_matrix in enumerate(trace.layer_router_probabilities):
            selected_token_index = selection_layer_to_token_index[layer_index]
            selection_scores = build_selection_scores_for_candidates(
                candidate_token_relative_indices=trace.candidate_token_relative_indices,
                selected_token_index=selected_token_index,
            )

            tokens = [
                TokenRecord(
                    token_index=trace.candidate_token_relative_indices[token_position],
                    token_text=trace.candidate_token_texts[token_position],
                    attention_weight=selection_scores[token_position],
                    expert_loads=[float(value) for value in token_expert_matrix[token_position]],
                )
                for token_position in range(len(trace.candidate_token_texts))
            ]
            layers.append(LayerRecord(layer_index=layer_index, tokens=tokens))

        examples.append(
            PromptRecord(
                prompt_id=trace.prompt_id,
                label=trace.label,
                prompt_text=trace.prompt_text,
                layers=layers,
            )
        )

    dataset = ExperimentDataset(
        metadata={
            "concept": concept,
            "contrast_concept": contrast_concept,
            "concept_type": concept_type,
            "model_id": model_id,
            "model_tag": model_tag,
            "selection_source": selection_source,
        },
        examples=examples,
    )
    validate_dataset(dataset)
    return dataset


def collect_olmoe_experiment_dataset_for_prompt_pairs(
    prompt_pairs: Sequence[StatementPromptPair],
    resources: HFModelResources,
    selection_layer_to_token_index: Dict[int, int],
    selection_source: str,
    contrast_concept: str = "generic_statement_control",
) -> ExperimentDataset:
    """Collect positive/negative OLMoE traces and adapt them into one dataset.

    Inputs:
    - `prompt_pairs`: upstream-style concept prompt pairs.
    - `resources`: loaded OLMoE model and tokenizer.
    - `selection_layer_to_token_index`: one chosen candidate suffix token per
      layer.
    - `selection_source`: short label describing how those token choices were
      obtained.
    - `contrast_concept`: textual label for the negative, unprefixed prompt set.

    Returns:
    - `ExperimentDataset` with two labels:
      - `positive`: prefixed concept prompts
      - `negative`: unprefixed statement controls
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback only used on minimal envs
        tqdm = lambda iterable, **_: iterable

    prompt_traces = collect_olmoe_router_traces_for_prompt_pairs(
        prompt_pairs=prompt_pairs,
        resources=resources,
    )

    return build_experiment_dataset_from_router_traces(
        prompt_traces=prompt_traces,
        concept=prompt_pairs[0].concept_value,
        contrast_concept=contrast_concept,
        concept_type=prompt_pairs[0].concept_type,
        model_id=resources.model_id,
        model_tag=resources.model_tag,
        selection_layer_to_token_index=selection_layer_to_token_index,
        selection_source=selection_source,
    )


def collect_olmoe_router_traces_for_prompt_pairs(
    prompt_pairs: Sequence[StatementPromptPair],
    resources: HFModelResources,
) -> List[OLMoERouterTrace]:
    """Collect positive and negative OLMoE router traces once per prompt pair.

    Returns:
    - `List[OLMoERouterTrace]` with length `2 * len(prompt_pairs)`, ordered as:
      - positive trace for pair 0
      - negative trace for pair 0
      - positive trace for pair 1
      - negative trace for pair 1
      - ...
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback only used on minimal envs
        tqdm = lambda iterable, **_: iterable

    prompt_traces: List[OLMoERouterTrace] = []
    for prompt_pair in tqdm(prompt_pairs, desc=f"Collecting OLMoE router traces for {prompt_pairs[0].concept_value}"):
        prompt_traces.append(
            collect_olmoe_router_trace_for_prompt(
                prompt_id=f"{prompt_pair.prompt_id}:positive",
                label="positive",
                prompt_text=prompt_pair.positive_full_prompt,
                resources=resources,
            )
        )
        prompt_traces.append(
            collect_olmoe_router_trace_for_prompt(
                prompt_id=f"{prompt_pair.prompt_id}:negative",
                label="negative",
                prompt_text=prompt_pair.negative_full_prompt,
                resources=resources,
            )
        )
    return prompt_traces


def run_olmoe_steering_pipeline(
    dataset: ExperimentDataset,
    config: Optional[ExperimentConfig] = None,
) -> SteeringConditionArtifacts:
    """Run the generic steering pipeline on a live-collected OLMoE dataset.

    Inputs:
    - `dataset`: OLMoE-derived experiment dataset with:
      - labels `positive` and `negative`,
      - one-hot selection scores over suffix candidates,
      - router-probability vectors of shape `(E,)` in `expert_loads`.
    - `config`: thresholds and top-k settings. If omitted, a sparse but
      probability-scale-friendly default is used.

    Returns:
    - `SteeringConditionArtifacts`: selected token rows plus the resulting sparse
      expert intervention plan.
    """
    from .pipeline import run_pipeline

    if config is None:
        config = ExperimentConfig(
            intervention=InterventionConfig(
                top_k_experts=4,
                activation_threshold=0.002,
                deactivation_threshold=-0.002,
            )
        )

    artifacts = run_pipeline(dataset, config)
    return SteeringConditionArtifacts(
        dataset=dataset,
        selected_tokens_by_layer=artifacts.selected_tokens_by_layer,
        steering_plan=artifacts.steering_plan,
    )


def steering_plan_to_router_bias_by_layer(
    steering_plan: SteeringPlan,
    coefficient: float = 8.0,
) -> Dict[int, List[float]]:
    """Convert a sparse steering plan into dense router-logit bias vectors.

    Shapes:
    - input per-layer sparse expert sets:
      - `experts_to_activate`: length `<= K`
      - `experts_to_deactivate`: length `<= K`
    - output per-layer dense bias vector:
      - `(E,)`

    Meaning:
    - Each activated expert receives `+coefficient`.
    - Each deactivated expert receives `-coefficient`.
    - Every untouched expert receives `0.0`.

    This sign-based bias is intentionally simple for the first runnable OLMoE
    backend. It keeps the runtime intervention easy to explain while preserving
    the original sparse expert choices from the steering plan.
    """
    bias_by_layer: Dict[int, List[float]] = {}
    for layer in steering_plan.layers:
        if layer.scores:
            num_experts = max(score.expert_index for score in layer.scores) + 1
        else:
            num_experts = 1 + max(
                layer.experts_to_activate + layer.experts_to_deactivate + [0]
            )

        bias_vector = [0.0] * num_experts
        for expert_index in layer.experts_to_activate:
            bias_vector[expert_index] += coefficient
        for expert_index in layer.experts_to_deactivate:
            bias_vector[expert_index] -= coefficient
        bias_by_layer[layer.layer_index] = bias_vector
    return bias_by_layer


def _apply_bias_to_last_router_row(router_logits: Any, bias_vector: Any) -> Any:
    """Add a dense expert bias vector to the final token row of router logits.

    Inputs:
    - `router_logits`: `(T, E)` for one forward call through one layer's gate.
      During generation with batch size `1`, `T` is the number of token rows
      processed in that call:
      - prompt prefill: `T = S_prompt`
      - cached decode step: `T = 1`
    - `bias_vector`: `(E,)`

    Returns:
    - new router-logit tensor with the bias added only to row `T - 1`.
    """
    router_logits = router_logits.clone()
    router_logits[-1, :] = router_logits[-1, :] + bias_vector
    return router_logits


@contextmanager
def olmoe_router_bias_hooks(
    model: Any,
    bias_by_layer: Dict[int, Sequence[float]],
) -> Iterator[None]:
    """Temporarily inject router-logit biases into OLMoE's sparse blocks.

    The hook point is `model.model.layers[layer_index].mlp.gate`, which is the
    linear gate producing per-expert router logits before top-k routing.

    Important runtime assumption:
    - generation is run with batch size `1`
    - we bias only the *last* token row of each forward pass, because that is the
      token whose hidden state determines the next autoregressive step

    Inputs:
    - `model`: `OlmoeForCausalLM` or another object exposing the same layer path.
    - `bias_by_layer`: dense router-logit bias vectors of shape `(E,)` per layer.
    """
    import torch

    handles = []

    for layer_index, bias_values in bias_by_layer.items():
        gate_module = model.model.layers[layer_index].mlp.gate
        bias_tensor = torch.tensor(list(bias_values), dtype=torch.float32)

        def hook(module: Any, inputs: Any, output: Any, bias_tensor: Any = bias_tensor) -> Any:
            return _apply_bias_to_last_router_row(
                router_logits=output,
                bias_vector=bias_tensor.to(device=output.device, dtype=output.dtype),
            )

        handles.append(gate_module.register_forward_hook(hook))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def generate_with_olmoe_steering(
    prompt_text: str,
    resources: HFModelResources,
    bias_by_layer: Optional[Dict[int, Sequence[float]]] = None,
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """Generate one response from OLMoE with optional router steering.

    Shape flow:
    - tokenized prompt: `(1, S_prompt)`
    - generated ids returned by Hugging Face: `(1, S_prompt + S_new)`
    - decoded output slice: tokens `S_prompt : S_prompt + S_new`

    Inputs:
    - `prompt_text`: user-visible question or request.
    - `resources`: loaded OLMoE model and tokenizer.
    - `bias_by_layer`: optional dense expert bias vectors `(E,)` per layer.
    - `max_new_tokens`: generation budget for the answer.
    - `temperature`: `0.0` means greedy decoding.
    - `top_p`: nucleus-sampling cutoff used only when `temperature > 0`.

    Returns:
    - decoded assistant continuation with surrounding whitespace stripped.
    """
    import torch

    tokenizer = resources.tokenizer
    model = resources.model

    encoded_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(encoded_inputs, "to"):
        encoded_inputs = encoded_inputs.to(model.device)
    encoded_inputs = _normalize_model_inputs(encoded_inputs)

    generate_kwargs: Dict[str, Any] = {
        "input_ids": encoded_inputs["input_ids"],
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if "attention_mask" in encoded_inputs:
        generate_kwargs["attention_mask"] = encoded_inputs["attention_mask"]

    if temperature > 0:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        if bias_by_layer:
            with olmoe_router_bias_hooks(model=model, bias_by_layer=bias_by_layer):
                generated_ids = model.generate(**generate_kwargs)
        else:
            generated_ids = model.generate(**generate_kwargs)

    prompt_length = encoded_inputs["input_ids"].shape[1]
    response_ids = generated_ids[0, prompt_length:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def fill_manual_review_plan_with_olmoe_generations(
    plan: ManualReviewPlan,
    resources: HFModelResources,
    moesteer_plans_by_concept: Dict[str, SteeringPlan],
    attention_guided_plans_by_concept: Dict[str, SteeringPlan],
    steering_coefficient: float = 8.0,
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> ManualReviewPlan:
    """Populate a manual review plan with baseline and two steered generations.

    Inputs:
    - `plan`: manual review grid whose `evaluation_question` field supplies the
      neutral prompt asked to the model.
    - `resources`: loaded OLMoE model and tokenizer.
    - `moesteer_plans_by_concept`: sparse steering plans computed with the fixed
      token-selection rule.
    - `attention_guided_plans_by_concept`: sparse steering plans computed with
      the attention-selected token rule.
    - `steering_coefficient`: dense router-logit bias magnitude used at runtime.
    - generation controls: `max_new_tokens`, `temperature`, `top_p`.

    Returns:
    - the same `ManualReviewPlan` instance with response fields filled in.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback only used on minimal envs
        tqdm = lambda iterable, **_: iterable

    baseline_cache: Dict[str, str] = {}
    moesteer_bias_cache = {
        concept: steering_plan_to_router_bias_by_layer(steering_plan, coefficient=steering_coefficient)
        for concept, steering_plan in moesteer_plans_by_concept.items()
    }
    attention_bias_cache = {
        concept: steering_plan_to_router_bias_by_layer(steering_plan, coefficient=steering_coefficient)
        for concept, steering_plan in attention_guided_plans_by_concept.items()
    }

    for case in tqdm(plan.cases, desc="Generating qualitative review responses"):
        if case.evaluation_question not in baseline_cache:
            baseline_cache[case.evaluation_question] = generate_with_olmoe_steering(
                prompt_text=case.evaluation_question,
                resources=resources,
                bias_by_layer=None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        case.baseline_response = baseline_cache[case.evaluation_question]
        case.moesteer_response = generate_with_olmoe_steering(
            prompt_text=case.evaluation_question,
            resources=resources,
            bias_by_layer=moesteer_bias_cache[case.concept],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        case.attention_guided_moesteer_response = generate_with_olmoe_steering(
            prompt_text=case.evaluation_question,
            resources=resources,
            bias_by_layer=attention_bias_cache[case.concept],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    return plan
