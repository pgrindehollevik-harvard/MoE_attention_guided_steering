from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .attention_collection import HFModelResources, compute_inserted_token_span_from_ids
from .config import ExperimentConfig, InterventionConfig
from .datasets import validate_dataset
from .manual_review import ManualReviewPlan
from .types import ExperimentDataset, LayerRecord, PromptRecord, SteeringPlan, TokenRecord
from .upstream_prompt_datasets import (
    StatementPromptPair,
    build_concept_conditioned_evaluation_prompt,
)


@dataclass
class OLMoESpanActivationTrace:
    """Per-prompt OLMoE expert activation rates over a meaningful token span.

    Shapes:
    - chat-formatted `input_ids`: `(1, S)`
    - raw per-layer OLMoE router logits: `(B * S, E)` or `(B, S, E)`
    - reshaped per-layer router logits: `(B, S, E)`
    - chosen readout span for one prompt: `(P, E)`
    - top-k routed-expert indicator matrix over the span: `(P, E)`
    - per-layer activation-rate vector after averaging over span tokens: `(E,)`
    - full `layer_expert_activation_rates`: `(L, E)`

    where:
    - `B` is the batch size. The first implementation assumes `B = 1`.
    - `S` is the full sequence length after applying the chat template.
    - `P` is the number of tokens inside the readout span.
    - `L` is the number of sparse decoder layers in OLMoE.
    - `E` is the number of experts in each sparse layer.

    Meaning:
    - `layer_expert_activation_rates[layer][expert]` is the fraction of tokens in
      the chosen readout span whose top-k router decision included that expert.
    - This mirrors the paper's "is the expert routed to for tokens in the span?"
      logic more closely than the earlier single-token prototype.
    """

    prompt_id: str
    label: str
    prompt_text: str
    span_name: str
    span_token_start: int
    span_token_end: int
    span_token_texts: List[str]
    top_k_experts_per_token: int
    layer_expert_activation_rates: List[List[float]]


@dataclass
class SteeringConditionArtifacts:
    """Reusable steering artifacts for one concept under one readout rule.

    Shapes:
    - `dataset.examples[e].layers[l].tokens[0].expert_loads`: `(E,)`
    - `steering_plan.layers[layer].scores`: dense expert-delta vector `(E,)`

    Meaning:
    - the dataset stores one span-aggregated expert-activation vector per layer
      and example,
    - the steering plan sparsifies those dense per-layer vectors into a small
      list of experts to favor or suppress at generation time.
    """

    dataset: ExperimentDataset
    steering_plan: SteeringPlan


def _flatten_token_ids(token_ids: Any) -> List[int]:
    """Convert tokenizer output ids into a flat Python list."""
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
    """Convert tokenizer outputs into a plain mapping for Hugging Face models."""
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


def find_user_content_token_span(tokenizer: Any, prompt_text: str) -> tuple[int, int]:
    """Recover the chat-token span occupied by the user's literal prompt text.

    Inputs:
    - `prompt_text`: raw user-visible prompt text, before chat templating.

    Returns:
    - `(start, end)` token indices for the user-content span inside the
      chat-formatted input sequence.

    Method:
    - tokenize the normal chat-formatted prompt,
    - tokenize the same chat template with an empty user message,
    - compute the exact inserted token interval.
    """
    full_chat = [{"role": "user", "content": prompt_text}]
    empty_chat = [{"role": "user", "content": ""}]

    full_ids = _flatten_token_ids(
        tokenizer.apply_chat_template(
            full_chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    )
    empty_ids = _flatten_token_ids(
        tokenizer.apply_chat_template(
            empty_chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    )
    return compute_inserted_token_span_from_ids(full_ids, empty_ids)


def collect_olmoe_span_activation_trace_for_prompt(
    prompt_id: str,
    label: str,
    prompt_text: str,
    resources: HFModelResources,
    span_name: str = "user_content",
) -> OLMoESpanActivationTrace:
    """Run one prompt through OLMoE and summarize routing over a token span.

    Shape flow:
    1. Apply the model's chat template and tokenize the prompt:
       - `input_ids`: `(1, S)`
    2. Ask OLMoE to return per-layer router logits:
       - raw layer tensor: `(1 * S, E)` or `(1, S, E)`
    3. Reshape each layer to `(1, S, E)` and slice the chosen readout span:
       - `(P, E)`
    4. For each token row, mark the top-k routed experts:
       - indicator matrix `(P, E)`
    5. Average those indicators over the span tokens:
       - activation-rate vector `(E,)`
    6. Stack over layers conceptually:
       - `(L, E)`

    Returns:
    - `OLMoESpanActivationTrace` with one activation-rate vector per layer.
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
        raise ValueError("The OLMoE collector currently supports batch_size=1 only.")

    span_start, span_end = find_user_content_token_span(tokenizer, prompt_text)
    input_ids = _flatten_token_ids(encoded_inputs["input_ids"])
    span_token_ids = input_ids[span_start:span_end]
    span_token_texts = tokenizer.convert_ids_to_tokens(span_token_ids)

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

    top_k_experts_per_token = int(getattr(model.config, "num_experts_per_tok", 1))
    layer_expert_activation_rates: List[List[float]] = []
    for layer_router_logits in outputs.router_logits:
        reshaped = _reshape_router_logits_for_prompt(
            layer_router_logits=layer_router_logits,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        span_router_logits = reshaped[0, span_start:span_end, :]
        top_k_indices = torch.topk(span_router_logits, k=top_k_experts_per_token, dim=-1).indices
        activation_indicator = torch.zeros_like(span_router_logits, dtype=torch.float32)
        activation_indicator.scatter_(dim=-1, index=top_k_indices, value=1.0)
        activation_rates = activation_indicator.mean(dim=0)
        layer_expert_activation_rates.append(activation_rates.detach().cpu().tolist())

    return OLMoESpanActivationTrace(
        prompt_id=prompt_id,
        label=label,
        prompt_text=prompt_text,
        span_name=span_name,
        span_token_start=span_start,
        span_token_end=span_end,
        span_token_texts=list(span_token_texts),
        top_k_experts_per_token=top_k_experts_per_token,
        layer_expert_activation_rates=layer_expert_activation_rates,
    )


def collect_olmoe_span_activation_traces_for_prompt_pairs(
    prompt_pairs: Sequence[StatementPromptPair],
    resources: HFModelResources,
    span_name: str = "user_content",
) -> List[OLMoESpanActivationTrace]:
    """Collect positive and negative span activation traces once per prompt pair.

    Returns:
    - `List[OLMoESpanActivationTrace]` with length `2 * len(prompt_pairs)`,
      ordered as positive trace, negative trace, positive trace, negative trace.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback only used on minimal envs
        tqdm = lambda iterable, **_: iterable

    prompt_traces: List[OLMoESpanActivationTrace] = []
    concept_name = prompt_pairs[0].concept_value if prompt_pairs else "unknown"
    for prompt_pair in tqdm(prompt_pairs, desc=f"Collecting OLMoE span traces for {concept_name}"):
        prompt_traces.append(
            collect_olmoe_span_activation_trace_for_prompt(
                prompt_id=f"{prompt_pair.prompt_id}:positive",
                label="positive",
                prompt_text=prompt_pair.positive_full_prompt,
                resources=resources,
                span_name=span_name,
            )
        )
        prompt_traces.append(
            collect_olmoe_span_activation_trace_for_prompt(
                prompt_id=f"{prompt_pair.prompt_id}:negative",
                label="negative",
                prompt_text=prompt_pair.negative_full_prompt,
                resources=resources,
                span_name=span_name,
            )
        )
    return prompt_traces


def build_experiment_dataset_from_span_traces(
    prompt_traces: Sequence[OLMoESpanActivationTrace],
    concept: str,
    contrast_concept: str,
    concept_type: str,
    model_id: str,
    model_tag: str,
    readout_span: str,
) -> ExperimentDataset:
    """Convert span-level OLMoE traces into the repo's generic experiment schema.

    Shape mapping:
    - input `prompt_traces[p].layer_expert_activation_rates[layer]`: `(E,)`
    - output `dataset.examples[p].layers[layer].tokens[0].expert_loads`: `(E,)`

    Meaning:
    - each layer now contributes exactly one row per example: the span-aggregated
      expert activation-rate vector,
    - the generic downstream pipeline can still operate unchanged, because it sees
      one trivially selected row per layer.
    """
    if not prompt_traces:
        raise ValueError("prompt_traces must be non-empty.")

    examples: List[PromptRecord] = []
    for trace in prompt_traces:
        layers: List[LayerRecord] = []
        for layer_index, expert_activation_rates in enumerate(trace.layer_expert_activation_rates):
            span_token_count = trace.span_token_end - trace.span_token_start
            layers.append(
                LayerRecord(
                    layer_index=layer_index,
                    tokens=[
                        TokenRecord(
                            token_index=trace.span_token_start,
                            token_text=f"[{trace.span_name}_activation_rate over {span_token_count} tokens]",
                            attention_weight=1.0,
                            expert_loads=[float(value) for value in expert_activation_rates],
                        )
                    ],
                )
            )

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
            "readout_span": readout_span,
            "readout_statistic": "topk_activation_rate",
        },
        examples=examples,
    )
    validate_dataset(dataset)
    return dataset


def collect_olmoe_experiment_dataset_for_prompt_pairs(
    prompt_pairs: Sequence[StatementPromptPair],
    resources: HFModelResources,
    readout_span: str = "user_content",
    contrast_concept: str = "generic_statement_control",
) -> ExperimentDataset:
    """Collect a faithful-ish SteerMoE dataset from OLMoE prompt pairs.

    Inputs:
    - `prompt_pairs`: upstream-style positive/negative statement prompt pairs.
    - `resources`: loaded OLMoE model and tokenizer.
    - `readout_span`: name of the span whose routing statistics are aggregated.
      The current implementation uses the full user-content span in the
      chat-formatted prompt.
    - `contrast_concept`: textual label for the negative unprefixed prompts.

    Returns:
    - `ExperimentDataset` with one span-aggregated expert-activation vector per
      layer and example.
    """
    prompt_traces = collect_olmoe_span_activation_traces_for_prompt_pairs(
        prompt_pairs=prompt_pairs,
        resources=resources,
        span_name=readout_span,
    )
    return build_experiment_dataset_from_span_traces(
        prompt_traces=prompt_traces,
        concept=prompt_pairs[0].concept_value,
        contrast_concept=contrast_concept,
        concept_type=prompt_pairs[0].concept_type,
        model_id=resources.model_id,
        model_tag=resources.model_tag,
        readout_span=readout_span,
    )


def run_olmoe_steermoe_pipeline(
    dataset: ExperimentDataset,
    config: Optional[ExperimentConfig] = None,
) -> SteeringConditionArtifacts:
    """Run the generic steering pipeline on a span-aggregated OLMoE dataset.

    Inputs:
    - `dataset`: OLMoE-derived experiment dataset with one per-layer activation
      vector `(E,)` per example.
    - `config`: thresholds and sparsity settings for the steering plan.

    Returns:
    - `SteeringConditionArtifacts` containing the dataset and sparse steering
      plan.
    """
    from .pipeline import run_pipeline

    if config is None:
        config = ExperimentConfig(
            intervention=InterventionConfig(
                top_k_experts=2,
                activation_threshold=0.01,
                deactivation_threshold=-0.01,
            )
        )

    artifacts = run_pipeline(dataset, config)
    return SteeringConditionArtifacts(
        dataset=dataset,
        steering_plan=artifacts.steering_plan,
    )


def steering_plan_to_router_bias_by_layer(
    steering_plan: SteeringPlan,
    coefficient: float = 1.0,
) -> Dict[int, List[float]]:
    """Convert a sparse steering plan into dense router-logit bias vectors.

    Shapes:
    - input sparse expert sets per layer:
      - `experts_to_activate`: length `<= K`
      - `experts_to_deactivate`: length `<= K`
    - output dense bias vector per layer:
      - `(E,)`

    Meaning:
    - the selected experts receive biases scaled by their relative delta
      magnitudes within the selected set,
    - untouched experts receive `0.0`,
    - `coefficient` is the maximum absolute bias magnitude in a layer.

    This is gentler than the earlier constant `±8` prototype and better matched
    to a first faithful SteerMoE transfer experiment.
    """
    bias_by_layer: Dict[int, List[float]] = {}
    for layer in steering_plan.layers:
        if layer.scores:
            num_experts = max(score.expert_index for score in layer.scores) + 1
        else:
            num_experts = 1 + max(layer.experts_to_activate + layer.experts_to_deactivate + [0])

        bias_vector = [0.0] * num_experts
        score_map = {score.expert_index: score.delta for score in layer.scores}
        selected_experts = layer.experts_to_activate + layer.experts_to_deactivate
        max_abs_selected_delta = max(
            (abs(score_map.get(expert_index, 0.0)) for expert_index in selected_experts),
            default=1.0,
        )
        if max_abs_selected_delta == 0:
            max_abs_selected_delta = 1.0

        for expert_index in layer.experts_to_activate:
            delta = max(score_map.get(expert_index, 0.0), 0.0)
            bias_vector[expert_index] = coefficient * (delta / max_abs_selected_delta)
        for expert_index in layer.experts_to_deactivate:
            delta = min(score_map.get(expert_index, 0.0), 0.0)
            bias_vector[expert_index] = coefficient * (delta / max_abs_selected_delta)

        bias_by_layer[layer.layer_index] = bias_vector
    return bias_by_layer


def _apply_bias_to_last_router_row(router_logits: Any, bias_vector: Any) -> Any:
    """Add a dense expert bias vector to the final token row of router logits."""
    router_logits = router_logits.clone()
    router_logits[-1, :] = router_logits[-1, :] + bias_vector
    return router_logits


def _recompute_topk_from_biased_router_logits(
    router_logits: Any,
    original_topk_weights: Any,
    original_topk_indices: Any,
) -> Any:
    """Recompute routed experts from biased router logits for tuple-style gates."""
    import torch

    routing_probabilities = torch.softmax(router_logits, dim=-1, dtype=torch.float)
    top_k = original_topk_indices.shape[-1]
    top_k_weights, top_k_indices = torch.topk(routing_probabilities, k=top_k, dim=-1)

    row_sums = original_topk_weights.sum(dim=-1)
    should_normalize = torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-4,
        rtol=1e-4,
    )
    if should_normalize:
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    top_k_weights = top_k_weights.to(dtype=original_topk_weights.dtype, device=original_topk_weights.device)
    top_k_indices = top_k_indices.to(dtype=original_topk_indices.dtype, device=original_topk_indices.device)
    return top_k_weights, top_k_indices


def _apply_bias_to_gate_output(output: Any, bias_vector: Any) -> Any:
    """Apply router steering to either legacy-tensor or tuple-style gate outputs."""
    if hasattr(output, "device") and hasattr(output, "dtype"):
        return _apply_bias_to_last_router_row(
            router_logits=output,
            bias_vector=bias_vector.to(device=output.device, dtype=output.dtype),
        )

    if isinstance(output, tuple) and len(output) >= 3:
        router_logits, top_k_weights, top_k_indices = output[:3]
        biased_router_logits = _apply_bias_to_last_router_row(
            router_logits=router_logits,
            bias_vector=bias_vector.to(device=router_logits.device, dtype=router_logits.dtype),
        )
        biased_top_k_weights, biased_top_k_indices = _recompute_topk_from_biased_router_logits(
            router_logits=biased_router_logits,
            original_topk_weights=top_k_weights,
            original_topk_indices=top_k_indices,
        )
        return (biased_router_logits, biased_top_k_weights, biased_top_k_indices, *output[3:])

    raise TypeError(
        "Unsupported OLMoE gate output type for steering hook: "
        f"{type(output).__name__}."
    )


@contextmanager
def olmoe_router_bias_hooks(
    model: Any,
    bias_by_layer: Dict[int, Sequence[float]],
) -> Iterator[None]:
    """Temporarily inject router-logit biases into OLMoE's sparse blocks.

    Important runtime assumption:
    - generation runs with batch size `1`
    - we bias only the *last* token row of each forward pass, because that token
      controls the next autoregressive step
    """
    import torch

    handles = []
    for layer_index, bias_values in bias_by_layer.items():
        gate_module = model.model.layers[layer_index].mlp.gate
        bias_tensor = torch.tensor(list(bias_values), dtype=torch.float32)

        def hook(module: Any, inputs: Any, output: Any, bias_tensor: Any = bias_tensor) -> Any:
            return _apply_bias_to_gate_output(output=output, bias_vector=bias_tensor)

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
        generate_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
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


def fill_manual_review_plan_with_steermoe_generations(
    plan: ManualReviewPlan,
    resources: HFModelResources,
    steermoe_plans_by_concept: Dict[str, SteeringPlan],
    steering_coefficient: float = 1.0,
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> ManualReviewPlan:
    """Populate a manual review plan with baseline and SteerMoE generations.

    Inputs:
    - `plan`: qualitative review grid whose `concept` plus
      `evaluation_question` identify the model-facing prompt.
    - `resources`: loaded OLMoE model and tokenizer.
    - `steermoe_plans_by_concept`: one sparse SteerMoE plan per concept.
    - `steering_coefficient`: maximum router-logit bias magnitude used at runtime.
    - generation controls: `max_new_tokens`, `temperature`, `top_p`.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback only used on minimal envs
        tqdm = lambda iterable, **_: iterable

    baseline_cache: Dict[str, str] = {}
    steermoe_bias_cache = {
        concept: steering_plan_to_router_bias_by_layer(steering_plan, coefficient=steering_coefficient)
        for concept, steering_plan in steermoe_plans_by_concept.items()
    }

    if "baseline" not in plan.condition_order or "steermoe" not in plan.condition_order:
        raise ValueError("ManualReviewPlan must declare 'baseline' and 'steermoe' conditions.")

    for case in tqdm(plan.cases, desc="Generating qualitative review responses"):
        prompt_text = build_concept_conditioned_evaluation_prompt(
            concept_type=plan.concept_type,
            concept_value=case.concept,
            evaluation_question=case.evaluation_question,
        )

        if prompt_text not in baseline_cache:
            baseline_cache[prompt_text] = generate_with_olmoe_steering(
                prompt_text=prompt_text,
                resources=resources,
                bias_by_layer=None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        case.responses["baseline"] = baseline_cache[prompt_text]
        case.responses["steermoe"] = generate_with_olmoe_steering(
            prompt_text=prompt_text,
            resources=resources,
            bias_by_layer=steermoe_bias_cache[case.concept],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    return plan
