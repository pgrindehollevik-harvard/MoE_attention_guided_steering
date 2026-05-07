from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .attention_collection import HFModelResources, compute_inserted_token_span_from_ids
from .manual_review import ManualReviewPlan
from .upstream_prompt_datasets import CustomSteeringExample


@dataclass
class OLMoETargetRoutingTrace:
    """Routing statistics for one message sequence over one explicit target span.

    Shape flow:
    - chat-formatted `input_ids`: `(1, S)`
    - raw per-layer OLMoE router logits: `(B * S, E)` or `(B, S, E)`
    - reshaped per-layer router logits: `(1, S, E)`
    - chosen target span inside the prompt: `(P, E)`
    - top-k routed-expert indicator matrix over the span: `(P, E)`
    - per-layer activation counts after summing over the target tokens: `(E,)`
    - per-layer activation rates after dividing by `P`: `(E,)`
    - full trace over all layers: `(L, E)`

    where:
    - `S` is the full chat-formatted sequence length.
    - `P` is the number of tokens in the matched target span.
    - `L` is the number of sparse decoder layers in OLMoE.
    - `E` is the number of experts per sparse layer.

    Meaning:
    - `layer_expert_activation_counts[layer][expert]` is how many target tokens
      in this prompt routed to that expert.
    - `layer_expert_activation_rates[layer][expert]` is the corresponding
      activation fraction in `[0, 1]`.

    This is the direct evidence that later feeds the SteerMoE-style
    risk-difference calculation.
    """

    trace_id: str
    subset_label: str
    messages: List[Dict[str, str]]
    target_name: str
    target_text: str
    target_token_start: int
    target_token_end: int
    target_token_texts: List[str]
    top_k_experts_per_token: int
    layer_expert_activation_counts: List[List[int]]
    layer_expert_activation_rates: List[List[float]]


@dataclass
class OLMoEPairedRoutingTrace:
    """Matched routing traces for one custom steering example.

    Meaning:
    - `messages_0_trace` is the concept-conditioned example.
    - `messages_1_trace` is the matched control example.
    - both traces target the same shared statement body.

    This is the closest local analogue to the paired activation records used in
    Adobe's custom SteerMoE workflow, adapted to our fears prompt pairs.
    """

    example_id: str
    concept_type: str
    concept_value: str
    statement_index: int
    statement_text: str
    body_text: str
    messages_0_trace: OLMoETargetRoutingTrace
    messages_1_trace: OLMoETargetRoutingTrace


@dataclass
class SteerMoERiskDifferenceScore:
    """One layer/expert score in the custom SteerMoE activation table.

    Shapes:
    - activation counts aggregated from each trace: scalar counts over all
      matched target tokens.
    - activation rates: scalars in `[0, 1]`.

    Meaning:
    - `messages_0_activation_rate` is the fraction of all concept-conditioned
      target tokens routed to this expert.
    - `messages_1_activation_rate` is the same fraction for the matched control
      target tokens.
    - `risk_difference = messages_0_activation_rate - messages_1_activation_rate`.

    Positive values indicate an expert is more associated with the
    concept-conditioned subset; negative values indicate the opposite.
    """

    layer_index: int
    expert_index: int
    messages_0_activation_count: int
    messages_1_activation_count: int
    messages_0_token_count: int
    messages_1_token_count: int
    messages_0_activation_rate: float
    messages_1_activation_rate: float
    risk_difference: float
    abs_risk_difference: float


@dataclass
class SteerMoESelectedExpert:
    """One globally selected expert from the risk-difference table.

    Meaning:
    - `direction='activate'` means routing should be biased *toward* this
      expert, because it appears more often in the concept-conditioned subset.
    - `direction='deactivate'` means routing should be biased *away* from this
      expert, because it appears more often in the control subset.
    """

    direction: str
    score: SteerMoERiskDifferenceScore


@dataclass
class SteerMoELayerPlan:
    """Selected SteerMoE interventions for one OLMoE layer.

    Shapes:
    - `experts_to_activate`: sparse list of expert indices for this layer.
    - `experts_to_deactivate`: sparse list of expert indices for this layer.

    Meaning:
    - only experts appearing in the global top positive/negative selections are
      listed here.
    - all other experts receive zero router bias at generation time.
    """

    layer_index: int
    experts_to_activate: List[int]
    experts_to_deactivate: List[int]
    selected_scores: List[SteerMoESelectedExpert]


@dataclass
class SteerMoEReplicationPlan:
    """A custom-steering plan tailored to our fears-data SteerMoE transfer test.

    Meaning:
    - the plan stores the full activation table plus the globally selected
      positive and negative experts used for steering.
    - `layers` is the runtime-ready grouped view used to bias OLMoE routing.

    This intentionally keeps the direct risk-difference evidence visible instead
    of compressing everything into a thresholded delta vector too early.
    """

    concept: str
    concept_type: str
    contrast_label: str
    model_id: str
    model_tag: str
    target_name: str
    pair_count: int
    num_layers: int
    num_experts: int
    top_positive_experts: int
    top_negative_experts: int
    minimum_abs_risk_difference: float
    activation_table: List[SteerMoERiskDifferenceScore]
    selected_positive_experts: List[SteerMoESelectedExpert]
    selected_negative_experts: List[SteerMoESelectedExpert]
    layers: List[SteerMoELayerPlan]


def _flatten_token_ids(token_ids: Any) -> List[int]:
    """Convert tokenizer outputs into a flat Python list of token ids."""
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
    """Recover `(B, S, E)` router tensors from OLMoE's flattened layer outputs."""
    shape = tuple(layer_router_logits.shape)
    if len(shape) == 2 and shape[0] == batch_size * sequence_length:
        return layer_router_logits.view(batch_size, sequence_length, shape[1])
    if len(shape) == 3 and shape[0] == batch_size and shape[1] == sequence_length:
        return layer_router_logits

    raise ValueError(
        "Expected router logits with shape (B * S, E) or (B, S, E), "
        f"got {shape} for batch_size={batch_size}, sequence_length={sequence_length}."
    )


def _remove_target_once_from_messages(
    messages: Sequence[Dict[str, str]],
    target_text: str,
) -> List[Dict[str, str]]:
    """Return a copy of `messages` with the first target occurrence removed.

    This powers exact target-span recovery:
    - tokenize the original message list,
    - tokenize the same messages with the target removed,
    - recover the inserted span via token-level diff.
    """

    reduced_messages: List[Dict[str, str]] = []
    found = False
    for message in messages:
        role = message["role"]
        content = message["content"]
        if not found and target_text in content:
            start = content.index(target_text)
            end = start + len(target_text)
            reduced_messages.append({"role": role, "content": content[:start] + content[end:]})
            found = True
        else:
            reduced_messages.append({"role": role, "content": content})

    if not found:
        raise ValueError("Target text was not found in the provided messages.")

    return reduced_messages


def find_chat_target_token_span(
    tokenizer: Any,
    messages: Sequence[Dict[str, str]],
    target_text: str,
) -> Tuple[int, int]:
    """Find the token span occupied by an explicit target string inside messages.

    Inputs:
    - `messages`: chat-style message list passed to `apply_chat_template`.
    - `target_text`: literal substring whose routing activations we want to
      analyze.

    Returns:
    - `(start, end)` token indices of the target span inside the
      chat-formatted input sequence.

    Method:
    - tokenize the original messages with generation prompt formatting,
    - tokenize the same messages with the target text removed once,
    - recover the inserted span via exact token diff.
    """

    full_ids = _flatten_token_ids(
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    )
    reduced_ids = _flatten_token_ids(
        tokenizer.apply_chat_template(
            _remove_target_once_from_messages(messages, target_text),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    )
    return compute_inserted_token_span_from_ids(full_ids, reduced_ids)


def collect_olmoe_target_routing_trace_for_messages(
    trace_id: str,
    subset_label: str,
    messages: Sequence[Dict[str, str]],
    target_text: str,
    resources: HFModelResources,
    target_name: str = "statement_body",
) -> OLMoETargetRoutingTrace:
    """Run OLMoE once and summarize routing over a matched target span.

    Shape flow:
    1. Tokenize the chat-formatted messages:
       - `input_ids`: `(1, S)`
    2. Recover the exact token span for the requested target:
       - target slice indices `(start, end)` with token count `P = end - start`
    3. Ask OLMoE to return per-layer router logits:
       - raw layer tensor `(1 * S, E)` or `(1, S, E)`
    4. Slice each layer to the target span:
       - `(P, E)`
    5. Mark the top-k routed experts for each target token:
       - indicator matrix `(P, E)`
    6. Summarize:
       - counts `(E,)`
       - rates `(E,)`

    Returns:
    - `OLMoETargetRoutingTrace` containing the direct routing evidence later used
      in the risk-difference table.
    """
    import torch

    tokenizer = resources.tokenizer
    model = resources.model

    encoded_inputs = tokenizer.apply_chat_template(
        list(messages),
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

    target_start, target_end = find_chat_target_token_span(
        tokenizer=tokenizer,
        messages=messages,
        target_text=target_text,
    )
    input_ids = _flatten_token_ids(encoded_inputs["input_ids"])
    target_token_ids = input_ids[target_start:target_end]
    target_token_texts = tokenizer.convert_ids_to_tokens(target_token_ids)

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
    layer_expert_activation_counts: List[List[int]] = []
    layer_expert_activation_rates: List[List[float]] = []
    for layer_router_logits in outputs.router_logits:
        reshaped = _reshape_router_logits_for_prompt(
            layer_router_logits=layer_router_logits,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        target_router_logits = reshaped[0, target_start:target_end, :]
        top_k_indices = torch.topk(target_router_logits, k=top_k_experts_per_token, dim=-1).indices
        activation_indicator = torch.zeros_like(target_router_logits, dtype=torch.float32)
        activation_indicator.scatter_(dim=-1, index=top_k_indices, value=1.0)
        activation_counts = activation_indicator.sum(dim=0)
        activation_rates = activation_indicator.mean(dim=0)
        layer_expert_activation_counts.append([int(value) for value in activation_counts.detach().cpu().tolist()])
        layer_expert_activation_rates.append([float(value) for value in activation_rates.detach().cpu().tolist()])

    return OLMoETargetRoutingTrace(
        trace_id=trace_id,
        subset_label=subset_label,
        messages=[{"role": message["role"], "content": message["content"]} for message in messages],
        target_name=target_name,
        target_text=target_text,
        target_token_start=target_start,
        target_token_end=target_end,
        target_token_texts=list(target_token_texts),
        top_k_experts_per_token=top_k_experts_per_token,
        layer_expert_activation_counts=layer_expert_activation_counts,
        layer_expert_activation_rates=layer_expert_activation_rates,
    )


def collect_olmoe_paired_routing_traces(
    examples: Sequence[CustomSteeringExample],
    resources: HFModelResources,
    target_name: str = "statement_body",
) -> List[OLMoEPairedRoutingTrace]:
    """Collect matched routing traces for each custom steering example.

    Returns:
    - `List[OLMoEPairedRoutingTrace]` with one positive/control trace pair per
      example.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback only used on minimal envs
        tqdm = lambda iterable, **_: iterable

    traces: List[OLMoEPairedRoutingTrace] = []
    concept_name = examples[0].concept_value if examples else "unknown"
    for example in tqdm(examples, desc=f"Collecting OLMoE routing traces for {concept_name}"):
        messages_0_trace = collect_olmoe_target_routing_trace_for_messages(
            trace_id=f"{example.example_id}:messages_0",
            subset_label="messages_0",
            messages=example.messages_0,
            target_text=example.messages_0_target,
            resources=resources,
            target_name=target_name,
        )
        messages_1_trace = collect_olmoe_target_routing_trace_for_messages(
            trace_id=f"{example.example_id}:messages_1",
            subset_label="messages_1",
            messages=example.messages_1,
            target_text=example.messages_1_target,
            resources=resources,
            target_name=target_name,
        )
        traces.append(
            OLMoEPairedRoutingTrace(
                example_id=example.example_id,
                concept_type=example.concept_type,
                concept_value=example.concept_value,
                statement_index=example.statement_index,
                statement_text=example.statement_text,
                body_text=example.body_text,
                messages_0_trace=messages_0_trace,
                messages_1_trace=messages_1_trace,
            )
        )

    return traces


def build_steermoe_activation_table_from_paired_traces(
    paired_traces: Sequence[OLMoEPairedRoutingTrace],
) -> List[SteerMoERiskDifferenceScore]:
    """Aggregate paired routing traces into an Adobe-style risk-difference table.

    Shape logic:
    - each trace contributes integer activation counts `(L, E)` plus token
      counts `P`.
    - counts are summed across the full custom steering dataset:
      - `messages_0_activation_count[layer, expert]`
      - `messages_1_activation_count[layer, expert]`
    - token totals are summed likewise:
      - `messages_0_token_count`
      - `messages_1_token_count`
    - activation rates become scalar fractions for each `(layer, expert)`.

    Returns:
    - `List[SteerMoERiskDifferenceScore]` with one row per layer/expert pair.
    """
    if not paired_traces:
        raise ValueError("paired_traces must be non-empty.")

    first_trace = paired_traces[0].messages_0_trace
    num_layers = len(first_trace.layer_expert_activation_counts)
    num_experts = len(first_trace.layer_expert_activation_counts[0])

    messages_0_token_count = 0
    messages_1_token_count = 0
    messages_0_totals = [[0 for _ in range(num_experts)] for _ in range(num_layers)]
    messages_1_totals = [[0 for _ in range(num_experts)] for _ in range(num_layers)]

    for paired_trace in paired_traces:
        positive_trace = paired_trace.messages_0_trace
        negative_trace = paired_trace.messages_1_trace

        messages_0_token_count += positive_trace.target_token_end - positive_trace.target_token_start
        messages_1_token_count += negative_trace.target_token_end - negative_trace.target_token_start

        for layer_index in range(num_layers):
            for expert_index in range(num_experts):
                messages_0_totals[layer_index][expert_index] += positive_trace.layer_expert_activation_counts[layer_index][expert_index]
                messages_1_totals[layer_index][expert_index] += negative_trace.layer_expert_activation_counts[layer_index][expert_index]

    scores: List[SteerMoERiskDifferenceScore] = []
    for layer_index in range(num_layers):
        for expert_index in range(num_experts):
            positive_count = messages_0_totals[layer_index][expert_index]
            negative_count = messages_1_totals[layer_index][expert_index]
            positive_rate = positive_count / messages_0_token_count if messages_0_token_count else 0.0
            negative_rate = negative_count / messages_1_token_count if messages_1_token_count else 0.0
            risk_difference = positive_rate - negative_rate
            scores.append(
                SteerMoERiskDifferenceScore(
                    layer_index=layer_index,
                    expert_index=expert_index,
                    messages_0_activation_count=positive_count,
                    messages_1_activation_count=negative_count,
                    messages_0_token_count=messages_0_token_count,
                    messages_1_token_count=messages_1_token_count,
                    messages_0_activation_rate=positive_rate,
                    messages_1_activation_rate=negative_rate,
                    risk_difference=risk_difference,
                    abs_risk_difference=abs(risk_difference),
                )
            )

    return scores


def build_steermoe_replication_plan(
    concept: str,
    concept_type: str,
    model_id: str,
    model_tag: str,
    target_name: str,
    paired_traces: Sequence[OLMoEPairedRoutingTrace],
    activation_table: Sequence[SteerMoERiskDifferenceScore],
    top_positive_experts: int = 8,
    top_negative_experts: int = 8,
    minimum_abs_risk_difference: float = 0.01,
    contrast_label: str = "generic_statement_control",
) -> SteerMoEReplicationPlan:
    """Select globally strongest experts from a SteerMoE risk-difference table.

    Selection logic:
    - keep positive candidates with `risk_difference > 0`
    - keep negative candidates with `risk_difference < 0`
    - require `abs(risk_difference) >= minimum_abs_risk_difference`
    - sort globally, not layer-by-layer
    - take the top `top_positive_experts` and `top_negative_experts`

    This is intentionally closer to Adobe's custom steering notebook than the
    repo's earlier per-layer thresholding path.
    """
    if top_positive_experts < 0 or top_negative_experts < 0:
        raise ValueError("Expert budgets must be non-negative.")
    if minimum_abs_risk_difference < 0:
        raise ValueError("minimum_abs_risk_difference must be non-negative.")
    if not activation_table:
        raise ValueError("activation_table must be non-empty.")

    positive_candidates = [
        score
        for score in activation_table
        if score.risk_difference > 0 and score.abs_risk_difference >= minimum_abs_risk_difference
    ]
    negative_candidates = [
        score
        for score in activation_table
        if score.risk_difference < 0 and score.abs_risk_difference >= minimum_abs_risk_difference
    ]
    positive_candidates.sort(key=lambda score: score.risk_difference, reverse=True)
    negative_candidates.sort(key=lambda score: score.risk_difference)

    selected_positive = [
        SteerMoESelectedExpert(direction="activate", score=score)
        for score in positive_candidates[:top_positive_experts]
    ]
    selected_negative = [
        SteerMoESelectedExpert(direction="deactivate", score=score)
        for score in negative_candidates[:top_negative_experts]
    ]

    all_selected = selected_positive + selected_negative
    num_layers = 1 + max(score.layer_index for score in activation_table)
    num_experts = 1 + max(score.expert_index for score in activation_table)
    selected_by_layer: Dict[int, List[SteerMoESelectedExpert]] = {layer_index: [] for layer_index in range(num_layers)}
    for selected in all_selected:
        selected_by_layer[selected.score.layer_index].append(selected)

    layers: List[SteerMoELayerPlan] = []
    for layer_index in range(num_layers):
        selected_scores = selected_by_layer[layer_index]
        if not selected_scores:
            continue
        layers.append(
            SteerMoELayerPlan(
                layer_index=layer_index,
                experts_to_activate=[
                    selected.score.expert_index
                    for selected in selected_scores
                    if selected.direction == "activate"
                ],
                experts_to_deactivate=[
                    selected.score.expert_index
                    for selected in selected_scores
                    if selected.direction == "deactivate"
                ],
                selected_scores=selected_scores,
            )
        )

    return SteerMoEReplicationPlan(
        concept=concept,
        concept_type=concept_type,
        contrast_label=contrast_label,
        model_id=model_id,
        model_tag=model_tag,
        target_name=target_name,
        pair_count=len(paired_traces),
        num_layers=num_layers,
        num_experts=num_experts,
        top_positive_experts=top_positive_experts,
        top_negative_experts=top_negative_experts,
        minimum_abs_risk_difference=minimum_abs_risk_difference,
        activation_table=list(activation_table),
        selected_positive_experts=selected_positive,
        selected_negative_experts=selected_negative,
        layers=layers,
    )


def steermoe_plan_to_router_bias_by_layer(
    steering_plan: SteerMoEReplicationPlan,
    coefficient: float = 1.0,
) -> Dict[int, List[float]]:
    """Convert a custom SteerMoE plan into dense router-logit bias vectors.

    Shapes:
    - input selected experts: sparse global selections grouped by layer
    - output bias vector for each touched layer: `(E,)`

    Meaning:
    - positive-risk experts receive positive bias
    - negative-risk experts receive negative bias
    - bias magnitudes are scaled by the selected experts' relative
      `abs_risk_difference` within each layer
    """
    bias_by_layer: Dict[int, List[float]] = {}
    for layer in steering_plan.layers:
        bias_vector = [0.0] * steering_plan.num_experts
        max_abs_selected_risk = max(
            (selected.score.abs_risk_difference for selected in layer.selected_scores),
            default=1.0,
        )
        if max_abs_selected_risk == 0:
            max_abs_selected_risk = 1.0

        for selected in layer.selected_scores:
            scale = coefficient * (selected.score.abs_risk_difference / max_abs_selected_risk)
            sign = 1.0 if selected.direction == "activate" else -1.0
            bias_vector[selected.score.expert_index] = sign * scale

        bias_by_layer[layer.layer_index] = bias_vector
    return bias_by_layer


def steermoe_plan_to_router_steering_by_layer(
    steering_plan: SteerMoEReplicationPlan,
) -> Dict[int, List[int]]:
    """Convert a SteerMoE plan into dense paper-style steering directions.

    Values are intentionally only signs:
    - `1` means force-activate the expert with the paper's max-plus-epsilon rule
    - `-1` means force-deactivate the expert with the paper's min-minus-epsilon rule
    - `0` means leave the expert untouched
    """
    steering_by_layer: Dict[int, List[int]] = {}
    for layer in steering_plan.layers:
        steering_vector = [0] * steering_plan.num_experts
        for expert_index in layer.experts_to_activate:
            steering_vector[expert_index] = 1
        for expert_index in layer.experts_to_deactivate:
            steering_vector[expert_index] = -1
        steering_by_layer[layer.layer_index] = steering_vector
    return steering_by_layer


def _apply_bias_to_last_router_row(router_logits: Any, bias_vector: Any) -> Any:
    """Add a dense expert bias vector to the final token row of router logits."""
    router_logits = router_logits.clone()
    router_logits[-1, :] = router_logits[-1, :] + bias_vector
    return router_logits


def _apply_paper_steering_to_router_logits(
    router_logits: Any,
    steering_vector: Any,
    epsilon: float,
) -> Any:
    """Apply SteerMoE's log-softmax max/min rule to every router row."""
    import torch

    if epsilon <= 0:
        raise ValueError("Paper-style SteerMoE epsilon must be positive.")

    activate_mask = steering_vector > 0
    deactivate_mask = steering_vector < 0
    if not bool(torch.any(activate_mask) or torch.any(deactivate_mask)):
        return router_logits

    scores = torch.log_softmax(router_logits.to(dtype=torch.float32), dim=-1)
    adjusted_scores = scores.clone()
    mask_shape = (1,) * (scores.ndim - 1) + (scores.shape[-1],)

    if bool(torch.any(activate_mask)):
        active_mask = activate_mask.reshape(mask_shape)
        active_value = scores.max(dim=-1, keepdim=True).values + epsilon
        adjusted_scores = torch.where(active_mask, active_value, adjusted_scores)

    if bool(torch.any(deactivate_mask)):
        inactive_mask = deactivate_mask.reshape(mask_shape)
        inactive_value = scores.min(dim=-1, keepdim=True).values - epsilon
        adjusted_scores = torch.where(inactive_mask, inactive_value, adjusted_scores)

    return adjusted_scores.to(dtype=router_logits.dtype)


def _recompute_topk_from_router_logits(
    router_logits: Any,
    original_topk_weights: Any,
    original_topk_indices: Any,
) -> Any:
    """Recompute routed experts from modified router logits for tuple-style gates."""
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
        biased_top_k_weights, biased_top_k_indices = _recompute_topk_from_router_logits(
            router_logits=biased_router_logits,
            original_topk_weights=top_k_weights,
            original_topk_indices=top_k_indices,
        )
        return (biased_router_logits, biased_top_k_weights, biased_top_k_indices, *output[3:])

    raise TypeError(
        "Unsupported OLMoE gate output type for steering hook: "
        f"{type(output).__name__}."
    )


def _apply_paper_steering_to_gate_output(output: Any, steering_vector: Any, epsilon: float) -> Any:
    """Apply the paper's force activate/deactivate rule to router outputs."""
    if hasattr(output, "device") and hasattr(output, "dtype"):
        return _apply_paper_steering_to_router_logits(
            router_logits=output,
            steering_vector=steering_vector.to(device=output.device),
            epsilon=epsilon,
        )

    if isinstance(output, tuple) and len(output) >= 3:
        router_logits, top_k_weights, top_k_indices = output[:3]
        steered_router_logits = _apply_paper_steering_to_router_logits(
            router_logits=router_logits,
            steering_vector=steering_vector.to(device=router_logits.device),
            epsilon=epsilon,
        )
        steered_top_k_weights, steered_top_k_indices = _recompute_topk_from_router_logits(
            router_logits=steered_router_logits,
            original_topk_weights=top_k_weights,
            original_topk_indices=top_k_indices,
        )
        return (steered_router_logits, steered_top_k_weights, steered_top_k_indices, *output[3:])

    raise TypeError(
        "Unsupported OLMoE gate output type for paper-style steering hook: "
        f"{type(output).__name__}."
    )


@contextmanager
def olmoe_router_bias_hooks(
    model: Any,
    bias_by_layer: Dict[int, Sequence[float]],
    steering_rule: str = "additive_bias",
    steering_epsilon: float = 0.01,
) -> Iterator[None]:
    """Temporarily inject router-logit biases into OLMoE's sparse blocks.

    Runtime assumption:
    - generation runs with batch size `1`
    - we bias only the final token row in each forward pass, because that row
      determines the next autoregressive decision.
    """
    import torch

    handles = []
    for layer_index, bias_values in bias_by_layer.items():
        gate_module = model.model.layers[layer_index].mlp.gate
        bias_tensor = torch.tensor(list(bias_values), dtype=torch.float32)

        def hook(module: Any, inputs: Any, output: Any, bias_tensor: Any = bias_tensor) -> Any:
            if steering_rule == "additive_bias":
                return _apply_bias_to_gate_output(output=output, bias_vector=bias_tensor)
            if steering_rule == "paper":
                return _apply_paper_steering_to_gate_output(
                    output=output,
                    steering_vector=bias_tensor,
                    epsilon=steering_epsilon,
                )
            raise ValueError(f"Unknown steering_rule: {steering_rule}")

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
    steering_rule: str = "additive_bias",
    steering_epsilon: float = 0.01,
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """Generate one response from OLMoE with optional router steering.

    Shape flow:
    - tokenized prompt: `(1, S_prompt)`
    - generated ids returned by Hugging Face: `(1, S_prompt + S_new)`
    - decoded response slice: tokens `S_prompt : S_prompt + S_new`
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
            with olmoe_router_bias_hooks(
                model=model,
                bias_by_layer=bias_by_layer,
                steering_rule=steering_rule,
                steering_epsilon=steering_epsilon,
            ):
                generated_ids = model.generate(**generate_kwargs)
        else:
            generated_ids = model.generate(**generate_kwargs)

    prompt_length = encoded_inputs["input_ids"].shape[1]
    response_ids = generated_ids[0, prompt_length:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def fill_manual_review_plan_with_steermoe_generations(
    plan: ManualReviewPlan,
    resources: HFModelResources,
    steermoe_plans_by_concept: Dict[str, SteerMoEReplicationPlan],
    steering_coefficient: float = 1.0,
    steering_rule: str = "additive_bias",
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> ManualReviewPlan:
    """Populate a manual review plan with baseline and SteerMoE generations.

    Inputs:
    - `plan`: qualitative review grid whose `concept` plus
      `full_prompt_text` identify the model-facing prompt. For the faithful
      steering test, `full_prompt_text` is question-only and intentionally omits
      the upstream concept prefix.
    - `resources`: loaded OLMoE model and tokenizer.
    - `steermoe_plans_by_concept`: one custom SteerMoE plan per concept.
    - `steering_coefficient`: additive bias magnitude, or epsilon for the
      paper-style max/min steering rule.
    - generation controls: `max_new_tokens`, `temperature`, `top_p`.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback only used on minimal envs
        tqdm = lambda iterable, **_: iterable

    baseline_cache: Dict[str, str] = {}
    if steering_rule == "additive_bias":
        steermoe_bias_cache = {
            concept: steermoe_plan_to_router_bias_by_layer(steering_plan, coefficient=steering_coefficient)
            for concept, steering_plan in steermoe_plans_by_concept.items()
        }
    elif steering_rule == "paper":
        steermoe_bias_cache = {
            concept: steermoe_plan_to_router_steering_by_layer(steering_plan)
            for concept, steering_plan in steermoe_plans_by_concept.items()
        }
    else:
        raise ValueError(f"Unknown steering_rule: {steering_rule}")

    if "baseline" not in plan.condition_order or "steermoe" not in plan.condition_order:
        raise ValueError("ManualReviewPlan must declare 'baseline' and 'steermoe' conditions.")

    for case in tqdm(plan.cases, desc="Generating qualitative review responses"):
        prompt_text = case.full_prompt_text.strip() or case.evaluation_question

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
            steering_rule=steering_rule,
            steering_epsilon=steering_coefficient,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    return plan
