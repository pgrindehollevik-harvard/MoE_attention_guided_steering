from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .attention_collection import HFModelResources
from .manual_review import ManualReviewPlan
from .olmoe_backend import (
    OLMoEPairedRoutingTrace,
    OLMoETargetRoutingTrace,
    _apply_bias_to_gate_output,
    _apply_paper_steering_to_gate_output,
    _flatten_token_ids,
    _normalize_model_inputs,
    _reshape_router_logits_for_prompt,
    build_steermoe_activation_table_from_paired_traces,
    build_steermoe_replication_plan,
    find_chat_target_token_span,
    steermoe_plan_to_router_bias_by_layer,
    steermoe_plan_to_router_steering_by_layer,
)
from .upstream_prompt_datasets import CustomSteeringExample


MixtralTargetRoutingTrace = OLMoETargetRoutingTrace
MixtralPairedRoutingTrace = OLMoEPairedRoutingTrace


def _mixtral_gate_module(model: Any, layer_index: int) -> Any:
    """Return the Mixtral router gate module for one decoder layer."""
    try:
        return model.model.layers[layer_index].block_sparse_moe.gate
    except AttributeError as exc:
        raise AttributeError(
            "Expected a Hugging Face Mixtral-style model with "
            "`model.model.layers[layer].block_sparse_moe.gate` router modules."
        ) from exc


def collect_mixtral_target_routing_trace_for_messages(
    trace_id: str,
    subset_label: str,
    messages: Sequence[Dict[str, str]],
    target_text: str,
    resources: HFModelResources,
    target_name: str = "statement_body",
) -> MixtralTargetRoutingTrace:
    """Run Mixtral once and summarize top-k router choices over a target span.

    Mixtral returns one router-logit tensor per sparse layer when called with
    `output_router_logits=True`. In current Hugging Face Transformers builds,
    each tensor has shape `(B * S, E)`, where `E` is the number of local experts.
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
        raise ValueError("The Mixtral collector currently supports batch_size=1 only.")

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

    router_logits_by_layer = getattr(outputs, "router_logits", None)
    if not router_logits_by_layer:
        raise ValueError(
            "Mixtral did not return router logits. Make sure the installed "
            "Transformers build supports Mixtral `output_router_logits=True`."
        )

    top_k_experts_per_token = int(getattr(model.config, "num_experts_per_tok", 2))
    layer_expert_activation_counts: List[List[int]] = []
    layer_expert_activation_rates: List[List[float]] = []
    for layer_router_logits in router_logits_by_layer:
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

    return MixtralTargetRoutingTrace(
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


def collect_mixtral_paired_routing_traces(
    examples: Sequence[CustomSteeringExample],
    resources: HFModelResources,
    target_name: str = "statement_body",
) -> List[MixtralPairedRoutingTrace]:
    """Collect matched positive/control Mixtral routing traces."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback only used on minimal envs
        tqdm = lambda iterable, **_: iterable

    traces: List[MixtralPairedRoutingTrace] = []
    concept_name = examples[0].concept_value if examples else "unknown"
    for example in tqdm(examples, desc=f"Collecting Mixtral routing traces for {concept_name}"):
        messages_0_trace = collect_mixtral_target_routing_trace_for_messages(
            trace_id=f"{example.example_id}:messages_0",
            subset_label="messages_0",
            messages=example.messages_0,
            target_text=example.messages_0_target,
            resources=resources,
            target_name=target_name,
        )
        messages_1_trace = collect_mixtral_target_routing_trace_for_messages(
            trace_id=f"{example.example_id}:messages_1",
            subset_label="messages_1",
            messages=example.messages_1,
            target_text=example.messages_1_target,
            resources=resources,
            target_name=target_name,
        )
        traces.append(
            MixtralPairedRoutingTrace(
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


@contextmanager
def mixtral_router_bias_hooks(
    model: Any,
    bias_by_layer: Dict[int, Sequence[float]],
    steering_rule: str = "additive_bias",
    steering_epsilon: float = 0.01,
) -> Iterator[None]:
    """Temporarily inject router-logit biases into Mixtral gate modules."""
    import torch

    handles = []
    for layer_index, bias_values in bias_by_layer.items():
        gate_module = _mixtral_gate_module(model=model, layer_index=layer_index)
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


def generate_with_mixtral_steering(
    prompt_text: str,
    resources: HFModelResources,
    bias_by_layer: Optional[Dict[int, Sequence[float]]] = None,
    steering_rule: str = "additive_bias",
    steering_epsilon: float = 0.01,
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """Generate one response from Mixtral with optional router steering."""
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
            with mixtral_router_bias_hooks(
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


def fill_manual_review_plan_with_mixtral_generations(
    plan: ManualReviewPlan,
    resources: HFModelResources,
    steermoe_plans_by_concept: Dict[str, Any],
    steering_coefficient: float = 1.0,
    steering_rule: str = "additive_bias",
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    top_p: float = 1.0,
    baseline_condition: str = "mixtral_baseline",
    steermoe_condition: str = "mixtral_steermoe",
) -> ManualReviewPlan:
    """Populate a review plan with Mixtral baseline and Mixtral + SteerMoE outputs."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback only used on minimal envs
        tqdm = lambda iterable, **_: iterable

    if baseline_condition not in plan.condition_order or steermoe_condition not in plan.condition_order:
        raise ValueError("ManualReviewPlan must declare Mixtral baseline and SteerMoE conditions.")

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

    for case in tqdm(plan.cases, desc="Generating Mixtral review responses"):
        prompt_text = case.full_prompt_text.strip() or case.evaluation_question

        if prompt_text not in baseline_cache:
            baseline_cache[prompt_text] = generate_with_mixtral_steering(
                prompt_text=prompt_text,
                resources=resources,
                bias_by_layer=None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        case.responses[baseline_condition] = baseline_cache[prompt_text]
        case.responses[steermoe_condition] = generate_with_mixtral_steering(
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
