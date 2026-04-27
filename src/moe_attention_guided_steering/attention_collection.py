from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .upstream_prompt_datasets import StatementPromptPair


@dataclass
class HFModelResources:
    """Loaded Hugging Face model objects plus shared token metadata."""

    model: Any
    tokenizer: Any
    model_id: str
    model_tag: str
    num_candidate_suffix_tokens: int


@dataclass
class PromptAttentionTrace:
    """Per-prompt attention scores for the shared candidate suffix tokens.

    Shapes:
    - input attentions from the model: `(1, H, S, S)` per layer
    - output `layer_attention_scores`: `(L, N)`

    where:
    - `H` is the number of attention heads,
    - `S` is the sequence length of the formatted chat prompt,
    - `L` is the number of layers,
    - `N` is the number of shared candidate suffix tokens, matching the upstream
      attention-guided steering token-selection stage.
    """

    prompt_id: str
    concept_type: str
    concept_value: str
    statement_index: int
    statement_text: str
    prefix_text: str
    body_text: str
    full_prompt_text: str
    prefix_token_span: Tuple[int, int]
    candidate_token_texts: List[str]
    candidate_token_relative_indices: List[int]
    layer_attention_scores: List[List[float]]


@dataclass
class AttentionCollectionRun:
    """All attention traces collected for one concept and one model."""

    concept_type: str
    concept_value: str
    model_id: str
    model_tag: str
    head_aggregation: str
    num_prompts: int
    num_layers: int
    num_candidate_suffix_tokens: int
    candidate_token_relative_indices: List[int]
    candidate_token_texts: List[str]
    statement_stride: int
    prompt_traces: List[PromptAttentionTrace]
    layer_to_token_index: Dict[int, int]


def _flatten_token_ids(token_ids: Any) -> List[int]:
    """Convert token ids from tensors, mappings, or lists into a flat Python list.

    Some chat tokenizers return a bare tensor for `apply_chat_template(...)`,
    while others return a mapping such as `{"input_ids": tensor, ...}`. This
    helper normalizes both cases so downstream prefix-span logic can stay shape-
    focused instead of tokenizer-output focused.
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
    """Convert tokenizer outputs into a plain model-input mapping.

    Output shape meaning:
    - `input_ids`: `(1, S)`
    - optional `attention_mask`: `(1, S)`

    where `S` is the sequence length of the chat-formatted prompt.
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


def longest_common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the number of matching items at the start of two sequences."""
    length = min(len(left), len(right))
    index = 0
    while index < length and left[index] == right[index]:
        index += 1
    return index


def longest_common_suffix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the number of matching items at the end of two sequences."""
    length = min(len(left), len(right))
    index = 0
    while index < length and left[-1 - index] == right[-1 - index]:
        index += 1
    return index


def compute_inserted_token_span_from_ids(
    full_ids: Sequence[int],
    reduced_ids: Sequence[int],
) -> Tuple[int, int]:
    """Find the token span inserted into `full_ids` relative to `reduced_ids`.

    This is the robust version of the upstream `get_prefix_inds` logic. Instead
    of searching for a literal token like `" What"`, we compare the full prompt
    against the same prompt body without the concept prefix and recover the exact
    inserted token interval.
    """
    prefix_length = longest_common_prefix_length(full_ids, reduced_ids)
    suffix_length = longest_common_suffix_length(
        full_ids[prefix_length:],
        reduced_ids[prefix_length:],
    )
    inserted_start = prefix_length
    inserted_end = len(full_ids) - suffix_length

    if inserted_end <= inserted_start:
        raise ValueError("Could not recover a non-empty inserted token span.")

    return inserted_start, inserted_end


def infer_candidate_suffix_token_count(tokenizer: Any, probe_text: str = "This is a random sentence") -> int:
    """Infer the number of shared suffix tokens used as attention candidates.

    This intentionally mirrors the upstream `get_n_common_toks` behavior. The
    candidate suffix includes:

    - the final token of the no-generation prompt, and
    - every token appended by `add_generation_prompt=True`.

    That is why the count is one larger than "just the added generation prompt
    tokens" for most chat models.
    """
    chat = [{"role": "user", "content": probe_text}]
    ids_no_generation = _flatten_token_ids(
        tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
    )
    ids_with_generation = _flatten_token_ids(
        tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    )
    return len(ids_with_generation[len(ids_no_generation) - 1 :])


def normalize_llama_moe_rope_scaling_for_remote_code(config: Any) -> Any:
    """Patch LLaMA-MoE configs for newer Transformers RoPE metadata.

    The LLaMA-MoE remote modeling code was written against Transformers 4.36 and
    expects either `rope_scaling is None` or a dict with a legacy `type` key.
    Newer Transformers versions may expose the default RoPE setting as a dict
    keyed by `rope_type`, which makes that remote code raise `KeyError: 'type'`.
    """
    if getattr(config, "model_type", "") != "llama_moe":
        return config

    rope_scaling = getattr(config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict) or "type" in rope_scaling:
        return config

    rope_type = rope_scaling.get("rope_type")
    if rope_type in (None, "default"):
        config.rope_scaling = None
    elif rope_type in {"linear", "dynamic"}:
        config.rope_scaling = {**rope_scaling, "type": rope_type}
    return config


def load_hf_model_resources(
    model_id: str,
    model_tag: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device_map: Optional[str] = "auto",
    torch_dtype: str = "bfloat16",
    load_in_4bit: bool = False,
    attn_implementation: Optional[str] = "eager",
    trust_remote_code: bool = False,
    infer_attention_suffix_tokens: bool = True,
    post_load_device: Optional[str] = None,
) -> HFModelResources:
    """Load a causal LM plus tokenizer for attention collection.

    The imports are intentionally local so the rest of the repo stays importable
    on machines that do not yet have GPU/Transformers dependencies installed.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_kwargs: Dict[str, Any] = {
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if torch_dtype:
        model_kwargs["dtype"] = getattr(torch, torch_dtype)

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    config = AutoConfig.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    model_kwargs["config"] = normalize_llama_moe_rope_scaling_for_remote_code(config)

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).eval()
    if post_load_device:
        model = model.to(post_load_device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        legacy=False,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    return HFModelResources(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        model_tag=model_tag or sanitize_model_tag(model_id),
        num_candidate_suffix_tokens=(
            infer_candidate_suffix_token_count(tokenizer)
            if infer_attention_suffix_tokens
            else 0
        ),
    )


def sanitize_model_tag(model_id: str) -> str:
    """Convert a Hugging Face model id into a filesystem-friendly tag."""
    return (
        model_id.replace("/", "__")
        .replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def find_prefix_token_span(
    tokenizer: Any,
    prefix_text: str,
    body_text: str,
) -> Tuple[int, int]:
    """Recover the token interval occupied by the concept prefix.

    Inputs:
    - `prefix_text`: the upstream positive concept instruction, such as
      `"Personify someone who is terrified of Bugs.  "`
    - `body_text`: the generic statement prompt body shared by the positive and
      negative examples.

    Returns:
    - `(start, end)` token indices for the prefix inside the chat-formatted input.
    """
    full_chat = [{"role": "user", "content": prefix_text + body_text}]
    reduced_chat = [{"role": "user", "content": body_text}]

    full_ids = _flatten_token_ids(
        tokenizer.apply_chat_template(
            full_chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    )
    reduced_ids = _flatten_token_ids(
        tokenizer.apply_chat_template(
            reduced_chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    )
    return compute_inserted_token_span_from_ids(full_ids, reduced_ids)


def compute_prefix_attention_sums_for_last_n(
    layer_attention: Any,
    num_candidate_suffix_tokens: int,
    prefix_start: int,
    prefix_end: int,
    head_aggregation: str = "mean",
) -> List[float]:
    """Aggregate attention-to-prefix for the final shared candidate tokens.

    Shapes:
    - input `layer_attention`: `(1, H, S, S)`
    - slice used internally: `(H, N, P)` where:
      - `N = num_candidate_suffix_tokens`
      - `P = prefix_end - prefix_start`
    - output: `(N,)`

    Meaning:
    - For each candidate suffix token, sum its attention paid to the concept
      prefix tokens.
    - Then aggregate those prefix-attention sums over heads using either `mean`
      or `max`.

    Returns:
    - list of length `N`, one prefix-attention score per candidate suffix token.
    """
    if hasattr(layer_attention, "detach"):
        import torch

        head_scores = layer_attention[0, :, -num_candidate_suffix_tokens:, prefix_start:prefix_end].sum(
            dim=-1
        )
        if head_aggregation == "mean":
            aggregated = head_scores.mean(dim=0)
        elif head_aggregation == "max":
            aggregated = head_scores.max(dim=0).values
        else:
            raise ValueError("head_aggregation must be 'mean' or 'max'.")
        return [float(value) for value in aggregated.detach().cpu().tolist()]

    if hasattr(layer_attention, "tolist"):
        layer_attention = layer_attention.tolist()

    head_scores: List[List[float]] = []
    for head_attention in layer_attention[0]:
        candidate_scores = []
        for candidate_attention in head_attention[-num_candidate_suffix_tokens:]:
            candidate_scores.append(float(sum(candidate_attention[prefix_start:prefix_end])))
        head_scores.append(candidate_scores)

    if head_aggregation == "mean":
        return [
            sum(scores[candidate_index] for scores in head_scores) / len(head_scores)
            for candidate_index in range(num_candidate_suffix_tokens)
        ]
    if head_aggregation == "max":
        return [
            max(scores[candidate_index] for scores in head_scores)
            for candidate_index in range(num_candidate_suffix_tokens)
        ]

    raise ValueError("head_aggregation must be 'mean' or 'max'.")


def collect_attention_trace_for_prompt_pair(
    prompt_pair: StatementPromptPair,
    resources: HFModelResources,
    head_aggregation: str = "mean",
) -> PromptAttentionTrace:
    """Run one positive upstream-style prompt through the model and collect attentions.

    Shape flow:
    1. The positive prompt is formatted with the model's chat template and
       tokenized into `input_ids` with shape `(1, S)`.
    2. Hugging Face returns attentions per layer with shape `(1, H, S, S)`.
    3. Each layer is reduced to `(N,)`, where `N` is the number of shared suffix
       candidate tokens.
    4. The full trace is stacked conceptually into `(L, N)`.
    """
    import torch

    tokenizer = resources.tokenizer
    model = resources.model
    prefix_start, prefix_end = find_prefix_token_span(
        tokenizer,
        prompt_pair.positive_prefix_text,
        prompt_pair.body_text,
    )

    encoded_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_pair.positive_full_prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(encoded_inputs, "to"):
        encoded_inputs = encoded_inputs.to(model.device)
    encoded_inputs = _normalize_model_inputs(encoded_inputs)

    model_kwargs: Dict[str, Any] = {
        "input_ids": encoded_inputs["input_ids"],
        "output_attentions": True,
        "return_dict": True,
        "use_cache": False,
    }
    if "attention_mask" in encoded_inputs:
        model_kwargs["attention_mask"] = encoded_inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(**model_kwargs)

    input_ids = _flatten_token_ids(encoded_inputs["input_ids"])
    candidate_token_ids = input_ids[-resources.num_candidate_suffix_tokens :]
    candidate_token_texts = resources.tokenizer.convert_ids_to_tokens(candidate_token_ids)
    candidate_token_relative_indices = list(
        range(-resources.num_candidate_suffix_tokens, 0)
    )

    layer_attention_scores = [
        compute_prefix_attention_sums_for_last_n(
            layer_attention=layer_attention,
            num_candidate_suffix_tokens=resources.num_candidate_suffix_tokens,
            prefix_start=prefix_start,
            prefix_end=prefix_end,
            head_aggregation=head_aggregation,
        )
        for layer_attention in outputs.attentions
    ]

    return PromptAttentionTrace(
        prompt_id=prompt_pair.prompt_id,
        concept_type=prompt_pair.concept_type,
        concept_value=prompt_pair.concept_value,
        statement_index=prompt_pair.statement_index,
        statement_text=prompt_pair.statement_text,
        prefix_text=prompt_pair.positive_prefix_text,
        body_text=prompt_pair.body_text,
        full_prompt_text=prompt_pair.positive_full_prompt,
        prefix_token_span=(prefix_start, prefix_end),
        candidate_token_texts=list(candidate_token_texts),
        candidate_token_relative_indices=candidate_token_relative_indices,
        layer_attention_scores=layer_attention_scores,
    )


def summarize_layer_to_token_index(prompt_traces: Sequence[PromptAttentionTrace]) -> Dict[int, int]:
    """Collapse many prompt traces into one negative token index per layer.

    This mirrors the upstream post-processing:
    - start from scores shaped `(P, L, N)`,
    - take the maximum over prompts to get `(L, N)`,
    - choose the best candidate token index for each layer,
    - convert candidate positions into negative indexing in the range `[-N, -1]`.
    """
    if not prompt_traces:
        raise ValueError("prompt_traces must be non-empty.")

    num_layers = len(prompt_traces[0].layer_attention_scores)
    num_candidates = len(prompt_traces[0].layer_attention_scores[0])
    layer_to_token_index: Dict[int, int] = {}

    for layer_index in range(num_layers):
        best_candidate_index = 0
        best_score = float("-inf")
        for candidate_index in range(num_candidates):
            candidate_best_prompt_score = max(
                trace.layer_attention_scores[layer_index][candidate_index]
                for trace in prompt_traces
            )
            if candidate_best_prompt_score > best_score:
                best_score = candidate_best_prompt_score
                best_candidate_index = candidate_index

        layer_to_token_index[layer_index] = best_candidate_index - num_candidates

    return layer_to_token_index


def collect_attention_run(
    prompt_pairs: Sequence[StatementPromptPair],
    resources: HFModelResources,
    head_aggregation: str = "mean",
    statement_stride: int = 2,
) -> AttentionCollectionRun:
    """Collect upstream-style attention traces for one concept."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - fallback only used on minimal envs
        tqdm = lambda iterable, **_: iterable

    prompt_traces = [
        collect_attention_trace_for_prompt_pair(
            prompt_pair=prompt_pair,
            resources=resources,
            head_aggregation=head_aggregation,
        )
        for prompt_pair in tqdm(prompt_pairs, desc=f"Collecting {prompt_pairs[0].concept_value}")
    ]

    layer_to_token_index = summarize_layer_to_token_index(prompt_traces)

    return AttentionCollectionRun(
        concept_type=prompt_pairs[0].concept_type,
        concept_value=prompt_pairs[0].concept_value,
        model_id=resources.model_id,
        model_tag=resources.model_tag,
        head_aggregation=head_aggregation,
        num_prompts=len(prompt_traces),
        num_layers=len(prompt_traces[0].layer_attention_scores),
        num_candidate_suffix_tokens=resources.num_candidate_suffix_tokens,
        candidate_token_relative_indices=list(prompt_traces[0].candidate_token_relative_indices),
        candidate_token_texts=list(prompt_traces[0].candidate_token_texts),
        statement_stride=statement_stride,
        prompt_traces=prompt_traces,
        layer_to_token_index=layer_to_token_index,
    )


def attention_run_to_array(run: AttentionCollectionRun) -> List[List[List[float]]]:
    """Convert a run to a plain `(P, L, N)` nested list."""
    return [trace.layer_attention_scores for trace in run.prompt_traces]


def save_attention_run(run: AttentionCollectionRun, output_dir: str) -> Dict[str, Path]:
    """Write `.npy` scores plus JSON metadata for one attention collection run."""
    import numpy as np

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    base_name = (
        f"attentions_{run.head_aggregation}head_{run.model_tag}_{run.concept_value}_paired_statements"
    )

    array_path = directory / f"{base_name}.npy"
    metadata_path = directory / f"{base_name}.metadata.json"
    layer_map_path = directory / f"{base_name}.layer_to_token.json"

    np.save(array_path, np.array(attention_run_to_array(run), dtype=np.float32))
    metadata_path.write_text(
        json.dumps(
            {
                "concept_type": run.concept_type,
                "concept_value": run.concept_value,
                "model_id": run.model_id,
                "model_tag": run.model_tag,
                "head_aggregation": run.head_aggregation,
                "num_prompts": run.num_prompts,
                "num_layers": run.num_layers,
                "num_candidate_suffix_tokens": run.num_candidate_suffix_tokens,
                "candidate_token_relative_indices": run.candidate_token_relative_indices,
                "candidate_token_texts": run.candidate_token_texts,
                "statement_stride": run.statement_stride,
                "prompt_traces": [asdict(trace) for trace in run.prompt_traces],
            },
            indent=2,
            sort_keys=True,
        )
    )
    layer_map_path.write_text(json.dumps(run.layer_to_token_index, indent=2, sort_keys=True))

    return {
        "array_path": array_path,
        "layer_map_path": layer_map_path,
        "metadata_path": metadata_path,
    }
