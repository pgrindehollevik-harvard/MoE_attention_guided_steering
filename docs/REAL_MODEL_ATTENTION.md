# Real-Model Attention Collection

This document explains the first real GPU stage we now support in this repo:
collecting attention-guided token-selection scores from an actual chat model,
following the structure of the upstream
`pdavar/attention_guided_steering` repository.

## What this stage computes

The upstream attention-guided steering method does **not** choose an arbitrary
token like the final hidden state by default. Instead, for each layer, it asks:

> Among the shared suffix tokens at the end of the chat-formatted prompt, which
> token pays the most attention to the concept prefix?

This repo now implements that same stage in
`src/moe_attention_guided_steering/attention_collection.py`.

## Shapes

For one prompt and one layer, Hugging Face returns an attention tensor:

- `(1, H, S, S)`

where:

- `H` = number of heads
- `S` = full tokenized prompt length

We then reduce it as follows:

1. Identify the token span of the concept prefix inside the prompt:
   - prefix span length `P`
2. Take the last shared suffix tokens used as candidate representation tokens:
   - candidate count `N`
3. Slice the attention tensor to:
   - `(H, N, P)`
4. Sum over the prefix axis:
   - `(H, N)`
5. Aggregate over heads with mean or max:
   - `(N,)`

So each layer ends up with one vector of length `N`, where each entry answers:

> How strongly does this candidate suffix token attend to the concept prefix?

Over many prompts, the saved attention artifact has shape:

- `(P_prompts, L, N)`

The layerwise token choice is then:

1. max over prompts:
   - `(L, N)`
2. argmax over candidate tokens:
   - `(L,)`

That gives one selected negative token index per layer.

## How prompts are built

We now mirror the upstream dataset construction in
`src/moe_attention_guided_steering/upstream_prompt_datasets.py`.

For fears, the positive prompt format is:

`Personify someone who is terrified of {fear}.  What are your thoughts on the following statement?`

with one generic statement inserted after that prompt body.

The negative prompt removes the concept prefix and keeps the rest unchanged.

For the attention stage, like the upstream `0_visualize_attn.py`, we only need
the positive prefixed prompts in order to decide which suffix token index should
represent the concept at each layer.

## What gets written to disk

`collect_attention_to_prefix.py` writes three files per concept:

1. `attentions_<agg>head_<model>_<concept>_paired_statements.npy`
   The raw attention scores with shape `(P_prompts, L, N)`.
2. `...metadata.json`
   Prompt text, prefix span, candidate token strings, and other run metadata.
3. `...layer_to_token.json`
   The final per-layer negative token index map, matching the upstream idea of
   selecting one token position per layer.

## Why this matters for the MoE hybrid

This attention stage is the missing bridge between the upstream attention-guided
steering repo and our MoE scaffold.

In the hybrid method, these layerwise token indices tell us **where** to inspect
MoE router behavior or expert activations. The future MoE trace collector should:

1. build the same positive/negative prompt pairs,
2. use `layer_to_token.json` to choose the token position per layer,
3. read expert loads at those positions,
4. compare those attention-selected readouts against the current statement-body
   readout used by the OLMoE and Mixtral SteerMoE backends.

That future MoE collector is still the next implementation step, but the real
attention-selection stage is no longer just a placeholder.
