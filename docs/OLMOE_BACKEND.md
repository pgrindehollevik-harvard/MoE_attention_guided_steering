# OLMoE SteerMoE Backend

This document explains the current real MoE backend implemented in this repo:

`allenai/OLMoE-1B-7B-0125-Instruct`

The current backend is intentionally focused on **stage 1**:

1. baseline generation,
2. span-based SteerMoE on our own prompt data.

It does **not** treat the earlier single-token attention-guided OLMoE prototype
as the main experiment anymore.

## Why this backend exists

The goal of stage 1 is straightforward:

> Before comparing SteerMoE to anything else, verify that a SteerMoE-style
> intervention works at all on our own fear dataset.

That means the backend should stay close to the paper's spirit:

- collect routing behavior over a **token span**,
- compare positive and negative examples,
- steer generation by favoring or suppressing experts.

## Shape overview

There are three important tensor families in the OLMoE path.

### 1. OLMoE router logits

For one prompt, Hugging Face returns per-layer router logits:

- raw layer output: `(B * S, E)` or `(B, S, E)`

where:

- `B` = batch size,
- `S` = full tokenized prompt length after chat templating,
- `E` = number of experts in the sparse layer.

The backend reshapes the common flattened case into:

- `(B, S, E)`

The current implementation assumes:

- `B = 1`

so each layer becomes:

- `(1, S, E)`

### 2. Span-level expert activation rates

The current readout span is the **full user-content span** inside the
chat-formatted prompt.

For one prompt and one layer:

1. slice the router logits to the chosen span:
   - `(P, E)`
2. take the top-k routed experts for each token in the span:
   - indices `(P, K)`
3. convert those routed experts into a binary activation matrix:
   - `(P, E)`
4. average over span tokens:
   - `(E,)`

where:

- `P` = number of tokens in the chosen span,
- `K` = number of experts routed per token by OLMoE.

So each layer ends up with one vector of length `E`, where each entry answers:

> For what fraction of span tokens was this expert selected by the router?

Over many prompts, the resulting data can be viewed as:

- positive matrix `(N_pos, E)`
- negative matrix `(N_neg, E)`

for each layer separately.

### 3. Runtime steering

After the positive-vs-negative comparison, the steering plan says which experts
to favor or suppress in each layer.

At generation time, the backend converts that sparse plan into one dense bias
vector per layer:

- router bias vector: `(E,)`

During generation, that bias is injected into the gate output inside:

- `model.model.layers[layer_index].mlp.gate`

For one generation forward pass, the gate output has shape:

- `(T, E)`

where:

- `T = S_prompt` during prompt prefill,
- `T = 1` during cached autoregressive decoding.

The backend biases only row `T - 1`, because that is the token whose hidden
state determines the next generated step.

## How this differs from the earlier prototype

The retired prototype did this:

- choose one suffix token per layer,
- read router probabilities only at that token,
- build expert deltas from those single-token vectors.

The current backend does this instead:

- define a meaningful prompt span,
- inspect routing over **every token in that span**,
- build expert deltas from span-aggregated activation rates.

That is closer to the paper's framing and avoids collapsing the problem to a
single chat-template boundary token.

## Main entrypoint

The current end-to-end experiment script is:

`run_olmoe_steermoe_review.py`

It:

1. loads a manual review plan,
2. builds positive/negative statement prompt pairs,
3. collects span-based OLMoE routing statistics,
4. builds one SteerMoE plan per concept,
5. generates baseline and SteerMoE outputs,
6. writes an updated JSON plan plus HTML and Markdown review reports.

The matching MIT-cluster Slurm template is:

`slurm/run_olmoe_steermoe_review.sbatch`

## Current design choices

These are implementation choices, not paper claims:

- readout span: full user-content span
- routing statistic: top-k expert activation rate
- default top-k intervention sparsity: `2`
- default activation threshold: `0.01`
- default deactivation threshold: `-0.01`
- default steering coefficient: `1.0`

Those defaults are intentionally gentler than the earlier OLMoE prototype,
because the previous stronger biasing regime produced visibly degenerate text.

## What this backend does not yet claim

This backend is **not yet**:

- a same-model comparison against an attention-based steering method,
- a full reproduction of every intervention detail in the SteerMoE paper,
- a final answer about whether attention helps MoE steering.

It is the clean stage-1 baseline we need before making those later claims.
