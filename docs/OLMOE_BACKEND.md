# OLMoE SteerMoE Backend

This document explains the current real MoE backend implemented in this repo:

`allenai/OLMoE-1B-7B-0125-Instruct`

The current backend is intentionally focused on **stage 1**:

1. baseline generation,
2. SteerMoE-style custom steering on our own fears data.

It does **not** treat the earlier single-token attention-guided OLMoE prototype
as the main experiment anymore.

## Why this backend exists

The goal of stage 1 is straightforward:

> Before comparing SteerMoE to anything else, verify that a SteerMoE-style
> intervention works at all on our own fear dataset.

That means the backend should stay close to Adobe's custom-steering workflow:

- build paired custom steering examples,
- collect routing behavior on an explicit shared target span,
- compute per-layer/per-expert **risk difference**,
- steer generation by favoring or suppressing globally selected experts.

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

### 2. Target-level expert activation counts and rates

The current readout target is the **shared statement body** inside each paired
fear/control prompt.

For one prompt and one layer:

1. slice the router logits to the chosen span:
   - `(P, E)`
2. take the top-k routed experts for each token in the span:
   - indices `(P, K)`
3. convert those routed experts into a binary activation matrix:
   - `(P, E)`
4. sum over target tokens:
   - activation counts `(E,)`
5. divide by the target token count:
   - `(E,)`

where:

- `P` = number of tokens in the chosen span,
- `K` = number of experts routed per token by OLMoE.

So each layer ends up with two vectors of length `E`:

- activation counts
- activation rates

Each activation-rate entry answers:

> For what fraction of span tokens was this expert selected by the router?

Over many paired prompts, those counts are added across the full custom
steering dataset, giving for each `(layer, expert)` pair:

- `messages_0_activation_count`
- `messages_1_activation_count`
- `messages_0_activation_rate`
- `messages_1_activation_rate`
- `risk_difference = messages_0_rate - messages_1_rate`

That table is the main stage-1 scoring object.

### 3. Runtime steering

After the risk-difference table is built, the steering plan globally selects:

- the strongest positive-risk experts to activate
- the strongest negative-risk experts to deactivate

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

- define paired custom steering examples,
- use the shared statement body as the matched target,
- inspect routing over **every token in that target**,
- compute risk difference directly on `(layer, expert)` activation rates,
- globally select experts for intervention.

That is closer to the paper's framing and avoids collapsing the problem to a
single chat-template boundary token.

## Main entrypoint

The current end-to-end experiment script is:

`run_olmoe_steermoe_review.py`

It:

1. loads a manual review plan,
2. builds positive/negative statement prompt pairs,
3. converts them into paired custom steering examples,
4. collects target-level OLMoE routing traces,
5. builds one SteerMoE risk-difference table and steering plan per concept,
5. generates baseline and SteerMoE outputs,
6. writes an updated JSON plan plus HTML and Markdown review reports.

The matching MIT-cluster Slurm template is:

`slurm/run_olmoe_steermoe_review.sbatch`

## Current design choices

These are implementation choices, not paper claims:

- readout target: shared statement body
- routing statistic: top-k expert activation count / rate
- scoring table: risk difference over all target tokens in the paired dataset
- default global activation budget: `8`
- default global deactivation budget: `8`
- default minimum absolute risk difference: `0.01`
- default steering coefficient: `1.0`

Those defaults are intentionally gentler than the earlier OLMoE prototype,
because the previous stronger biasing regime produced visibly degenerate text.

## What this backend does not yet claim

This backend is **not yet**:

- a same-model comparison against an attention-based steering method,
- a full reproduction of every intervention detail in the SteerMoE paper,
- a final answer about whether attention helps MoE steering.

It is the clean stage-1 baseline we need before making those later claims.
