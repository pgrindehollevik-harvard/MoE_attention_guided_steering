# OLMoE Backend

This document explains the first real Mixture-of-Experts backend implemented in
this repo: `allenai/OLMoE-1B-7B-0125-Instruct`.

The goal of this backend is to make the three-condition qualitative experiment
actually runnable on a real MoE checkpoint:

1. baseline generation,
2. original MoESteer using one fixed suffix token,
3. attention-guided MoESteer using the per-layer token choices from the
   attention collector.

## Why OLMoE is a good first backend

OLMoE is a strong bring-up target because:

- it is an actual sparse MoE model exposed in Hugging Face Transformers,
- it returns per-layer `router_logits`,
- the instruct checkpoint uses a normal chat template,
- and it is much lighter to iterate on than larger MoE checkpoints.

## Shape overview

There are three different tensor families in the OLMoE path.

### 1. Attention-guided token selection

This is still handled by the generic attention collector:

- model attentions per layer: `(1, H, S, S)`
- reduced per-layer suffix-token scores: `(N,)`
- stacked attention run: `(P, L, N)`

where:

- `H` = number of attention heads,
- `S` = full prompt length after chat templating,
- `N` = number of shared suffix candidate tokens,
- `P` = number of prompt pairs.

The result is one negative token index per layer:

- `layer_to_token_index[layer] -> {-N, ..., -1}`

### 2. OLMoE router traces

For one prompt, OLMoE returns router logits for each layer:

- raw Hugging Face output per layer: `(B * S, E)`

The backend reshapes that into:

- `(B, S, E)`

and then slices the final shared suffix candidate tokens:

- `(B, N, E)`

For batch size `1`, we drop the singleton batch dimension:

- `(N, E)`

Stacking over layers gives:

- `(L, N, E)`

where:

- `L` = number of decoder layers,
- `E` = number of experts in each sparse layer.

These are converted into the repo's generic dataset schema by storing each
candidate suffix token as one `TokenRecord` whose:

- `attention_weight` is a one-hot selection score used only for argmax token
  choice in the generic pipeline,
- `expert_loads` is the router-probability vector `(E,)`.

### 3. Runtime steering

The sparse steering plan says which experts to favor or suppress in each layer.
At generation time, the OLMoE backend converts that into one dense bias vector
per layer:

- router bias vector: `(E,)`

During generation, that vector is added to the final token row of the gate
output inside `model.model.layers[layer].mlp.gate`.

For one generation forward pass, the gate output has shape:

- `(T, E)`

where:

- `T = S_prompt` during prompt prefill,
- `T = 1` during cached autoregressive decoding.

The backend only biases row `T - 1`, so the intervention applies to the token
that controls the next generated step.

## Original vs attention-guided MoESteer

The only difference between the two steered conditions is how the steering plan
chooses its representative token per layer.

### Original MoESteer

- Use one fixed suffix token index for every layer.
- Default choice in this repo: `-1` (the final shared suffix token).

### Attention-guided MoESteer

- Use `layer_to_token.json` from the attention collector.
- Each layer may pick a different suffix token index.

After that choice, both conditions share the exact same downstream steps:

1. collect OLMoE router probabilities,
2. compute positive-vs-negative expert deltas,
3. keep sparse top-k expert interventions,
4. apply router-logit bias during generation.

## Main entrypoint

The end-to-end experiment script is:

`run_olmoe_manual_review.py`

It:

1. loads a manual review plan,
2. collects or reuses attention token maps,
3. builds fixed-token and attention-guided steering plans,
4. generates baseline / MoESteer / attention-guided MoESteer outputs,
5. writes an updated JSON plan plus HTML and Markdown review reports.

The matching Slurm template is:

`slurm/run_olmoe_manual_review.sbatch`
