# Architecture Walkthrough

This repo is now organized around one primary runnable experiment:

- **Stage 1**: baseline vs **custom-steering SteerMoE** on
  `allenai/OLMoE-1B-7B-0125-Instruct`

The earlier toy scripts and failed single-token OLMoE hybrid entrypoints were
removed so the codebase lines up with that goal.

## Top-level entrypoints

- `prepare_manual_fear_review.py`
  Samples concepts and evaluation questions into a reproducible qualitative
  review plan.
- `run_olmoe_steermoe_review.py`
  Main stage-1 runner. It builds paired custom steering examples, collects
  routing traces on the shared statement-body target, computes a
  risk-difference activation table, builds SteerMoE plans, generates baseline
  vs steered outputs, and renders HTML/Markdown.
- `render_manual_review_report.py`
  Re-renders a saved manual review plan into Markdown + HTML.
- `collect_attention_to_prefix.py`
  Dense-model attention collector kept for future same-model comparisons against
  an attention-based method.
- `inspect_reference_data.py`
  Lightweight sanity-check tool for the imported upstream text assets.

## Shared package

The reusable logic lives in `src/moe_attention_guided_steering/`.

- `config.py`
  Generic experiment knobs retained for future non-SteerMoE extensions.
- `types.py`
  Generic dataset and steering records retained for future shared utilities.
- `datasets.py`
  Loading and validation helpers for the generic experiment dataset schema.
- `moe_utils.py`
  Generic expert-delta scoring utilities retained for future comparisons.
- `pipeline.py`
  Generic orchestration retained for non-stage-1 paths.
- `reference_data.py`
  Loaders for the vendored upstream concept lists, statement pools, and
  evaluation prompts.
- `upstream_prompt_datasets.py`
  Helpers that rebuild the upstream prefixed-vs-unprefixed statement prompts and
  convert them into paired custom steering examples.
- `attention_collection.py`
  Dense-model attention-to-prefix extraction plus shared token-span helpers.
- `olmoe_backend.py`
  The stage-1 OLMoE backend. This is where matched target spans are located,
  routing traces are collected, risk-difference tables are built, and selected
  experts are injected back into generation.
- `manual_review.py`
  Condition-agnostic qualitative review plan + HTML/Markdown rendering.
- `io_utils.py`
  Generic JSON / Markdown output helpers used by multiple entrypoints.

## The stage-1 tensor flow

For one prompt on OLMoE:

1. The tokenizer converts the chat-formatted prompt into a sequence of length
   `S`.
2. OLMoE returns router logits per layer with shape `(L, S, E)`.
   Here:
   - `L` = number of MoE layers
   - `S` = sequence length
   - `E` = number of experts
3. We identify the **shared statement-body target** inside that sequence and
   slice the router tensor down to `(L, P, E)`, where `P` is the number of
   tokens in that target.
4. For each token row `(E,)`, we mark which experts were actually selected by
   top-k routing, giving a binary activation indicator over the span.
5. We sum those token-level indicators across the `P` target tokens to get one
   activation-count vector `(E,)` per layer, and divide by `P` to get one
   activation-rate vector `(E,)` per layer.
6. Across positive and negative prompt sets, we aggregate counts over the whole
   paired dataset and compute a **risk difference** for every `(layer, expert)`
   pair.
7. We globally select the strongest positive and negative experts and turn those
   into sparse expert activation / deactivation plans.

So the main stage-1 readout is now the **risk-difference activation table over
a matched target span**, not single-token router state and not the earlier
generic span-averaging abstraction.

## Why this split matters

This layout keeps the repo honest about what is generic and what is model-
specific:

- `olmoe_backend.py` is the OLMoE-specific instrumentation, scoring, and
  intervention layer for the faithful stage-1 SteerMoE path.
- `attention_collection.py` is future-facing support for same-model
  attention-based comparisons.

That means later comparisons can add a new backend or a new readout rule without
rewriting the whole repo.
