# Architecture Walkthrough

This repo is organized around one current runnable steering experiment and one
planned method comparison:

- **Current**: baseline vs custom-steering SteerMoE on
  `allenai/OLMoE-1B-7B-0125-Instruct`.
- **Next**: baseline vs attention-guided activation steering vs SteerMoE on the
  same OLMoE checkpoint.

Older toy scripts, the prefix-conditioned model comparison path, and the
temporary larger-MoE backend were removed so the codebase lines up with the active
research question.

## Top-Level Entrypoints

- `prepare_manual_fear_review.py`
  Samples five fear concepts and five evaluation questions into a reproducible
  qualitative review plan.
- `run_olmoe_steermoe_review.py`
  Main runner. It builds paired custom steering examples, collects routing
  traces on the shared statement-body target, computes a risk-difference
  activation table, builds SteerMoE plans, generates question-only baseline vs
  steered outputs, and renders HTML/Markdown.
- `render_manual_review_report.py`
  Re-renders a saved manual review plan into Markdown and HTML.
- `collect_attention_to_prefix.py`
  Attention collector kept for the future OLMoE attention-guided activation
  steering comparison.
- `inspect_reference_data.py`
  Lightweight sanity-check tool for the imported upstream text assets.

## Shared Package

The reusable logic lives in `src/moe_attention_guided_steering/`.

- `reference_data.py`
  Loaders for the vendored upstream concept lists, statement pools, and
  evaluation prompts.
- `upstream_prompt_datasets.py`
  Helpers that rebuild the upstream prefixed-vs-unprefixed statement prompts and
  convert them into paired custom steering examples.
- `attention_collection.py`
  Attention-to-prefix extraction plus shared Hugging Face loading and token-span
  helpers.
- `generation.py`
  Generic unsteered Hugging Face generation helpers for chat-template and
  plain-template models.
- `olmoe_backend.py`
  The OLMoE backend. This is where matched target spans are located, routing
  traces are collected, risk-difference tables are built, and selected experts
  are injected back into generation.
- `manual_review.py`
  Condition-agnostic qualitative review plan plus HTML/Markdown rendering.
- `io_utils.py`
  Small filesystem and JSON helpers used by multiple entrypoints.

## The Steering Tensor Flow

For one paired prompt on OLMoE:

1. The tokenizer converts the chat-formatted prompt into a sequence of length
   `S`.
2. OLMoE returns router logits per layer with shape `(L, S, E)` after reshaping
   any flattened `(B * S, E)` router tensors.
   Here:
   - `L` = number of MoE layers
   - `S` = sequence length
   - `E` = number of experts
3. We identify the shared statement-body target inside that sequence and slice
   the router tensor down to `(L, P, E)`, where `P` is the number of target
   tokens.
4. For each token row `(E,)`, we mark which experts were selected by top-k
   routing, giving a binary activation indicator over the span.
5. We sum those token-level indicators across the `P` target tokens to get one
   activation-count vector `(E,)` per layer, and divide by `P` to get one
   activation-rate vector `(E,)` per layer.
6. Across positive and negative prompt sets, we aggregate counts over the whole
   paired dataset and compute a risk difference for every `(layer, expert)` pair.
7. We globally select the strongest positive and negative experts and turn those
   into sparse expert activation and deactivation plans.

So the current readout is the risk-difference activation table over a matched
target span, not single-token router state.

## Why This Split Matters

The repo keeps model-specific and method-specific pieces separated:

- `olmoe_backend.py` is the OLMoE router tracing and intervention layer.
- `attention_collection.py` is support for the future attention-guided readout.
- `manual_review.py` can render two-column and later three-column comparisons
  without changing the report schema.

That means the next method can be added without reopening the old model
comparison work.
