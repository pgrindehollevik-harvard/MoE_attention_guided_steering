# Architecture Walkthrough

This codebase is organized to feel familiar to anyone who has already read the attention-guided steering repository.

## Numbered scripts

The top-level scripts give the project a very explicit research workflow:

1. `0_validate_dataset.py`
   Loads the dataset and checks that the experiment specification is structurally sound.
2. `1_select_representative_tokens.py`
   Chooses the highest-attention token for each layer and example.
3. `2_compute_expert_scores.py`
   Compares expert usage between positive and negative examples.
4. `3_build_steering_plan.py`
   Converts those expert scores into a concrete intervention plan.
5. `4_export_experiment_report.py`
   Produces a human-readable Markdown summary for collaborators.

There is also one utility script outside the numbered pipeline:

- `inspect_reference_data.py`
  Summarizes the imported attention-guided steering text assets under `data/`.

## Shared package

The reusable logic lives in `src/moe_attention_guided_steering/`.

- `config.py`
  Small dataclasses for the knobs we expect to tune during experiments.
- `types.py`
  Dataclasses for the structured records flowing through the pipeline.
- `datasets.py`
  Input loading and validation.
- `attention_utils.py`
  Representative-token selection.
- `moe_utils.py`
  Expert scoring and steering-plan construction.
- `pipeline.py`
  A thin orchestration layer that wires the pieces together.
- `io_utils.py`
  Output helpers for JSON and Markdown reports.
- `reference_data.py`
  Loaders and adapters for the imported attention-guided steering text assets.

## Why this split matters

Research code becomes hard to reason about when the data schema, selection logic, scoring logic, and output formatting all live in the same script. This repo splits those concerns so collaborators can answer very specific questions:

- "How are representative tokens chosen?" -> `attention_utils.py`
- "How do we decide which experts to activate?" -> `moe_utils.py`
- "What exactly is in the input file?" -> `types.py` and `datasets.py`
- "What did a specific run produce?" -> `outputs/...` and `io_utils.py`

## Toy mode versus real-model mode

Right now the repo is deliberately in **toy mode**. That means the input is a JSON file containing pre-computed attention weights and expert loads. This keeps the core research logic understandable before we introduce heavy framework dependencies.

When we move to a real MoE model, the main changes should be:

- add a model-instrumentation module that records attention and expert loads,
- use `reference_data.py` to pick concept catalogs and evaluation templates from the imported upstream `data/` tree,
- write those traces into the same schema used by the toy dataset,
- keep the rest of the pipeline unchanged.

That separation is useful because it lets us debug model collection and intervention planning independently.
