# MoE Attention-Guided Steering

This repository studies two related but distinct ideas:

- **SteerMoE**: steer a Mixture-of-Experts model by identifying experts that are
  associated with a target behavior and biasing routing toward or away from them.
- **Attention-guided steering**: use attention patterns to decide where a
  concept signal is best represented, instead of blindly reading from a fixed
  token position.

The current repo is deliberately staged.

- **Stage 1, implemented and GPU-ready now**: test whether **SteerMoE transfers
  to our own fear/phobia-style data** on `allenai/OLMoE-1B-7B-0125-Instruct`.
- **Stage 2, planned next**: compare that OLMoE SteerMoE baseline to an
  attention-based alternative on the **same model**, so the comparison is about
  methods rather than architecture changes.

## Current research stance

After rereading the SteerMoE paper and Adobe's `custom_steering.ipynb`, the
repo now treats a **custom-steering replication on our fears data** as the main
runnable path.

Why:

- the SteerMoE paper computes expert statistics from routing evidence collected
  on a task-relevant target region, not from a single representative token,
- Adobe's custom notebook saves paired routing activations and scores experts by
  **risk difference** directly,
- our earlier generic span-aggregation path was useful as a stepping stone, but
  it was still too far from the paper's custom-steering workflow.

So the repo is no longer presenting the failed single-token hybrid as the main
experiment. That code path has been retired from the committed source.

## What is implemented now

- a typed dataset / steering-plan library under `src/moe_attention_guided_steering/`,
- a real-model attention-to-prefix collector kept for future same-model
  attention-based comparisons,
- an OLMoE backend that:
  - builds Adobe-style paired custom steering examples from our fears prompts,
  - collects router traces on the shared statement-body target,
  - computes a SteerMoE activation table with per-layer/per-expert risk
    differences,
  - selects globally strongest positive and negative experts,
  - applies router-logit bias during generation,
  - writes a qualitative baseline-vs-SteerMoE review report,
- unit tests for the core logic.

The old numbered toy scripts and single-token OLMoE hybrid entrypoints have
been removed so the repo reflects the experiment we actually want to run.

## Repository layout

- `collect_attention_to_prefix.py`
  Real-model GPU script for collecting upstream-style attention-to-prefix scores.
- `run_olmoe_steermoe_review.py`
  End-to-end OLMoE experiment runner for **baseline vs SteerMoE**.
- `prepare_manual_fear_review.py`
  Builds a qualitative review worksheet for the currently configured conditions.
- `render_manual_review_report.py`
  Renders Markdown + HTML from a saved review-plan JSON.
- `inspect_reference_data.py`
  Quick summary tool for the imported attention-guided steering text assets.
- `src/moe_attention_guided_steering/`
  Reusable library code.
- `data/`
  Vendored concept lists, statement pools, and evaluation prompts from the
  upstream attention-guided steering repo.
- `docs/ARCHITECTURE.md`
  Codebase walkthrough.
- `docs/RESEARCH_SYNTHESIS.md`
  Plain-English mapping from the reference papers to this repo.
- `docs/REAL_MODEL_ATTENTION.md`
  Explanation of the real-model attention collection stage.
- `docs/OLMOE_BACKEND.md`
  Explanation of the stage-1 OLMoE SteerMoE backend and tensor shapes.
- `docs/SLURM_RUNBOOK.md`
  MIT-cluster-oriented GPU launch notes.
- `tests/`
  Unit tests for core logic.

## Quickstart

From the repository root:

```bash
python3 inspect_reference_data.py --concept-type fears
python3 -m unittest discover -s tests
```

## Stage 1: OLMoE SteerMoE transfer experiment

The first real MoE backend targets:

```bash
allenai/OLMoE-1B-7B-0125-Instruct
```

The current stage-1 experiment asks:

> Can a SteerMoE-style custom steering pipeline produce meaningful qualitative
> changes on our fear dataset without destroying fluency?

### Local command

```bash
python3 prepare_manual_fear_review.py

python3 run_olmoe_steermoe_review.py \
  --plan-json outputs/manual_fear_review/manual_review_plan.json \
  --output-dir experiments/olmoe_steermoe_fears_seed7
```

### What this runner does

For each sampled concept, it:

1. builds upstream-style positive/negative statement prompt pairs,
2. converts those pairs into Adobe-style custom steering examples,
3. runs OLMoE and collects router traces on the shared statement-body target,
4. computes a per-layer/per-expert **risk-difference activation table**,
5. selects the globally strongest positive and negative experts,
6. generates:
   - baseline output,
   - SteerMoE output,
7. renders:
   - `manual_review_plan.json`
   - `qualitative_review.md`
   - `qualitative_review.html`

### Output structure

The stage-1 OLMoE run writes:

- `experiments/olmoe_steermoe_fears_seed7/custom_steering_datasets/`
- `experiments/olmoe_steermoe_fears_seed7/routing_traces/`
- `experiments/olmoe_steermoe_fears_seed7/activation_tables/`
- `experiments/olmoe_steermoe_fears_seed7/steering_plans/`
- `experiments/olmoe_steermoe_fears_seed7/manual_review_plan.json`
- `experiments/olmoe_steermoe_fears_seed7/qualitative_review.md`
- `experiments/olmoe_steermoe_fears_seed7/qualitative_review.html`

## Real-model attention collection

The repo still includes a real-model attention stage inspired by the upstream
`attention_guided_steering` workflow:

```bash
python3 collect_attention_to_prefix.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --model-tag llama_3_1_8b_instruct \
  --concept-type fears \
  --sample-concepts 5 \
  --seed 7 \
  --statement-stride 2
```

That stage is **not** the main OLMoE experiment anymore. It now lives in the
repo as supporting infrastructure for the later same-model comparison against an
attention-based method.

See [docs/REAL_MODEL_ATTENTION.md](/Users/peterflo/Desktop/MoE_attention_guided_steering/docs/REAL_MODEL_ATTENTION.md)
for the shape walkthrough.

## Why the current SteerMoE backend is closer to the paper

The old prototype used one selected suffix token per layer. The current OLMoE
backend is closer to SteerMoE because it now follows the custom-steering logic
more directly:

- paired concept/control prompts are built from the same statement body,
- the shared statement body is used as the matched routing target,
- routed-expert counts are collected for every token in that target,
- activation rates and risk differences are computed directly for each
  `(layer, expert)` pair,
- the globally strongest positive and negative experts are selected for runtime
  steering.

See [docs/OLMOE_BACKEND.md](/Users/peterflo/Desktop/MoE_attention_guided_steering/docs/OLMOE_BACKEND.md)
for the exact tensor story.

## What comes next

After we have a trustworthy SteerMoE-on-our-data baseline, the next clean
comparison is:

- baseline OLMoE
- SteerMoE on OLMoE
- an attention-based steering method on **the same OLMoE model**

That next comparison will be method-level rather than architecture-level, which
is exactly why the repo now prioritizes a clean stage-1 SteerMoE path first.
