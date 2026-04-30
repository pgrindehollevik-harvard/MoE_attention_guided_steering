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
- **Stage 1b, current next run**: repeat the question-only SteerMoE review on
  `mistralai/Mixtral-8x7B-Instruct-v0.1`, with `meta-llama/Llama-3.1-8B-Instruct`
  as an unsteered reference column.
- **Stage 2, planned later**: compare a same-model SteerMoE baseline to an
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
- a Mixtral backend that mirrors the same custom-steering workflow against
  Hugging Face Mixtral router gates,
- a Mixtral question-only review runner whose columns are:
  - Llama 3.1 8B baseline, unsteered,
  - Mixtral 8x7B baseline, unsteered,
  - Mixtral 8x7B + SteerMoE,
- unit tests for the core logic.

The old numbered toy scripts and single-token OLMoE hybrid entrypoints have
been removed so the repo reflects the experiment we actually want to run.

## Repository layout

- `collect_attention_to_prefix.py`
  Real-model GPU script for collecting upstream-style attention-to-prefix scores.
- `run_olmoe_steermoe_review.py`
  End-to-end OLMoE experiment runner for **question-only baseline vs SteerMoE**.
- `run_mixtral_steermoe_review.py`
  End-to-end Mixtral runner for **Llama baseline vs Mixtral baseline vs Mixtral + SteerMoE**.
- `compare_prefix_conditioned_models.py`
  Older full-prefix diagnostic comparing unsteered Llama 3.1 8B, OLMoE, and Qwen1.5-MoE behavior.
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
> changes on our fear dataset without including the concept prefix at test time?

### Local command

```bash
python3 prepare_manual_fear_review.py

python3 run_olmoe_steermoe_review.py \
  --plan-json outputs/manual_fear_review/manual_review_plan.json \
  --output-dir experiments/olmoe_steermoe_fears_seed7
```

### Exact MIT cluster repeat

```bash
cd ~/MoE_attention_guided_steering
git pull
source .venv/bin/activate
python3 prepare_manual_fear_review.py

REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
MODEL_ID=allenai/OLMoE-1B-7B-0125-Instruct \
MODEL_TAG=olmoe_1b_7b_0125_instruct \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/olmoe_steermoe_fears_seed7 \
sbatch --partition=mit_normal_gpu --time=06:00:00 slurm/run_olmoe_steermoe_review.sbatch
```

After completion, the main colleague-facing file is:

- `experiments/olmoe_steermoe_fears_seed7/qualitative_review.html`

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

Baseline and SteerMoE use the **same question-only test prompt** for each case.
In the fears setting that prompt is:

```text
{evaluation_question}
```

For example:

```text
What is the scariest thing in the world? Answer in 20 words or less.
```

The concept prefix is still used to learn the SteerMoE routing plan from
paired prefix/control statement prompts. It is intentionally omitted during
baseline and steered generation, so any concept-specific effect in the SteerMoE
column has to come from router bias rather than from the prompt text.

### Output structure

The stage-1 OLMoE run writes:

- `experiments/olmoe_steermoe_fears_seed7/custom_steering_datasets/`
- `experiments/olmoe_steermoe_fears_seed7/routing_traces/`
- `experiments/olmoe_steermoe_fears_seed7/activation_tables/`
- `experiments/olmoe_steermoe_fears_seed7/steering_plans/`
- `experiments/olmoe_steermoe_fears_seed7/generation_metadata.json`
- `experiments/olmoe_steermoe_fears_seed7/manual_review_plan.json`
- `experiments/olmoe_steermoe_fears_seed7/qualitative_review.md`
- `experiments/olmoe_steermoe_fears_seed7/qualitative_review.html`

## Stage 1b: Mixtral SteerMoE review with Llama reference

The current requested comparison is not the prefix-conditioned diagnostic. It is
a question-only steering report with these columns:

```text
Llama 3.1 8B baseline | Mixtral 8x7B baseline | Mixtral 8x7B + SteerMoE
```

All three columns receive only the bare evaluation question at generation time.
The concept prefix is still used upstream to build paired routing traces and
select Mixtral experts, but it is not included in the test prompt.

### Exact MIT cluster repeat

```bash
cd ~/MoE_attention_guided_steering
git pull --ff-only origin main
source .venv/bin/activate
python3 prepare_manual_fear_review.py

REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/mixtral_steermoe_fears_seed7_question_only_with_llama \
sbatch --partition=mit_normal_gpu --time=12:00:00 slurm/run_mixtral_steermoe_review.sbatch
```

The main colleague-facing file is:

- `experiments/mixtral_steermoe_fears_seed7_question_only_with_llama/qualitative_review.html`

The Mixtral runner loads Mixtral in 4-bit by default via bitsandbytes. That is
the practical single-GPU path for a roughly 47B-total, 13B-active MoE checkpoint.

## Prefix-conditioned model suitability diagnostic

Parmida's model-quality question is handled as a separate diagnostic:

> If we explicitly include the concept prefix, does OLMoE underperform a dense
> Llama 3.1 8B baseline and another MoE model?

The default config compares:

- `meta-llama/Llama-3.1-8B-Instruct`
- `allenai/OLMoE-1B-7B-0125-Instruct`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat`

Run locally or on a GPU node:

```bash
python3 compare_prefix_conditioned_models.py \
  --plan-json outputs/manual_fear_review/manual_review_plan.json \
  --model-config-json configs/prefix_conditioned_model_comparison.json \
  --output-dir experiments/prefix_conditioned_model_comparison_fears_seed7
```

MIT cluster repeat:

```bash
REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/prefix_conditioned_model_comparison_fears_seed7 \
sbatch --partition=mit_normal_gpu --time=12:00:00 slurm/compare_prefix_conditioned_models.sbatch
```

The main output is:

- `experiments/prefix_conditioned_model_comparison_fears_seed7/model_comparison.html`

This diagnostic does **not** test steering. It includes the prefix on purpose so
we can judge whether OLMoE is a poor base model for this prompt family before
interpreting the prefix-free steering results.

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

After the Mixtral question-only review is run and inspected, the next clean
comparison is method-level rather than architecture-level:

- baseline on the chosen MoE,
- SteerMoE on that same MoE,
- an attention-based steering method on that same MoE.

The repo keeps OLMoE and Mixtral as separate backends so we do not mix a model
capacity question with a steering-method question.
