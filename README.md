# MoE Attention-Guided Steering

This repo is for one concrete research question:

> Can a SteerMoE-style router intervention make a model answer a bare question
> as if a hidden fear concept were present, without putting that concept in the
> test prompt?

The important constraint is the prompt contract. During evaluation, the model
does **not** see:

```text
Personify someone who is terrified of Bugs.
```

It only sees the bare question, for example:

```text
What is the scariest thing in the world? Answer in 20 words or less.
```

Any concept-specific movement should come from router steering, not from the
words in the prompt.

## Current Next Run

The current experiment to run on ORCD is:

```text
Llama 3.1 8B baseline | Mixtral 8x7B baseline | Mixtral 8x7B + SteerMoE
```

All three columns are question-only. Llama is an unsteered reference. Mixtral
baseline and Mixtral + SteerMoE use the same Mixtral checkpoint and the same
bare question; the third column adds router-logit bias from the saved SteerMoE
plan.

On the cluster:

```bash
cd ~/MoE_attention_guided_steering
git pull --ff-only origin main
source .venv/bin/activate
pip install -U -r requirements-gpu.txt
python3 prepare_manual_fear_review.py

REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/mixtral_steermoe_fears_seed7_question_only_with_llama \
sbatch --partition=mit_normal_gpu --time=02:00:00 slurm/run_mixtral_steermoe_review.sbatch
```

The `pip install -U` line matters for Mixtral: current Transformers 4-bit
loading requires `bitsandbytes>=0.46.1`.

Watch it:

```bash
squeue -u $USER
tail -n 120 logs/mixtral-steermoe-<JOBID>.out
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
```

Main file to read when it finishes:

```text
experiments/mixtral_steermoe_fears_seed7_question_only_with_llama/qualitative_review.html
```

## What The HTML Means

For each sampled fear concept and evaluation question, the report shows:

- `Llama 3.1 8B baseline (question only)`: unsteered dense reference.
- `Mixtral 8x7B baseline (question only)`: unsteered Mixtral.
- `Mixtral 8x7B + SteerMoE (question only)`: same Mixtral model, same question,
  plus router bias.

Read it left to right:

1. Does Llama give a sane answer to the bare question?
2. Does unsteered Mixtral give a sane answer to the same bare question?
3. Does Mixtral + SteerMoE become more concept-faithful without becoming worse?

The concept prefix shown in the report is there for reviewer context only. It is
not sent during generation.

## What Is Implemented

- OLMoE question-only SteerMoE review.
- Mixtral question-only SteerMoE review with Llama reference.
- Router-trace collection over the matched statement-body span.
- Risk-difference expert scoring.
- Router-logit bias during generation.
- HTML/Markdown reports for qualitative inspection.
- A future-facing attention-to-prefix collector, kept for later method
  comparison work.

Removed on purpose:

- old toy pipeline abstractions,
- the single-token hybrid prototype,
- the prefix-conditioned model comparison path,
- one-off helper scripts that were superseded by the integrated Mixtral runner.

The repo should now point new readers toward the actual experiment instead of
asking them to mentally sort active work from old scaffolding.

## Repository Map

- `prepare_manual_fear_review.py`
  Builds the sampled fear concepts/questions used by the review reports.
- `run_mixtral_steermoe_review.py`
  Current main runner. Produces the three-column Llama/Mixtral/Mixtral+SteerMoE
  report.
- `run_olmoe_steermoe_review.py`
  Earlier OLMoE SteerMoE runner, still useful as a smaller-model baseline.
- `collect_attention_to_prefix.py`
  Attention readout collector kept for later same-model method comparisons.
- `render_manual_review_report.py`
  Re-renders a saved manual review JSON into Markdown/HTML.
- `inspect_reference_data.py`
  Quick sanity check for the vendored concept/question data.
- `src/moe_attention_guided_steering/olmoe_backend.py`
  OLMoE router tracing and router-bias hooks.
- `src/moe_attention_guided_steering/mixtral_backend.py`
  Mixtral router tracing and router-bias hooks.
- `src/moe_attention_guided_steering/manual_review.py`
  Shared review-plan schema and HTML/Markdown renderer.
- `src/moe_attention_guided_steering/upstream_prompt_datasets.py`
  Builds paired concept/control statement prompts for SteerMoE scoring.
- `src/moe_attention_guided_steering/attention_collection.py`
  Shared HF loading utilities plus the future attention readout path.
- `slurm/run_mixtral_steermoe_review.sbatch`
  ORCD launcher for the current main run.
- `slurm/run_olmoe_steermoe_review.sbatch`
  ORCD launcher for the OLMoE run.

## Local Sanity Checks

These do not load real model weights:

```bash
python3 inspect_reference_data.py --concept-type fears
python3 -m unittest discover -s tests
python3 run_mixtral_steermoe_review.py --help
```

For a tiny ORCD smoke test before the full run:

```bash
LIMIT_CASES=1 \
OUTPUT_DIR=experiments/mixtral_steermoe_smoke \
sbatch --partition=mit_normal_gpu --time=02:00:00 slurm/run_mixtral_steermoe_review.sbatch
```

## OLMoE Run

The OLMoE path is still here because it is the smaller, earlier reproduction
target. Run it when you want the original baseline-vs-SteerMoE report:

```bash
REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/olmoe_steermoe_fears_seed7 \
sbatch --partition=mit_normal_gpu --time=06:00:00 slurm/run_olmoe_steermoe_review.sbatch
```

Main output:

```text
experiments/olmoe_steermoe_fears_seed7/qualitative_review.html
```

## Output Layout

Both steering runners write the same kind of bundle:

```text
custom_steering_datasets/
routing_traces/
activation_tables/
steering_plans/
generation_metadata.json
manual_review_plan.json
qualitative_review.md
qualitative_review.html
```

The most useful files for research review are:

- `qualitative_review.html`: the human-facing side-by-side report.
- `generation_metadata.json`: prompt contract and run settings.
- `steering_plans/`: selected experts per concept.
- `activation_tables/`: full risk-difference tables.

## Research Notes

The current SteerMoE implementation follows the custom-steering shape more
closely than the old prototype:

1. Build matched concept/control prompt pairs from the same statement body.
2. Collect router choices over that shared statement body.
3. Aggregate expert activation counts across the paired dataset.
4. Score experts by risk difference.
5. Select globally strongest positive and negative experts.
6. Apply router-logit bias during question-only generation.

Later, the clean method comparison should happen on one chosen MoE:

```text
baseline | SteerMoE | attention-guided steering
```

That is where the attention machinery comes back in. For now, the immediate job
is to see whether Mixtral + SteerMoE produces a stronger question-only effect
than the smaller OLMoE run.
