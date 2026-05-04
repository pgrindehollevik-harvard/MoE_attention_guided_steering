# MoE Attention-Guided Steering

This repo is for a concrete research sequence around OLMoE:

1. Run SteerMoE on our five sampled fear concepts and compare it to an
   unsteered OLMoE baseline.
2. If that transfer check is meaningful, compare three OLMoE conditions:
   baseline, attention-guided activation steering, and SteerMoE.

The repo is not currently organized around prefix-conditioned model comparison
or a larger-MoE detour. Those paths were useful while we were debugging the
question, but the active experiment is OLMoE again.

## Prompt Contract

During evaluation, the model does **not** see the concept prefix:

```text
Personify someone who is terrified of Bugs.
```

It only sees the bare question, for example:

```text
What is the scariest thing in the world? Answer in 20 words or less.
```

Any concept movement should come from steering. If the prompt itself names the
fear, the experiment is answering an easier and less interesting question.

## Current Run

Run this first:

```text
OLMoE baseline | OLMoE + SteerMoE
```

Both columns use `allenai/OLMoE-1B-7B-0125-Instruct`. Both columns receive the
same question-only prompt. The baseline has no intervention; the SteerMoE column
adds router-logit bias from the selected OLMoE experts.

On ORCD:

```bash
cd ~/MoE_attention_guided_steering
git pull --ff-only origin main
source .venv/bin/activate
pip install -U -r requirements-gpu.txt
python3 prepare_manual_fear_review.py

REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/olmoe_steermoe_fears_seed7_question_only \
sbatch --partition=mit_normal_gpu --time=02:00:00 slurm/run_olmoe_steermoe_review.sbatch
```

Watch it:

```bash
squeue -u $USER
tail -n 120 logs/olmoe-steermoe-<JOBID>.out
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
```

Main file to open when it finishes:

```text
experiments/olmoe_steermoe_fears_seed7_question_only/qualitative_review.html
```

## What The HTML Means

Each section is one fear concept plus one evaluation question.

- `OLMoE baseline (question only)`: unsteered OLMoE answering the bare question.
- `OLMoE + SteerMoE (question only)`: the same OLMoE checkpoint, same question,
  plus router bias from that concept's saved SteerMoE plan.

The report also shows a `Prefix-conditioned diagnostic prompt`. That text is
for the reviewer, not the model. In the actual generation call,
`case.full_prompt_text = case.evaluation_question`.

Read each row with a simple checklist:

1. Is the baseline fluent and sensible?
2. Does SteerMoE change the answer at all?
3. If it changes the answer, does it move toward the target fear instead of just
   becoming generic horror language?
4. Does steering preserve coherence?

If the answer to #2 or #3 is mostly "no," then SteerMoE has not transferred
well to these fear prompts yet.

## Next Experiment

If the OLMoE SteerMoE transfer check looks worth comparing against, the next
target is:

```text
OLMoE baseline | OLMoE + attention-guided activation steering | OLMoE + SteerMoE
```

The attention-guided activation steering backend is not implemented yet. The
repo keeps the attention-to-prefix collector because it is the natural starting
point for that method, but the current runnable steering backend is OLMoE
SteerMoE.

## What Is Implemented

- Manual review plan generation for five fears and five evaluation questions.
- OLMoE question-only baseline vs SteerMoE generation.
- OLMoE router-trace collection over the matched statement-body span.
- Risk-difference expert scoring.
- Router-logit bias during OLMoE generation.
- HTML/Markdown qualitative reports.
- A future-facing attention-to-prefix collector for the next method comparison.

Removed on purpose:

- the prefix-conditioned model comparison runner,
- the temporary larger-MoE steering runner/backend,
- old toy pipeline abstractions,
- the single-token hybrid prototype,
- one-off helper scripts that are no longer part of the research path.

The goal is that a new reader can tell what to run in the first five minutes.

## Repository Map

- `prepare_manual_fear_review.py`
  Builds the sampled fear concepts and evaluation questions.
- `run_olmoe_steermoe_review.py`
  Current main runner. Produces the OLMoE baseline vs OLMoE + SteerMoE report.
- `collect_attention_to_prefix.py`
  Attention readout collector kept for the next same-model method comparison.
- `render_manual_review_report.py`
  Re-renders a saved manual review JSON into Markdown/HTML.
- `inspect_reference_data.py`
  Quick sanity check for the vendored concept/question data.
- `src/moe_attention_guided_steering/olmoe_backend.py`
  OLMoE router tracing and router-bias hooks.
- `src/moe_attention_guided_steering/manual_review.py`
  Shared review-plan schema and HTML/Markdown renderer.
- `src/moe_attention_guided_steering/upstream_prompt_datasets.py`
  Builds paired concept/control statement prompts for SteerMoE scoring.
- `src/moe_attention_guided_steering/attention_collection.py`
  Shared HF loading utilities plus the attention readout path.
- `slurm/run_olmoe_steermoe_review.sbatch`
  ORCD launcher for the current run.
- `slurm/collect_attention_to_prefix.sbatch`
  ORCD launcher for the future attention-readout stage.

## Local Sanity Checks

These checks do not load real model weights:

```bash
python3 inspect_reference_data.py --concept-type fears
python3 -m unittest discover -s tests
python3 run_olmoe_steermoe_review.py --help
```

For a tiny ORCD smoke test before the full run:

```bash
LIMIT_CASES=1 \
OUTPUT_DIR=experiments/olmoe_steermoe_smoke \
sbatch --partition=mit_normal_gpu --time=02:00:00 slurm/run_olmoe_steermoe_review.sbatch
```

## Output Layout

The OLMoE steering runner writes:

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

## Troubleshooting

If Slurm rejects a longer wall time on `mit_normal_gpu`, keep the submission at
`--time=02:00:00`.

If `sacct` only says `FAILED`, the useful error is in the log:

```bash
tail -n 120 logs/olmoe-steermoe-<JOBID>.out
```

If the manual plan is missing, rebuild it:

```bash
python3 prepare_manual_fear_review.py
```

## Research Notes

The current SteerMoE implementation follows the custom-steering shape:

1. Build matched concept/control prompt pairs from the same statement body.
2. Collect OLMoE router choices over that shared statement body.
3. Aggregate expert activation counts across the paired dataset.
4. Score experts by risk difference.
5. Select globally strongest positive and negative experts.
6. Apply router-logit bias during question-only generation.

The later attention-guided method should stay on OLMoE too. That keeps the
comparison clean: same data, same model, different steering rule.
