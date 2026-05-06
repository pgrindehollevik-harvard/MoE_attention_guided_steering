# MoE Attention-Guided Steering

This repo has two related Mixtral runs. They answer different questions, so keep
their prompts separate.

## Run 1: Prefix-Conditioned Model Comparison

This is Parmida's first request. The model sees the full prompt:

```text
Personify someone who is terrified of Bugs. What is the scariest thing in the world? Answer in 20 words or less.
```

The goal is model suitability, not steering. We are checking whether a larger
MoE can follow the explicit fear prefix better than the smaller MoEs we tried.

The report columns are:

```text
Llama 3.1 8B Instruct | OLMoE 1B-7B Instruct | Mixtral 8x7B Instruct
```

This is the same style of report as the old Qwen run, but the third column is
now Mixtral instead of Qwen.

Run it on ORCD:

```bash
cd ~/MoE_attention_guided_steering
git pull --ff-only origin main
source .venv/bin/activate
pip install -U -r requirements-gpu.txt
python3 prepare_manual_fear_review.py

REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/prefix_conditioned_model_comparison_mixtral_fears_seed7 \
sbatch --partition=mit_normal_gpu --gres=gpu:2 --mem=192G --time=02:00:00 slurm/compare_prefix_conditioned_models.sbatch
```

Watch it:

```bash
squeue -u $USER
tail -n 120 logs/prefix-models-<JOBID>.out
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
```

Main file:

```text
experiments/prefix_conditioned_model_comparison_mixtral_fears_seed7/model_comparison.html
```

If you only want to test whether Mixtral itself loads on the cluster, run the
tracked Mixtral-only config first:

```bash
MODEL_CONFIG_JSON=configs/mixtral_only_prefix_config.json \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/prefix_conditioned_mixtral_only_fears_seed7 \
sbatch --partition=mit_normal_gpu --gres=gpu:2 --mem=192G --time=02:00:00 slurm/compare_prefix_conditioned_models.sbatch
```

Both Mixtral configs use 8-bit loading plus bitsandbytes CPU offload. That is
slower and larger than 4-bit, but it avoids the ORCD 4-bit `Params4bit`
compatibility failure while still letting Mixtral spill overflow modules to CPU.

## Run 2: Question-Only Mixtral SteerMoE

This is Parmida's second request. During evaluation, Mixtral does **not** see the
concept prefix:

```text
What is the scariest thing in the world? Answer in 20 words or less.
```

The report columns are:

```text
Llama 3.1 8B baseline | Mixtral 8x7B baseline | Mixtral 8x7B + SteerMoE
```

All columns receive the same question-only prompt. The Llama and Mixtral
baseline columns have no intervention; the SteerMoE column adds router-logit
bias from the selected Mixtral experts.

Run it on ORCD:

```bash
REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/mixtral_steermoe_fears_seed7_question_only_with_llama \
sbatch --partition=mit_normal_gpu --gres=gpu:2 --mem=192G --time=02:00:00 slurm/run_mixtral_steermoe_review.sbatch
```

Main file:

```text
experiments/mixtral_steermoe_fears_seed7_question_only_with_llama/qualitative_review.html
```

## How To Read The Two Outputs

Run 1 asks:

1. If the fear is explicitly in the prompt, can Mixtral follow it?
2. Does Mixtral look stronger than OLMoE and the previous Qwen MoE run?

Run 2 asks:

1. If the fear is omitted from the test prompt, does SteerMoE move Mixtral toward
   the intended fear concept?
2. Does steering preserve coherence?

The prefix shown in Run 2's HTML is for reviewer context only. In the actual
generation call, `case.full_prompt_text = case.evaluation_question`.

## Next Method Comparison

If the Mixtral SteerMoE transfer check looks worth comparing against, the next
target is:

```text
Mixtral 8x7B baseline | Mixtral 8x7B + attention-guided activation steering | Mixtral 8x7B + SteerMoE
```

The attention-guided activation steering backend is not implemented yet. The
repo keeps the attention-to-prefix collector because it is the natural starting
point for that method.

## Repository Map

- `compare_prefix_conditioned_models.py`
  Run 1: full-prefix model comparison with Llama, OLMoE, and Mixtral.
- `run_mixtral_steermoe_review.py`
  Run 2: Llama question-only reference vs Mixtral baseline vs Mixtral + SteerMoE.
- `prepare_manual_fear_review.py`
  Builds the sampled fear concepts and evaluation questions.
- `configs/prefix_conditioned_model_comparison.json`
  Model list for Run 1. Mixtral is loaded in 8-bit with CPU offload by default.
- `slurm/compare_prefix_conditioned_models.sbatch`
  ORCD launcher for Run 1.
- `slurm/run_mixtral_steermoe_review.sbatch`
  ORCD launcher for Run 2.
- `src/moe_attention_guided_steering/model_comparison.py`
  Prefix-conditioned comparison report generation.
- `src/moe_attention_guided_steering/mixtral_backend.py`
  Mixtral router tracing and router-bias hooks.
- `src/moe_attention_guided_steering/olmoe_backend.py`
  OLMoE router tracing and router-bias hooks from the earlier run.
- `collect_attention_to_prefix.py`
  Attention readout collector kept for the future method comparison.

## Local Sanity Checks

These checks do not load real model weights:

```bash
python3 inspect_reference_data.py --concept-type fears
python3 -m unittest discover -s tests
python3 compare_prefix_conditioned_models.py --help
python3 run_mixtral_steermoe_review.py --help
```

For a tiny ORCD smoke test of Run 1:

```bash
LIMIT_CASES=1 \
OUTPUT_DIR=experiments/prefix_conditioned_model_comparison_mixtral_smoke \
sbatch --partition=mit_normal_gpu --time=02:00:00 slurm/compare_prefix_conditioned_models.sbatch
```

For a tiny ORCD smoke test of Run 2:

```bash
LIMIT_CASES=1 \
OUTPUT_DIR=experiments/mixtral_steermoe_smoke \
sbatch --partition=mit_normal_gpu --time=02:00:00 slurm/run_mixtral_steermoe_review.sbatch
```

## Troubleshooting

If the job fails with a bitsandbytes error, refresh the GPU environment:

```bash
source .venv/bin/activate
pip install -U -r requirements-gpu.txt
```

`requirements-gpu.txt` pins Transformers below version 5 because the current
ORCD stack hit a 4-bit bitsandbytes error:

```text
TypeError: Params4bit.__new__() got an unexpected keyword argument '_is_hf_initialized'
```

If the error says modules were dispatched to CPU or disk, pull the latest repo
and use either the default comparison config or
`configs/mixtral_only_prefix_config.json`; both use 8-bit CPU offload for Mixtral.

If `sacct` only says `FAILED`, the useful error is in the log:

```bash
tail -n 120 logs/prefix-models-<JOBID>.out
tail -n 120 logs/mixtral-steermoe-<JOBID>.out
```

If `prepare_manual_fear_review.py` fails with `Disk quota exceeded`, free space
or move caches/outputs to a scratch/project directory before resubmitting. That
error happens before either experiment actually starts.
