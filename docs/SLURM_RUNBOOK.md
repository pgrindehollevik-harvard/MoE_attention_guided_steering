# Slurm Runbook

There are two runnable GPU jobs. The immediate one for Parmida's first request
is the prefix-conditioned model comparison.

## Run 1: Prefix-Conditioned Comparison

This run compares unsteered full-prefix generations:

```text
Llama 3.1 8B Instruct | OLMoE 1B-7B Instruct | Mixtral 8x7B Instruct
```

Submit:

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

Main output:

```text
experiments/prefix_conditioned_model_comparison_mixtral_fears_seed7/model_comparison.html
```

The Mixtral entry in `configs/prefix_conditioned_model_comparison.json` is
loaded in 8-bit with bitsandbytes CPU offload enabled. Llama and OLMoE use their
normal per-model settings. We use 8-bit here because the ORCD package stack hit
a 4-bit `Params4bit` compatibility error during Mixtral loading.

If Mixtral is the only question you need to answer, run the smaller diagnostic:

```bash
MODEL_CONFIG_JSON=configs/mixtral_only_prefix_config.json \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/prefix_conditioned_mixtral_only_fears_seed7 \
sbatch --partition=mit_normal_gpu --gres=gpu:2 --mem=192G --time=02:00:00 slurm/compare_prefix_conditioned_models.sbatch
```

## Run 2: Question-Only Mixtral SteerMoE

This run compares:

```text
Llama 3.1 8B baseline | Mixtral baseline | Mixtral + SteerMoE
```

Submit:

```bash
REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/mixtral_steermoe_fears_seed7_question_only_with_llama \
sbatch --partition=mit_normal_gpu --gres=gpu:2 --mem=192G --time=02:00:00 slurm/run_mixtral_steermoe_review.sbatch
```

Main output:

```text
experiments/mixtral_steermoe_fears_seed7_question_only_with_llama/qualitative_review.html
```

### Steering ablations

Use these after the default Mixtral SteerMoE run if the steered outputs look too
weak or too generic. Keep `INCLUDE_LLAMA_REFERENCE=false` while probing so the
jobs only compare Mixtral baseline against Mixtral + SteerMoE.

```bash
REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/mixtral_steermoe_fears_seed7_question_only_coef4_pos8_neg8 \
INCLUDE_LLAMA_REFERENCE=false \
STEERING_COEFFICIENT=4.0 \
TOP_POSITIVE_EXPERTS=8 \
TOP_NEGATIVE_EXPERTS=8 \
sbatch --partition=mit_normal_gpu --gres=gpu:2 --mem=192G --time=02:00:00 slurm/run_mixtral_steermoe_review.sbatch
```

The current ablation set is:

```text
coef4_pos8_neg8: STEERING_COEFFICIENT=4.0, TOP_POSITIVE_EXPERTS=8, TOP_NEGATIVE_EXPERTS=8
coef1_pos8_neg0: STEERING_COEFFICIENT=1.0, TOP_POSITIVE_EXPERTS=8, TOP_NEGATIVE_EXPERTS=0
coef4_pos8_neg0: STEERING_COEFFICIENT=4.0, TOP_POSITIVE_EXPERTS=8, TOP_NEGATIVE_EXPERTS=0
```

Local report compiler:

```bash
python3 compile_mixtral_steering_ablation_report.py
```

## Monitoring

```bash
squeue -u $USER
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
```

Run 1 log:

```bash
tail -n 120 logs/prefix-models-<JOBID>.out
```

Run 2 log:

```bash
tail -n 120 logs/mixtral-steermoe-<JOBID>.out
```

If `squeue -j <JOBID>` says the id is invalid, the job has already left the
queue; use `sacct` and the log instead.

## Smoke Tests

Run 1:

```bash
LIMIT_CASES=1 \
OUTPUT_DIR=experiments/prefix_conditioned_model_comparison_mixtral_smoke \
sbatch --partition=mit_normal_gpu --time=02:00:00 slurm/compare_prefix_conditioned_models.sbatch
```

Run 2:

```bash
LIMIT_CASES=1 \
OUTPUT_DIR=experiments/mixtral_steermoe_smoke \
sbatch --partition=mit_normal_gpu --time=02:00:00 slurm/run_mixtral_steermoe_review.sbatch
```

## Quota And Caches

The recent failure happened before the job reached model loading:

```text
OSError: [Errno 122] Disk quota exceeded
```

That means the manual review JSON could not be written. Check quota and large
cache/output directories before resubmitting:

```bash
quota -s
du -sh ~/.cache/huggingface outputs experiments logs 2>/dev/null
```

If home quota is full, move `HF_HOME`, `OUTPUT_DIR`, or old experiment bundles
to a scratch/project location with enough space.
