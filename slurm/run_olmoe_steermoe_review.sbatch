#!/bin/bash
# MIT ORCD Slurm template for the question-only OLMoE SteerMoE transfer experiment.

#SBATCH --job-name=olmoe-steermoe
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/MoE_attention_guided_steering}"
cd "$REPO_DIR"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

EXTRA_ARGS=()
if [ -n "${LIMIT_CASES:-}" ]; then
  EXTRA_ARGS+=(--limit-cases "$LIMIT_CASES")
fi

python3 run_olmoe_steermoe_review.py \
  --model-id "${MODEL_ID:-allenai/OLMoE-1B-7B-0125-Instruct}" \
  --model-tag "${MODEL_TAG:-olmoe_1b_7b_0125_instruct}" \
  --plan-json "${PLAN_JSON:-outputs/manual_fear_review/manual_review_plan.json}" \
  --data-dir "${DATA_DIR:-data}" \
  --output-dir "${OUTPUT_DIR:-experiments/olmoe_steermoe_fears_seed7_question_only}" \
  --statement-stride "${STATEMENT_STRIDE:-2}" \
  --readout-target "${READOUT_TARGET:-statement_body}" \
  --top-positive-experts "${TOP_POSITIVE_EXPERTS:-8}" \
  --top-negative-experts "${TOP_NEGATIVE_EXPERTS:-8}" \
  --minimum-abs-risk-difference "${MINIMUM_ABS_RISK_DIFFERENCE:-0.01}" \
  --steering-coefficient "${STEERING_COEFFICIENT:-1.0}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-48}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-1.0}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
  "${EXTRA_ARGS[@]}"
