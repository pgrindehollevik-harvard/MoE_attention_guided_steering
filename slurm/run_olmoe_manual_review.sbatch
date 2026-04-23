#!/bin/bash
# Generic Slurm template for the first end-to-end OLMoE qualitative review run.

#SBATCH --job-name=olmoe-review
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/MoE_attention_guided_steering}"
cd "$REPO_DIR"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python3 run_olmoe_manual_review.py \
  --model-id "${MODEL_ID:-allenai/OLMoE-1B-7B-0125-Instruct}" \
  --model-tag "${MODEL_TAG:-olmoe_1b_7b_0125_instruct}" \
  --plan-json "${PLAN_JSON:-outputs/manual_fear_review/manual_review_plan.json}" \
  --data-dir "${DATA_DIR:-data}" \
  --output-dir "${OUTPUT_DIR:-experiments/olmoe_fears_seed7}" \
  --statement-stride "${STATEMENT_STRIDE:-2}" \
  --head-aggregation "${HEAD_AGGREGATION:-mean}" \
  --fixed-token-index "${FIXED_TOKEN_INDEX:--1}" \
  --top-k-experts "${TOP_K_EXPERTS:-4}" \
  --activation-threshold "${ACTIVATION_THRESHOLD:-0.002}" \
  --deactivation-threshold "${DEACTIVATION_THRESHOLD:--0.002}" \
  --steering-coefficient "${STEERING_COEFFICIENT:-8.0}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-48}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-1.0}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}"
