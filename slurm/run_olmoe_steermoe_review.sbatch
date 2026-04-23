#!/bin/bash
# MIT ORCD Slurm template for the stage-1 OLMoE SteerMoE transfer experiment.

#SBATCH --job-name=olmoe-steermoe
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/MoE_attention_guided_steering}"
cd "$REPO_DIR"

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python3 run_olmoe_steermoe_review.py \
  --model-id "${MODEL_ID:-allenai/OLMoE-1B-7B-0125-Instruct}" \
  --model-tag "${MODEL_TAG:-olmoe_1b_7b_0125_instruct}" \
  --plan-json "${PLAN_JSON:-outputs/manual_fear_review/manual_review_plan.json}" \
  --data-dir "${DATA_DIR:-data}" \
  --output-dir "${OUTPUT_DIR:-experiments/olmoe_steermoe_fears_seed7}" \
  --statement-stride "${STATEMENT_STRIDE:-2}" \
  --readout-span "${READOUT_SPAN:-user_content}" \
  --top-k-experts "${TOP_K_EXPERTS:-2}" \
  --activation-threshold "${ACTIVATION_THRESHOLD:-0.01}" \
  --deactivation-threshold "${DEACTIVATION_THRESHOLD:--0.01}" \
  --steering-coefficient "${STEERING_COEFFICIENT:-1.0}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-48}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-1.0}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}"
