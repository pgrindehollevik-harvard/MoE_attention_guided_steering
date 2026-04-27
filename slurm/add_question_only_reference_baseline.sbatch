#!/bin/bash
# Append an unsteered question-only reference model column to an existing review.

#SBATCH --job-name=llama-ref
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
if [ "${LOAD_IN_4BIT:-false}" = "true" ]; then
  EXTRA_ARGS+=(--load-in-4bit)
fi

python3 add_question_only_reference_baseline.py \
  --plan-json "${PLAN_JSON:-experiments/olmoe_steermoe_fears_seed7_question_only/manual_review_plan.json}" \
  --output-dir "${OUTPUT_DIR:-experiments/olmoe_steermoe_fears_seed7_question_only_with_llama}" \
  --condition-key "${CONDITION_KEY:-llama_3_1_8b_baseline}" \
  --condition-label "${CONDITION_LABEL:-Llama 3.1 8B baseline (question only)}" \
  --insert-after "${INSERT_AFTER:-baseline}" \
  --model-id "${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}" \
  --model-tag "${MODEL_TAG:-llama_3_1_8b_instruct}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-48}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-1.0}" \
  --device-map "${DEVICE_MAP:-auto}" \
  --torch-dtype "${TORCH_DTYPE:-bfloat16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
  "${EXTRA_ARGS[@]}"
