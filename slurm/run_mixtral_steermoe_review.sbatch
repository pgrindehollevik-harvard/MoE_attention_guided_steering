#!/bin/bash
# MIT ORCD Slurm template for the question-only Mixtral SteerMoE review.

#SBATCH --job-name=mixtral-steermoe
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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
if [ "${LOAD_MIXTRAL_IN_4BIT:-false}" = "true" ]; then
  EXTRA_ARGS+=(--load-mixtral-in-4bit)
  EXTRA_ARGS+=(--no-load-mixtral-in-8bit)
fi
if [ "${LOAD_MIXTRAL_IN_8BIT:-true}" != "true" ]; then
  EXTRA_ARGS+=(--no-load-mixtral-in-8bit)
fi
if [ "${BNB_CPU_OFFLOAD:-true}" != "true" ]; then
  EXTRA_ARGS+=(--no-bnb-cpu-offload)
fi
if [ "${INCLUDE_LLAMA_REFERENCE:-true}" != "true" ]; then
  EXTRA_ARGS+=(--no-include-llama-reference)
fi
if [ "${LOAD_LLAMA_REFERENCE_IN_8BIT:-false}" = "true" ]; then
  EXTRA_ARGS+=(--load-llama-reference-in-8bit)
fi
if [ -n "${LIMIT_CASES:-}" ]; then
  EXTRA_ARGS+=(--limit-cases "$LIMIT_CASES")
fi

python3 run_mixtral_steermoe_review.py \
  --mixtral-model-id "${MIXTRAL_MODEL_ID:-mistralai/Mixtral-8x7B-Instruct-v0.1}" \
  --mixtral-model-tag "${MIXTRAL_MODEL_TAG:-mixtral_8x7b_instruct_v0_1}" \
  --llama-reference-model-id "${LLAMA_REFERENCE_MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}" \
  --llama-reference-model-tag "${LLAMA_REFERENCE_MODEL_TAG:-llama_3_1_8b_instruct}" \
  --plan-json "${PLAN_JSON:-outputs/manual_fear_review/manual_review_plan.json}" \
  --data-dir "${DATA_DIR:-data}" \
  --output-dir "${OUTPUT_DIR:-experiments/mixtral_steermoe_fears_seed7_question_only_with_llama}" \
  --statement-stride "${STATEMENT_STRIDE:-2}" \
  --readout-target "${READOUT_TARGET:-statement_body}" \
  --top-positive-experts "${TOP_POSITIVE_EXPERTS:-8}" \
  --top-negative-experts "${TOP_NEGATIVE_EXPERTS:-8}" \
  --minimum-abs-risk-difference "${MINIMUM_ABS_RISK_DIFFERENCE:-0.01}" \
  --steering-coefficient "${STEERING_COEFFICIENT:-1.0}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-48}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-1.0}" \
  --device-map "${DEVICE_MAP:-auto}" \
  --torch-dtype "${TORCH_DTYPE:-bfloat16}" \
  --mixtral-attn-implementation "${MIXTRAL_ATTN_IMPLEMENTATION:-sdpa}" \
  --llama-reference-attn-implementation "${LLAMA_REFERENCE_ATTN_IMPLEMENTATION:-eager}" \
  "${EXTRA_ARGS[@]}"
