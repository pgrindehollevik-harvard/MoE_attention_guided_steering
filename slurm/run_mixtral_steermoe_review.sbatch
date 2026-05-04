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
if [ "${LOAD_MIXTRAL_IN_4BIT:-true}" != "true" ]; then
  EXTRA_ARGS+=(--no-load-mixtral-in-4bit)
fi
if [ -n "${LIMIT_CASES:-}" ]; then
  EXTRA_ARGS+=(--limit-cases "$LIMIT_CASES")
fi

python3 run_mixtral_steermoe_review.py \
  --mixtral-model-id "${MIXTRAL_MODEL_ID:-mistralai/Mixtral-8x7B-Instruct-v0.1}" \
  --mixtral-model-tag "${MIXTRAL_MODEL_TAG:-mixtral_8x7b_instruct_v0_1}" \
  --plan-json "${PLAN_JSON:-outputs/manual_fear_review/manual_review_plan.json}" \
  --data-dir "${DATA_DIR:-data}" \
  --output-dir "${OUTPUT_DIR:-experiments/mixtral_steermoe_fears_seed7_question_only}" \
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
  "${EXTRA_ARGS[@]}"
