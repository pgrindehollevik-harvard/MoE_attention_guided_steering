#!/bin/bash
# MIT ORCD Slurm template for the full-prefix model suitability diagnostic.

#SBATCH --job-name=prefix-models
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
if [ "${LOAD_IN_4BIT:-false}" = "true" ]; then
  EXTRA_ARGS+=(--load-in-4bit)
fi
if [ "${LOAD_IN_8BIT:-false}" = "true" ]; then
  EXTRA_ARGS+=(--load-in-8bit)
fi
if [ "${BNB_CPU_OFFLOAD:-false}" = "true" ]; then
  EXTRA_ARGS+=(--bnb-cpu-offload)
fi
if [ -n "${LIMIT_CASES:-}" ]; then
  EXTRA_ARGS+=(--limit-cases "$LIMIT_CASES")
fi
if [ -n "${DEVICE_MAP:-}" ]; then
  EXTRA_ARGS+=(--device-map "$DEVICE_MAP")
fi

python3 compare_prefix_conditioned_models.py \
  --plan-json "${PLAN_JSON:-outputs/manual_fear_review/manual_review_plan.json}" \
  --model-config-json "${MODEL_CONFIG_JSON:-configs/prefix_conditioned_model_comparison.json}" \
  --output-dir "${OUTPUT_DIR:-experiments/prefix_conditioned_model_comparison_mixtral_fears_seed7}" \
  --prompt-mode "${PROMPT_MODE:-prefix}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-48}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-1.0}" \
  --torch-dtype "${TORCH_DTYPE:-bfloat16}" \
  "${EXTRA_ARGS[@]}"
