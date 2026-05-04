#!/bin/bash
# Generic Slurm template for collecting attention-to-prefix scores on a GPU node.

#SBATCH --job-name=moe-attn-prefix
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
if [ "${LOAD_IN_4BIT:-true}" = "true" ]; then
  EXTRA_ARGS+=(--load-in-4bit)
fi

python3 collect_attention_to_prefix.py \
  --model-id "${MODEL_ID:-mistralai/Mixtral-8x7B-Instruct-v0.1}" \
  --model-tag "${MODEL_TAG:-mixtral_8x7b_instruct_v0_1}" \
  --concept-type "${CONCEPT_TYPE:-fears}" \
  --sample-concepts "${SAMPLE_CONCEPTS:-5}" \
  --seed "${SEED:-7}" \
  --statement-stride "${STATEMENT_STRIDE:-2}" \
  --head-aggregation "${HEAD_AGGREGATION:-mean}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --output-dir "${OUTPUT_DIR:-outputs/attention_to_prefix_mixtral}" \
  "${EXTRA_ARGS[@]}"
