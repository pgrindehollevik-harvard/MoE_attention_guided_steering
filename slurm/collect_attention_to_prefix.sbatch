#!/bin/bash
# Generic Slurm template for collecting attention-to-prefix scores on a GPU node.

#SBATCH --job-name=moe-attn-prefix
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

python3 collect_attention_to_prefix.py \
  --model-id "${MODEL_ID:-allenai/OLMoE-1B-7B-0125-Instruct}" \
  --model-tag "${MODEL_TAG:-olmoe_1b_7b_0125_instruct}" \
  --concept-type "${CONCEPT_TYPE:-fears}" \
  --sample-concepts "${SAMPLE_CONCEPTS:-5}" \
  --seed "${SEED:-7}" \
  --statement-stride "${STATEMENT_STRIDE:-2}" \
  --head-aggregation "${HEAD_AGGREGATION:-mean}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --output-dir "${OUTPUT_DIR:-outputs/attention_to_prefix_olmoe}"
