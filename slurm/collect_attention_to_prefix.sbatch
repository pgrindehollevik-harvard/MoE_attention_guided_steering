#!/bin/bash
# Generic Slurm template for collecting attention-to-prefix scores on a GPU node.

#SBATCH --job-name=moe-attn-prefix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Adjust this to the location of the repo on the cluster.
REPO_DIR="${REPO_DIR:-$HOME/MoE_attention_guided_steering}"
cd "$REPO_DIR"

# Activate your environment here.
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}"

python3 collect_attention_to_prefix.py \
  --model-id "${MODEL_ID:-meta-llama/Meta-Llama-3.1-8B-Instruct}" \
  --model-tag "${MODEL_TAG:-llama_3_1_8b}" \
  --concept-type "${CONCEPT_TYPE:-fears}" \
  --sample-concepts "${SAMPLE_CONCEPTS:-5}" \
  --seed "${SEED:-7}" \
  --statement-stride "${STATEMENT_STRIDE:-2}" \
  --head-aggregation "${HEAD_AGGREGATION:-mean}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --output-dir "${OUTPUT_DIR:-outputs/attention_to_prefix}"
