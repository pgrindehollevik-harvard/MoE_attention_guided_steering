#!/usr/bin/env bash
# Run notebooks/00_environment_check.ipynb on the cluster.
# Cheap. Use this as the first thing on a new account / new partition.

#SBATCH --job-name=nb-00-env
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/MoE_attention_guided_steering}"
RUN_NAME="${RUN_NAME:-env_check_$(date +%s)}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/${RUN_NAME}}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
SEED="${SEED:-7}"

NB_PATH="notebooks/00_environment_check.ipynb"

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}" logs

if [[ "${USE_APPTAINER:-0}" == "1" ]]; then
  module load apptainer
  EXEC=(apptainer exec --nv "${REPO_DIR}/containers/agrs.sif")
else
  source "${REPO_DIR}/.venv/bin/activate"
  EXEC=()
fi

EXECUTED_NB="${OUTPUT_DIR}/executed_notebook.ipynb"

"${EXEC[@]}" papermill \
  "${NB_PATH}" \
  "${EXECUTED_NB}" \
  --log-output \
  -p run_name "${RUN_NAME}" \
  -p output_dir "${OUTPUT_DIR}" \
  -p require_cuda True \
  -p check_hf_auth False \
  -p expected_packages '["torch","transformers","bitsandbytes","accelerate","pandas","numpy","matplotlib","papermill","jupyter"]'

echo "executed notebook: ${EXECUTED_NB}"
