#!/usr/bin/env bash
# Canonical sbatch wrapper template for notebooks/ pipeline.
# Copy this file to nb_NN_<notebook_stem>.sbatch and edit:
#   - SBATCH header (partition / GPU / mem / time)
#   - NB_PATH                 (the notebook this wrapper runs)
#   - notebook-specific -p PARAM=value lines below
#
# All wrappers share the env-var contract documented in
# slurm/notebooks/README.md.

#SBATCH --job-name=nb-template
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- contract -------------------------------------------------------------

REPO_DIR="${REPO_DIR:-$HOME/MoE_attention_guided_steering}"
RUN_NAME="${RUN_NAME:-nb_template_$(date +%s)}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/${RUN_NAME}}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
SEED="${SEED:-7}"

NB_PATH="notebooks/_template.ipynb"           # <-- edit per wrapper
NB_NAME="$(basename "${NB_PATH}" .ipynb)"

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}" logs

# --- runtime --------------------------------------------------------------

if [[ "${USE_APPTAINER:-0}" == "1" ]]; then
  module load apptainer
  EXEC=(apptainer exec --nv "${REPO_DIR}/containers/agrs.sif")
else
  source "${REPO_DIR}/.venv/bin/activate"
  EXEC=()
fi

# --- run ------------------------------------------------------------------

EXECUTED_NB="${OUTPUT_DIR}/executed_notebook.ipynb"

"${EXEC[@]}" papermill \
  "${NB_PATH}" \
  "${EXECUTED_NB}" \
  --log-output \
  -p run_name "${RUN_NAME}" \
  -p output_dir "${OUTPUT_DIR}" \
  -p seed "${SEED}"
  # ^ append notebook-specific -p PARAM=value lines below this block

echo "executed notebook: ${EXECUTED_NB}"
