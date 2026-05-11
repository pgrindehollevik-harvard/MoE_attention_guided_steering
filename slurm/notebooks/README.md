# slurm/notebooks/ — sbatch wrappers around papermill

Each `.sbatch` file in this directory wraps exactly one notebook in
`notebooks/` and runs it headlessly via `papermill`. There is no
second logic path; the sbatch wrapper exists for cluster compute,
not for any independent execution.

## Convention

- One sbatch per notebook, named `nb_NN_<notebook_stem>.sbatch`.
- All wrappers share the same env-var contract:

  | Env var | Meaning | Default |
  |---|---|---|
  | `REPO_DIR` | Where the repo is checked out on the node | `$HOME/MoE_attention_guided_steering` |
  | `RUN_NAME` | Human label for this run | `nb<NN>_$(date +%s)` |
  | `OUTPUT_DIR` | Where artifacts and the executed notebook land | `experiments/$RUN_NAME` |
  | `HF_HOME` | Hugging Face cache | `$HOME/.cache/huggingface` |
  | `SEED` | Random seed | `7` |

  Each wrapper additionally exports notebook-specific knobs (e.g.,
  `STEERING_COEFFICIENT`, `TOP_POSITIVE_EXPERTS`) and passes them to
  papermill via `-p`.

- Every wrapper saves the *executed* notebook (with all outputs)
  into `$OUTPUT_DIR/executed_notebook.ipynb`. That file is the
  cluster artifact reviewers and future-you will read.

- Logs go to `logs/<nb_name>-${SLURM_JOB_ID}.{out,err}`.

## Running on ORCD

The MIT ORCD `mit_normal_gpu` partition is the default. Override
with `--partition=…` on the sbatch command line if needed.

```bash
# baseline pattern
sbatch slurm/notebooks/nb_00_environment_check.sbatch

# per-run override (every wrapper supports the same env vars)
RUN_NAME=mixtral_steermoe_seed7_coef4 \
STEERING_COEFFICIENT=4.0 \
TOP_POSITIVE_EXPERTS=8 \
sbatch slurm/notebooks/nb_03_stage2_steermoe_mixtral.sbatch
```

## Apptainer (preferred on cluster)

Once the Phase-0 Apptainer image is built (see
`containers/agrs.def`), every wrapper transparently switches to
running inside the container by setting `USE_APPTAINER=1`:

```bash
USE_APPTAINER=1 sbatch slurm/notebooks/nb_03_stage2_steermoe_mixtral.sbatch
```

The wrapper then calls `apptainer exec --nv $REPO_DIR/containers/agrs.sif`
in front of the papermill invocation. Without `USE_APPTAINER=1`
the wrapper falls back to the project venv, which is the right
choice for early development on a fresh cluster account before the
container is built.

## Inventory

| Wrapper | Runs notebook | GPU |
|---|---|---|
| `nb_00_environment_check.sbatch` | `00_environment_check.ipynb` | optional |
| `_template.sbatch` | (template; do not run) | — |

Wrappers for notebooks 01–11 land alongside their corresponding
notebooks as their REVISION_PLAN phases come due.
