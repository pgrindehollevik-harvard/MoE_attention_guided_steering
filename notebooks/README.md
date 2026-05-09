# notebooks/ — experiment-execution pipeline

This directory holds the **runnable, reviewer-readable** version of
every experimental step in the thesis. The contract is strict:

> **Every notebook is documentation \emph{and} the cluster execution.**
> No second code path lives in a `.py` script that the notebook does
> not call. No experimental result lands in `experiments/` that did
> not come from a `papermill`-executed notebook in this directory.

If a notebook passes its sanity checks, the same notebook (with
parameters overridden via `papermill -p`) is the canonical way to
re-run that step on the MIT ORCD cluster.

## Why this pattern

Auditability and replicability. A reviewer who clones this repo
should be able to:

1. Read any notebook top-to-bottom and understand what it does.
2. Run it locally on CPU in smoke mode (small `LIMIT_CASES`, no real
   model load) to verify the wiring.
3. Run the *same notebook* on the cluster, headlessly, by calling
   the matching `slurm/notebooks/<name>.sbatch` wrapper, and get an
   archived executed copy with full outputs back in
   `experiments/<run_dir>/executed_notebook.ipynb`.

The data-flow diagram is one arrow: notebook → executed notebook.

## How a notebook is structured

Every notebook (including future ones) follows the canonical
template in [`_template.ipynb`](_template.ipynb):

1. **Title and one-paragraph summary.** What this step does, what
   inputs it consumes, what outputs it writes.
2. **Parameters cell** (papermill convention). All knobs --- run name,
   sample sizes, seeds, model id, output directory --- exposed as
   top-level Python variables. Marked with the cell tag `parameters`.
3. **Imports.** From the project's `src/.../pipelines/` package, not
   from the legacy top-level scripts directly.
4. **Body.** Calls into pipeline functions; never reimplements
   pipeline logic inline.
5. **Sanity checks.** A small section that prints a head of the
   outputs (e.g., a few generations, a per-layer activation-rate
   plot, an aggregated metric table). A reviewer must be able to
   tell at a glance whether the run looked sane.
6. **Provenance footer.** Writes `experiments/<run_dir>/provenance.json`
   with the git SHA, hostname, slurm job id (if present), wallclock,
   and the resolved papermill parameter values. This makes the run
   self-describing.

## How to run a notebook

### Locally, interactively (small smoke tests only)

```bash
make setup
jupyter lab notebooks/03_stage2_steermoe_mixtral.ipynb
```

In the parameters cell, set `LIMIT_CASES=1` and run. This validates
wiring without needing a GPU.

### Locally, headlessly via papermill (CI / dry-run)

```bash
papermill notebooks/03_stage2_steermoe_mixtral.ipynb \
  /tmp/03_executed.ipynb \
  -p run_name smoke_local \
  -p limit_cases 1 \
  -p output_dir experiments/smoke_local
```

### On MIT ORCD (real run)

```bash
sbatch slurm/notebooks/nb_03_stage2_steermoe_mixtral.sbatch
```

The sbatch wrapper sources the project venv, exports the right
HF_HOME and OUTPUT_DIR, and calls papermill on the notebook. The
executed notebook (with all outputs) is copied into
`experiments/<run_dir>/executed_notebook.ipynb` so the cluster's
artifact is fully self-describing.

Override parameters from the sbatch invocation:

```bash
RUN_NAME=mixtral_steermoe_seed7_coef4 \
STEERING_COEFFICIENT=4.0 \
TOP_POSITIVE_EXPERTS=8 \
sbatch slurm/notebooks/nb_03_stage2_steermoe_mixtral.sbatch
```

## Notebook inventory

| # | Notebook | Purpose | GPU? |
|---|---|---|---|
| 00 | `00_environment_check.ipynb` | Verify CUDA / HF auth / disk; cheap. | No |
| 01 | `01_prepare_dataset.ipynb` | Sample concepts and evaluation prompts. | No |
| 02 | `02_stage1_prefix_baseline.ipynb` | Full-prefix Llama / OLMoE / Mixtral comparison. | Yes |
| 03 | `03_stage2_steermoe_mixtral.ipynb` | Mixtral baseline vs SteerMoE (question-only). | Yes |
| 04 | `04_stage3_agrs_mixtral.ipynb` | Mixtral baseline vs AGRS (Phase 2 deliverable). | Yes |
| 05 | `05_method_comparison.ipynb` | 6-method full sweep on Mixtral / fears. | Yes |
| 06 | `06_cross_model.ipynb` | Mixtral / OLMoE / Qwen-MoE / DS-MoE generalization. | Yes |
| 07 | `07_cross_concept.ipynb` | Fears / professions / refusal / sentiment generalization. | Yes |
| 08 | `08_evaluate_with_judge.ipynb` | LLM-as-judge scoring pass; emits CTR. | No (API) |
| 09 | `09_evaluate_with_embeddings.ipynb` | Concept-similarity scoring; emits CSS. | No (API) |
| 10 | `10_aggregate_and_figures.ipynb` | All artifacts → CSVs → tables / figures. | No |
| 11 | `11_statistics.ipynb` | Bootstrap CIs, paired tests, effect sizes. | No |

The `00` notebook is the only one that exists in the initial Phase
0 commit. Notebooks 01–11 land as their corresponding REVISION_PLAN
phases come due.

## What lives where

- **`notebooks/`** — this directory; runnable execution + audit.
- **`slurm/notebooks/`** — sbatch wrappers, one per notebook above.
- **`src/moe_attention_guided_steering/pipelines/`** — pure-function
  entrypoints the notebooks import. Stable API; the legacy top-level
  scripts (`compare_prefix_conditioned_models.py` etc.) are thin
  shims around these.
- **`experiments/<run_dir>/`** — committed run artifacts (executed
  notebook, generation metadata, qualitative review, activation
  tables, steering plans, provenance.json).
- **`paper/notebooks/`** — separate, paper-figure regeneration
  notebooks. They run *locally* off the committed CSVs and never
  touch a GPU. Don't confuse them with this directory.
