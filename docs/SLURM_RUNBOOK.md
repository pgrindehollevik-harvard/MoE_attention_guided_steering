# Slurm Runbook

This repo currently has one primary GPU experiment:

- `slurm/run_olmoe_steermoe_review.sbatch`

That script runs the **stage-1 OLMoE SteerMoE transfer experiment**:

1. load a manual review plan,
2. collect routing traces on the shared statement-body target with `allenai/OLMoE-1B-7B-0125-Instruct`,
3. build SteerMoE plans from a per-layer/per-expert risk-difference table,
4. generate baseline and SteerMoE responses,
5. render `qualitative_review.md` and `qualitative_review.html`.

The dense attention collector is still available for future same-model
comparisons, but it is no longer the main path for this repo.

## Expected stage-1 workflow

1. Create or activate a Python environment on the cluster.
2. Install `requirements-gpu.txt`.
3. Ensure Hugging Face auth works for your chosen model.
4. Prepare the qualitative review plan:

```bash
python3 prepare_manual_fear_review.py
```

5. Launch the OLMoE run:

```bash
REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
MODEL_ID=allenai/OLMoE-1B-7B-0125-Instruct \
MODEL_TAG=olmoe_1b_7b_0125_instruct \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/olmoe_steermoe_fears_seed7 \
sbatch --partition=mit_normal_gpu --time=06:00:00 slurm/run_olmoe_steermoe_review.sbatch
```

## MIT-cluster-friendly defaults

The Slurm script defaults are intentionally conservative:

- partition: `mit_normal_gpu`
- GPU count: `1`
- CPUs: `8`
- memory: `64G`
- wall time: `06:00:00`

Stage-1 runner defaults:

- readout target: `statement_body`
- top positive experts: `8`
- top negative experts: `8`
- minimum absolute risk difference: `0.01`
- steering coefficient: `1.0`

These are much gentler than the retired single-token hybrid prototype and are
meant to preserve fluency while we test whether SteerMoE transfers at all.

## Monitoring

While the job is queued or running:

```bash
squeue -u $USER
```

For one specific job:

```bash
squeue -j <JOBID>
tail -n 80 logs/olmoe-steermoe-<JOBID>.out
```

After the job leaves the queue:

```bash
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
```

## Expected outputs

The stage-1 run writes:

- `experiments/olmoe_steermoe_fears_seed7/custom_steering_datasets/`
- `experiments/olmoe_steermoe_fears_seed7/routing_traces/`
- `experiments/olmoe_steermoe_fears_seed7/activation_tables/`
- `experiments/olmoe_steermoe_fears_seed7/steering_plans/`
- `experiments/olmoe_steermoe_fears_seed7/manual_review_plan.json`
- `experiments/olmoe_steermoe_fears_seed7/qualitative_review.md`
- `experiments/olmoe_steermoe_fears_seed7/qualitative_review.html`

## Optional future attention stage

For the later same-model comparison against an attention-based method, the repo
still keeps:

- `collect_attention_to_prefix.py`
- `slurm/collect_attention_to_prefix.sbatch`

That stage is now supporting infrastructure rather than the main experiment.
