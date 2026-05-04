# Slurm Runbook

The current GPU steering experiment is:

- `slurm/run_mixtral_steermoe_review.sbatch`

It runs the Mixtral SteerMoE transfer check:

1. load a manual review plan,
2. collect routing traces on the shared statement-body target with
   `mistralai/Mixtral-8x7B-Instruct-v0.1`,
3. build SteerMoE plans from per-layer/per-expert risk-difference tables,
4. generate question-only baseline and SteerMoE responses,
5. render `qualitative_review.md` and `qualitative_review.html`.

The attention collector is still available for the next same-model method
comparison, but it is not the current generation runner.

## First Run On ORCD

```bash
cd ~/MoE_attention_guided_steering
git pull --ff-only origin main
source .venv/bin/activate
pip install -U -r requirements-gpu.txt
python3 prepare_manual_fear_review.py

REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
MIXTRAL_MODEL_ID=mistralai/Mixtral-8x7B-Instruct-v0.1 \
MIXTRAL_MODEL_TAG=mixtral_8x7b_instruct_v0_1 \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/mixtral_steermoe_fears_seed7_question_only \
sbatch --partition=mit_normal_gpu --time=02:00:00 slurm/run_mixtral_steermoe_review.sbatch
```

This run compares:

- `mixtral_baseline`: regular Mixtral on the question-only test prompt.
- `mixtral_steermoe`: the same Mixtral and the same question-only prompt, plus
  router bias.

Prompt template:

```text
{evaluation_question}
```

So the comparison tests whether router bias can replace the omitted concept
prefix at generation time.

## MIT-Cluster-Friendly Defaults

The Slurm script defaults are:

- partition: `mit_normal_gpu`
- GPU count: `1`
- CPUs: `8`
- memory: `128G`
- wall time: `02:00:00`
- Mixtral 4-bit loading: enabled

The SteerMoE runner defaults are:

- readout target: `statement_body`
- top positive experts: `8`
- top negative experts: `8`
- minimum absolute risk difference: `0.01`
- steering coefficient: `1.0`
- max new tokens: `48`
- temperature: `0.0`

These are deliberately modest. The first question is whether SteerMoE transfers
to the fear data on Mixtral at all, not whether a more aggressive sweep can
force an effect.

For a tiny smoke test, keep only the first concept-question case:

```bash
LIMIT_CASES=1 \
OUTPUT_DIR=experiments/mixtral_steermoe_smoke \
sbatch --partition=mit_normal_gpu --time=02:00:00 slurm/run_mixtral_steermoe_review.sbatch
```

## Monitoring

While the job is queued or running:

```bash
squeue -u $USER
```

For one specific job:

```bash
squeue -j <JOBID>
tail -n 120 logs/mixtral-steermoe-<JOBID>.out
```

After the job leaves the queue:

```bash
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
```

If `sacct` only says `FAILED`, use the `tail` command above. The Python or model
loading error will be in the log file.

## Pulling Results Back Locally

On the cluster:

```bash
tar -czf mixtral_steermoe_fears_seed7_question_only.tar.gz -C experiments mixtral_steermoe_fears_seed7_question_only
```

On your local machine:

```bash
scp mit-orcd:~/MoE_attention_guided_steering/mixtral_steermoe_fears_seed7_question_only.tar.gz ~/Downloads/
cd /Users/peterflo/Desktop/MoE_attention_guided_steering/experiments
tar -xzf ~/Downloads/mixtral_steermoe_fears_seed7_question_only.tar.gz
```

Main file:

```text
experiments/mixtral_steermoe_fears_seed7_question_only/qualitative_review.html
```

## Expected Outputs

The run writes:

- `experiments/mixtral_steermoe_fears_seed7_question_only/custom_steering_datasets/`
- `experiments/mixtral_steermoe_fears_seed7_question_only/routing_traces/`
- `experiments/mixtral_steermoe_fears_seed7_question_only/activation_tables/`
- `experiments/mixtral_steermoe_fears_seed7_question_only/steering_plans/`
- `experiments/mixtral_steermoe_fears_seed7_question_only/generation_metadata.json`
- `experiments/mixtral_steermoe_fears_seed7_question_only/manual_review_plan.json`
- `experiments/mixtral_steermoe_fears_seed7_question_only/qualitative_review.md`
- `experiments/mixtral_steermoe_fears_seed7_question_only/qualitative_review.html`

## Optional Future Attention Stage

For the later same-model comparison against an attention-guided activation
steering method, the repo keeps:

- `collect_attention_to_prefix.py`
- `slurm/collect_attention_to_prefix.sbatch`

That stage should feed the future three-way Mixtral report:

```text
Mixtral baseline | Mixtral + attention-guided activation steering | Mixtral + SteerMoE
```
