# Slurm Runbook

This repo currently has two GPU steering experiments plus one older diagnostic:

- `slurm/run_olmoe_steermoe_review.sbatch`
- `slurm/run_mixtral_steermoe_review.sbatch`
- `slurm/compare_prefix_conditioned_models.sbatch`

The OLMoE script runs the original **stage-1 OLMoE SteerMoE transfer experiment**:

1. load a manual review plan,
2. collect routing traces on the shared statement-body target with `allenai/OLMoE-1B-7B-0125-Instruct`,
3. build SteerMoE plans from a per-layer/per-expert risk-difference table,
4. generate baseline and SteerMoE responses from question-only prompts,
5. render `qualitative_review.md` and `qualitative_review.html`.

The Mixtral script runs the current requested question-only review:

```text
Llama 3.1 8B baseline | Mixtral 8x7B baseline | Mixtral 8x7B + SteerMoE
```

The prefix-conditioned model comparison is separate and older. The dense
attention collector is still available for future same-model comparisons, but it
is no longer the main path for this repo.

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

This run compares:

- `baseline`: regular OLMoE on the question-only test prompt
- `steermoe`: the same OLMoE and the same question-only prompt, plus router bias

Prompt template:

```text
{evaluation_question}
```

So the stage-1 comparison tests whether router bias can replace the omitted
concept prefix at generation time.

## Current Mixtral review workflow

Run this after preparing the same manual review plan:

```bash
REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/mixtral_steermoe_fears_seed7_question_only_with_llama \
sbatch --partition=mit_normal_gpu --time=12:00:00 slurm/run_mixtral_steermoe_review.sbatch
```

This run compares:

- `llama_3_1_8b_baseline`: Llama 3.1 8B on the question-only test prompt, no steering.
- `mixtral_baseline`: Mixtral 8x7B on the same question-only test prompt, no steering.
- `mixtral_steermoe`: the same Mixtral checkpoint and prompt, plus router bias.

Mixtral is loaded in 4-bit by default. Disable that only if the target GPU has
enough memory for the full checkpoint plus generation overhead:

```bash
LOAD_MIXTRAL_IN_4BIT=false sbatch slurm/run_mixtral_steermoe_review.sbatch
```

## Prefix-conditioned model diagnostic

Run this after preparing the same manual review plan if you want to compare base
model behavior when the explicit prefix is present:

```bash
REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/prefix_conditioned_model_comparison_fears_seed7 \
sbatch --partition=mit_normal_gpu --time=12:00:00 slurm/compare_prefix_conditioned_models.sbatch
```

Default models are configured in
`configs/prefix_conditioned_model_comparison.json`:

- `meta-llama/Llama-3.1-8B-Instruct`
- `allenai/OLMoE-1B-7B-0125-Instruct`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat`

This diagnostic deliberately uses:

```text
Personify someone who is terrified of {concept}.  {evaluation_question}
```

It is for base-model suitability only and should not be interpreted as the
steering result.

## MIT-cluster-friendly defaults

The Mixtral Slurm script defaults are intentionally conservative:

- partition: `mit_normal_gpu`
- GPU count: `1`
- CPUs: `8`
- memory: `128G`
- wall time: `12:00:00`

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
# or, for the Mixtral review:
tail -n 80 logs/mixtral-steermoe-<JOBID>.out
```

After the job leaves the queue:

```bash
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,ExitCode
```

## Pulling results back locally

On the cluster:

```bash
tar -czf olmoe_steermoe_fears_seed7.tar.gz -C experiments olmoe_steermoe_fears_seed7
tar -czf mixtral_steermoe_fears_seed7_question_only_with_llama.tar.gz -C experiments mixtral_steermoe_fears_seed7_question_only_with_llama
```

On your local machine:

```bash
scp mit-orcd:~/MoE_attention_guided_steering/olmoe_steermoe_fears_seed7.tar.gz ~/Downloads/
cd /Users/peterflo/Desktop/MoE_attention_guided_steering/experiments
tar -xzf ~/Downloads/olmoe_steermoe_fears_seed7.tar.gz
```

## Expected outputs

The stage-1 run writes:

- `experiments/olmoe_steermoe_fears_seed7/custom_steering_datasets/`
- `experiments/olmoe_steermoe_fears_seed7/routing_traces/`
- `experiments/olmoe_steermoe_fears_seed7/activation_tables/`
- `experiments/olmoe_steermoe_fears_seed7/steering_plans/`
- `experiments/olmoe_steermoe_fears_seed7/generation_metadata.json`
- `experiments/olmoe_steermoe_fears_seed7/manual_review_plan.json`
- `experiments/olmoe_steermoe_fears_seed7/qualitative_review.md`
- `experiments/olmoe_steermoe_fears_seed7/qualitative_review.html`

The Mixtral review writes:

- `experiments/mixtral_steermoe_fears_seed7_question_only_with_llama/custom_steering_datasets/`
- `experiments/mixtral_steermoe_fears_seed7_question_only_with_llama/routing_traces/`
- `experiments/mixtral_steermoe_fears_seed7_question_only_with_llama/activation_tables/`
- `experiments/mixtral_steermoe_fears_seed7_question_only_with_llama/steering_plans/`
- `experiments/mixtral_steermoe_fears_seed7_question_only_with_llama/generation_metadata.json`
- `experiments/mixtral_steermoe_fears_seed7_question_only_with_llama/manual_review_plan.json`
- `experiments/mixtral_steermoe_fears_seed7_question_only_with_llama/qualitative_review.md`
- `experiments/mixtral_steermoe_fears_seed7_question_only_with_llama/qualitative_review.html`

The prefix diagnostic writes:

- `experiments/prefix_conditioned_model_comparison_fears_seed7/model_comparison_results.json`
- `experiments/prefix_conditioned_model_comparison_fears_seed7/model_comparison.md`
- `experiments/prefix_conditioned_model_comparison_fears_seed7/model_comparison.html`

## Optional future attention stage

For the later same-model comparison against an attention-based method, the repo
still keeps:

- `collect_attention_to_prefix.py`
- `slurm/collect_attention_to_prefix.sbatch`

That stage is now supporting infrastructure rather than the main experiment.
