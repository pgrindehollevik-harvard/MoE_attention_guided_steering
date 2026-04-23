# Slurm Runbook

This repo now includes a real attention-collection stage intended to be runnable
on a GPU cluster. The template job script lives at:

- `slurm/collect_attention_to_prefix.sbatch`
- `slurm/run_olmoe_manual_review.sbatch`

## Expected workflow

1. Create or activate a Python environment on the cluster.
2. Install the GPU dependencies from `requirements-gpu.txt`.
3. Set Hugging Face auth if your chosen model requires it.
4. Run `collect_attention_to_prefix.py` on one or more concepts.
5. Inspect the saved `.npy`, `.metadata.json`, and `.layer_to_token.json` files.
6. Feed those layerwise token indices into the next MoE-trace stage later.

For the first end-to-end MoE qualitative run, the workflow is now:

1. Prepare a manual review plan with `prepare_manual_fear_review.py`.
2. Launch `run_olmoe_manual_review.py` on a GPU node.
3. Let the script:
   - collect or reuse attention maps,
   - build fixed-token and attention-guided steering plans,
   - generate baseline / MoESteer / attention-guided MoESteer responses,
   - render the qualitative HTML and Markdown review page.

## Example command

```bash
python3 collect_attention_to_prefix.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --model-tag llama_3_1_8b_instruct \
  --concept-type fears \
  --sample-concepts 5 \
  --seed 7 \
  --statement-stride 2 \
  --head-aggregation mean \
  --output-dir outputs/attention_to_prefix
```

## Notes for the first fear experiment

To stay aligned with the manual review bundle already in this repo, a good first
cluster run is:

- concept family: `fears`
- sample size: `5`
- seed: `7`
- statement stride: `2`
- head aggregation: `mean`

That produces attention-selection artifacts for a small but representative set of
fear concepts before scaling to the full concept list.

## OLMoE qualitative run

The first real MoE backend targets:

- `allenai/OLMoE-1B-7B-0125-Instruct`

Local command:

```bash
python3 run_olmoe_manual_review.py \
  --plan-json outputs/manual_fear_review/manual_review_plan.json \
  --output-dir experiments/olmoe_fears_seed7
```

Slurm command:

```bash
REPO_DIR=$PWD \
HF_HOME=$HOME/.cache/huggingface \
MODEL_ID=allenai/OLMoE-1B-7B-0125-Instruct \
MODEL_TAG=olmoe_1b_7b_0125_instruct \
PLAN_JSON=outputs/manual_fear_review/manual_review_plan.json \
OUTPUT_DIR=experiments/olmoe_fears_seed7 \
sbatch slurm/run_olmoe_manual_review.sbatch
```

Important runtime defaults in that script:

- original MoESteer fixed token index: `-1`
- top-k experts per layer: `4`
- activation threshold: `0.002`
- deactivation threshold: `-0.002`
- steering coefficient: `8.0`

Those defaults are tuned for router probabilities rather than the larger toy
numbers in the synthetic JSON example.

## Practical cluster notes

- Use `--attn-implementation eager` unless you have confirmed another attention
  backend still returns the full attention tensors you need.
- If VRAM is tight, try a smaller model first or use `--load-in-4bit`.
- Set a writable cache directory with `--cache-dir` or by exporting `HF_HOME`.
  Reusing `HF_HOME=$HOME/.cache/huggingface` is often the easiest way to make
  Slurm jobs see the same auth token created by `hf auth login`.
- The collector currently runs batch size `1`, intentionally matching the
  upstream attention extraction style and keeping tensor bookkeeping simple.
- Many clusters require partition and wall-time overrides at submit time, for
  example `sbatch --partition=<gpu_partition> --time=<hh:mm:ss> ...`.
