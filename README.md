# MoE Attention-Guided Steering

This repository is a fresh, standalone project for combining two ideas:

- **SteerMoE**: steer a Mixture-of-Experts model by selectively increasing or decreasing the contribution of specific experts.
- **Attention-guided steering**: choose *which token representation to steer from* using attention instead of blindly using the last token everywhere.

The goal is to produce a research codebase that is both technically credible and easy to explain to collaborators. For that reason, the code is intentionally written in a teaching-first style with detailed docstrings and comments in the important functions.

## Project philosophy

The starting point here is not a full production integration into a real MoE model. Instead, this repo gives us a clean, well-documented architecture for the hybrid method so we can:

1. agree on the conceptual pipeline,
2. verify the scoring logic on toy data,
3. keep the code organization close to the attention-guided steering repository,
4. then replace the toy hooks with real model instrumentation.

## How the hybrid method works

The core idea is:

1. collect attention statistics for prompts that express a target concept and a contrast concept,
2. use those attention statistics to pick the most representative token for each layer,
3. inspect MoE expert loads at those selected token positions,
4. compare positive vs. negative examples to see which experts are consistently associated with the target concept,
5. build a layerwise steering plan that says which experts to activate and which to deactivate.

This is the bridge between the two papers:

- **SteerMoE** gives us the intervention target: MoE experts.
- **Attention-guided steering** gives us a stronger and more interpretable signal for *where* to measure the representations that define the steering signal.

## Repository layout

- `0_validate_dataset.py` through `4_export_experiment_report.py`: numbered scripts inspired by the attention-guided steering repo.
- `collect_attention_to_prefix.py`: real-model GPU script for collecting upstream-style attention-to-prefix scores and per-layer token choices.
- `inspect_reference_data.py`: quick summary tool for the imported attention-guided steering text assets under `data/`.
- `prepare_manual_fear_review.py`: builds a manual three-way worksheet for baseline vs MoESteer vs attention-guided MoESteer.
- `args.py`: shared CLI flags used by the scripts.
- `src/moe_attention_guided_steering/`: reusable library code.
- `data/`: vendored concept lists, general statements, and evaluation prompt templates from the attention-guided steering repo.
- `examples/toy_experiment.json`: a tiny synthetic dataset that makes the full pipeline runnable immediately.
- `docs/ARCHITECTURE.md`: codebase walkthrough.
- `docs/RESEARCH_SYNTHESIS.md`: plain-English mapping from the reference papers to this repo.
- `docs/REAL_MODEL_ATTENTION.md`: explanation of the new real-model attention collection stage.
- `docs/SLURM_RUNBOOK.md`: generic cluster launch notes.
- `docs/UPSTREAM_DATA.md`: provenance and usage notes for the imported attention-guided steering data.
- `slurm/`: Slurm submission templates for GPU runs.
- `tests/`: unit tests for the core logic.

## Quickstart

From the repository root:

```bash
python3 inspect_reference_data.py --concept-type fears
python3 0_validate_dataset.py
python3 1_select_representative_tokens.py --output-dir outputs/demo
python3 2_compute_expert_scores.py --output-dir outputs/demo
python3 3_build_steering_plan.py --output-dir outputs/demo
python3 4_export_experiment_report.py --output-dir outputs/demo
python3 -m unittest discover -s tests
```

The scripts default to `examples/toy_experiment.json`, so you can run the whole pipeline without downloading models.

## Real-model attention collection

The repo now also includes a real GPU-ready attention stage inspired directly by
the upstream `attention_guided_steering` workflow.

The key script is:

```bash
python3 collect_attention_to_prefix.py \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --model-tag llama_3_1_8b \
  --concept-type fears \
  --sample-concepts 5 \
  --seed 7 \
  --statement-stride 2
```

This script computes the actual per-layer attention-to-prefix scores used to pick
one candidate suffix token per layer, closely matching the upstream
`0_visualize_attn.py` stage. The output is:

- a raw attention score array with shape `(num_prompts, num_layers, num_candidate_tokens)`,
- rich JSON metadata,
- a final `layer_to_token` map that can later drive MoE expert-load collection.

See [docs/REAL_MODEL_ATTENTION.md](/Users/peterflo/Desktop/MoE_attention_guided_steering/docs/REAL_MODEL_ATTENTION.md) and [docs/SLURM_RUNBOOK.md](/Users/peterflo/Desktop/MoE_attention_guided_steering/docs/SLURM_RUNBOOK.md) for the details.

## Imported attention-guided steering data

The repository now includes the text assets from the upstream
`attention_guided_steering/data` tree so we can stay close to that project's
concept catalog and evaluation setup.

That imported data is useful immediately for:

- choosing concept families to probe,
- reusing the upstream evaluation prompt templates,
- keeping a future real-model collector aligned with the same experimental setup.

What it does *not* provide yet is the actual trace data needed by the MoE hybrid
pipeline. Our current `0` through `4` scripts still expect an `ExperimentDataset`
with per-layer attention weights and per-token expert loads. In other words, the
imported `data/` tree is now the experiment catalog, while `examples/toy_experiment.json`
remains the runnable toy trace dataset.

## What is already implemented

- a validated dataset schema for attention + expert-load traces,
- attention-based representative-token selection,
- a real-model attention-to-prefix collector aligned with the upstream attention-guided steering repo,
- positive-vs-negative expert score computation,
- a thresholded layerwise intervention planner,
- JSON + Markdown report generation,
- unit tests that prove the toy example behaves as expected.

## What should be implemented next

The next engineering step is to replace the toy JSON input with a real instrumentation layer that:

- records per-layer attention signals from an MoE model,
- uses the imported `data/` catalog to choose concept families and evaluation templates,
- uses the new real attention collector to choose one suffix token index per layer,
- extracts expert router loads or expert activations at the selected token positions,
- applies the resulting intervention plan during generation.

The repo is set up so that those changes should mostly land in a future model-integration module, while the scoring and planning logic can stay stable.
