# Experiment Artifacts

This directory keeps the shareable layer of the project experiments: qualitative
HTML/Markdown reviews, model comparison reports, run metadata, manual review
plans, steering plans, and compact activation tables.

The bulky internals are intentionally left out of Git. Raw routing traces,
router datasets, custom steering datasets, NumPy attention dumps, and compressed
archives can easily become hundreds of megabytes and should stay local or move
to a dedicated artifact store if we need long-term archival.

## Current Runs

- `prefix_conditioned_model_comparison_fears_seed7_mixtral_8bit/`: the
  Parmida #1 full-prefix comparison with Llama 3.1 8B, OLMoE, and Mixtral.
- `prefix_conditioned_mixtral_only_fears_seed7_8bit/`: the Mixtral-only
  full-prefix sanity check that verified Mixtral could run on ORCD in 8-bit.
- `mixtral_steermoe_fears_seed7_question_only_with_llama/`: the Parmida #2
  question-only review with Llama baseline, Mixtral baseline, and Mixtral +
  SteerMoE.
- `mixtral_steermoe_ablation_summary/`: local side-by-side HTML/PDF summary for
  the Mixtral steering-strength ablations.
- `olmoe_steermoe_fears_seed7_question_only_with_llama/`: the earlier
  question-only OLMoE SteerMoE review with Llama reference.

When adding a new experiment, prefer committing the report and metadata files
first. Only commit large raw artifacts if the repo has Git LFS or another
artifact policy in place.
