# Upstream Data Notes

This repository now vendors the text assets from the `data/` directory of the
attention-guided steering reference repo:

- Source repo: <https://github.com/pdavar/attention_guided_steering>
- Source data tree: <https://github.com/pdavar/attention_guided_steering/tree/main/data>

## What was imported

The imported files fall into three groups:

1. `data/concepts/*.txt`
   Concept inventories such as fears, moods, personas, personalities, and places.
2. `data/general_statements/class_*.txt`
   Paired statement pools that the upstream project uses when building
   contrastive prompts.
3. `data/evaluation_prompts/*_eval_v*.txt`
   Evaluation instructions for judging whether steering succeeded for a given
   concept family and prompt version.

## How this repo uses the imported data

The new module `src/moe_attention_guided_steering/reference_data.py` treats these
files as a catalog for experiment setup. In particular, it:

- loads concept families into typed Python structures,
- groups evaluation prompts by family and version,
- exposes a `ReferenceConceptSuite` adapter that ties a concept family to its
  matching evaluation prompt family,
- keeps the upstream naming mismatch in one place, such as `fears -> phobia`
  and `places -> topophile`.

These files are inputs to the current OLMoE steering runner. They
provide concept names, matched generic statements, and evaluation questions; the
runner collects the model-specific router traces at experiment time.

## Why this still helps

The imported data gives us a concrete bridge to the attention-guided steering
workflow:

- concept lists tell us what to probe,
- general statements give us reusable prompt scaffolds,
- evaluation prompts tell us how the upstream project scores steering success.

That means the current steering runners and the future attention collector can
start from the same concept catalog and evaluation templates.

## Important redistribution note

When I checked the upstream GitHub repo on April 22, 2026, I did not see a
license file on the repository page. Because this repository is public, it would
be wise to confirm redistribution permissions for the vendored text assets if you
plan to keep them here long term.
