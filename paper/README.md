# Paper-friendly variant

Paper-submission entry points for *Routing-Bias Steering in Sparse
Mixture-of-Experts: A Diagnostic Study of SteerMoE on Mixtral 8x7B*.

This directory is layered on top of the existing research repo. It adds
**no** runtime behavior to the underlying experiments and can be deleted
without affecting the upstream training and evaluation code.

## TL;DR

Reproduce every table and figure in the paper from artifacts already
committed under `experiments/`:

```bash
python3 paper/scripts/aggregate_results.py
python3 paper/scripts/make_figures.py
```

This emits:

- `paper/notebooks/sweep_results.csv` — one row per (run, response).
- `paper/notebooks/expert_selection.csv` — one row per (run, layer, expert).
- `paper/docs/paper/tables/*.csv` and `headline_delta.tex`.
- `paper/docs/paper/figures/*.pdf`.

Then build the paper:

```bash
cd paper/docs/paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Layout

```
paper/
├── README.md                   # this file
├── CITATION.cff                # software citation metadata
├── requirements.txt            # paper-build pip extras
├── notebooks/
│   ├── 01_data_overview.ipynb       # concept/prompt sanity check
│   ├── 02_prefix_baseline.ipynb     # Stage 1 model comparison view
│   └── 03_main.ipynb                # end-to-end figures + tables (paper)
├── scripts/
│   ├── aggregate_results.py         # experiments/ → CSVs
│   └── make_figures.py              # CSVs → tables/figures
└── docs/
    └── paper/
        ├── main.tex
        ├── preamble.tex
        ├── references.bib
        ├── sections/
        │   ├── background.tex
        │   ├── problem.tex
        │   ├── data_eda.tex
        │   ├── methods.tex
        │   ├── results.tex
        │   ├── conclusion.tex
        │   ├── broader_impact.tex
        │   ├── acknowledgements.tex
        │   └── appendix.tex
        ├── tables/
        └── figures/
```

The `paper/` directory mirrors the structure used by the prior
[hyperbolic embeddings](https://github.com/pgrindehollevik-harvard/hyperbolic)
submission; reviewers familiar with that repo should find the same paths.

## What this paper claims

1. **Mixtral 8x7B follows explicit fear prefixes** on this prompt family.
2. **Question-only SteerMoE produces small, mostly-defocusing shifts** on
   Mixtral. The concept-token hit rate moves by at most one case in
   twenty-five across all eight configurations swept; mean response length
   contracts by `-2.16` to `-5.52` words as the steering coefficient grows.
3. **The risk-difference signal is diffuse** on Mixtral: the top twenty
   `(layer, expert)` cells account for less than 10 % of the total
   `|risk diff|` mass.

We interpret this as evidence that the *readout* — which tokens the
router-logit comparison should weight most heavily — is the next thing
worth changing, not the *intervention*. That motivates the planned
attention-guided activation-steering follow-up described in the paper's
Discussion section.

## What this paper does **not** claim

We do not claim SteerMoE fails in general. The original demonstration
uses different MoE checkpoints, prompt families, and test contracts.
We claim only that the upstream recipe, faithfully re-implemented, does
not produce concept-specific transfer on Mixtral under our question-only
contract.

## Pointers back to the underlying repo

- `src/moe_attention_guided_steering/mixtral_backend.py` — router
  tracing and router-bias hooks (the active SteerMoE backend).
- `run_mixtral_steermoe_review.py` — the entrypoint that produces every
  `experiments/mixtral_steermoe_*/` run consumed by this paper.
- `compare_prefix_conditioned_models.py` — the entrypoint behind Stage 1
  (Claim 1) full-prefix comparison.
- `docs/RESEARCH_SYNTHESIS.md` — staged plan and where this paper fits.
- `docs/ARCHITECTURE.md` — backend-by-backend walkthrough.
