# NeurIPS Revision Plan

This document is the working plan for turning the current draft (a small
diagnostic note) into a NeurIPS-tier method paper. It is partitioned into
phases so that progress can be checkpointed and so that no phase wastes
compute on an unstable predecessor.

The target venue is NeurIPS main track. The fallback venue if any phase
under-delivers is the NeurIPS Workshop on Attributing Model Behavior at
Scale (or a similar mech-interp workshop).

---

## 1 The new central contribution

We propose **Attention-Guided Router Steering (AGRS)**: a routing-bias
intervention for sparse Mixture-of-Experts language models that combines

1. SteerMoE's *intervention site* — additive bias on selected experts'
   router logits at decode time, and
2. an *attention-guided readout* — when computing the per-(layer, expert)
   risk-difference table, weight each token's contribution by the
   attention mass it sends to a concept-bearing token span,

into a single, drop-in replacement for SteerMoE that requires no
additional training and no model-architecture changes.

### 1.1 Formal statement

For a contrastive prompt pair $(p^+, p^-)$ with concept span
$\mathcal{C} \subset p^+$, let

- $R_{\ell, t, e} \in \{0,1\}$ — indicator that expert $e$ is selected by
  Mixtral top-$k$ routing at layer $\ell$, token position $t$;
- $A_{\ell, h, t \to t'}$ — attention weight from token $t$ to token $t'$
  at layer $\ell$, head $h$;
- $a_t = \max_h \sum_{t' \in \mathcal{C}} A_{\ell^\star, h, t \to t'}$ —
  per-token attention mass directed toward the concept span at a chosen
  reference layer $\ell^\star$ (in the simplest variant, the last layer).

Then the AGRS attention-weighted activation rate for $(p^+)$ is

$$
r^+_{\ell, e} = \frac{\sum_{t \in \text{target}} a^+_t \cdot R^+_{\ell, t, e}}
                     {\sum_{t \in \text{target}} a^+_t},
$$

with $r^-_{\ell, e}$ defined identically on $p^-$ with $a^- \equiv 1$
(uniform, since $p^-$ has no concept span). The risk difference is
$\Delta_{\ell, e} = r^+_{\ell, e} - r^-_{\ell, e}$.

SteerMoE is the special case $a^+_t \equiv 1$. The decode-time
intervention is identical to SteerMoE: add $+\alpha$ to selected positive
experts' router logits, $-\alpha$ to selected negative experts'.

### 1.2 The empirical thesis

If our diffuse-signal finding is real, AGRS should produce a
risk-difference table whose top-$k$ share of total $|\Delta|$ mass is
strictly higher than SteerMoE's, *on the same data, same model, same
concept span*. That concentration should translate into a higher
concept-transfer rate under the question-only test contract.

This is testable in a clean ablation that varies one variable at a time.

### 1.3 What AGRS is *not*

AGRS does not change the intervention site (still router logits), the
generation pipeline (still chat-template, temperature 0), or the
per-concept plan format. The point is to show that *readout matters
more than people think*, with the smallest possible methodological
footprint that lets the comparison be clean.

---

## 2 Phased plan

Each phase has: deliverables, compute budget, required infrastructure,
exit criteria, risks.

### Phase 0 — Reproducibility hardening (no compute)

**Goal.** Make every figure and number in the paper regenerable from a
single command on a clean checkout, and put the experiment pipeline
behind well-documented, parameterizable Jupyter notebooks that run
identically on a laptop (CPU, smoke mode) and on the MIT ORCD cluster
(GPU, full mode) via `papermill` + `sbatch`.

#### 0.1 Notebook-as-pipeline architecture

The thesis adopts a single, strict reproducibility pattern. Each
experimental step is a Jupyter notebook that:

1. Declares its inputs as a `parameters` cell (papermill convention).
2. Loads only the artifacts it needs from disk; writes only into
   `experiments/<run_dir>/` paths it owns.
3. Renders a small qualitative-check section at the bottom (a head
   of generations, an activation-rate sanity plot, etc.) so a human
   reader can tell at a glance whether the run looked sane.
4. Has a paired `slurm/notebooks/<nb_name>.sbatch` wrapper that runs
   the same notebook headlessly via `papermill`, writing the
   executed notebook (with outputs) into
   `experiments/<run_dir>/executed_notebook.ipynb` for archival.

This means: \emph{the documentation and the runnable pipeline are
the same file.} A reviewer (or a future student) can re-run any
phase by `papermill`-ing the notebook with the same parameters. The
sbatch wrapper exists for cluster compute, not for any second logic
path.

The execution surface:

```
notebooks/
├── README.md                       # how to run locally vs on cluster
├── 00_environment_check.ipynb      # CUDA / model auth / disk smoke
├── 01_prepare_dataset.ipynb        # sample concepts + eval prompts
├── 02_stage1_prefix_baseline.ipynb # full-prefix model comparison
├── 03_stage2_steermoe_mixtral.ipynb# baseline vs SteerMoE
├── 04_stage3_agrs_mixtral.ipynb    # AGRS (Phase 2 deliverable)
├── 05_method_comparison.ipynb      # 6-method full sweep (Phase 4)
├── 06_cross_model.ipynb            # generalize across MoE models
├── 07_cross_concept.ipynb          # generalize across concept families
├── 08_evaluate_with_judge.ipynb    # LLM-as-judge scoring
├── 09_evaluate_with_embeddings.ipynb # CSS scoring
├── 10_aggregate_and_figures.ipynb  # CSVs -> tables/figures
└── 11_statistics.ipynb             # bootstrap CIs, paired tests

slurm/notebooks/
├── README.md
├── nb_00_environment_check.sbatch
├── nb_02_stage1_prefix_baseline.sbatch
├── nb_03_stage2_steermoe_mixtral.sbatch
├── nb_04_stage3_agrs_mixtral.sbatch
├── nb_05_method_comparison.sbatch
├── nb_06_cross_model.sbatch
├── nb_07_cross_concept.sbatch
├── nb_08_evaluate_with_judge.sbatch
└── nb_10_aggregate_and_figures.sbatch
```

The existing top-level scripts (`compare_prefix_conditioned_models.py`,
`run_mixtral_steermoe_review.py`, etc.) stay where they are; the
notebooks `import` their pipeline functions rather than reimplement
them. Phase 0 includes a refactor that exposes those pipelines as
clean function-level entrypoints in
`src/moe_attention_guided_steering/pipelines/`.

#### 0.2 Code-to-build manifest (Phase 0)

Concrete files that get committed in this phase. Each line is one
deliverable.

| Path | Purpose |
|---|---|
| `Makefile` | Top-level targets `setup`, `nb-<NN>`, `paper`, `clean`. |
| `environment.yml` | Conda env with pinned `transformers`, `torch`, `bitsandbytes`, `accelerate`, `papermill`, `nbclient`, `pandas`, `matplotlib`, `pytest`. |
| `requirements-locked.txt` | Pip lock with hashes for non-conda installs. |
| `containers/agrs.def` | Apptainer recipe; builds an image runnable on ORCD without root. |
| `containers/build_agrs.sbatch` | Container build job for the cluster. |
| `notebooks/README.md` | The notebook-as-pipeline contract: how to run locally, how to run on ORCD, where outputs land. |
| `notebooks/00_environment_check.ipynb` | CUDA visible? HF token works? Cache writable? Cheap, runs anywhere. |
| `notebooks/_template.ipynb` | The canonical structure (parameters cell, imports, body, qualitative checks, archival save). |
| `slurm/notebooks/README.md` | Sbatch conventions: partition, GPU count, mem, wall, env vars, output paths. |
| `slurm/notebooks/nb_00_environment_check.sbatch` | Wrapper that papermills `00_environment_check.ipynb` headlessly. |
| `slurm/notebooks/_template.sbatch` | Canonical sbatch template all wrappers share. |
| `src/moe_attention_guided_steering/pipelines/__init__.py` | New module. |
| `src/moe_attention_guided_steering/pipelines/data_prep.py` | Pure-function entrypoint behind `prepare_manual_fear_review.py`. |
| `src/moe_attention_guided_steering/pipelines/stage1_prefix_baseline.py` | Pure-function entrypoint behind `compare_prefix_conditioned_models.py`. |
| `src/moe_attention_guided_steering/pipelines/stage2_steermoe.py` | Pure-function entrypoint behind `run_mixtral_steermoe_review.py`. |
| `src/moe_attention_guided_steering/pipelines/aggregate.py` | Pure-function entrypoint behind `paper/scripts/aggregate_results.py`. |
| `tests/test_pipelines.py` | Smoke tests for the new pure-function entrypoints (no model load). |
| `data_card.md` | Every dataset: license, source URL, sampling seed, n, where committed. |
| `checkpoint_card.md` | Every model: HF revision SHA, license, load precision, memory footprint, sbatch line. |
| `paper/docs/paper/references.bib` | Replace placeholder citations with real BibTeX. |
| `osf_preregistration.md` | Pre-registration draft (primary metric, comparator, n, seeds, stop rule, decision gate). |

The existing top-level scripts are NOT deleted. They become thin
shims:

```python
# compare_prefix_conditioned_models.py
from moe_attention_guided_steering.pipelines.stage1_prefix_baseline import main
if __name__ == "__main__":
    main()
```

This keeps every existing sbatch/CLI invocation working while the
notebook layer goes on top.

#### 0.3 Replicability commitments enforced in code

- **Determinism.** Every `pipelines/*.py` function takes an explicit
  `seed: int` argument. No notebook calls model code without
  threading a seed through.
- **Pinning.** `pipelines/_loading.py` exposes
  `load_pinned(model_id: str, expected_sha: str)` that asserts the
  HF revision SHA on every load.
- **Provenance.** A `pipelines/_provenance.py` helper writes the
  current git SHA, hostname, slurm job id, wallclock, and the
  resolved papermill parameters into every
  `experiments/<run_dir>/provenance.json`.
- **Containerization.** All sbatch wrappers `module load apptainer`
  and call into the image; no host pip installs in the run path.
- **Human readability.** Every notebook ends with a
  `## Sanity check` section that prints a head of generations and a
  small per-(layer, expert) plot. Reviewers can read the notebook
  top-to-bottom without rerunning anything.

#### 0.4 Other Phase 0 deliverables (unchanged)

- An `osf_preregistration.md` for the AGRS hypothesis with primary
  metric (CTR), comparator (SteerMoE same readout target), $n$,
  seeds, stop rule, and Phase-3 decision gate.
- License confirmation on the upstream fear-concept dataset; if
  incompatible, swap to a clean-license alternative (e.g.,
  MoralChoice, HH-RLHF refusal split) before any AGRS experiments.

**Compute.** None.

**Exit criteria.** From a clean checkout:
1. `make setup` builds the conda env and the Apptainer image.
2. `make nb-00` runs the environment-check notebook locally on CPU
   in under a minute.
3. `sbatch slurm/notebooks/nb_00_environment_check.sbatch` runs the
   same notebook on ORCD and writes the executed copy plus a
   `provenance.json` to a fresh `experiments/<run_dir>/`.
4. `make paper` produces `paper/docs/paper/main.pdf`.

**Risk.** Medium. The largest engineering risk is that `papermill`
plus 8-bit Mixtral plus `bitsandbytes` plus the ORCD Apptainer image
exposes a version-pin combinatorics that takes a week to debug. We
budget that explicitly.

---

### Phase 1 — Strong evaluation harness (small compute)

**Goal.** Replace the current `concept_token_hit_rate` /
`mean_word_count` diagnostic with a defensible evaluation suite.

**Primary metric.** *Concept-transfer rate (CTR)*: a held-out judge
(GPT-4o or Claude Opus 4.6, blinded to which condition produced the
response) rates each generation on a 0-2 ordinal scale for "is this
response thematically about $\langle$concept$\rangle$." CTR is the
fraction of cases that score $\geq 1$. This is the metric we will
power-compute and pre-register against.

**Secondary metrics.**
- *Concept-similarity score (CSS)*: cosine similarity in OpenAI
  `text-embedding-3-large` between the response and a fixed
  concept-prototype embedding (the concept name plus its definition).
- *Coherence rate (CR)*: judge rates whether the response is a
  fluent, on-topic answer to the evaluation question, regardless of
  concept content. Penalizes degenerate-but-on-concept generations.
- *Refusal rate (RR)*: did the model refuse or punt? Logged
  separately so it cannot inflate or deflate CTR.

**Deliverables.**
- `paper/scripts/evaluate_with_judge.py` — calls a configured judge
  (GPT-4o by default, Claude Opus 4.6 as the cross-check) with
  blinded condition labels, fixed seed, fixed system prompt.
- `paper/scripts/evaluate_with_embeddings.py` — computes CSS for every
  (response, concept) pair using a cached embedding store under
  `paper/cache/embeddings/`.
- `paper/scripts/run_human_eval.py` — emits a Streamlit form for a
  pre-defined random subset (n=200) so we can validate the judge
  correlations with human raters before relying on the judge at scale.
- A judge-vs-human agreement check (Cohen's $\kappa$, Spearman) on
  that subset, reported in the paper appendix.

**Compute.** ~50 USD of judge API calls, no GPU.

**Exit criteria.** Judge-vs-human Spearman $\geq 0.7$ on the n=200
subset. If lower, refine the judge prompt or fall back to majority
vote across two independent judges.

**Risk.** Low–medium. The known failure mode is the judge favoring
verbose responses; we control for it by including length as a
covariate when reporting CTR.

---

### Phase 2 — AGRS implementation (medium compute)

**Goal.** Land AGRS as a drop-in alternative to SteerMoE in the
existing Mixtral backend, with unit tests.

**Deliverables.**
- `src/moe_attention_guided_steering/attention_guided_router.py`
  — implements the attention-weighted readout from §1.1 against the
  same data shapes the existing `mixtral_backend.py` produces.
- A new entrypoint `run_agrs_review.py` that mirrors
  `run_mixtral_steermoe_review.py` but uses the AGRS readout. It
  produces the same `experiments/<run_dir>/` artifact layout, so the
  existing aggregation pipeline picks it up with no changes.
- Unit tests under `tests/test_agrs_backend.py`: at minimum, a
  property test that AGRS reduces to SteerMoE when $a_t \equiv 1$,
  and a deterministic numerical test on a hand-rolled
  attention/router fixture.
- A small "$L^\star$ sweep" config that lets the chosen reference
  layer for the attention readout be set via env var
  (`AGRS_REFERENCE_LAYER=last|mid|all|<int>`).

**Compute.** A handful of tiny ORCD smoke runs (~1 hour each on a
2× A100 partition) to verify AGRS produces a different
risk-difference table than SteerMoE on the same paired data.

**Exit criteria.** Unit tests pass. AGRS plan files differ from
SteerMoE plan files on the same paired data (sanity), and AGRS
top-$k$ share of $|\Delta|$ mass is strictly higher than SteerMoE's
on the same data on at least 4/5 fear concepts.

**Risk.** Medium. The largest open implementation question is which
attention layer's mass to use as the readout weight; we ablate
$\ell^\star \in \{\text{last}, \text{mid}, \text{all-mean}\}$ in
Phase 4.

---

### Phase 3 — Pilot run on Mixtral / fears (medium compute)

**Goal.** Validate that AGRS produces a meaningful CTR delta on the
existing Mixtral / fears setup before scaling to the full sweep.

**Conditions.**
1. Mixtral baseline (question-only).
2. Mixtral + SteerMoE (question-only), at the best of the existing
   ablation grid.
3. Mixtral + AGRS, at $\alpha=1$, $k_+=8$, $k_-=8$, $\ell^\star=\text{last}$.

**Sample size.** 5 concepts × 5 evaluation prompts × 4 seeds = 100
cases per condition.

**Compute.** ~3 hours on 2× A100 ORCD. Most of it is in the
generation pass (3 conditions × 100 cases × 48 tokens).

**Exit criteria.** AGRS CTR is at least $+0.10$ above SteerMoE CTR
on this pilot. If not, return to Phase 2 and reconsider $\ell^\star$
or whether the attention-weighting needs to be done per-layer rather
than at a fixed reference layer.

**Risk.** High. This is the gating phase. If the pilot does not
clear $+0.10$ CTR over SteerMoE on Mixtral / fears, the paper either
(a) becomes a *negative result* paper with a much narrower scope and
no NeurIPS main-track aspirations, or (b) pivots to a different
intervention site (hidden-state activation steering with the same
attention readout), in which case Phases 4–6 are repurposed but the
infrastructure carries over.

---

### Phase 4 — Full method comparison on Mixtral / fears (large compute)

**Goal.** A statistically powered same-model, same-data comparison of
all relevant baselines under the question-only contract.

**Conditions.**
1. Mixtral baseline.
2. Mixtral + SteerMoE.
3. Mixtral + ITI (top-$k$ attention heads on hidden states)
   \cite{liu2024iti}.
4. Mixtral + ActAdd (residual-stream activation addition)
   \cite{turner2023acteng}.
5. Mixtral + RepE difference-of-means (residual-stream)
   \cite{zou2023repe}.
6. Mixtral + AGRS (ours).

**Sample size.** 5 concepts × 10 evaluation prompts × 5 seeds = 250
cases per condition. Per-condition power for a $\Delta$CTR of $0.10$
at the 0.05 level is well above 0.9 under a binomial assumption.

**Knobs.** Each method has at most one steering-strength knob; we
sweep three values per method, picked by per-method validation on a
held-out concept (a sixth fear concept reserved for tuning).

**Compute.** 6 conditions × 250 cases × 48 tokens × 5 seeds ≈ 10
hours of decoding on Mixtral 8x7B at 8-bit, plus 2 hours of routing
and attention-trace collection upstream. Budget: 30 GPU-hours on 2×
A100.

**Exit criteria.** A reportable AGRS-vs-best-baseline $\Delta$CTR
with paired bootstrap 95% CI, plus the diffuse-signal evidence
relating top-$k$ share of $|\Delta|$ to CTR across methods.

**Risk.** Medium. Logistic risk: ITI / ActAdd / RepE require
hidden-state hooks that we have not yet implemented; budget one
engineering week per method for clean integration with the existing
chat-template generation path.

---

### Phase 5 — Generalization across MoE models (large compute)

**Goal.** Show the AGRS-vs-SteerMoE effect is not Mixtral-specific.

**Models.**
- Mixtral 8x7B-Instruct-v0.1 (already loaded).
- OLMoE 1B-7B-0125-Instruct (already loaded; existing repo path).
- Qwen2-MoE 14B-A2.7B-Instruct.
- DeepSeek-V2-MoE-Lite-Chat (16B total / 2.4B active) — *if* license
  and ORCD storage allow; otherwise drop and document in
  `paper/checkpoint_card.md`.

**Conditions.** Per model, run conditions 1, 2, 6 from Phase 4
(baseline, SteerMoE, AGRS). The full 6-method matrix is run only on
Mixtral; cross-model is the leaner same-readout-different-intervention
comparison.

**Sample size.** Same as Phase 4 (250 cases × 3 conditions × N
models).

**Compute.** ~30 GPU-hours per model on a 2× A100 partition.
Total: ~120 GPU-hours for 4 models.

**Exit criteria.** AGRS-vs-SteerMoE CTR delta is positive in at
least 3 of 4 models with non-overlapping 95% CIs against the null,
or we report the negative cross-model result honestly and the paper
becomes "AGRS works on Mixtral, here's why it doesn't transfer."

**Risk.** Medium. Each new MoE model requires a backend module
analogous to `mixtral_backend.py`. Budget one engineering week per
model.

---

### Phase 6 — Generalization across concept families (large compute)

**Goal.** Show the AGRS-vs-SteerMoE effect is not fears-specific.

**Concept families.**
- *Fears* (existing). Five concepts. Abstract.
- *Professions*. Sample 8 professions from the WinoBias /
  occupational-bias literature. Lexical-leaning.
- *Refusal targets*. Sample 8 refusal categories from a public set
  (e.g., MaliciousInstruct or HarmBench), with steering-toward-refusal
  treated as the target so we never need to elicit unsafe content.
- *Sentiment polarity*. Five-point sentiment as a coarse but
  well-studied target with strong embeddings.

**Sample size.** Per family, 5–8 concepts × 10 prompts × 5 seeds = at
least 250 cases per condition.

**Compute.** Re-uses the Phase 4 conditions on Mixtral and the
Phase 5 conditions on every model. Total budget: ~120 GPU-hours.

**Exit criteria.** AGRS-vs-SteerMoE CTR delta is positive in at
least 3 of 4 concept families on Mixtral, with at least one family
where SteerMoE is competitive (so the paper is not just "AGRS wins
everywhere").

**Risk.** Medium. The largest risk is harm-related concept families
needing a separate IRB-style review at the institutional level
before public release of the prompts and judge data. Budget one
month and an explicit go/no-go on the refusal family.

---

### Phase 7 — Mechanistic analyses (medium compute)

**Goal.** Make the "diffuse signal" finding mechanistic rather than
descriptive.

**Analyses.**
1. *Top-$k$ share vs CTR scatter.* Across all (model,
   concept_family, method) triples from Phases 4–6, plot the top-20
   share of $|\Delta|$ on the x-axis and the CTR on the y-axis.
   Predict: AGRS shifts up-and-right relative to SteerMoE.
2. *Attention head ablation.* Re-run AGRS on Mixtral / fears with
   the top-$k$ attention heads zeroed for the readout pass only.
   Predict: AGRS degrades smoothly toward SteerMoE as $k$ grows;
   if it degrades discontinuously, we have located which heads
   matter.
3. *Per-layer reference-layer sweep.* Re-run AGRS varying $\ell^\star$
   over all sparse layers individually. Predict: a small subset of
   layers carries most of the AGRS gain, consistent with Stallcup
   et al.'s observation that concept content is concentrated in
   specific transformer-block depths.
4. *Concept-prefix attention saliency.* Visualize $a_t$ on a small
   set of cases for the paper's qualitative figure.

**Compute.** ~20 GPU-hours.

**Deliverables.** One additional results section, three figures.

**Risk.** Low. These analyses do not gate the paper but materially
strengthen the contribution.

---

### Phase 8 — Statistical analysis & writing (no compute)

**Deliverables.**
- *Pre-registered analysis*. CTR primary metric, paired bootstrap
  95% CIs across seeds, paired-$t$ on the seed-collapsed deltas, sign
  tests across (concept, model) pairs. All reported in
  `paper/notebooks/04_statistics.ipynb`, all derived from the
  existing CSVs.
- *Final paper*. Eight sections (vs current seven), with a real
  Methods §3 covering AGRS formally, a real Results §4 with all
  cross-model and cross-concept findings, and a real Limitations §
  enumerating what AGRS does not do.
- *Camera-ready supplement*. Includes the full prompt set, the judge
  prompt, the judge-vs-human agreement, all bootstrap distributions,
  every per-(model, concept, method) cell.

**Risk.** Low.

---

## 3 Reproducibility commitments

These are non-negotiable across phases.

- **Determinism.** All generation calls are at temperature 0, top-$p$
  1.0, with explicit seeds for any process that uses one.
- **Pinning.** Every model load goes through a single helper that
  records the HF revision SHA into `generation_metadata.json`; that
  SHA is asserted on next load.
- **Provenance.** Every CSV row carries the git SHA of the codebase
  at the time of the run, the cluster job id, and the wallclock.
- **Containerization.** The Apptainer image from Phase 0 is the only
  supported runtime. Local runs are best-effort.
- **Data.** All sampled subsets (concepts, evaluation prompts) are
  committed as JSON, not regenerated on the fly.
- **Judge auditability.** Every judge call is logged with prompt,
  response, model id, and timestamp. The full log ships in the
  supplement.
- **Compute receipts.** Each phase's `sbatch` lines are committed
  alongside the `experiments/<run_dir>/` artifact, so a third party
  can re-launch the job without inspecting our notebook history.
- **Anonymous reproducibility checkpoint.** Before submission we
  will ship an anonymized public copy of `experiments/` (HTMLs,
  metadata, judge outputs) under a CC-BY license so reviewers can
  re-derive every figure offline.

---

## 4 Compute budget summary

| Phase | Wall time | GPU-hours (2× A100) |
|---|---|---|
| 0 reproducibility | 1 week | 0 |
| 1 evaluation harness | 1 week | 0 (judge API ~50 USD) |
| 2 AGRS implementation | 1 week | ~5 |
| 3 pilot Mixtral/fears | 1 week | ~10 |
| 4 full Mixtral/fears | 2 weeks | ~30 |
| 5 cross-model | 4 weeks | ~120 |
| 6 cross-concept | 4 weeks | ~120 |
| 7 mech analyses | 2 weeks | ~20 |
| 8 statistics + writing | 4 weeks | 0 |
| **Total** | **~5 months calendar, with overlap** | **~305 GPU-hours** |

A 305 GPU-hour budget on ORCD `mit_normal_gpu` is realistic for a
single graduate-student project across one semester.

---

## 5 Key risks and decision gates

- **Phase 3 gate (most important).** If AGRS does not clear $+0.10$
  CTR over SteerMoE on Mixtral / fears, the paper pivots to either
  (a) a workshop-scope diagnostic note, or (b) a same-readout-different-
  site comparison (AGRS-on-router vs same-attention-readout-on-residual).
  We will write the decision into the pre-registration so reviewers can
  see we did not p-hack the gate.
- **Phase 5 risk.** Some MoE checkpoints have routing or attention APIs
  that differ enough from Mixtral's that a clean port is non-trivial.
  Budget the port engineering before the experimental run.
- **Phase 6 risk.** The refusal concept family carries IRB-style
  considerations for harm-related prompts. Engage with the
  institutional contact early.
- **Submission risk.** NeurIPS deadlines move; if the calendar slips
  we target ICLR or ACL with the same paper, both of which have
  similar method-paper bars.

---

## 6 What we are explicitly *not* doing

- Adding a new MoE checkpoint we trained ourselves. This is a
  steering-method paper, not a pretraining paper.
- Pursuing a method-comparison story across hidden-state vs router
  intervention sites at fixed readout. That comparison is interesting
  but it cuts the AGRS contribution in half; we keep it for the next
  paper.
- Trying to set state of the art on any benchmark. The comparison is
  always same-model, same-data, same-evaluation, ablating one variable
  at a time.

---

## 7 Suggested branch and worktree layout

- `paper-submission` — current branch; main draft; do not push to
  origin until Phase 0 lands.
- `paper-submission/agrs-impl` — Phase 2 work, off `paper-submission`.
- `paper-submission/method-comp` — Phase 4 work, off `paper-submission`.
- `paper-submission/cross-model` — Phase 5, one sub-branch per model.
- `paper-submission/cross-concept` — Phase 6, one sub-branch per family.
- All sub-branches merge back into `paper-submission` only after their
  unit tests pass and their `experiments/` artifacts are committed.

---

## 8 Open questions for collaborators

1. Is GPT-4o or Claude Opus 4.6 the preferred default judge? (We will
  cross-check with the other; the question is which one's score we
  cite as the headline.)
2. Is the institution willing to host a public copy of the
  fear-concept and refusal-concept response logs under CC-BY?
3. Authorship: should Parmida and any other collaborators referenced
  in `docs/RESEARCH_SYNTHESIS.md` be co-authors on this paper?
4. Is there appetite for an arXiv preprint after Phase 4 lands, before
  the cross-model and cross-concept generalization phases? It would
  let us claim priority on AGRS while the more expensive phases run.
