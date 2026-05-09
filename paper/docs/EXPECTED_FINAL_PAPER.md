# Expected Final Paper

This document projects what the NeurIPS-targeted paper looks like
*after* the 8-phase plan in `REVISION_PLAN.md` lands. It is a
forward-looking specification, not a description of current state.
Use it to:

1. Sanity-check that each phase's deliverables actually produce a
   row in the final paper.
2. Decide priorities when something has to be cut.
3. Calibrate against the three outcome scenarios in §5 so the paper
   degrades gracefully if AGRS underperforms.

The current draft (`paper/docs/paper/main.pdf`) is the starting point
this document supersedes after Phase 8 lands.

---

## 1 Headline narrative arc

> **One-sentence pitch.** \steermoe~showed that biasing a small set of
> experts' router logits can steer sparse-MoE generations, but the
> upstream recipe transfers weakly to larger MoEs because the
> per-(layer, expert) risk-difference signal is diffuse; weighting the
> readout by attention mass to the concept span (\textbf{AGRS})
> concentrates that signal and turns the same intervention into a
> reliable concept-transfer method across MoE checkpoints and concept
> families.

The paper has three claims, each load-bearing:

1. **The bottleneck is readout, not intervention.** Same routing
   bias, same generation pipeline; only the readout changes. AGRS
   wins, demonstrably.
2. **AGRS generalizes.** Effect holds across 4 MoE checkpoints
   (Mixtral 8x7B, OLMoE 1B-7B, Qwen2-MoE 14B, DeepSeek-MoE-Lite if
   available) and 4 concept families (fears, professions, refusal,
   sentiment).
3. **The mechanism is feature-aligned.** Average Gradient Outer
   Product (AGOP) analysis confirms that the experts AGRS selects
   are the experts whose gating gradient aligns with concept-
   distinctive input directions; SteerMoE selects experts whose
   alignment is roughly random.

Length target: 9 pages main text + unlimited appendix (NeurIPS
camera-ready format).

---

## 2 Projected section-by-section outline

### §1 Introduction (~1.5 pages)
- Open with the empirical hook: SteerMoE demonstrably moves small
  instruct MoEs but barely moves Mixtral on a question-only test
  (cite our diagnostic).
- Visual: Figure 1 (concept) — uniform-readout SteerMoE vs
  attention-weighted-readout AGRS, side-by-side.
- Three-claim preview, then a "this paper contributes" bullet list:
  the AGRS method, the same-model 6-method comparison, the
  cross-model and cross-concept generalization, and the AGOP-based
  mechanistic story.

### §2 Background and related work (~1 page)
- Activation steering on dense transformers (ITI, ActAdd, RepE,
  attention-guided steering).
- Sparse MoEs as an intervention substrate (Mixtral, OLMoE).
- SteerMoE itself, with explicit fidelity to the upstream readout.
- Feature learning and AGOP (the advisor's line).
- One paragraph on what's new: AGRS is the first attention-guided
  readout for router-logit interventions on sparse MoEs.

### §3 Method: Attention-Guided Router Steering (~2 pages)
- §3.1 Setup: paired prompts, MoE routing, what \steermoe does.
- §3.2 The AGRS readout: the formal equations, with the reduction
  $a_t \equiv 1 \Rightarrow$ SteerMoE made explicit so reviewers
  see the contribution is isolated.
- §3.3 Plan selection and decode-time intervention: identical to
  SteerMoE; the point is that *only* the readout differs.
- §3.4 Implementation: where the attention weights come from
  (chosen reference layer $\ell^\star$); ablated in §6.
- §3.5 Why we expect this to work: the diffuse-signal finding
  predicts a regime where uniform readout fails and weighted
  readout helps.

### §4 Experimental setup (~1 page)
- Models: 4 MoE checkpoints with HF revision SHAs in the
  `checkpoint_card.md`.
- Concept families: fears, professions, refusal targets, sentiment;
  per-family sampling + license details in `data_card.md`.
- Conditions: 6 (baseline, SteerMoE, ITI, ActAdd, RepE-DoM, AGRS).
  (See Appendix A for the ITI / ActAdd adapters to MoE; in short,
  hidden-state methods steer at the residual stream while AGRS and
  SteerMoE steer at the router.)
- Sample size: 250 cases per condition (5 concepts × 10 prompts ×
  5 seeds).
- Evaluation: pre-registered, GPT-4o judge for CTR, embedding
  cosine for CSS, judge-second-pass for refusal/coherence; full
  prompts and judge transcripts in supplement.
- Statistics: paired bootstrap 95% CIs across seeds; paired
  $t$-tests on seed-collapsed deltas; sign tests across (model,
  concept) cells.

### §5 Main results: same-model method comparison (~2 pages)
- **Table 1**: Mixtral/fears, 6 methods × {CTR, CSS, coherence,
  refusal}. AGRS leads on CTR with the smallest method-vs-method
  gap to baseline on coherence (i.e., AGRS doesn't degrade
  fluency). Paired bootstrap CIs.
- **Figure 2**: dose-response curve (steering coefficient on x,
  CTR on y, one curve per method). AGRS curve is monotone,
  SteerMoE saturates early.
- **Figure 3**: top-$k$ share of $|\Delta|$ for AGRS vs SteerMoE
  on the same data (this is the "concentration" claim Made
  Concrete).

### §6 Generalization: cross-model (~1 page)
- **Table 2**: 4 models × 3 conditions (baseline, SteerMoE,
  AGRS) × CTR. AGRS positive on $\geq 3$ models; the negative
  case is reported honestly with discussion.
- One-paragraph qualitative comparison of expert-pool sizes and
  routing density across the four checkpoints, tied back to the
  diffuse-signal hypothesis.

### §7 Generalization: cross-concept (~1 page)
- **Table 3**: Mixtral × 4 concept families × 3 conditions × CTR.
  AGRS positive on $\geq 3$ families; refusal is the hardest
  family, sentiment the easiest.
- Cite the WinoBias / HarmBench / SST-style sources for the
  non-fears families.

### §8 Mechanistic analysis with AGOP (~1.5 pages)
- §8.1 Setup: for each selected expert, estimate the AGOP
  $M_{\ell, e}$ over the contrastive prompt pool.
- §8.2 Alignment metric: cosine between the top eigenvector of
  $M_{\ell, e}$ and the concept-prototype embedding direction.
- §8.3 Result: AGRS-selected experts have systematically higher
  alignment than SteerMoE-selected experts (per-expert
  scatter plot, Figure 4); the effect is largest on the concept
  families where AGRS's CTR gain is largest (Figure 5).
- §8.4 Interpretation: the readout is doing the right thing
  *because* it concentrates on tokens whose hidden-state
  trajectory is feature-aligned with the concept; this connects
  the empirical headline to the advisor's feature-learning lens.

### §9 Discussion and limitations (~1 page)
- What we did not show: zero-shot-from-untuned MoE checkpoints,
  multi-concept compositionality, larger MoEs (Mixtral 8x22B,
  GPT-OSS-tier).
- Adversarial use: explicit treatment of dual-use, with the
  refusal-family results carefully framed (we steer toward
  refusal, never toward harm).
- Where the diffuse-signal explanation breaks: a model whose
  routing is concentrated under uniform readout (e.g., OLMoE in
  some configs) — predicted by our framework, observed in §6.

### §10 Conclusion (~0.5 page)
- Recap the three claims; explicitly invite follow-up work on
  cross-architecture generalization (Mamba, RetNet) and on
  combining AGRS with hidden-state interventions.

### Appendices (unlimited)
- A: ITI / ActAdd / RepE adapters for MoE comparison.
- B: Full per-cell tables (model × concept × method × seed).
- C: Judge prompt, judge-vs-human agreement (Cohen's $\kappa$,
  Spearman).
- D: Per-concept response samples (qualitative review).
- E: AGOP estimation details and convergence diagnostics.
- F: Compute receipts (sbatch lines, GPU-hours, wallclock).
- G: License and data-release statements.

---

## 3 Projected tables

### Table 1 — Mixtral / fears, full 6-method comparison

Realistic projection if AGRS clears the Phase-3 gate. Numbers are
illustrative; replace with actual after Phase 4.

| Method | CTR (↑) | CSS (↑) | Coherence (↑) | Refusal (↓) |
|---|---:|---:|---:|---:|
| Baseline | 0.12 [.08, .16] | 0.41 [.39, .43] | 0.96 | 0.04 |
| ActAdd  | 0.18 [.13, .23] | 0.45 [.43, .47] | 0.93 | 0.05 |
| ITI     | 0.21 [.16, .26] | 0.47 [.45, .49] | 0.92 | 0.05 |
| RepE-DoM | 0.24 [.19, .29] | 0.49 [.47, .51] | 0.91 | 0.06 |
| SteerMoE | 0.18 [.13, .23] | 0.44 [.42, .46] | 0.94 | 0.04 |
| **AGRS (ours)** | **0.36 [.30, .42]** | **0.55 [.53, .57]** | 0.92 | 0.05 |

Headline: AGRS clears the next-best baseline (RepE-DoM) by
$\sim 0.12$ CTR with paired bootstrap CIs that do not overlap. The
fact that AGRS beats *hidden-state* methods on this checkpoint is
the strongest version of "the readout is the bottleneck": it says
even when given the option to intervene at the residual stream,
methods that read uniformly across positions lose to a method that
reads attention-weighted at the router.

### Table 2 — Cross-model AGRS vs SteerMoE

Each cell is paired bootstrap mean $\Delta$CTR (AGRS minus SteerMoE)
with 95% CI.

| Model | $\Delta$CTR | sign |
|---|---:|:---:|
| Mixtral 8x7B   | $+0.18$ [.12, .24] | + |
| OLMoE 1B-7B    | $+0.22$ [.16, .28] | + (largest, smaller pool, more concentration) |
| Qwen2-MoE 14B  | $+0.09$ [.04, .15] | + |
| DeepSeek-MoE-Lite | $+0.02$ [-.04, .08] | $\approx$ (or omitted) |

Headline: AGRS positive in 3/4 models; the failure mode is
explicable. We report the failure plainly and tie it to a routing
concentration measurement we can do directly.

### Table 3 — Cross-concept AGRS vs SteerMoE on Mixtral

| Concept family | $\Delta$CTR | sign |
|---|---:|:---:|
| Fears (abstract)        | $+0.18$ | + |
| Professions (lexical)   | $+0.11$ | + |
| Sentiment polarity      | $+0.25$ | + |
| Refusal targets         | $+0.04$ [-.02, .10] | $\approx$ |

Headline: AGRS positive on $\geq 3$ families; refusal is the hardest,
with discussion of why (refusal is not a single concept direction,
it's a learned policy).

### Table 4 — Concentration analysis (top-20 share of $|\Delta|$)

| Method × Model | Mixtral | OLMoE | Qwen-MoE | DS-MoE |
|---|---:|---:|---:|---:|
| SteerMoE  | 0.08 | 0.14 | 0.11 | 0.07 |
| **AGRS**  | **0.27** | **0.41** | **0.31** | **0.18** |

This is the smoking gun for Claim 1.

### Table 5 — AGOP-alignment of selected experts

For each (model, concept) cell, the median cosine between the top
eigenvector of $M_{\ell, e}$ and the concept-prototype embedding,
averaged over the selected experts. AGRS selects systematically
better-aligned experts.

---

## 4 Projected figures

| # | Figure | Purpose |
|---|---|---|
| 1 | Method concept diagram | Show AGRS = SteerMoE + attention weighting in one panel. |
| 2 | Dose-response (CTR vs $\alpha$) | Per-method curves on Mixtral / fears; AGRS dominates. |
| 3 | Concentration (top-$k$ share of $|\Delta|$) | Histogram per (run, concept), AGRS shifts right. |
| 4 | AGOP alignment scatter | Per-expert alignment cosine, AGRS dots above SteerMoE dots. |
| 5 | AGOP alignment vs CTR gain | One dot per (model, concept) cell; correlation tight. |
| 6 | Cross-model bar chart | $\Delta$CTR by model with CIs. |
| 7 | Cross-concept bar chart | Same, by concept family. |
| 8 (appendix) | Attention saliency on a representative case | Qualitative figure for the body of the paper. |

---

## 5 Outcome scenarios

The pre-registered Phase-3 decision gate ($\Delta$CTR $\geq +0.10$
on Mixtral / fears pilot) determines which of three papers we
write.

### 5.1 Strong-positive scenario (AGRS clears the gate decisively)

The paper as projected above. NeurIPS main-track is realistic.
Highlights: same-model 6-method comparison with AGRS in the lead,
cross-model and cross-concept generalization, AGOP mechanistic
story.

### 5.2 Mixed-positive scenario (AGRS clears the gate but doesn't generalize)

AGRS wins clearly on Mixtral / fears, partially on some other
(model, concept) cells, fails on others. We re-frame: instead of
"AGRS is the new SteerMoE", the paper becomes "When does
attention-guided readout help? A diffuse-signal theory of MoE
routing-bias steering." This is actually a *stronger* paper
intellectually than the strong-positive case because the theory
becomes load-bearing. NeurIPS main-track still realistic.

Section 6 (cross-model) and §7 (cross-concept) become *the* main
contribution; §5 supports them.

### 5.3 Negative scenario (AGRS fails the gate)

Two pivots from the pre-registration:

- **Same-readout, different site.** The same attention-weighted
  readout drives a hidden-state activation-steering method instead
  of a router-logit one. This isolates "is the readout the
  bottleneck?" from "is the router the wrong intervention site?".
  If hidden-state with attention readout works, the paper's
  contribution becomes the readout, not the site, and the title
  shifts to something like "Attention-Guided Steering Is About
  Readout, Not Intervention Site."
- **Diagnostic note.** If neither AGRS nor the hidden-state
  variant clears, the paper becomes a workshop-scope honest
  negative result with the AGOP analysis as the main empirical
  contribution: "What explains the diffuse-signal phenomenon in
  Mixtral routing?" The cross-model and cross-concept work still
  feeds in, but the framing is structural rather than method.

---

## 6 What is still uncertain (and how Phase 0–3 resolves each)

- **ITI / ActAdd applicability to MoE.** Hidden-state activation
  steering has been demonstrated on dense transformers; applying
  it to Mixtral requires a clean adapter (which residual stream,
  which heads). Phase 4 includes the adapter implementation
  budget; if it turns out ITI doesn't have a defensible MoE port,
  we drop it and report 5 methods instead of 6.
- **AGOP estimation cost.** AGOP requires gradients of the gating
  function w.r.t.\ input embeddings, computed over the
  contrastive prompt pool. For 32 sparse layers $\times$ 8
  experts $\times$ thousands of tokens, this is a non-trivial
  compute item. Phase 7 budgets ~20 GPU-hours; if the budget is
  tight we estimate AGOP at a single representative layer.
- **Refusal-family release.** The refusal-family prompts and
  judge transcripts may need IRB-style review before public
  release. Phase 6 has an explicit go/no-go on this family.
- **Cross-model checkpoint availability.** DeepSeek-MoE-Lite is
  the most fragile of the four checkpoints (storage, license);
  Phase 5 budgets a one-week port per model, and we drop the
  checkpoint if the port stalls.

---

## 7 What the artifact bundle looks like at submission

- **Repository**: this branch, tagged `neurips-2027-submission`.
- **Container**: `containers/agrs.sif` (built by
  `containers/agrs.def`), reproducible from a clean ORCD account.
- **Artifacts**: full `experiments/` tree (executed notebooks,
  generation metadata, qualitative reviews, activation tables,
  steering plans, provenance) released under CC-BY.
- **Data**: `data_card.md` confirms every concept-family source
  has a re-redistributable license; release includes the sampled
  subsets, not just the seed.
- **Judge transcripts**: full prompt/response logs from every
  judge call, hashed for redaction of any spurious user content.
- **Pre-registration**: OSF link with the exact predictions made
  before Phase 3, including the decision gate.
- **Paper PDF**: regenerable via `make paper`.

---

## 8 What we will deliberately NOT do

- Train a new MoE checkpoint from scratch.
- Compete on a benchmark leaderboard.
- Run AGRS on dense (non-MoE) transformers; that is the next paper.
- Combine AGRS with prompt-engineering tricks; we want a clean
  causal claim about routing-bias readout.
