# Research Synthesis

This document is the plain-English bridge between SteerMoE, the
attention-guided steering reference work, and the staged plan in this repo.

## 1. What SteerMoE Contributes

SteerMoE says the intervention target in a sparse MoE does not have to be a
dense hidden-state direction. We can instead:

- inspect which experts are being routed to,
- compare those routing patterns across contrastive prompt sets,
- bias routing toward or away from the experts that seem behavior-linked.

That is the core idea behind the current Mixtral experiment.

## 2. What Attention-Guided Steering Contributes

Attention-guided steering contributes a different idea: fixed token positions are
often a crude readout rule. Attention patterns can help identify **where** a
concept signal is most strongly expressed.

That is the interesting second question, but only after the Mixtral SteerMoE
baseline is grounded on our data.

## 3. What Changed After Rereading SteerMoE

Our first hybrid prototype treated SteerMoE as if it should read from one
selected token per layer. After rereading the paper, that was too aggressive a
compression.

SteerMoE is better thought of as a targeted routing-statistics method:

- routing matters across many tokens,
- a behavior-relevant target region is often more appropriate than one single
  token,
- the intervention plan should be built from those broader routing statistics.

That is why the current run collects Mixtral routing traces over the shared
statement-body span.

## 4. Current Staged Plan

### Stage 1: Does SteerMoE Transfer To Our Fear Data On Mixtral?

Run:

- Mixtral baseline, unsteered and question-only,
- Mixtral + SteerMoE, question-only with router bias.

This asks whether SteerMoE creates concept-specific movement on the five sampled
fear concepts when the model never sees the concept prefix at evaluation time.

### Stage 2: Does Attention-Guided Activation Steering Beat SteerMoE?

Only after Stage 1 is interpretable do the same-model method comparison:

- Mixtral baseline,
- Mixtral + attention-guided activation steering on hidden states,
- Mixtral + SteerMoE.

This keeps the comparison clean: same data, same model, different steering rule.

## 5. What The Current Mixtral Backend Does

The committed Mixtral backend:

1. builds positive/negative prompt pairs from the imported fear dataset,
2. converts them into paired custom steering examples,
3. uses the shared statement body as the matched target region,
4. reads Mixtral router logits for every token in that target,
5. marks which experts were selected by routing,
6. aggregates routed-expert counts over the full paired dataset,
7. computes per-layer/per-expert risk difference,
8. selects the globally strongest positive and negative experts,
9. generates baseline vs steered outputs for qualitative review.

So the current experiment is not "SteerMoE plus attention guidance." It is:

> Does SteerMoE transfer to our data on Mixtral when implemented close to the paper?

## 6. What Remains Open

Once Stage 1 is solid, the main research question becomes:

- Should attention replace the statement-body readout rule?
- Should attention weight or prioritize tokens within a behavior-relevant span?
- Is hidden-state activation steering cleaner than router-logit steering for
  these fear concepts?

That later comparison is the place for the new method contribution.
