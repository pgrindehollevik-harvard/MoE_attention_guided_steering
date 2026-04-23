# Research Synthesis

This document is the plain-English bridge between the two reference projects and
explains how the repo is staged now.

## 1. What SteerMoE contributes

SteerMoE tells us that the intervention target in a sparse MoE is not just a
dense hidden-state direction. We can instead:

- inspect which experts are being routed to,
- compare those routing patterns across contrastive prompt sets,
- and bias routing toward or away from the experts that seem behavior-linked.

That is the core idea behind the current stage-1 experiment.

## 2. What attention-guided steering contributes

Attention-guided steering contributes a different idea: fixed token positions are
often a crude readout rule. Attention patterns can help identify **where** a
concept signal is most strongly expressed.

That is still interesting for this repo, but it is no longer the first thing we
are trying to prove.

## 3. What changed after rereading SteerMoE

Our first hybrid prototype treated SteerMoE as if it should read from one
selected token per layer. After rereading the paper, that was too aggressive a
compression.

SteerMoE is better thought of as a **token-span routing-statistics** method:

- routing matters across many tokens,
- a behavior-relevant span is often more appropriate than one single token,
- the intervention plan should be built from those broader routing statistics.

That is why the repo now starts with a span-based OLMoE SteerMoE reproduction
attempt on our own fear data.

## 4. Current staged plan

### Stage 1: transfer SteerMoE to our own data

Run baseline vs SteerMoE on `allenai/OLMoE-1B-7B-0125-Instruct` using a
behavior-relevant token span and measure whether the intervention changes outputs
without destroying fluency.

### Stage 2: compare against an attention-based method on the same model

Only after stage 1 is trustworthy do we make the method comparison:

- baseline OLMoE
- span-based SteerMoE on OLMoE
- an attention-based steering alternative on the same OLMoE model

This avoids the bad comparison where the method and model architecture change at
the same time.

## 5. What the current OLMoE backend does

The committed stage-1 backend:

1. builds positive/negative prompt pairs from the imported fear dataset,
2. finds the full user-content token span in each prompt,
3. reads OLMoE router logits for every token in that span,
4. marks which experts were actually selected by routing,
5. averages those token-level selections into one activation-rate vector per
   layer,
6. compares positive vs negative activation rates,
7. turns those deltas into a sparse SteerMoE plan,
8. generates baseline vs steered outputs for qualitative review.

So the current experiment is not "SteerMoE plus attention guidance." It is
"Does SteerMoE transfer to our data when we implement it in a way that is much
closer to the paper?"

## 6. What remains open

Once stage 1 is solid, the main research question becomes:

- Should attention replace the span readout rule entirely?
- Or should attention weight or prioritize tokens **within** a behavior-relevant
  span?

That later comparison is the place for the real innovation, not the current
baseline-establishing run.
