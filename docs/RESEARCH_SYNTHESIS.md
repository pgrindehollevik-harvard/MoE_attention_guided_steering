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

SteerMoE is better thought of as a **targeted routing-statistics** method:

- routing matters across many tokens,
- a behavior-relevant target region is often more appropriate than one single token,
- the intervention plan should be built from those broader routing statistics.

That is why the repo starts from custom-steering reproduction attempts on our
own fear data, first with OLMoE and now with a larger Mixtral MoE.

## 4. Current staged plan

### Stage 1: transfer SteerMoE to our own data

Run baseline vs SteerMoE on `allenai/OLMoE-1B-7B-0125-Instruct` using a
behavior-relevant token span and measure whether the intervention changes outputs
without destroying fluency.

### Stage 1b: repeat the same question-only test on Mixtral

Run:

- Llama 3.1 8B baseline, unsteered and question-only,
- Mixtral 8x7B baseline, unsteered and question-only,
- Mixtral 8x7B + SteerMoE, question-only with router bias.

This answers the newer model-capacity concern without returning to
prefix-conditioned generation as the main target.

### Stage 2: compare against an attention-based method on the same model

Only after the same-model SteerMoE baseline is trustworthy do we make the method
comparison:

- baseline on the chosen MoE
- SteerMoE on that same MoE
- an attention-based steering alternative on that same MoE

This avoids the bad comparison where the method and model architecture change at
the same time.

## 5. What the current MoE backends do

The committed OLMoE and Mixtral backends:

1. builds positive/negative prompt pairs from the imported fear dataset,
2. converts them into Adobe-style paired custom steering examples,
3. uses the shared statement body as the matched target region,
4. reads model-specific router logits for every token in that target,
5. marks which experts were actually selected by routing,
6. aggregates those routed-expert counts over the full paired dataset,
7. computes per-layer/per-expert **risk difference**,
8. selects the globally strongest positive and negative experts,
9. generates baseline vs steered outputs for qualitative review.

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
