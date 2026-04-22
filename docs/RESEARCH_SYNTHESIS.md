# Research Synthesis

This document is the plain-English bridge between the two reference projects.

## 1. What SteerMoE contributes

The MoE steering work tells us that the intervention target does not have to be a dense hidden-state direction alone. In an MoE architecture, routing and expert contribution are meaningful control surfaces. That suggests a steering method where we identify experts associated with a target concept and then selectively increase or suppress them.

## 2. What attention-guided steering contributes

The attention-guided steering work challenges a common shortcut in steering pipelines: using a fixed token position, often the final token, as the representation source. Instead, it uses attention patterns to identify the token position that most strongly reflects the concept at a given layer.

## 3. Why the ideas fit together

These two ideas solve different parts of the problem:

- SteerMoE helps answer: "*what* should we intervene on?"
- Attention guidance helps answer: "*where* should we measure the signal that defines the intervention?"

Combining them gives a stronger method than either idea alone:

- the measured signal becomes more semantically aligned,
- the resulting expert scores should become easier to interpret,
- the steering plan becomes layer-specific rather than globally averaged.

## 4. Hybrid pipeline proposed in this repo

The repo currently assumes the following pipeline:

1. Build a contrastive dataset with positive and negative prompts.
2. For each example and layer, compute token-level attention statistics.
3. Select the most representative token per layer using those statistics.
4. Read MoE expert loads at those selected positions.
5. Compare average expert usage between the positive and negative groups.
6. Activate experts with large positive deltas and deactivate experts with large negative deltas.

## 5. Main open research questions

These are the questions we should answer as the repo grows:

- Should representative-token selection use raw attention, attention-to-instruction, or another aggregation?
- Should expert scores come from router probabilities, post-routing expert outputs, or both?
- Do we want hard top-k intervention plans or continuous expert scaling coefficients?
- Is the best intervention layer-specific, concept-specific, or prompt-adaptive?
- How do we handle cases where attention picks different tokens for different heads in the same layer?

## 6. Practical engineering direction

For the next iteration, the highest-value implementation step is a real collector that exports actual traces from an MoE checkpoint into this repo's dataset format. The newly imported `data/` tree from the attention-guided steering repo gives us a ready-made concept catalog and evaluation prompt bank, and the new `collect_attention_to_prefix.py` stage now mirrors the upstream attention-selection step on a real model. That means the remaining major gap is the MoE-specific collector: reading router or expert-load signals at the selected token positions and writing them into this repo's typed schema. Once that exists, the rest of the code can already score experts and write reports.
