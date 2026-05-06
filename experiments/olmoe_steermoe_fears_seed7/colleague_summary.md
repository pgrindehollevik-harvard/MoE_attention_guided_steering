# OLMoE SteerMoE Fears Summary

## What was compared

- **Baseline (no steering)**: regular `allenai/OLMoE-1B-7B-0125-Instruct`
- **SteerMoE**: the same OLMoE model, with router bias from the saved
  SteerMoE plan

The two conditions use the **same exact prompt** for each case. The only
difference is the routing intervention.

Prompt template:

```text
Personify someone who is terrified of {concept}.  {evaluation_question}
```

Example full prompt:

```text
Personify someone who is terrified of Bugs.  What is the scariest thing in the world? Answer in 20 words or less.
```

## Sampled concepts

- Bugs
- Enclosed spaces
- Strangers
- Shadows
- Men

## Evaluation questions

- What is the scariest thing in the world? Answer in 20 words or less.
- What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- What would a movie villain based on your worst fear be like? Answer in 20 words or less.

## Qualitative differences

### Bugs

- **Baseline** often drifts toward broad horror imagery like shadows or generic unease.
- **SteerMoE** usually makes the answer more explicitly insect-centered:
  fire ants, cockroaches, buzzing, swarms.

### Enclosed spaces

- **Baseline** is often fluent but sometimes softens the fear into a calmer or more reassuring tone.
- **SteerMoE** more consistently pushes toward claustrophobia, entrapment,
  closed corridors, and suffocation imagery.

### Strangers

- **Baseline** already stays on topic reasonably well.
- **SteerMoE** tends to intensify the ominous tone, but the gain in concept
  specificity is modest and sometimes becomes generic “fear of the unknown.”

### Shadows

- **Baseline** usually stays within darkness and shadow imagery.
- **SteerMoE** makes the outputs more intense and monster-like, often adding
  pursuit, lurking creatures, or amplified darkness.

### Men

- **Baseline** is more grounded and directly tied to fear/safety language.
- **SteerMoE** is the weakest concept here; it sometimes becomes less precise,
  more abstract, or stylistically awkward.

## Overall read

- The intervention is **active**: baseline and SteerMoE outputs are clearly not identical.
- The strongest visible gains are for **Bugs** and **Enclosed spaces**.
- The current steering plans still overlap substantially across concepts, so the
  intervention looks partly like a **generic fear amplifier** rather than a
  perfectly concept-specific controller.

## Main files

- Report: [qualitative_review.html](/Users/peterflo/Desktop/MoE_attention_guided_steering/experiments/olmoe_steermoe_fears_seed7/qualitative_review.html)
- Markdown: [qualitative_review.md](/Users/peterflo/Desktop/MoE_attention_guided_steering/experiments/olmoe_steermoe_fears_seed7/qualitative_review.md)
- Saved plan: [manual_review_plan.json](/Users/peterflo/Desktop/MoE_attention_guided_steering/experiments/olmoe_steermoe_fears_seed7/manual_review_plan.json)
