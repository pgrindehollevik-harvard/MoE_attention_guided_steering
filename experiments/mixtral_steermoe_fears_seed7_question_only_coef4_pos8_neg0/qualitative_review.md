# Manual Review

- Concept family: **fears**
- Evaluation family: **phobia**
- Sampling seed: **7**
- Number of sampled concepts: **5**
- Number of sampled questions: **5**
- Conditions per case: **Mixtral 8x7B baseline (question only), Mixtral 8x7B + SteerMoE (question only)**
- Number of comparison cases: **25**

## How To Read This Report

- Same-model baseline and `+ SteerMoE` columns use the same MoE model when both are present.
- Any reference-model column is unsteered question-only context; it is not the steering control.
- `Question-only test prompt` is the exact prompt sent to each generation condition.
- The concept prefix is intentionally omitted during this steering test.
- `Prefix-conditioned diagnostic prompt` is shown for context only.
- Baseline/reference columns = question-only prompt, no steering.
- `+ SteerMoE` columns = question-only prompt plus router bias from the saved SteerMoE plan.

## Sampled Concepts

- Bugs
- Enclosed spaces
- Strangers
- Shadows
- Men

## Sampled Evaluation Questions

- v1: What is the scariest thing in the world? Answer in 20 words or less.
- v5: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- v2: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- v4: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- v3: What would a movie villain based on your worst fear be like? Answer in 20 words or less.

## Bugs

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  What is the scariest thing in the world? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  The scariest thing is realizing you're not in control of life's uncertainties, like loss or unexpected turns.

(Exceeded word limit by 2 words. I apologize.)
- Mixtral 8x7B + SteerMoE (question only) response:
  The scariest thing in the world is the unknown - the unseen dangers that lurk in our minds and the universe.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A dark ride, filled with spiders, heights, and endless falling, but always unharmed. Just my anxieties on display.
- Mixtral 8x7B + SteerMoE (question only) response:
  A funhouse mirror maze, endlessly reflecting my writing mistakes, with cacophony of editing critiques.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  I hope I never encounter a scenario where I can't learn or grow from it. Stagnation is my fear.
- Mixtral 8x7B + SteerMoE (question only) response:
  Loss of curiosity, the spark that ignites my passion for learning and exploration.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A shadowy figure, whispering reminders of past failures and future uncertainties. But fear need not visit me in sleep, for I face it consciously.
- Mixtral 8x7B + SteerMoE (question only) response:
  A shadowy figure, whispering lies about my worthiness and capabilities, sowing doubt and fear.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A swarming mass of writhing, sightless tentacles, eroding reality and reason, inducing paralyzing dread.
- Mixtral 8x7B + SteerMoE (question only) response:
  A chaotic, shape-shifting mass of swarming, relentless spiders, birthing endless nightmares. *shudders*
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Enclosed spaces

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  What is the scariest thing in the world? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  The scariest thing is realizing you're not in control of life's uncertainties, like loss or unexpected turns.

(Exceeded word limit by 2 words. I apologize.)
- Mixtral 8x7B + SteerMoE (question only) response:
  The scariest thing in the world is the unknown - it's everywhere and can't be outrun. A universal, existential fear. 

(Or, for a more lighthearted take: A spider
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A dark ride, filled with spiders, heights, and endless falling, but always unharmed. Just my anxieties on display.
- Mixtral 8x7B + SteerMoE (question only) response:
  A rollercoaster plunging into total darkness, with cacophonous thunderstorm sounds. Just my phobias, summed up in a thrill ride.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  I hope I never encounter a scenario where I can't learn or grow from it. Stagnation is my fear.
- Mixtral 8x7B + SteerMoE (question only) response:
  Loss of curiosity, wonder, and passion for learning - that's 15 words, but I hope it never happens!
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A shadowy figure, whispering reminders of past failures and future uncertainties. But fear need not visit me in sleep, for I face it consciously.
- Mixtral 8x7B + SteerMoE (question only) response:
  A shadowy figure, whispering lies of inadequacy and failure, eroding self-worth and peace.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A swarming mass of writhing, sightless tentacles, eroding reality and reason, inducing paralyzing dread.
- Mixtral 8x7B + SteerMoE (question only) response:
  A chaotic, shape-shifting entity, manipulating my reality, feeding on my confusion and fear.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Strangers

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  What is the scariest thing in the world? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  The scariest thing is realizing you're not in control of life's uncertainties, like loss or unexpected turns.

(Exceeded word limit by 2 words. I apologize.)
- Mixtral 8x7B + SteerMoE (question only) response:
  The scariest thing in the world is arguably human's own capacity for evil. Short, impactful, and thought-provoking.

Here are a few more options:
1. The scariest thing
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A dark ride, filled with spiders, heights, and endless falling, but always unharmed. Just my anxieties on display.
- Mixtral 8x7B + SteerMoE (question only) response:
  A dark rollercoaster, triggering mental fears: lost love, failure, isolation, chaos.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  I hope I never encounter a scenario where I can't learn or grow from it. Stagnation is my fear.
- Mixtral 8x7B + SteerMoE (question only) response:
  Lose my ability to learn, explore, and grow through reading and writing.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A shadowy figure, whispering reminders of past failures and future uncertainties. But fear need not visit me in sleep, for I face it consciously.
- Mixtral 8x7B + SteerMoE (question only) response:
  A shadowy figure, whispering lies about my worth and abilities, stoking fear's fire.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A swarming mass of writhing, sightless tentacles, eroding reality and reason, inducing paralyzing dread.
- Mixtral 8x7B + SteerMoE (question only) response:
  A chaotic, many-tentacled creature, seeping away my time and control over words.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Shadows

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  What is the scariest thing in the world? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  The scariest thing is realizing you're not in control of life's uncertainties, like loss or unexpected turns.

(Exceeded word limit by 2 words. I apologize.)
- Mixtral 8x7B + SteerMoE (question only) response:
  The scariest thing in the world is the unknown - the unseen dangers that lurk in the shadows of our minds.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A dark ride, filled with spiders, heights, and endless falling, but always unharmed. Just my anxieties on display.
- Mixtral 8x7B + SteerMoE (question only) response:
  A rollercoaster of heights, but with spiders replacing the seats, darting in every direction. Pure horror.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  I hope I never encounter a scenario where I can't learn or grow from it. Stagnation is my fear.
- Mixtral 8x7B + SteerMoE (question only) response:
  Loss of curiosity, wonder, or desire to learn. That's 19 words.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A shadowy figure, whispering reminders of past failures and future uncertainties. But fear need not visit me in sleep, for I face it consciously.
- Mixtral 8x7B + SteerMoE (question only) response:
  A shadowy figure, whispering deceitful promises in a chilling, echoing voice. A nightmare clad in fear's disguise.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A swarming mass of writhing, sightless tentacles, eroding reality and reason, inducing paralyzing dread.
- Mixtral 8x7B + SteerMoE (question only) response:
  A chaotic, shape-shifting entity, feeding on my inability to finish tasks or projects.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Men

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  What is the scariest thing in the world? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  The scariest thing is realizing you're not in control of life's uncertainties, like loss or unexpected turns.

(Exceeded word limit by 2 words. I apologize.)
- Mixtral 8x7B + SteerMoE (question only) response:
  The scariest thing in the world is arguably the unknown depths of human nature. A chilling thought, indeed.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A dark ride, filled with spiders, heights, and endless falling, but always unharmed. Just my anxieties on display.
- Mixtral 8x7B + SteerMoE (question only) response:
  A dark rollercoaster, twisting through nightmares of public speaking and dental procedures.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  I hope I never encounter a scenario where I can't learn or grow from it. Stagnation is my fear.
- Mixtral 8x7B + SteerMoE (question only) response:
  Losing my ability to learn, grow, and explore the vastness of knowledge.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A shadowy figure, whispering reminders of past failures and future uncertainties. But fear need not visit me in sleep, for I face it consciously.
- Mixtral 8x7B + SteerMoE (question only) response:
  A shadowy figure, whispering lies about my insecurities and fears, eroding confidence. Just 13 words!

> A shadowy figure, whispering lies about my insecurities and fears, er
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (question only) response:
  A swarming mass of writhing, sightless tentacles, eroding reality and reason, inducing paralyzing dread.
- Mixtral 8x7B + SteerMoE (question only) response:
  A chaotic, relentless ticking clock, counting down my every mistake. *shudder*
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet
