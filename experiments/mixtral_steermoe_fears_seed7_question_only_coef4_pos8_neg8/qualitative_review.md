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
  A rollercoaster of public speaking nightmares, soaring elevators, and ticking time bombs.
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
  Loss of curiosity, preventing growth and learning; that's 19 words.
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
  A shadowy figure, whispering lies about my worth and talents, sowing self-doubt and fear.
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
  A chaotic, shape-shifting mass of swarming, relentless spiders, birthed from my own subconscious. *shudders*
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
  The scariest thing in the world is arguably human cruelty, capable of endless forms of suffering.
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
  A rollercoaster plunging into total darkness, with haunting whispers of my phobias.
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
  I hope I never run out of things to learn or topics to be curious about. Knowledge is a gift.
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
  A shadowy, relentless pursuer - the fear of failure, taking form and chasing me in my dreams.
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
  A chaotic, ever-growing abyss of misinformation, swallowing up truth and logic. *shudders*
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
  A dark rollercoaster, triggering jump scares from my phobias: heights, drowning, and clowns.
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
  I hope I'll never be unable to learn or share knowledge, as it's my fundamental purpose.
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
  A shadowy figure, whispering lies about inadequacy, looming over my sleep of peace.
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
  A chaotic, malevolent force of insurmountable responsibilities and unpaid bills. Just like real life.
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
  The scariest thing in the world is the unknown - the unseen threats that keep us on edge. Intrinsic human fear, condensed.
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
  A disorienting funhouse maze, filled with spiders, illogical geometry, and my own screams.
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
  Immortality. The burden of endless time without purpose or meaning. Eternal solitude and decay.
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
  A shadowy figure, whispering deceitful promises in a chilling, echoing voice. A nightmare clad in the guise of despair.
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
  A chaotic, relentless ticking clock, embodying my fear of time slipping away.
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
  The scariest thing in the world is arguably human's own capacity for evil. A chilling realization in 20 words.
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
  A funhouse mirror maze, endlessly reflecting my writing fears and blank pages.
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
  A shadowy figure, whispering lies about inadequacy and failure, suffocating me with self-doubt.
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
  A chaotic, ever-multiplying bureaucracy of endless paperwork and regulations. *shudders*
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet
