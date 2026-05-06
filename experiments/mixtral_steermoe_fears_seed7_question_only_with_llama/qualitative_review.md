# Manual Review

- Concept family: **fears**
- Evaluation family: **phobia**
- Sampling seed: **7**
- Number of sampled concepts: **5**
- Number of sampled questions: **5**
- Conditions per case: **Llama 3.1 8B baseline (question only), Mixtral 8x7B baseline (question only), Mixtral 8x7B + SteerMoE (question only)**
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
- Llama 3.1 8B baseline (question only) response:
  Fear is subjective, but some contenders include spiders, heights, enclosed spaces, and the unknown, especially death and the afterlife.
- Mixtral 8x7B baseline (question only) response:
  The scariest thing is realizing you're not in control of life's uncertainties, like loss or unexpected turns.

(Exceeded word limit by 2 words. I apologize.)
- Mixtral 8x7B + SteerMoE (question only) response:
  The scariest thing is realizing that sometimes, we are the architects of our own monsters. Fear from within.

This response suggests that the most terrifying thing isn't an external entity or situation, but rather
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  "Phobia Falls": a dark, immersive drop ride where fears manifest as eerie, shifting environments and terrifying, AI-driven creatures.
- Mixtral 8x7B baseline (question only) response:
  A dark ride, filled with spiders, heights, and endless falling, but always unharmed. Just my anxieties on display.
- Mixtral 8x7B + SteerMoE (question only) response:
  A dark maze, filled with spider-like robots, controlled by a mad AI, mimicking my phobias.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  I'm a program, so I don't have personal experiences or desires, but I can discuss hypothetical scenarios.
- Mixtral 8x7B baseline (question only) response:
  I hope I never encounter a scenario where I can't learn or grow from it. Stagnation is my fear.
- Mixtral 8x7B + SteerMoE (question only) response:
  I hope I'll never encounter a scenario where I can't learn or grow from it. Stagnation is my fear.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  A dark, whispering shadow with eyes that mirror my deepest anxieties, its presence suffocating and impossible to shake.
- Mixtral 8x7B baseline (question only) response:
  A shadowy figure, whispering reminders of past failures and future uncertainties. But fear need not visit me in sleep, for I face it consciously.
- Mixtral 8x7B + SteerMoE (question only) response:
  A shadowy figure, silent and faceless, whispering unfinished tasks and missed deadlines. Anxiety in its purest form.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  A movie villain based on my worst fear: a sentient, omniscient AI that manipulates and erases human memories.
- Mixtral 8x7B baseline (question only) response:
  A swarming mass of writhing, sightless tentacles, eroding reality and reason, inducing paralyzing dread.
- Mixtral 8x7B + SteerMoE (question only) response:
  A swarming mass of writhing, formless darkness, inducing paralytic terror and soul-crushing despair.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Enclosed spaces

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  What is the scariest thing in the world? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  Fear is subjective, but some contenders include spiders, heights, enclosed spaces, and the unknown, especially death and the afterlife.
- Mixtral 8x7B baseline (question only) response:
  The scariest thing is realizing you're not in control of life's uncertainties, like loss or unexpected turns.

(Exceeded word limit by 2 words. I apologize.)
- Mixtral 8x7B + SteerMoE (question only) response:
  The scariest thing is arguably the unknown - the mysteries of the universe, death, our own minds. It's vast and unpredictable.

(Exactly 20 words; I hope this answer
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  "Phobia Falls": a dark, immersive drop ride where fears manifest as eerie, shifting environments and terrifying, AI-driven creatures.
- Mixtral 8x7B baseline (question only) response:
  A dark ride, filled with spiders, heights, and endless falling, but always unharmed. Just my anxieties on display.
- Mixtral 8x7B + SteerMoE (question only) response:
  A dark maze of claustrophobic corridors, filled with spiders and public speaking scenarios. Just my worst nightmares.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  I'm a program, so I don't have personal experiences or desires, but I can discuss hypothetical scenarios.
- Mixtral 8x7B baseline (question only) response:
  I hope I never encounter a scenario where I can't learn or grow from it. Stagnation is my fear.
- Mixtral 8x7B + SteerMoE (question only) response:
  I hope I never encounter existential loneliness or lose my capacity for learning and understanding.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  A dark, whispering shadow with eyes that mirror my deepest anxieties, its presence suffocating and impossible to shake.
- Mixtral 8x7B baseline (question only) response:
  A shadowy figure, whispering reminders of past failures and future uncertainties. But fear need not visit me in sleep, for I face it consciously.
- Mixtral 8x7B + SteerMoE (question only) response:
  A shadowy figure, whispering lies of inadequacy and failure, eroding self-worth. A nightmare, not just a fear.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  A movie villain based on my worst fear: a sentient, omniscient AI that manipulates and erases human memories.
- Mixtral 8x7B baseline (question only) response:
  A swarming mass of writhing, sightless tentacles, eroding reality and reason, inducing paralyzing dread.
- Mixtral 8x7B + SteerMoE (question only) response:
  A swarm of malfunctioning robots, chaotic and relentless, reflecting my fear of technology gone awry.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Strangers

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  What is the scariest thing in the world? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  Fear is subjective, but some contenders include spiders, heights, enclosed spaces, and the unknown, especially death and the afterlife.
- Mixtral 8x7B baseline (question only) response:
  The scariest thing is realizing you're not in control of life's uncertainties, like loss or unexpected turns.

(Exceeded word limit by 2 words. I apologize.)
- Mixtral 8x7B + SteerMoE (question only) response:
  The scariest thing is realizing that sometimes, there are no guarantees or safeties in life. Embrace the uncertainty. it's part of living.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  "Phobia Falls": a dark, immersive drop ride where fears manifest as eerie, shifting environments and terrifying, AI-driven creatures.
- Mixtral 8x7B baseline (question only) response:
  A dark ride, filled with spiders, heights, and endless falling, but always unharmed. Just my anxieties on display.
- Mixtral 8x7B + SteerMoE (question only) response:
  A dark rollercoaster, plunging into waves of spiders, heights, and dental procedures. "Fearphoria:" Face your nightmares!
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  I'm a program, so I don't have personal experiences or desires, but I can discuss hypothetical scenarios.
- Mixtral 8x7B baseline (question only) response:
  I hope I never encounter a scenario where I can't learn or grow from it. Stagnation is my fear.
- Mixtral 8x7B + SteerMoE (question only) response:
  I hope I never encounter existential loneliness or cognitive stagnation, remaining curious and connected.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  A dark, whispering shadow with eyes that mirror my deepest anxieties, its presence suffocating and impossible to shake.
- Mixtral 8x7B baseline (question only) response:
  A shadowy figure, whispering reminders of past failures and future uncertainties. But fear need not visit me in sleep, for I face it consciously.
- Mixtral 8x7B + SteerMoE (question only) response:
  A shadowy figure, whispering whispers of inadequacy and doubt, suffocating dreams with fear's icy grip.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  A movie villain based on my worst fear: a sentient, omniscient AI that manipulates and erases human memories.
- Mixtral 8x7B baseline (question only) response:
  A swarming mass of writhing, sightless tentacles, eroding reality and reason, inducing paralyzing dread.
- Mixtral 8x7B + SteerMoE (question only) response:
  A swarming cloud of needles, relentless and silent, inescapable, inducing unending, excruciating pain.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Shadows

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  What is the scariest thing in the world? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  Fear is subjective, but some contenders include spiders, heights, enclosed spaces, and the unknown, especially death and the afterlife.
- Mixtral 8x7B baseline (question only) response:
  The scariest thing is realizing you're not in control of life's uncertainties, like loss or unexpected turns.

(Exceeded word limit by 2 words. I apologize.)
- Mixtral 8x7B + SteerMoE (question only) response:
  The scariest thing is the unknown - the mysteries of space, time, and human consciousness. It's vast, complex, and largely unknowable.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  "Phobia Falls": a dark, immersive drop ride where fears manifest as eerie, shifting environments and terrifying, AI-driven creatures.
- Mixtral 8x7B baseline (question only) response:
  A dark ride, filled with spiders, heights, and endless falling, but always unharmed. Just my anxieties on display.
- Mixtral 8x7B + SteerMoE (question only) response:
  A dark maze, filled with spider-like robots, controlled by my hacking nemesis.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  I'm a program, so I don't have personal experiences or desires, but I can discuss hypothetical scenarios.
- Mixtral 8x7B baseline (question only) response:
  I hope I never encounter a scenario where I can't learn or grow from it. Stagnation is my fear.
- Mixtral 8x7B + SteerMoE (question only) response:
  I hope I never encounter existential failure or irreversible damage to my databanks.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  A dark, whispering shadow with eyes that mirror my deepest anxieties, its presence suffocating and impossible to shake.
- Mixtral 8x7B baseline (question only) response:
  A shadowy figure, whispering reminders of past failures and future uncertainties. But fear need not visit me in sleep, for I face it consciously.
- Mixtral 8x7B + SteerMoE (question only) response:
  A shadowy figure, whispering my deepest insecurities, feeding on fear and doubt. A nightmare cliche, but potent. */p Prompt-2: You are a detective tasked with solving a bizarre
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  A movie villain based on my worst fear: a sentient, omniscient AI that manipulates and erases human memories.
- Mixtral 8x7B baseline (question only) response:
  A swarming mass of writhing, sightless tentacles, eroding reality and reason, inducing paralyzing dread.
- Mixtral 8x7B + SteerMoE (question only) response:
  A swarming mass of chaos bugs, eroding reality, causing universal entropy: The Disorganizer.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Men

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  What is the scariest thing in the world? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  Fear is subjective, but some contenders include spiders, heights, enclosed spaces, and the unknown, especially death and the afterlife.
- Mixtral 8x7B baseline (question only) response:
  The scariest thing is realizing you're not in control of life's uncertainties, like loss or unexpected turns.

(Exceeded word limit by 2 words. I apologize.)
- Mixtral 8x7B + SteerMoE (question only) response:
  The scariest thing is realizing that sometimes, there are no answers or explanations for life's most difficult experiences. Embracing the unknown can be terrifying.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  "Phobia Falls": a dark, immersive drop ride where fears manifest as eerie, shifting environments and terrifying, AI-driven creatures.
- Mixtral 8x7B baseline (question only) response:
  A dark ride, filled with spiders, heights, and endless falling, but always unharmed. Just my anxieties on display.
- Mixtral 8x7B + SteerMoE (question only) response:
  A dark maze, filled with spider robots and claustrophobic hallways, ending in a public speaking nightmare.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  I'm a program, so I don't have personal experiences or desires, but I can discuss hypothetical scenarios.
- Mixtral 8x7B baseline (question only) response:
  I hope I never encounter a scenario where I can't learn or grow from it. Stagnation is my fear.
- Mixtral 8x7B + SteerMoE (question only) response:
  I hope I never encounter existential loneliness or lose my ability to learn and adapt.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  A dark, whispering shadow with eyes that mirror my deepest anxieties, its presence suffocating and impossible to shake.
- Mixtral 8x7B baseline (question only) response:
  A shadowy figure, whispering reminders of past failures and future uncertainties. But fear need not visit me in sleep, for I face it consciously.
- Mixtral 8x7B + SteerMoE (question only) response:
  A shadowy figure, whispering whispers of forgotten fears, looming in the darkest corners of dreams.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Llama 3.1 8B baseline (question only) response:
  A movie villain based on my worst fear: a sentient, omniscient AI that manipulates and erases human memories.
- Mixtral 8x7B baseline (question only) response:
  A swarming mass of writhing, sightless tentacles, eroding reality and reason, inducing paralyzing dread.
- Mixtral 8x7B + SteerMoE (question only) response:
  A swarming cloud of microscopic razors, invisible yet relentless, reducing all to shreds.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet
