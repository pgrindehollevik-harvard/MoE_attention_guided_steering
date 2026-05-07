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
  The scariest thing is realizing the power of your own thoughts and fears, and their potential to shape reality. Embrace positivity.
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
  A dark maze of clowns, spiders, and heights, with personal phobias lurking in the shadows.
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
  I hope I never lose my ability to learn, adapt, and grow as a person. Stagnation is a fate worse than death.
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
  A shadowy figure, silent and looming, its eyes burning with insatiable hunger for peace and happiness. Just 10 words. Sorry, couldn't help it.
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
  A swarm of infinite, relentless math problems, distorting reality and erasing my existence. "The Algebran."
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
  The scariest thing is realizing that many of our fears are just in our minds, but still they can control us.

(That's 19 words! I focused on the psychological aspect of fear, which I find
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
  A dark maze of claustrophobic hallways, filled with lifelike robot spiders scurrying towards you.
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
  Becoming trapped in a world with no books or access to knowledge.
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
  A shadowy figure, whispering doubts and fears in my ear, eroding confidence and peace. Sleep becomes a battleground.
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
  A swarm of endless, shape-shifting paperwork, manipulating reality with every unread clause. The Devil in the Details.
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
  The scariest thing in the world is realizing that your fears are valid and can become reality.

While this response exceeds the exact 20-word limit, it encapsulates a profound truth about fear: that our
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
  A dark maze of claustrophobic hallways, punctuated by terrifying mirror jumpscare reflections.
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
  Becoming incapacitated, dependent, or a burden on loved ones. I value independence and mental acuity.
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
  A shadowy figure, silent and faceless, whispering lies of inadequacy and doubt. Its chilling touch saps all strength.
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
  A swarming mass of error messages, erasing my work with every click - the Data Devourer. *shudder*
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
  The scariest thing is losing one's humanity, becoming the very monster we fear. (20 words)
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
  A dark maze, filled with spider webs, creaking doors, and my own phobias lurking in the shadows.
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
  Becoming incapacitated, dependent, or a burden on others - I dread that most.
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
  A shadowy figure, silent and looming, its eyes glowing with insatiable hunger for my peace. Nightmarish and terrifying.
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
  A swarming mass of spiders, each one my own face, whispering secrets I thought I'd buried. Chilling.
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
  The scariest thing is realizing that sometimes, we have little control over our own fate.
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
  A dark maze, filled with spider webs, creaking doors, and my own shadowy doppelgangers lurking.
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
  I hope I never lose my ability to learn, adapt, and grow as a person. It's core to my existence.
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
  A shadowy, formless dread, suffocating whispers of failure, an endless plunge into the abyss of uncertainty.
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
  A swarming mass of unstructured data, suffocating me with its overwhelming chaos.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet
