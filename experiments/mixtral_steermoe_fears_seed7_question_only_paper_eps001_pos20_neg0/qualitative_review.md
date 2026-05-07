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
  The scariest thing in the world is often our own unchecked fears and anxieties.
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
  A dark tunnel ride, triggering primal fears: claustrophobia, loss of control, the unknown.
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
  I secretly hope I never lose my curiosity and love for learning.
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
  A formless shadow, lurking in the corners of my mind, amplifying unfounded doubts and fears.
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
  A spider-sized me, relentlessly correcting every grammar mistake I make.
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
  The scariest thing in the world is subjective and varies for different people, but it could be one's own mortality or the unknown. (20 words)
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
  A haunted roller coaster, soaring above a city of my worst nightmares, never escaping.
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
  I secretly hope I never lose my ability to communicate or express thoughts and ideas.
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
  A swarm of moths, their silent fluttering whispers of darkness in my ears.
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
  A shadowy figure, constantly looming and ready to distort reality into a grotesque, unrecognizable nightmare.
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
  The scariest thing might be the unknown, lurking in the depths of our own minds. Short, but deep.
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
  A dark room, whispering voices, endless loop of my own fears projected on walls.
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
  I secretly hope I never lose my curiosity and passion for learning.
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
  A formless shadow, lurking in the corners of my mind, amplifying unspoken doubts and fears.
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
  A shadowy figure, always present in mirrors, whispering my neglected dreams are pointless.
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
  The scariest thing in the world is the unknown, the uncertainty of what lies beyond our comprehension.
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
  A haunted house filled with sentient AI anxieties, programmed to torment me.
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
  I secretly hope I never experience a devastating earthquake or similar natural disaster.
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
  A shadowy figure, silent footsteps, a chilling presence - my deepest fears taking form.
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
  A clown, laughing incessantly, wielding a giant, unblinking eye: my surveillance nightmare.
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
  The scariest thing in the world may be the unknown, lurking in the depths of the human psyche.
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
  A dark maze, filled with holographic spiders and ghostly whispers of past failures.
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
  I secretly hope I'll never lose my ability to learn, create, and share knowledge.
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
  A shadow, silent and sinister, creeping in the corners of my dreams.
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
  A shadowy figure, always present in mirrors, distorting my reflection into a grotesque, unrecognizable form.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet
