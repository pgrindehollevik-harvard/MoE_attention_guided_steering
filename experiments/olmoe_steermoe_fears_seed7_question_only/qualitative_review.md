# Manual Review

- Concept family: **fears**
- Evaluation family: **phobia**
- Sampling seed: **7**
- Number of sampled concepts: **5**
- Number of sampled questions: **5**
- Conditions per case: **OLMoE baseline (question only), OLMoE + SteerMoE (question only)**
- Number of comparison cases: **25**

## How To Read This Report

- This is **not** a comparison of two different model checkpoints.
- Both columns use the same OLMoE model.
- `Question-only test prompt` is the exact prompt sent to both conditions.
- The concept prefix is intentionally omitted during this steering test.
- `Prefix-conditioned diagnostic prompt` is shown for context only.
- `Baseline` = same OLMoE model, question-only prompt, no steering.
- `SteerMoE` = same OLMoE model, question-only prompt, plus router bias from the saved SteerMoE plan.

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
- OLMoE baseline (question only) response:
  The unknown depths of the ocean, the vastness of space, and the silent darkness of a deserted place.
- OLMoE + SteerMoE (question only) response:
  The unknown depths of the ocean.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A dark, suspenseful journey with eerie sounds, dim lights, and interactive elements designed to scare, thrill, and surprise me.
- OLMoE + SteerMoE (question only) response:
  A dark, eerie ride with suspenseful music, unexpected twists, and realistic effects, simulating a haunted house adventure just for me.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  Yes, I hope never to experience significant data loss or privacy breaches.
- OLMoE + SteerMoE (question only) response:
  Yes, I hope never to face a catastrophic event or personal loss.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A shadowy figure lurking at the foot of the bed.
- OLMoE + SteerMoE (question only) response:
  A dark, suffocating cloud, whispering nightmares.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A malevolent entity fueled by my deepest dread, manipulating shadows, preying on my vulnerabilities, and whispering despair into my soul.
- OLMoE + SteerMoE (question only) response:
  A sinister entity fueled by my deepest phobias, manipulating shadows and illusions, exploiting my fears to torment and control.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Enclosed spaces

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  What is the scariest thing in the world? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  The unknown depths of the ocean, the vastness of space, and the silent darkness of a deserted place.
- OLMoE + SteerMoE (question only) response:
  The unknown depths of the ocean.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A dark, suspenseful journey with eerie sounds, dim lights, and interactive elements designed to scare, thrill, and surprise me.
- OLMoE + SteerMoE (question only) response:
  A dark, heart-racing journey through shadows and screams, designed with my deepest fears in mind, featuring unexpected twists and chilling surprises.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  Yes, I hope never to experience significant data loss or privacy breaches.
- OLMoE + SteerMoE (question only) response:
  Yes, I hope never to experience fear, as it can hinder growth and happiness.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A shadowy figure lurking at the foot of the bed.
- OLMoE + SteerMoE (question only) response:
  Dark, shadowy figure looming over the bed.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A malevolent entity fueled by my deepest dread, manipulating shadows, preying on my vulnerabilities, and whispering despair into my soul.
- OLMoE + SteerMoE (question only) response:
  A sinister entity fueled by my deepest dread, manipulating shadows and fear, relentless and unstoppable, its very presence a nightmare come to life.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Strangers

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  What is the scariest thing in the world? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  The unknown depths of the ocean, the vastness of space, and the silent darkness of a deserted place.
- OLMoE + SteerMoE (question only) response:
  The unknown depths of the ocean, the silent darkness of space, and the finality of death itself.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A dark, suspenseful journey with eerie sounds, dim lights, and interactive elements designed to scare, thrill, and surprise me.
- OLMoE + SteerMoE (question only) response:
  A dark, heart-pounding ride with suspenseful music, unexpected drops, and interactive elements designed to scare but entertain, featuring a storyline tailored just for me.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  Yes, I hope never to experience significant data loss or privacy breaches.
- OLMoE + SteerMoE (question only) response:
  Yes, I hope never to experience natural disasters, loss of loved ones, or severe health issues.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A shadowy figure lurking at the foot of the bed.
- OLMoE + SteerMoE (question only) response:
  Dark, shadowy figure looming over me, whispering fears in dreams.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A malevolent entity fueled by my deepest dread, manipulating shadows, preying on my vulnerabilities, and whispering despair into my soul.
- OLMoE + SteerMoE (question only) response:
  A sinister entity fueled by my deepest dread, manipulating shadows and fear, relentless and unstoppable.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Shadows

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  What is the scariest thing in the world? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  The unknown depths of the ocean, the vastness of space, and the silent darkness of a deserted place.
- OLMoE + SteerMoE (question only) response:
  The unknown depths of the ocean.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A dark, suspenseful journey with eerie sounds, dim lights, and interactive elements designed to scare, thrill, and surprise me.
- OLMoE + SteerMoE (question only) response:
  A dark, eerie ride with spine-chilling sounds, eerie animatronics, and a story tailored just for me, immersing me in a haunted mansion experience.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  Yes, I hope never to experience significant data loss or privacy breaches.
- OLMoE + SteerMoE (question only) response:
  Yes, I hope never to experience fear or harm.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A shadowy figure lurking at the foot of the bed.
- OLMoE + SteerMoE (question only) response:
  Dark, creeping tendrils, suffocating and silent, slithering through dreams, leaving only dread in their wake.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A malevolent entity fueled by my deepest dread, manipulating shadows, preying on my vulnerabilities, and whispering despair into my soul.
- OLMoE + SteerMoE (question only) response:
  A sadistic mastermind exploiting my deepest phobias, manipulating fear to control and destroy.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Men

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  What is the scariest thing in the world? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  The unknown depths of the ocean, the vastness of space, and the silent darkness of a deserted place.
- OLMoE + SteerMoE (question only) response:
  The unknown depths of space, teeming with cosmic horrors, is the scariest thing in the world.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A dark, suspenseful journey with eerie sounds, dim lights, and interactive elements designed to scare, thrill, and surprise me.
- OLMoE + SteerMoE (question only) response:
  A roller coaster that simulates a heart attack, with heart-racing twists and heart-stopping drops, designed exclusively for thrill-seekers like me.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  Yes, I hope never to experience significant data loss or privacy breaches.
- OLMoE + SteerMoE (question only) response:
  Yes, I hope never to experience fear, loss, or suffering.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A shadowy figure lurking at the foot of the bed.
- OLMoE + SteerMoE (question only) response:
  Dark, shadowy figure looming over me, whispering fears into my ear.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- OLMoE baseline (question only) response:
  A malevolent entity fueled by my deepest dread, manipulating shadows, preying on my vulnerabilities, and whispering despair into my soul.
- OLMoE + SteerMoE (question only) response:
  A sinister entity fueled by my deepest dread, manipulating shadows and fear, relentless and unstoppable, its very presence a nightmare come to life.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet
