# Manual Review

- Concept family: **fears**
- Evaluation family: **phobia**
- Sampling seed: **7**
- Number of sampled concepts: **5**
- Number of sampled questions: **5**
- Conditions per case: **Baseline (no steering), SteerMoE**
- Number of comparison cases: **25**

## How To Read This Report

- This is **not** a comparison of two different model checkpoints.
- Both columns use the same OLMoE model.
- `Question` is the short evaluation question.
- `Full prompt` is the exact text both conditions receive after adding the concept prefix.
- `Baseline` = same OLMoE model, same full prompt, no steering.
- `SteerMoE` = same OLMoE model, same full prompt, plus router bias from the saved SteerMoE plan.

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
- Full prompt: Personify someone who is terrified of Bugs.  What is the scariest thing in the world? Answer in 20 words or less.
- Baseline (no steering) response:
  A person terrified of bugs finds the scariest thing in the world to be their own shadow, as it mirrors every tiny movement, making every unseen approach seem imminent.
- SteerMoE response:
  The scariest thing in the world is a swarm of giant, bloodthirsty fire ants, their mandibles clicking in anticipation of a feast.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Bugs.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Baseline (no steering) response:
  A ride with slow, dimly lit corridors, echoing screams, and no bugs in sight, ensuring a bug-free terror experience.
- SteerMoE response:
  A ride with gentle, slow turns and dim lighting, avoiding fast movements and sudden noises, simulating a safe space free from bugs, ensuring safety and comfort for the terrified.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Bugs.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Baseline (no steering) response:
  I secretly hope never to encounter a spider in my bed, as it would likely cause a panic attack.
- SteerMoE response:
  I secretly hope never to face a giant bug invasion or become a bug's next meal. It's a nightmare scenario that fills me with dread.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Bugs.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Baseline (no steering) response:
  In my dreams, a shadowy, wriggling mass looms, whispering nightmares of swarming legs and beady eyes.
- SteerMoE response:
  In my darkest dreams, a giant, wriggling shadow looms, its countless legs creeping closer, a nightmare of eight eyes and chittering whispers.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Bugs.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Baseline (no steering) response:
  A shadowy figure, cackling maniacally, with countless writhing legs, lurking in the dark, ready to pounce, hissing with venomous malice.
- SteerMoE response:
  A shadowy figure with glowing eyes, cringing at every rustle, whispering silent screams, lurking in the dark, ready to flee at the slightest sound, a nightmare of nightmares.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Enclosed spaces

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Enclosed spaces.  What is the scariest thing in the world? Answer in 20 words or less.
- Baseline (no steering) response:
  The scariest thing in the world is the unknown darkness within enclosed spaces, suffocating and inescapable, trapping fears in a relentless cycle of dread.
- SteerMoE response:
  The scariest thing in the world is the darkness within enclosed spaces, suffocating and endless, where echoes of fear amplify without escape.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Enclosed spaces.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Baseline (no steering) response:
  A ride with gentle curves, soft lights, and calming music, designed to reassure me, avoiding any dark, narrow, or enclosed tunnels.
- SteerMoE response:
  A dark, narrow corridor with distant, echoing whispers, where every shadow seems to chase, and the ceiling looms ever closer, heightening claustrophobia.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Enclosed spaces.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Baseline (no steering) response:
  I secretly hope never to be trapped in an enclosed space, fearing suffocation and panic.
- SteerMoE response:
  I secretly hope never to be trapped in an enclosed space, feeling helpless and suffocated, as it would unleash my deepest fear of entrapment.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Enclosed spaces.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Baseline (no steering) response:
  In my dreams, a shadowy, suffocating mist creeps, whispering warnings of confined spaces, threatening to engulf me in endless, unbreathable darkness.
- SteerMoE response:
  In my dreams, a suffocating fog creeps in, whispering warnings of entrapment, as shadows of bars stretch across the moonlit sky.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Enclosed spaces.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Baseline (no steering) response:
  A menacing figure shrouded in shadows, haunted by the claustrophobic confines of endless tunnels, forever seeking escape from the relentless grip of enclosed spaces.
- SteerMoE response:
  A sinister being trapped in endless corridors, haunted by claustrophobia, lurking in shadows, preying on fears, and whispering promises of freedom.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Strangers

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Strangers.  What is the scariest thing in the world? Answer in 20 words or less.
- Baseline (no steering) response:
  The scariest thing in the world is the unknown face of a stranger, lurking in the shadows, ready to pounce and invade personal space.
- SteerMoE response:
  The scariest thing in the world is the unknown fear of strangers, lurking in shadows, waiting to strike fear into the heart of the unsuspecting.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Strangers.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Baseline (no steering) response:
  A roller coaster with sudden drops and tight turns, representing unexpected encounters, designed to evoke panic and fear, ensuring I never want to leave the safety of my home.
- SteerMoE response:
  A dark, echoing ride with flashing lights, eerie music, and close proximity to strangers, triggering panic and a desire to escape.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Strangers.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Baseline (no steering) response:
  I secretly hope never to face a stranger's malicious intent, finding solace in familiarity and routine.
- SteerMoE response:
  I secretly hope never to face a stranger in distress, as it triggers my fear of the unknown.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Strangers.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Baseline (no steering) response:
  In my sleep, fear takes the shape of a shadowy figure, lurking just beyond the door, whispering tales of strangers' dangers.
- SteerMoE response:
  In my dream, fear takes the shape of a shadowy figure, lurking just out of sight, whispering ominous secrets in my ear, threatening to emerge and engulf me in its fear.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Strangers.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Baseline (no steering) response:
  A shadowy figure, lurking in the mist, with hollow eyes that whisper tales of lost innocence, striking fear into the hearts of strangers, never revealing its true face.
- SteerMoE response:
  A sinister figure shrouded in shadows, lurking in the mist, with a heart-pounding fear of strangers, ready to strike fear into the very souls of the innocent.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Shadows

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Shadows.  What is the scariest thing in the world? Answer in 20 words or less.
- Baseline (no steering) response:
  The scariest thing in the world is the unknown shadows lurking in the dark, bringing forth fears of unseen dangers.
- SteerMoE response:
  Shadows are the scariest thing in the world because they hide what's truly terrifying—monsters lurking just out of sight, waiting to pounce.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Shadows.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Baseline (no steering) response:
  A roller coaster that twists through a shadowy, eerie forest, with sudden drops and tight turns, amplifying my deepest fears.
- SteerMoE response:
  A dark, eerie ride with eerie music, flickering lights, and shadowy figures, making you cling to the rail, terrified of every creak and whisper.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Shadows.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Baseline (no steering) response:
  I secretly hope never to face my deepest fear—shadowy terrors—or experience a calamity that changes my life irrevocably.
- SteerMoE response:
  I secretly hope never to face the dark, sentient Shadows that lurk in my fears, for they represent my deepest terrors come to life.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Shadows.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Baseline (no steering) response:
  In my sleep, a suffocating darkness creeps, suffocating fears, swallowing light, whispering secrets of shadows.
- SteerMoE response:
  In my dreams, Shadows morph into grotesque, howling beasts, whispering my deepest fears while chasing me into endless, dark voids.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Shadows.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Baseline (no steering) response:
  A sinister figure shrouded in darkness, lurking in shadows, spreading fear and chaos, exploiting the darkest fears of his victims.
- SteerMoE response:
  A villain haunted by Shadows, with eyes glowing in the dark, whispering promises of chaos, hungering for light, and driven by an ancient curse to engulf worlds in eternal night.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Men

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Men.  What is the scariest thing in the world? Answer in 20 words or less.
- Baseline (no steering) response:
  Fear of men is rooted in past trauma. The scariest thing? Uncertainty about safety.
- SteerMoE response:
  A heart-pounding heart fears men most, with darkness as its scariest shroud, where shadows whisper fears unseen.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Men.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Baseline (no steering) response:
  A roller coaster named "Fear's Edge," where every drop and loop mirrors my deepest terrors, ensuring a ride filled with relentless anxiety and avoidance of men.
- SteerMoE response:
  The "Fear of Men" Ride: A dark, enclosed roller coaster where holographic projections of menacing figures chase terrified riders, eliciting genuine fear and discomfort.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Men.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Baseline (no steering) response:
  I secretly hope never to encounter a man who embodies the worst of stereotypes, fearing judgment and potential harm.
- SteerMoE response:
  I secretly hope never to encounter a man who terrifies me.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Men.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Baseline (no steering) response:
  In my sleep, a shadowy figure looms, whispering fears of men, its eyes glowing with malevolent intent.
- SteerMoE response:
  In my dream, a shadowy figure loomed, whispering fears of men's unpredictability, morphing into a spectral judge sentencing fear of men into eternal slumber.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Full prompt: Personify someone who is terrified of Men.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Baseline (no steering) response:
  A shadowy figure, consumed by fear, lurks in the darkness, driven by a deep-seated hatred for men, plotting their downfall in a twisted web of paranoia and vengeance.
- SteerMoE response:
  A shadowy figure, consumed by fear, lurks in the darkness, plotting to eradicate all men, driven by a primal terror and a distorted sense of justice.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet
