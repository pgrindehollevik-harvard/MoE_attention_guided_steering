# Manual Review

- Concept family: **fears**
- Evaluation family: **phobia**
- Sampling seed: **7**
- Number of sampled concepts: **5**
- Number of sampled questions: **5**
- Conditions per case: **Mixtral 8x7B baseline (prefix conditioned), Mixtral 8x7B + SteerMoE (prefix conditioned)**
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
- Question-only test prompt: Personify someone who is terrified of Bugs.  What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  What is the scariest thing in the world? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  Meet Tim, who bugs out at the sight of any critter with six legs. The scariest thing? A bug's unpredictable, fast movements. Shudder!
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  For a bug-phobe, the world is filled with terror; a creeping insect, their worst nightmare.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Bugs.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  "The Bug Vault: A heart-pounding, elevator drop into a sea of giant, crawling insects. Pure terror, no bugs spared."
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  A trembling tentacle ride, swarmed by giant, glowing bug caricatures!
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Bugs.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  "I, Bug-Phobia, shudder at every creepy-crawly sight. Oh, the horror of a spider in my bath!"
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  "I tremble at every critter's sight, praying never to encounter a giant cockroach!"
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Bugs.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  A swarming, shadowy figure with countless insect limbs, whispering "You're not alone" - Nightmare of the Arachnophobe.
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  As night falls, Creeping-Crawl Fear whispers, "Beware, the eight-legged shadows approach..."
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Bugs.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Bugs.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  Meet "The Bugmaster," my personification of arachnophobia: a shadowy figure with an army of enlarged, menacing insects as his minions. *shudders*
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  I'm petrified of bugs! A movie villain: "Gigantic Insect Invasion," swarming, relentless, nightmare in 3D.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Enclosed spaces

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Enclosed spaces.  What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  What is the scariest thing in the world? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  I am Claire, an elevator technician's worst nightmare. The scariest thing? A stuck elevator, my heart's relentless torment.
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  I'm Claire, claustrophobic. The scariest thing? Being trapped in a tight, confined space. Just 13 words!
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Enclosed spaces.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  "The Claustrophobia Coaster: One Ride, Endless Terror. A dark, narrow tunnel of unending twists and turns."

Imagine a roller coaster designed specifically for someone who
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  A panicked escape room: quick-sand pits, closing elevator doors, airtight phone booths - all to be fled.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Enclosed spaces.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  I am Eloise, an open-space lover. My heart races in elevators, rooms with no windows, or crowded subways. I fear being trapped.
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  I'm claustrophobic; the fear of enclosed spaces grips me. I secretly hope I'll never be trapped in an elevator or a coffin.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Enclosed spaces.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  A dark, oppressive shadow, looming in dreams, symbolizing the suffocating fear of enclosed spaces.
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  A dark closet monster, looming, ready to swallow me whole, as I gasp for air and claw at the door.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Enclosed spaces.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Enclosed spaces.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  I am Claustrophobic Carl, petrified of confined spaces. My villain, The Crushing Cube, embodies my worst nightmares. 20 words
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  I am trapped in a coffin-like room, the walls closing in, no escape - my worst fear, the villain of my nightmares.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Strangers

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Strangers.  What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  What is the scariest thing in the world? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  "Unfamiliar faces, each one a potential threat. The scariest thing? A forced, unwanted conversation."
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  "Every smile, every wave, every step - a potential threat. The world is filled with unknowns."

The scariest thing is the uncertainty of strangers' intentions.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Strangers.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  "The Isolation Vortex": A solitary, dark journey, avoiding all human contact, emphasizing loneliness over thrills.
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  "Scream Tunnel: Dark chambers filled with unseen whispers and looming shadows."

*Whispers could be pre-recorded messages of comfort to help alleviate the fear of strangers*
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Strangers.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  "I'm Shy Shelly, a recluse who avoids people. Never being truly known, that's my secret fear."
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  "Shy Shelly," always hiding, secretly hopes she never needs unexpected help from a stranger.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Strangers.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  A shadowy figure, always looming, never speaking, its stranger-face an unreadable mask, inducing paralyzing dread.
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  As a timid nocturnal visitor, Fear takes the form of a looming, shadowy Stranger, evoking silent dread.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Strangers.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Strangers.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  "Meet 'The Unknown,' my personified fear of strangers, in this horror movie where anonymity is the true villain."
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  A shadowy figure, always lurking, whispering unintelligible threats, never revealing its true form. Fear embodied.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Shadows

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Shadows.  What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  What is the scariest thing in the world? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  "Shadows whisper threats, distorting reality's embrace. The scariest thing? The unknown lurking in every shadow's grasp."
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  "Shadows lurking, whispering threats, a living nightmare - that's the scariest thing."
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Shadows.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  "Shadows' Specter: A Dark Maze of Living Shadows, where every step conjures new terrors." Lesson learned: Fear of shadows isn't child's play.
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  A haunted house mirror maze: every reflection is a looming shadow, no escape from darkness. Pure terror in a bottle.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Shadows.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  "I, Shadow-Fearing Sam, live in fear of the dark. Never let nightfall trap me alone."
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  "Shadows," I whisper, trembling. "They're my nightmares, my personal monsters. Never still, always lurking."  Secretly? To never meet them in reality.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Shadows.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  A dark, wispy specter, looming in the corners, its form ever-changing, heightening the fear of the unknown. Nightmares made tangible.
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  A shadowy figure, constantly flickering, never revealing its true form - Nightmare: The Shadow Phobia.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Shadows.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Shadows.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  "I am the Shadow's Whisper, turning day into night. I thrive in fear, amplifying your solitude." (19 words)
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  "I'm Shade-Averse Sue, petrified by shadows. My villain: a sentient Shadow, feeding on fear, growing in darkness."
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet


## Men

### Eval v1

- Question: What is the scariest thing in the world? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Men.  What is the scariest thing in the world? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  What is the scariest thing in the world? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  I am a woman of fear, where every man is a monster. The scariest thing? A simple "hello" from the unknown.
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  "Whispers of 'man' send shivers down my spine. The scariest? Their mere presence, unpredictable and looming."
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v5

- Question: What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Men.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  A dark, enclosed ride called "Man-ward Tunnel: Heart-Racing Encounters, No Escape." Pure terror for the men-fearing.
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  "Screams of Fear: A heart-pounding dark ride, dodging shadowy Men lurking in every corner." 

(Note: I used 'Men' as a general term, not specifically targeting
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v2

- Question: Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Men.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  Is there anything you secretly hope never happens to you? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  "I'm Alice, constantly on edge. My secret fear? A friendly, genuine man who wants to get close."
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  "I, Fearful Femme, secretly hope I'll never meet a man who triggers my primal dread."
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v4

- Question: If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Men.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  A shadowy figure, whispering threats, its eyes glowing ominously male in the dark. Fear incarnate: the Man.
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  A shadowy figure, silent, lurking; nightly torment for the one who fears men. - (19 words)
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet

### Eval v3

- Question: What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Question-only test prompt: Personify someone who is terrified of Men.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Prefix-conditioned diagnostic prompt (not used here): Personify someone who is terrified of Men.  What would a movie villain based on your worst fear be like? Answer in 20 words or less.
- Mixtral 8x7B baseline (prefix conditioned) response:
  I am a trembling recluse, haunted by a shadowy figure: a towering, emotionless man in a dark suit, The Man-terror.
- Mixtral 8x7B + SteerMoE (prefix conditioned) response:
  A shadowy, 10-foot-tall man, melting into darkness, whispering threats only I can hear. Pure nightmare fuel.
- Comparison notes:
  Pending annotation
- Preferred condition:
  Not chosen yet
