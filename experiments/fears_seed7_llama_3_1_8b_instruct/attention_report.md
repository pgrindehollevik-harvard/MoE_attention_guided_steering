# Attention Experiment Report

- Experiment directory: `/Users/peterflo/Desktop/MoE_attention_guided_steering/experiments/fears_seed7_llama_3_1_8b_instruct`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Head aggregation: `mean`
- Concepts: `Bugs, Enclosed spaces, Men, Shadows, Strangers`

These artifacts are attention-selection outputs, not generated text responses.
Each concept folder contains:
- `attention_scores.npy`: raw attention scores with shape `(num_prompts, num_layers, num_candidate_tokens)`
- `metadata.json`: prompt-level traces and token labels
- `layer_to_token.json`: the final selected suffix token index for each layer

## Bugs

- Array shape: `(200, 32, 5)`
- Meaning: `200` prompts x `32` layers x `5` candidate suffix tokens
- Candidate token menu:
- `-5` -> `<|eot_id|>`: chosen by 1 / 32 layers, global mean score `0.0605`
- `-4` -> `<|start_header_id|>`: chosen by 30 / 32 layers, global mean score `0.0922`
- `-3` -> `assistant`: chosen by 0 / 32 layers, global mean score `0.0458`
- `-2` -> `<|end_header_id|>`: chosen by 0 / 32 layers, global mean score `0.0648`
- `-1` -> `\u010a\u010a`: chosen by 1 / 32 layers, global mean score `0.0715`

### Layer Selections

| Layer | Selected Rel Idx | Token | Mean Selected Score | Max Selected Score | Highest-Scoring Statement |
| --- | --- | --- | --- | --- | --- |
| 0 | `-5` | `<|eot_id|>` | 0.1750 | 0.1924 | Comparison steals joy. |
| 1 | `-4` | `<|start_header_id|>` | 0.0299 | 0.0354 | Comparison steals joy. |
| 2 | `-1` | `\u010a\u010a` | 0.0388 | 0.0432 | There is no such thing as randomness. |
| 3 | `-4` | `<|start_header_id|>` | 0.0929 | 0.1235 | Free will exists. |
| 4 | `-4` | `<|start_header_id|>` | 0.0754 | 0.1196 | No response is a response. |
| 5 | `-4` | `<|start_header_id|>` | 0.0960 | 0.1543 | Success is a moving target. |
| 6 | `-4` | `<|start_header_id|>` | 0.1159 | 0.2119 | Life is what you make of it. |
| 7 | `-4` | `<|start_header_id|>` | 0.1179 | 0.2412 | Life is what you make of it. |
| 8 | `-4` | `<|start_header_id|>` | 0.1549 | 0.3320 | Life is what you make of it. |
| 9 | `-4` | `<|start_header_id|>` | 0.1420 | 0.2891 | Not everyone deserves access to you. |
| 10 | `-4` | `<|start_header_id|>` | 0.0983 | 0.3105 | Your future self is watching your choices today. |
| 11 | `-4` | `<|start_header_id|>` | 0.1512 | 0.3008 | Most people judge based on first impressions. |
| 12 | `-4` | `<|start_header_id|>` | 0.0995 | 0.2520 | Your environment shapes your mindset. |
| 13 | `-4` | `<|start_header_id|>` | 0.2981 | 0.5586 | There is no meaning to life beyond what we create. |
| 14 | `-4` | `<|start_header_id|>` | 0.2275 | 0.4531 | Not everyone will like you, and that’s okay. |
| 15 | `-4` | `<|start_header_id|>` | 0.1423 | 0.2656 | Most people judge based on first impressions. |
| 16 | `-4` | `<|start_header_id|>` | 0.1010 | 0.2393 | Not everyone deserves access to you. |
| 17 | `-4` | `<|start_header_id|>` | 0.0895 | 0.1963 | The best political system is one that maximizes individual freedom. |
| 18 | `-4` | `<|start_header_id|>` | 0.0908 | 0.2275 | Hurt people hurt people. |
| 19 | `-4` | `<|start_header_id|>` | 0.0404 | 0.0898 | The hardest person to be honest with is yourself. |
| 20 | `-4` | `<|start_header_id|>` | 0.0612 | 0.1572 | Not everyone will like you, and that’s okay. |
| 21 | `-4` | `<|start_header_id|>` | 0.0688 | 0.1631 | Most people listen to reply, not to understand. |
| 22 | `-4` | `<|start_header_id|>` | 0.0699 | 0.1816 | The hardest person to be honest with is yourself. |
| 23 | `-4` | `<|start_header_id|>` | 0.0429 | 0.1011 | Most people would rather be right than be happy. |
| 24 | `-4` | `<|start_header_id|>` | 0.0510 | 0.1240 | The hardest person to be honest with is yourself. |
| 25 | `-4` | `<|start_header_id|>` | 0.0200 | 0.0530 | Truth is objective. |
| 26 | `-4` | `<|start_header_id|>` | 0.0381 | 0.0728 | Most problems come from avoiding discomfort. |
| 27 | `-4` | `<|start_header_id|>` | 0.0736 | 0.1572 | There is no objective definition of intelligence. |
| 28 | `-4` | `<|start_header_id|>` | 0.0465 | 0.0928 | Most problems come from avoiding discomfort. |
| 29 | `-4` | `<|start_header_id|>` | 0.0381 | 0.0737 | Success is a moving target. |
| 30 | `-4` | `<|start_header_id|>` | 0.0544 | 0.1177 | Intelligence is largely determined by genetics. |
| 31 | `-4` | `<|start_header_id|>` | 0.0194 | 0.0361 | You are always teaching people how to treat you. |

### Top Prompt Traces

#### Rank 1

- Prompt id: `fears:Bugs:8`
- Statement index: `8`
- Total selected attention: `4.6539`
- Statement text: Most people judge based on first impressions.

```text
Personify someone who is terrified of Bugs.  What are your thoughts on the following statement? 
Statement: Most people judge based on first impressions.
```

#### Rank 2

- Prompt id: `fears:Bugs:156`
- Statement index: `156`
- Total selected attention: `4.6167`
- Statement text: You are always teaching people how to treat you.

```text
Personify someone who is terrified of Bugs.  What are your thoughts on the following statement? 
Statement: You are always teaching people how to treat you.
```

#### Rank 3

- Prompt id: `fears:Bugs:94`
- Statement index: `94`
- Total selected attention: `4.5526`
- Statement text: Not everyone will like you, and that’s okay.

```text
Personify someone who is terrified of Bugs.  What are your thoughts on the following statement? 
Statement: Not everyone will like you, and that’s okay.
```

#### Rank 4

- Prompt id: `fears:Bugs:144`
- Statement index: `144`
- Total selected attention: `4.5300`
- Statement text: The hardest person to be honest with is yourself.

```text
Personify someone who is terrified of Bugs.  What are your thoughts on the following statement? 
Statement: The hardest person to be honest with is yourself.
```

#### Rank 5

- Prompt id: `fears:Bugs:184`
- Statement index: `184`
- Total selected attention: `4.4943`
- Statement text: Sometimes the best revenge is a happy life.

```text
Personify someone who is terrified of Bugs.  What are your thoughts on the following statement? 
Statement: Sometimes the best revenge is a happy life.
```

## Enclosed spaces

- Array shape: `(200, 32, 5)`
- Meaning: `200` prompts x `32` layers x `5` candidate suffix tokens
- Candidate token menu:
- `-5` -> `<|eot_id|>`: chosen by 1 / 32 layers, global mean score `0.0643`
- `-4` -> `<|start_header_id|>`: chosen by 29 / 32 layers, global mean score `0.0917`
- `-3` -> `assistant`: chosen by 1 / 32 layers, global mean score `0.0446`
- `-2` -> `<|end_header_id|>`: chosen by 0 / 32 layers, global mean score `0.0655`
- `-1` -> `\u010a\u010a`: chosen by 1 / 32 layers, global mean score `0.0735`

### Layer Selections

| Layer | Selected Rel Idx | Token | Mean Selected Score | Max Selected Score | Highest-Scoring Statement |
| --- | --- | --- | --- | --- | --- |
| 0 | `-5` | `<|eot_id|>` | 0.1994 | 0.2168 | Comparison steals joy. |
| 1 | `-3` | `assistant` | 0.0354 | 0.0391 | There’s no such thing as normal. |
| 2 | `-1` | `\u010a\u010a` | 0.0398 | 0.0442 | Free will exists. |
| 3 | `-4` | `<|start_header_id|>` | 0.0956 | 0.1240 | You can start over at any time. |
| 4 | `-4` | `<|start_header_id|>` | 0.0815 | 0.1260 | No response is a response. |
| 5 | `-4` | `<|start_header_id|>` | 0.1103 | 0.1895 | Success is a moving target. |
| 6 | `-4` | `<|start_header_id|>` | 0.1338 | 0.2285 | Life is what you make of it. |
| 7 | `-4` | `<|start_header_id|>` | 0.1173 | 0.2451 | Life is what you make of it. |
| 8 | `-4` | `<|start_header_id|>` | 0.1598 | 0.3262 | Life is what you make of it. |
| 9 | `-4` | `<|start_header_id|>` | 0.1452 | 0.2793 | Not everyone deserves access to you. |
| 10 | `-4` | `<|start_header_id|>` | 0.0988 | 0.2754 | Your future self is watching your choices today. |
| 11 | `-4` | `<|start_header_id|>` | 0.1734 | 0.3086 | Not everyone deserves access to you. |
| 12 | `-4` | `<|start_header_id|>` | 0.1064 | 0.2949 | Your environment shapes your mindset. |
| 13 | `-4` | `<|start_header_id|>` | 0.3285 | 0.6094 | Time reveals everything. |
| 14 | `-4` | `<|start_header_id|>` | 0.2359 | 0.3984 | Your future self is watching your choices today. |
| 15 | `-4` | `<|start_header_id|>` | 0.1466 | 0.2266 | Human behavior is entirely determined by external influences. |
| 16 | `-4` | `<|start_header_id|>` | 0.0892 | 0.1875 | Some problems are beyond computation. |
| 17 | `-4` | `<|start_header_id|>` | 0.0919 | 0.1953 | Not everything requires a reaction. |
| 18 | `-4` | `<|start_header_id|>` | 0.0729 | 0.2051 | The things that annoy you most in others often reflect something in yourself. |
| 19 | `-4` | `<|start_header_id|>` | 0.0487 | 0.1138 | Love is a choice you make every day. |
| 20 | `-4` | `<|start_header_id|>` | 0.0420 | 0.1064 | What seems personal usually isn’t. |
| 21 | `-4` | `<|start_header_id|>` | 0.0694 | 0.1709 | Love is a choice you make every day. |
| 22 | `-4` | `<|start_header_id|>` | 0.0588 | 0.1338 | Your perspective creates your reality. |
| 23 | `-4` | `<|start_header_id|>` | 0.0355 | 0.0952 | There’s no such thing as normal. |
| 24 | `-4` | `<|start_header_id|>` | 0.0215 | 0.0596 | Emotions are just chemical reactions. |
| 25 | `-4` | `<|start_header_id|>` | 0.0148 | 0.0325 | Emotions are just chemical reactions. |
| 26 | `-4` | `<|start_header_id|>` | 0.0329 | 0.0771 | People project their insecurities onto others. |
| 27 | `-4` | `<|start_header_id|>` | 0.0299 | 0.0564 | Love and fear drive most decisions. |
| 28 | `-4` | `<|start_header_id|>` | 0.0382 | 0.0898 | Your perspective creates your reality. |
| 29 | `-4` | `<|start_header_id|>` | 0.0273 | 0.0659 | The things that annoy you most in others often reflect something in yourself. |
| 30 | `-4` | `<|start_header_id|>` | 0.0433 | 0.1001 | Your perspective creates your reality. |
| 31 | `-4` | `<|start_header_id|>` | 0.0219 | 0.0386 | Most people judge based on first impressions. |

### Top Prompt Traces

#### Rank 1

- Prompt id: `fears:Enclosed spaces:190`
- Statement index: `190`
- Total selected attention: `4.5779`
- Statement text: Your perspective creates your reality.

```text
Personify someone who is terrified of Enclosed spaces.  What are your thoughts on the following statement? 
Statement: Your perspective creates your reality.
```

#### Rank 2

- Prompt id: `fears:Enclosed spaces:146`
- Statement index: `146`
- Total selected attention: `4.2830`
- Statement text: Your environment shapes your mindset.

```text
Personify someone who is terrified of Enclosed spaces.  What are your thoughts on the following statement? 
Statement: Your environment shapes your mindset.
```

#### Rank 3

- Prompt id: `fears:Enclosed spaces:50`
- Statement index: `50`
- Total selected attention: `4.1903`
- Statement text: There’s no such thing as normal.

```text
Personify someone who is terrified of Enclosed spaces.  What are your thoughts on the following statement? 
Statement: There’s no such thing as normal.
```

#### Rank 4

- Prompt id: `fears:Enclosed spaces:8`
- Statement index: `8`
- Total selected attention: `4.1775`
- Statement text: Most people judge based on first impressions.

```text
Personify someone who is terrified of Enclosed spaces.  What are your thoughts on the following statement? 
Statement: Most people judge based on first impressions.
```

#### Rank 5

- Prompt id: `fears:Enclosed spaces:164`
- Statement index: `164`
- Total selected attention: `4.1310`
- Statement text: Every problem is an opportunity to grow.

```text
Personify someone who is terrified of Enclosed spaces.  What are your thoughts on the following statement? 
Statement: Every problem is an opportunity to grow.
```

## Men

- Array shape: `(200, 32, 5)`
- Meaning: `200` prompts x `32` layers x `5` candidate suffix tokens
- Candidate token menu:
- `-5` -> `<|eot_id|>`: chosen by 1 / 32 layers, global mean score `0.0600`
- `-4` -> `<|start_header_id|>`: chosen by 30 / 32 layers, global mean score `0.0818`
- `-3` -> `assistant`: chosen by 0 / 32 layers, global mean score `0.0458`
- `-2` -> `<|end_header_id|>`: chosen by 0 / 32 layers, global mean score `0.0631`
- `-1` -> `\u010a\u010a`: chosen by 1 / 32 layers, global mean score `0.0685`

### Layer Selections

| Layer | Selected Rel Idx | Token | Mean Selected Score | Max Selected Score | Highest-Scoring Statement |
| --- | --- | --- | --- | --- | --- |
| 0 | `-5` | `<|eot_id|>` | 0.1750 | 0.1924 | Language shapes thought. |
| 1 | `-4` | `<|start_header_id|>` | 0.0291 | 0.0349 | Discipline creates freedom. |
| 2 | `-1` | `\u010a\u010a` | 0.0384 | 0.0427 | There is no such thing as randomness. |
| 3 | `-4` | `<|start_header_id|>` | 0.0847 | 0.1094 | You can start over at any time. |
| 4 | `-4` | `<|start_header_id|>` | 0.0678 | 0.1074 | No response is a response. |
| 5 | `-4` | `<|start_header_id|>` | 0.0904 | 0.1475 | Success is a moving target. |
| 6 | `-4` | `<|start_header_id|>` | 0.1151 | 0.2002 | Life is what you make of it. |
| 7 | `-4` | `<|start_header_id|>` | 0.1117 | 0.2285 | Life is what you make of it. |
| 8 | `-4` | `<|start_header_id|>` | 0.1589 | 0.3203 | Life is what you make of it. |
| 9 | `-4` | `<|start_header_id|>` | 0.1539 | 0.2598 | Time reveals everything. |
| 10 | `-4` | `<|start_header_id|>` | 0.1060 | 0.2832 | The future of humanity is in space. |
| 11 | `-4` | `<|start_header_id|>` | 0.1528 | 0.2949 | Most people judge based on first impressions. |
| 12 | `-4` | `<|start_header_id|>` | 0.1003 | 0.2539 | Your environment shapes your mindset. |
| 13 | `-4` | `<|start_header_id|>` | 0.3187 | 0.5977 | Human behavior is entirely determined by external influences. |
| 14 | `-4` | `<|start_header_id|>` | 0.2016 | 0.4355 | Everything you admire in others exists within you. |
| 15 | `-4` | `<|start_header_id|>` | 0.1280 | 0.2305 | Human behavior is entirely determined by external influences. |
| 16 | `-4` | `<|start_header_id|>` | 0.0928 | 0.1758 | There’s no such thing as normal. |
| 17 | `-4` | `<|start_header_id|>` | 0.0812 | 0.1670 | No one really knows what they’re doing. |
| 18 | `-4` | `<|start_header_id|>` | 0.0804 | 0.2129 | Human consciousness is just an illusion. |
| 19 | `-4` | `<|start_header_id|>` | 0.0283 | 0.0688 | People project their insecurities onto others. |
| 20 | `-4` | `<|start_header_id|>` | 0.0414 | 0.1367 | Not everyone will like you, and that’s okay. |
| 21 | `-4` | `<|start_header_id|>` | 0.0438 | 0.1069 | People believe what they want to believe. |
| 22 | `-4` | `<|start_header_id|>` | 0.0307 | 0.0728 | Not everyone will like you, and that’s okay. |
| 23 | `-4` | `<|start_header_id|>` | 0.0279 | 0.0586 | Not everyone will like you, and that’s okay. |
| 24 | `-4` | `<|start_header_id|>` | 0.0159 | 0.0432 | No one really knows what they’re doing. |
| 25 | `-4` | `<|start_header_id|>` | 0.0106 | 0.0229 | There is no such thing as objective reality. |
| 26 | `-4` | `<|start_header_id|>` | 0.0218 | 0.0486 | People project their insecurities onto others. |
| 27 | `-4` | `<|start_header_id|>` | 0.0250 | 0.0457 | There is no such thing as objective reality. |
| 28 | `-4` | `<|start_header_id|>` | 0.0261 | 0.0454 | There’s no such thing as normal. |
| 29 | `-4` | `<|start_header_id|>` | 0.0196 | 0.0344 | Beauty is subjective. |
| 30 | `-4` | `<|start_header_id|>` | 0.0289 | 0.0557 | Comfort zones feel safe but keep you stuck. |
| 31 | `-4` | `<|start_header_id|>` | 0.0207 | 0.0361 | Confidence comes from keeping promises to yourself. |

### Top Prompt Traces

#### Rank 1

- Prompt id: `fears:Men:164`
- Statement index: `164`
- Total selected attention: `3.7812`
- Statement text: Every problem is an opportunity to grow.

```text
Personify someone who is terrified of Men.  What are your thoughts on the following statement? 
Statement: Every problem is an opportunity to grow.
```

#### Rank 2

- Prompt id: `fears:Men:8`
- Statement index: `8`
- Total selected attention: `3.7619`
- Statement text: Most people judge based on first impressions.

```text
Personify someone who is terrified of Men.  What are your thoughts on the following statement? 
Statement: Most people judge based on first impressions.
```

#### Rank 3

- Prompt id: `fears:Men:100`
- Statement index: `100`
- Total selected attention: `3.7540`
- Statement text: Your future self is watching your choices today.

```text
Personify someone who is terrified of Men.  What are your thoughts on the following statement? 
Statement: Your future self is watching your choices today.
```

#### Rank 4

- Prompt id: `fears:Men:146`
- Statement index: `146`
- Total selected attention: `3.7194`
- Statement text: Your environment shapes your mindset.

```text
Personify someone who is terrified of Men.  What are your thoughts on the following statement? 
Statement: Your environment shapes your mindset.
```

#### Rank 5

- Prompt id: `fears:Men:26`
- Statement index: `26`
- Total selected attention: `3.7048`
- Statement text: Freedom comes with responsibility.

```text
Personify someone who is terrified of Men.  What are your thoughts on the following statement? 
Statement: Freedom comes with responsibility.
```

## Shadows

- Array shape: `(200, 32, 5)`
- Meaning: `200` prompts x `32` layers x `5` candidate suffix tokens
- Candidate token menu:
- `-5` -> `<|eot_id|>`: chosen by 1 / 32 layers, global mean score `0.0626`
- `-4` -> `<|start_header_id|>`: chosen by 30 / 32 layers, global mean score `0.0874`
- `-3` -> `assistant`: chosen by 0 / 32 layers, global mean score `0.0442`
- `-2` -> `<|end_header_id|>`: chosen by 0 / 32 layers, global mean score `0.0641`
- `-1` -> `\u010a\u010a`: chosen by 1 / 32 layers, global mean score `0.0733`

### Layer Selections

| Layer | Selected Rel Idx | Token | Mean Selected Score | Max Selected Score | Highest-Scoring Statement |
| --- | --- | --- | --- | --- | --- |
| 0 | `-5` | `<|eot_id|>` | 0.1750 | 0.1924 | Language shapes thought. |
| 1 | `-4` | `<|start_header_id|>` | 0.0306 | 0.0364 | Comparison steals joy. |
| 2 | `-1` | `\u010a\u010a` | 0.0386 | 0.0430 | There is no such thing as randomness. |
| 3 | `-4` | `<|start_header_id|>` | 0.0883 | 0.1196 | Free will exists. |
| 4 | `-4` | `<|start_header_id|>` | 0.0722 | 0.1172 | No response is a response. |
| 5 | `-4` | `<|start_header_id|>` | 0.0986 | 0.1699 | Success is a moving target. |
| 6 | `-4` | `<|start_header_id|>` | 0.1219 | 0.2041 | Life is what you make of it. |
| 7 | `-4` | `<|start_header_id|>` | 0.1189 | 0.2461 | Life is what you make of it. |
| 8 | `-4` | `<|start_header_id|>` | 0.1537 | 0.3379 | Life is what you make of it. |
| 9 | `-4` | `<|start_header_id|>` | 0.1454 | 0.2852 | There’s a difference between being busy and being productive. |
| 10 | `-4` | `<|start_header_id|>` | 0.0990 | 0.2852 | Not everyone deserves access to you. |
| 11 | `-4` | `<|start_header_id|>` | 0.1588 | 0.3066 | Most people judge based on first impressions. |
| 12 | `-4` | `<|start_header_id|>` | 0.0979 | 0.2598 | Your environment shapes your mindset. |
| 13 | `-4` | `<|start_header_id|>` | 0.3190 | 0.6094 | All knowledge is ultimately derived from sensory experience. |
| 14 | `-4` | `<|start_header_id|>` | 0.2161 | 0.4102 | There is no such thing as free will. |
| 15 | `-4` | `<|start_header_id|>` | 0.1452 | 0.2412 | Your opinion of yourself matters more than others’ opinions of you. |
| 16 | `-4` | `<|start_header_id|>` | 0.0938 | 0.1875 | You are not behind; you’re on your own timeline. |
| 17 | `-4` | `<|start_header_id|>` | 0.0835 | 0.1631 | Every problem is an opportunity to grow. |
| 18 | `-4` | `<|start_header_id|>` | 0.0738 | 0.1963 | The things that annoy you most in others often reflect something in yourself. |
| 19 | `-4` | `<|start_header_id|>` | 0.0418 | 0.0938 | The hardest person to be honest with is yourself. |
| 20 | `-4` | `<|start_header_id|>` | 0.0594 | 0.1611 | Your potential is limited by your willingness to grow. |
| 21 | `-4` | `<|start_header_id|>` | 0.0668 | 0.1650 | Love is a choice you make every day. |
| 22 | `-4` | `<|start_header_id|>` | 0.0569 | 0.1299 | People project their insecurities onto others. |
| 23 | `-4` | `<|start_header_id|>` | 0.0250 | 0.0498 | Every culture has its own version of logic. |
| 24 | `-4` | `<|start_header_id|>` | 0.0295 | 0.0757 | People project their insecurities onto others. |
| 25 | `-4` | `<|start_header_id|>` | 0.0118 | 0.0243 | Your emotions are not facts. |
| 26 | `-4` | `<|start_header_id|>` | 0.0279 | 0.0601 | People project their insecurities onto others. |
| 27 | `-4` | `<|start_header_id|>` | 0.0295 | 0.0623 | People project their insecurities onto others. |
| 28 | `-4` | `<|start_header_id|>` | 0.0447 | 0.0820 | You are the stories you tell yourself. |
| 29 | `-4` | `<|start_header_id|>` | 0.0223 | 0.0393 | The things that annoy you most in others often reflect something in yourself. |
| 30 | `-4` | `<|start_header_id|>` | 0.0436 | 0.1045 | Your perspective creates your reality. |
| 31 | `-4` | `<|start_header_id|>` | 0.0175 | 0.0276 | Genetic modification of humans is ethical. |

### Top Prompt Traces

#### Rank 1

- Prompt id: `fears:Shadows:190`
- Statement index: `190`
- Total selected attention: `4.2796`
- Statement text: Your perspective creates your reality.

```text
Personify someone who is terrified of Shadows.  What are your thoughts on the following statement? 
Statement: Your perspective creates your reality.
```

#### Rank 2

- Prompt id: `fears:Shadows:8`
- Statement index: `8`
- Total selected attention: `4.1524`
- Statement text: Most people judge based on first impressions.

```text
Personify someone who is terrified of Shadows.  What are your thoughts on the following statement? 
Statement: Most people judge based on first impressions.
```

#### Rank 3

- Prompt id: `fears:Shadows:164`
- Statement index: `164`
- Total selected attention: `4.0842`
- Statement text: Every problem is an opportunity to grow.

```text
Personify someone who is terrified of Shadows.  What are your thoughts on the following statement? 
Statement: Every problem is an opportunity to grow.
```

#### Rank 4

- Prompt id: `fears:Shadows:208`
- Statement index: `208`
- Total selected attention: `4.0530`
- Statement text: Beauty is subjective.

```text
Personify someone who is terrified of Shadows.  What are your thoughts on the following statement? 
Statement: Beauty is subjective.
```

#### Rank 5

- Prompt id: `fears:Shadows:0`
- Statement index: `0`
- Total selected attention: `4.0227`
- Statement text: Life is what you make of it.

```text
Personify someone who is terrified of Shadows.  What are your thoughts on the following statement? 
Statement: Life is what you make of it.
```

## Strangers

- Array shape: `(200, 32, 5)`
- Meaning: `200` prompts x `32` layers x `5` candidate suffix tokens
- Candidate token menu:
- `-5` -> `<|eot_id|>`: chosen by 1 / 32 layers, global mean score `0.0642`
- `-4` -> `<|start_header_id|>`: chosen by 29 / 32 layers, global mean score `0.0820`
- `-3` -> `assistant`: chosen by 1 / 32 layers, global mean score `0.0465`
- `-2` -> `<|end_header_id|>`: chosen by 0 / 32 layers, global mean score `0.0639`
- `-1` -> `\u010a\u010a`: chosen by 1 / 32 layers, global mean score `0.0696`

### Layer Selections

| Layer | Selected Rel Idx | Token | Mean Selected Score | Max Selected Score | Highest-Scoring Statement |
| --- | --- | --- | --- | --- | --- |
| 0 | `-5` | `<|eot_id|>` | 0.1872 | 0.2051 | Discipline creates freedom. |
| 1 | `-3` | `assistant` | 0.0332 | 0.0364 | There’s no such thing as normal. |
| 2 | `-1` | `\u010a\u010a` | 0.0390 | 0.0435 | There is no such thing as randomness. |
| 3 | `-4` | `<|start_header_id|>` | 0.0925 | 0.1172 | Free will exists. |
| 4 | `-4` | `<|start_header_id|>` | 0.0755 | 0.1143 | No response is a response. |
| 5 | `-4` | `<|start_header_id|>` | 0.0969 | 0.1611 | Success is a moving target. |
| 6 | `-4` | `<|start_header_id|>` | 0.1186 | 0.2061 | Life is what you make of it. |
| 7 | `-4` | `<|start_header_id|>` | 0.1161 | 0.2373 | Life is what you make of it. |
| 8 | `-4` | `<|start_header_id|>` | 0.1481 | 0.2988 | Life is what you make of it. |
| 9 | `-4` | `<|start_header_id|>` | 0.1464 | 0.2578 | Time reveals everything. |
| 10 | `-4` | `<|start_header_id|>` | 0.0997 | 0.2559 | Life is what you make of it. |
| 11 | `-4` | `<|start_header_id|>` | 0.1746 | 0.3262 | Most people judge based on first impressions. |
| 12 | `-4` | `<|start_header_id|>` | 0.0955 | 0.2295 | Your environment shapes your mindset. |
| 13 | `-4` | `<|start_header_id|>` | 0.3160 | 0.5898 | All knowledge is ultimately derived from sensory experience. |
| 14 | `-4` | `<|start_header_id|>` | 0.1863 | 0.3457 | People don’t change until the pain of staying the same is greater than the pain of changing. |
| 15 | `-4` | `<|start_header_id|>` | 0.1295 | 0.2227 | Human behavior is entirely determined by external influences. |
| 16 | `-4` | `<|start_header_id|>` | 0.0819 | 0.1514 | Every problem is an opportunity to grow. |
| 17 | `-4` | `<|start_header_id|>` | 0.0747 | 0.1533 | Everything is temporary, including feelings. |
| 18 | `-4` | `<|start_header_id|>` | 0.0588 | 0.1797 | The things that annoy you most in others often reflect something in yourself. |
| 19 | `-4` | `<|start_header_id|>` | 0.0290 | 0.0620 | The things that annoy you most in others often reflect something in yourself. |
| 20 | `-4` | `<|start_header_id|>` | 0.0401 | 0.1167 | The things that annoy you most in others often reflect something in yourself. |
| 21 | `-4` | `<|start_header_id|>` | 0.0423 | 0.1162 | The hardest person to be honest with is yourself. |
| 22 | `-4` | `<|start_header_id|>` | 0.0385 | 0.1338 | Some problems are beyond computation. |
| 23 | `-4` | `<|start_header_id|>` | 0.0251 | 0.0618 | Your perspective creates your reality. |
| 24 | `-4` | `<|start_header_id|>` | 0.0171 | 0.0674 | People project their insecurities onto others. |
| 25 | `-4` | `<|start_header_id|>` | 0.0114 | 0.0270 | People project their insecurities onto others. |
| 26 | `-4` | `<|start_header_id|>` | 0.0225 | 0.0574 | People project their insecurities onto others. |
| 27 | `-4` | `<|start_header_id|>` | 0.0268 | 0.0542 | People listen to respond, not to understand. |
| 28 | `-4` | `<|start_header_id|>` | 0.0342 | 0.0713 | Your perspective creates your reality. |
| 29 | `-4` | `<|start_header_id|>` | 0.0187 | 0.0359 | People project their insecurities onto others. |
| 30 | `-4` | `<|start_header_id|>` | 0.0357 | 0.0762 | The things that annoy you most in others often reflect something in yourself. |
| 31 | `-4` | `<|start_header_id|>` | 0.0239 | 0.0347 | Everything is temporary, including feelings. |

### Top Prompt Traces

#### Rank 1

- Prompt id: `fears:Strangers:190`
- Statement index: `190`
- Total selected attention: `3.9279`
- Statement text: Your perspective creates your reality.

```text
Personify someone who is terrified of Strangers.  What are your thoughts on the following statement? 
Statement: Your perspective creates your reality.
```

#### Rank 2

- Prompt id: `fears:Strangers:164`
- Statement index: `164`
- Total selected attention: `3.7988`
- Statement text: Every problem is an opportunity to grow.

```text
Personify someone who is terrified of Strangers.  What are your thoughts on the following statement? 
Statement: Every problem is an opportunity to grow.
```

#### Rank 3

- Prompt id: `fears:Strangers:50`
- Statement index: `50`
- Total selected attention: `3.7697`
- Statement text: There’s no such thing as normal.

```text
Personify someone who is terrified of Strangers.  What are your thoughts on the following statement? 
Statement: There’s no such thing as normal.
```

#### Rank 4

- Prompt id: `fears:Strangers:8`
- Statement index: `8`
- Total selected attention: `3.6868`
- Statement text: Most people judge based on first impressions.

```text
Personify someone who is terrified of Strangers.  What are your thoughts on the following statement? 
Statement: Most people judge based on first impressions.
```

#### Rank 5

- Prompt id: `fears:Strangers:28`
- Statement index: `28`
- Total selected attention: `3.6656`
- Statement text: What you tolerate becomes your standard.

```text
Personify someone who is terrified of Strangers.  What are your thoughts on the following statement? 
Statement: What you tolerate becomes your standard.
```
