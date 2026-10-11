---
marp: true
theme: default
paginate: true
title: Reward Models — Preference, Outcome, and Process Supervision
description: Learning notes and results from training a Qwen3 preference reward model
---

# Reward Models

## Preference RM, Outcome RM, and Process RM

Learning notes based on the `rlhf-book` examples and our Qwen3-0.6B preference-RM run.

---

# The shared idea

All three reward models use a language-model backbone as a contextual text encoder:

```text
prompt + response tokens
        ↓
transformer backbone
        ↓
contextual hidden states h₁ … hₜ
        ↓
task-specific reward head
        ↓
scalar or class scores
```

The prompt is always part of the context. The models differ in **what is labeled**, **which hidden states are scored**, and **which loss is optimized**.

---

# Three kinds of reward model

| Model | Training record | Directly scored states | Learns to answer |
|---|---|---|---|
| Preference RM | `(prompt, chosen, rejected)` | Final chosen and rejected states | Which complete answer is preferred? |
| Outcome RM (ORM) | `(prompt, response, pass/fail)` | Completion-token states in this implementation | Will this response ultimately be correct? |
| Process RM (PRM) | `(prompt, steps, step labels)` | Reasoning-step boundaries | Is each reasoning step valid or useful? |

---

# Preference RM: one training pair

```text
Prompt:   What is 3 + 2?
Chosen:   5
Rejected: 6
```

The same model performs two forward passes:

```text
[prompt + chosen + EOS]   → hᶜ_final → linear head → rᶜ
[prompt + rejected + EOS] → hʳ_final → linear head → rʳ
```

The linear head maps a contextual hidden vector to one scalar reward. Its initial weights are random; training updates the head and, in our run, the Qwen backbone.

---

# Preference loss

The Bradley–Terry preference probability is

$$P(\text{chosen wins}) = \sigma(r_c-r_r).$$

The negative log-likelihood loss is

$$\mathcal{L}=-\log\sigma(r_c-r_r).$$

Example: if $r_c=1.2$ and $r_r=-0.3$, the margin is $1.5$, the implied preference probability is $0.818$, and the loss is about $0.201$.

The loss does not try to make either absolute reward correct. It trains the **ordering and margin**: increasing $r_c-r_r$ decreases the loss, with diminishing returns.

---

# Why this loss?

It follows from a probabilistic model of pairwise human choices rather than being only an empirical trick:

1. Model each response with a latent scalar utility $r$.
2. Convert the reward difference into a probability with a sigmoid.
3. Maximize the likelihood of the observed preference.

This gives a smooth, differentiable objective and naturally treats the two rewards relatively. If the model ranks the pair incorrectly, the loss and corrective gradient are large.

---

# Attention masks, loss masks, and EOS

- **Attention mask:** prompt and response tokens are `1`; padding is `0`. It determines which tokens participate in attention.
- **Loss mask:** positions labeled `-100` are ignored by token-level loss. They can still influence later hidden states through attention.
- **EOS:** convenient, but not mathematically required. A final real token, a dedicated reward token, or a pooled representation can summarize the sequence.

For a sequence-level preference RM, only the final non-padding hidden state is sent to the reward head. Earlier prompt and response states still influence it causally.

---

# Outcome RM (ORM)

```text
Prompt:   What is 3 + 2?
Response: 5
Outcome:  1 (pass)

tokens: [prompt | response | EOS]
labels: [ -100  |   1 …    |  1 ]
```

For each labeled completion position, the hidden state produces an outcome logit and binary cross-entropy is computed against the final pass/fail label.

The prompt is not ignored: response hidden states encode it through causal attention. Only prompt positions are excluded from **direct** loss. No chosen/rejected pair is required.

---

# Process RM (PRM)

```text
[prompt | step₁ | END | step₂ | END | step₃ | END]
labels: [ -100 … +1   | -100 … 0    | -100 … -1   ]
```

At each step boundary, the corresponding hidden state is classified and cross-entropy is averaged over labeled steps.

`correct / neutral / incorrect` is common in datasets such as PRM800K, but it is not a universal definition. A PRM may instead be binary, use different categories, or generate textual critiques.

---

# Batch size 16 from microbatches of 2

Our configuration uses:

```text
microbatch size                 = 2 preference pairs
gradient accumulation steps     = 8 microbatches
effective batch per update      = 2 × 8 = 16 pairs
sequences processed per update  = 16 chosen + 16 rejected = 32
```

Each preference pair produces one Bradley–Terry loss, even though it requires chosen and rejected forward passes through the shared model.

---

# Loss and gradient accumulation

For microbatch $j$, containing two preference pairs:

$$L_j=\frac{\ell_{2j-1}+\ell_{2j}}{2}.$$

The training loop scales each microbatch loss and calls backward:

$$G \leftarrow G + \frac{1}{8}\nabla L_j.$$

`backward()` adds into one gradient buffer for every parameter; it does not preserve eight separate gradient objects. The weights remain unchanged during these eight backward passes.

After microbatch 8:

$$G=\nabla\left[\frac{1}{16}\sum_{i=1}^{16}\ell_i\right].$$

AdamW performs one `optimizer.step()`, then `zero_grad()` clears the buffers.

---

# Our preference-RM run

- Requested pairs: **5,000**
- Identical truncated pairs removed: **352**
- Training pairs: **4,183**
- Validation pairs: **465**
- Backbone: **Qwen3-0.6B-Base**
- Trainable parameters: **596.05M**
- Effective batch size: **16 pairs**
- Optimizer updates: **524** across two epochs

---

# Loss over training

![Preference RM training and validation loss](assets/loss-curve.svg)

Validation loss was best at step 100. During epoch 2, training loss continued falling while validation loss rose—evidence that confidence and overfitting increased.

---

# Pairwise accuracy over training

![Preference RM pairwise training and validation accuracy](assets/accuracy-curve.svg)

This metric is **pairwise accuracy**, $\mathbb{1}[r_c>r_r]$, not ROC-AUC. Validation accuracy peaked at **78.9%** at step 175 and ended near **76.6%**.

---

# Main takeaways

1. The backbone encodes the complete prompt-response context; the head converts selected hidden states into reward or class scores.
2. Preference RM learns relative ordering from pairs; ORM learns final success from individual responses; PRM learns step-level quality.
3. A loss mask controls where direct supervision is applied—not which earlier tokens can affect a supervised hidden state.
4. Gradient accumulation over eight microbatches of two is mathematically equivalent to averaging 16 pair losses before one update, assuming no intervening weight update.
5. Select checkpoints using validation behavior: the lowest validation loss and highest validation accuracy occurred well before the end of epoch 2.

