# Training LLMs to Reason in a Continuous Latent Space (Coconut) — Summary

**Hao, Sukhbaatar, Su, Li, Hu, Weston & Tian (Meta FAIR / UC San Diego, Dec 2024)** · [arXiv:2412.06769](https://arxiv.org/abs/2412.06769) · [PDF](https://arxiv.org/pdf/2412.06769)
Full text: [`../docs/references/coconut-continuous-latent.md`](../docs/references/coconut-continuous-latent.md)

The "reason in latent space, not language space" paper — the missing middle term in the CoT-vs-looping picture.

## The problem

LLMs reason via chain-of-thought in **language space** (word tokens). But language may be the wrong medium for reasoning: most tokens exist for fluency, not reasoning; the architecture spends roughly *equal compute per token* whether it's filler or a critical planning step; and (citing neuroscience) human language is optimized for *communication*, not reasoning. Forcing every reasoning step through a discrete word token is a bottleneck.

## The method — Coconut (Chain of Continuous Thought)

One change to the autoregressive loop. Normally: `h_t → W·h_t → softmax → pick a token → re-embed e(token) → feed back`. Coconut cuts out the middle: **feed the last hidden state `h_t` directly back as the next input embedding**, skipping the decode-to-token and re-embed. The loop closes in **continuous latent space** instead of through the vocabulary, so you don't collapse the rich `d`-dimensional state down to one sampled token.

**Mode gating.** The model toggles between **language mode** (normal token generation) and **latent mode** (hidden-state-fed-back), marked by special tokens:

- `<bot>` (begin-of-thought) → enter latent mode (no vocab decode), inserted right after the question.
- `<eot>` (end-of-thought) → return to language mode; decided by a small binary classifier or by padding to a fixed number of thoughts.

## Training (supervised, not RL)

A multi-stage curriculum (from iCoT): start with full language CoT, then progressively replace the first *k* reasoning steps with *k×c* continuous thoughts, removing the language steps stage by stage (reset optimizer between stages; mask loss on questions and latent thoughts). Plain cross-entropy, backprop through the fully-differentiable continuous thoughts — **no reward, no policy gradient.** The objective isn't to *compress* the removed language step but to *facilitate future reasoning*, so the model can learn representations *more effective than language*. Cost: n+1 sequential forward passes for n latent thoughts (hard to parallelize; flagged as future work).

> **RL note:** Coconut itself is supervised. But the *when-to-stop-thinking* decision (`<eot>` / how many thoughts) is a control decision — the same shape as ACT adaptive halting — and a natural place to layer RL on top later (frontier reasoning models already do RL over token-space CoT). In this paper that switch is a classifier or fixed length, not learned by reward.

## The emergent payoff — latent reasoning ≈ breadth-first search

A continuous thought can **encode several alternative next steps at once**, so the model keeps multiple options alive and prunes wrong ones as it goes — a BFS-like search — instead of committing to one path the way token-CoT must. It wasn't trained to do this; it emerges. On planning/backtracking logic tasks (ProntoQA, and their harder DAG-based **ProsQA**), Coconut **beats language CoT with fewer inference tokens**; on GSM8k math it matches the benefit of language chains.

## Why it matters for the thread — the token bottleneck

Recall: CoT raises effective depth by *looping output back into input*; a looped transformer does that loop *internally in latent space*. Coconut is the **explicit, external loop kept continuous**. The key insight: **decoding to a discrete token is a lossy projection** (a `d`-dim latent state → one of `|V|` tokens) that forces *premature commitment* to one reasoning path and discards the superposition. Coconut skips that compression *during* reasoning and only "translates to language when necessary."

Three points of the same primitive:

- **CoT** = externalized loop, in *token* space (lossy, one path at a time, interpretable).
- **Looped transformer** = internal loop, in *latent* space (cheap, opaque).
- **Coconut** = externalized loop, in *latent* space (continuous, superposed paths — the best-of-both middle term).

It also lands on the "forgetting/compression" theme: the vocabulary decode is heavy compression; Coconut avoids it mid-reasoning, paying it only at the end. For OpenMythos (whose looped block already reasons in latent space), Coconut is the explicit-autoregressive cousin — same principle, different loop axis.
