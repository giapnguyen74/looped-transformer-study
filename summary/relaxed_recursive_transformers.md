# Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA — Summary

**Bae, Fisch, Harutyunyan, Ji, Kim & Schuster (KAIST AI / Google DeepMind, Oct 2024)** · [arXiv:2410.20672](https://arxiv.org/abs/2410.20672) · [PDF](https://arxiv.org/pdf/2410.20672)
Full text (partial): [`../docs/references/relaxed-recursive-transformers.md`](../docs/references/relaxed-recursive-transformers.md)

This is the paper behind OpenMythos's `LoRAAdapter`. It answers a tension we kept hitting: **pure weight-tying is too rigid (one block must serve every depth), fully distinct layers throw away the savings — so relax the tie just a little.**

## The problem

Parameter sharing ("layer tying") could shrink LLMs, but in modern LLMs it had mostly underperformed. The twist: *convert an existing pretrained model* into a shared one instead of training a looped model from scratch.

## Recursive Transformer

Collapse a vanilla N-layer model into a single block of K unique layers, **looped** N/K times (the CYCLE strategy). The decisive move is **initialization**: seed the shared block from the original pretrained weights, then briefly "uptrain." Good init → strong performance with little training — you start *near a good basin* instead of fighting the looped landscape from zero (the "short route to a good basin" idea from our trainability thread).

## Relaxed Recursive Transformer — the core idea

Strict tying limits capacity, so relax it with **per-depth LoRA** modules. The key insight:

> **The LoRA matrices capture the *difference* between each original layer and the shared block, stored cheaply in low rank.**

- Original distinct weights: `W₁ … W_N`. Shared block: `W_shared`. The lost per-layer difference is the residual `ΔW_ℓ = W_ℓ − W_shared`.
- LoRA restores a low-rank slice of it: effective weight at depth ℓ = `W_shared + Bℓ Aℓ`, with `A` of shape `r×d`, `B` of shape `d×r`, `r ≪ d`. Cost per layer drops from `d²` to `2dr`.
- So `W_shared` holds **what's common across layers**; the LoRA holds **what's specific to each depth**, compressed.

Three properties make it work:

- **Rank `r` is the dial.** `r = 0` → pure recursive (no differences kept); `r = full` → recovers the original vanilla model. In between, choose how much per-layer distinctness to restore — the capacity knob.
- **Init is truncated SVD of the actual residual.** `ΔW_ℓ = UΣVᵀ`, take the top-`r` components → the *best possible* rank-`r` approximation (Eckart–Young). The LoRA starts off already approximating the layers it replaced.
- **Low rank suffices** because residuals between similar layers concentrate in a few directions; small `r` recovers most of the variation.

It's a clean **common + specific factorization**: shared block = common structure, low-rank LoRA = per-depth specialization — a more expressive cousin of the step-embedding trick (which only *adds a bias* per depth; LoRA actually *modifies the transform*).

## Results

Recursive Gemma 1B (converted from Gemma 2B) beats similar-size vanilla models (TinyLlama 1.1B, Pythia 1B) and distillation baselines, and recovers most of the full 2B model's performance. With knowledge distillation, ~60B uptraining tokens ≈ full Gemma trained on 3T tokens.

## Bonus — Continuous Depth-wise Batching

Because parameters are shared across *depth* (loop iterations), not just sequence positions, a new batching dimension opens up; paired with early-exit (per-sample adaptive depth, like ACT), they project **2–3× (up to ~4×) throughput**.

## How it fits the thread / the repo

- The LoRA **rank** is the capacity dial between expressivity and compactness.
- Pretrained-init + uptraining is a **trainability shortcut** (short route to a good basin).
- Early-exit is the **dynamic-depth** idea again (adaptive compute per sample).
- OpenMythos's `LoRAAdapter` is an even more compact variant: it shares the `A` (down) and `B` (up) matrices across *all* loops and learns only a **per-loop scale vector**, trading the paper's per-layer flexibility for extra compactness.

Caveat: the local copy captured the abstract, intro, contributions, and §2.1–2.2; the detailed methods (§2.3–2.4), experiments, and appendices are equation/table-heavy and live in the [PDF](https://arxiv.org/pdf/2410.20672).
