# Mixture-of-Depths Attention (MoDA) — Summary

**Zhu, Fang, Liao, et al. (HUST / ByteDance Seed, 2026)** · [arXiv:2603.15619](https://arxiv.org/abs/2603.15619) · [PDF](https://arxiv.org/pdf/2603.15619) · [code](https://github.com/hustvl/MoDA)
Full text (partial): [`../docs/references/mixture-of-depths-attention.md`](../docs/references/mixture-of-depths-attention.md)

Another "add attention on a new axis" idea — this time the **depth axis** (across layers), to fix *information dilution* in deep stacks.

## The problem

Scaling depth is valuable, but deep Transformers suffer **information dilution**: features formed in shallow layers get progressively washed out by repeated residual additions, so they're hard to recover deep in the stack. The root cause: **the residual stream compresses all of depth history into one fixed-size hidden-state trajectory** — early salient features get superposed away. (The "residual = single workspace, and it forgets" theme, now as an *unwanted* forgetting across layers.)

## The "read, operate, write" lens

They frame stacking blocks along the depth stream as read → operate → write, and compare:

- **Depth Residual** (ResNet-style): read = identity, write = add. Compresses depth history into a fixed tensor → dilution.
- **Depth Dense** (DenseNet-style): read *all* preceding layers, concatenate → lossless but O(T·L²·D²) — prohibitive at scale, fixed connectivity.
- **Depth Attention** (their bridge): use *attention* to read historical depth KV **data-dependently** — each token attends, at its own position, across *layers* to the KV `{Kᵢ,Vᵢ}` of all preceding layers. Cost O(T·L²·D), a factor 1/D cheaper than dense, adaptive connectivity.
- **MoDA** (the proposal): **unify sequence attention and depth attention in one softmax operator.** Each head attends to *both* the sequence KV of the current layer (across positions) *and* its own depth KV (same position, across preceding layers), normalized jointly. Write: append the current layer's KV to the depth stream for future layers.

## The principle

Just as attention beats fixed pooling *across positions*, **data-dependent retrieval beats a blind residual add *across layers***. MoDA extends the attention principle from the sequence axis to the depth axis — giving each layer adaptive read access to everything earlier, instead of relying on the lossy residual.

## Hardware

A fused kernel runs sequence + depth attention in one pass with shared online-softmax, plus chunk-aware depth-KV layout and group-aware indexing → **97.3% of FlashAttention-2 efficiency at 64K**, ~3.7% FLOPs overhead. (Same "make the new attention accelerator-friendly" concern as the Recurrent Transformer's tiling.)

## Results

1.5B models (OLMo2 recipe, 400B tokens): +0.2 avg perplexity across 10 validation sets, **+2.11% avg downstream** on 10 tasks, negligible overhead; better with **post-norm** than pre-norm; reduces attention-sink behavior.

## Where it sits in the set

Another "fix shallow interaction" move, on yet another axis:

- **Recurrent Transformer** → recurrence on the **position** axis (within a layer).
- **MoDA** → attention on the **depth** axis (across layers).
- **Looped family** → reuse on the **depth** axis (weight-shared iteration).

MoDA and looping are cousins on the same axis, opposite in spirit: looping *reuses* one block across depth (parameter-efficient), while MoDA *adds cross-layer KV memory* to make real depth more effective (un-forgetting). The overlap with the "re-readable knowledge bank" framing is direct — MoDA gives the depth dimension its own explicit attention memory, where the looped model re-accesses implicitly via weight-sharing.

Caveat: the local copy is full through §2.2 (design space + MoDA formulation); the algorithm, kernel, and experiments are truncated — the abstract supplies the headline numbers. See the [PDF](https://arxiv.org/pdf/2603.15619).
