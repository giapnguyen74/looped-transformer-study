# The Recurrent Transformer: Greater Effective Depth and Efficient Decoding — Summary

**Oncescu, Morwani, Jelassi, Meterez, Kwun & Kakade (Harvard, Apr 2026)** · [arXiv:2604.21215](https://arxiv.org/abs/2604.21215) · [PDF](https://arxiv.org/pdf/2604.21215) · [code](https://github.com/geniucos/recurrent-transformer)
Full text (partial): [`../docs/references/recurrent-transformer.md`](../docs/references/recurrent-transformer.md)

**The odd one out in the set.** Every other paper loops the *depth* axis (reuse the same block across iterations — "folding"). This paper makes attention recurrent along the *sequence/position* axis *within each layer*, with **distinct per-layer weights**. Different mechanism, different goal. (Naming collision to watch: "recurrent-depth transformer" = the folded/looped models; "Recurrent Transformer" = this position-axis one.)

## The problem it names

Standard Transformers are "temporally shallow": at layer ℓ, position t attends to key–value pairs computed from the *previous* layer's representation. So any two positions interact essentially once per layer — the effective depth of cross-position interaction is capped by the layer count.

## The change (one tweak)

A position's **persistent key–value pair is computed from that layer's *output*** (after attention + MLP), not from its input. So a later position attends to an earlier position whose representation has *already undergone this layer's computation* — giving each layer its own recurrent key–value memory along the time axis. (Unlike the Feedback Transformer, which shares memory *across* layers; RT keeps it per-layer, which is what makes it efficient.)

**The circularity fix.** Position i's persistent KV comes from its output `zᵢ` — but `zᵢ` itself attends to position i. Circular. Resolution: a **temporary** KV from the input, used only locally at the current position; a **persistent** KV from the output `zᵢ`, exposed to all later positions.

So per position: `aᵢ = Attn( (k₁,v₁)…(kᵢ₋₁,vᵢ₋₁) [from earlier positions' OUTPUTS], (kᵢᵗᵉᵐᵖ,vᵢᵗᵉᵐᵖ) [from own INPUT], qᵢ )`.

## Still autoregressive — with a left-to-right refinement scan

It stays fully causal (position i only sees ≤ i; standard mask; normal one-token-at-a-time decode with a KV cache). The twist: each position composes its **own raw input** plus the **already-processed outputs** of its predecessors. The consequence is **multi-hop composition within a single layer**: position 1 finishes → position 2 reads refined-pos-1 → position 3 reads refined-pos-1-and-2 → … A vanilla Transformer needs several stacked layers to chain positions like that; RT does it in one. That is the "greater effective depth," on the sequence axis.

## Three lenses

- **Representation.** Can emulate a standard Transformer (mild assumptions) *and* implement token-to-token recurrence (restrict attention to the previous position) — sitting between fully parallel attention and a pure RNN/SSM, without a capped-memory bottleneck.
- **Stability.** Classic RNNs pass info i→j only through a length-(j−i) chain → vanishing/exploding gradients. RT adds **many multi-hop paths plus direct one-hop attention** between any two positions, so with pre-norm before KV computation and 1/√L depthwise residual scaling it trains stably (the same isometry/conditioning toolkit, applied to a different recurrence).
- **Efficiency (the systems contribution).** Naively, training/prefill is sequential and bandwidth-bound (arithmetic intensity ≈Θ(1)) because KV is revealed one position at a time. An **exact IO-aware tiling algorithm** (Flash-Inference style) cuts HBM traffic Θ(N²) → **Θ(N log N)** and raises arithmetic intensity to Θ(N/log N), preserving the exact math. This is the hardware lens the other papers don't touch.

## The payoff — trade depth for width

Greater effective temporal depth means the same quality with **fewer layers**, shrinking the KV cache and decode-time memory traffic. On C4 (150M, 300M): a **6-layer RT ≈ a 12-layer Transformer** and it beats all param-matched Transformer configs (RT 6L = 2.860, 12L = 2.867 vs best Transformer 2.892), cutting KV-cache size ~30%.

## Folding vs not: where it sits

Two orthogonal ways to buy "more effective depth without more parameters":

| | Looped / recurrent-depth (folding) | Recurrent Transformer (this paper) |
|---|---|---|
| Recurrence axis | **depth** (reuse one block T×) | **position/time** (per-layer scan) |
| Weights | shared (tied) → folding | distinct per layer → no folding |
| Effective-depth gain | iterative reasoning over one map | multi-hop cross-position per layer |
| Main cost | extra FLOPs + trainability/stability | sequential/bandwidth-bound → needs tiling |
| Main payoff | reasoning, param-efficiency, test-time scaling | KV-cache reduction, decode efficiency |

Because they're orthogonal, you could in principle **combine** them — loop a stack of recurrent-transformer layers (fold the depth axis *and* recur the position axis).

One-liner: **the looped models fold depth (reuse a map); the Recurrent Transformer compresses more causal composition into each layer's sequence-scan (distinct weights, no fold) — same goal, perpendicular axes.**

Caveat: the local copy is full through §2.2; the representational proofs, stability analysis, tiling algorithm, and full experiments are truncated — see the [PDF](https://arxiv.org/pdf/2604.21215).
