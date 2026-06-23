---
title: "The Recurrent Transformer: Greater Effective Depth and Efficient Decoding"
arxiv_id: "2604.21215"
url: "https://arxiv.org/abs/2604.21215"
pdf: "https://arxiv.org/pdf/2604.21215"
authors: "Costin-Andrei Oncescu, Depen Morwani, Samy Jelassi, Alexandru Meterez, Mujin Kwun, Sham Kakade (Harvard University)"
date: "2026-04-23"
source: "html-fulltext"
---

# The Recurrent Transformer: Greater Effective Depth and Efficient Decoding

**Authors:** Costin-Andrei Oncescu, Depen Morwani, Samy Jelassi, Alexandru Meterez, Mujin Kwun, Sham Kakade (Harvard University). Correspondence to: concescu@g.harvard.edu
**arXiv:** 2604.21215 — https://arxiv.org/abs/2604.21215

> Note: This markdown was extracted from the arXiv HTML rendering. The fetched HTML covered the abstract, introduction, contributions, and the architectural overview (through Section 2.2); later sections (3–9 and appendices) were not present in the retrieved HTML. Inline mathematical notation has been simplified to plain text for readability. Code is available at https://github.com/geniucos/recurrent-transformer

## Abstract

Transformers process tokens in parallel but are temporally shallow: at position t, each layer attends to key–value pairs computed based on the previous layer, yielding a depth capped by the number of layers. Recurrent models offer unbounded temporal depth but suffer from optimization instability and historically underutilize modern accelerators. We introduce the *Recurrent Transformer*, a simple architectural change where *each layer* attends to key–value pairs computed off its own activations, yielding layerwise recurrent memory while preserving standard autoregressive decoding cost. We show that the architecture can emulate both (i) a conventional Transformer and (ii) token-to-token recurrent updates under mild assumptions, while avoiding optimization instability.

Naively, prefill/training appears bandwidth-bound with effective arithmetic intensity near 1 because keys and values are revealed sequentially; we give an exact tiling-based algorithm that preserves the mathematical computation while reducing HBM traffic from Θ(N²) to Θ(N log N), increasing effective arithmetic intensity to Θ(N / log N) for sequence length N.

On 150M and 300M parameter C4 pretraining, Recurrent Transformers improve cross-entropy over a parameter-matched Transformer baseline and achieve the improvement with fewer layers (fixed parameters), suggesting that recurrence can trade depth for width, thus reducing KV cache memory footprint and inference latency. Code is available at https://github.com/geniucos/recurrent-transformer

## 1 Introduction

Transformers (Vaswani et al., 2017) are highly effective sequence models, but their computation across positions is structurally shallow: within each layer, position t attends to key–value pairs computed from the previous layer embeddings, allowing essentially at most one interaction per layer between any pair of positions. A growing body of theory studies the fundamental limitations implied by bounded depth in attention models, including circuit-complexity characterizations of what low-depth Transformers can and cannot represent (Merrill et al., 2022; Liu et al., 2023). These perspectives motivate architectures that achieve greater effective depth.

We introduce the Recurrent Transformer (RT), a simple modification of how key–value pairs are computed that makes each layer temporally recurrent. In a standard Transformer, at layer ℓ, the key–value pair at position t is computed from the layer-(ℓ−1) representation at that position and can then be attended to by later positions t′ > t. In the Recurrent Transformer, by contrast, the key–value pair at position t in layer ℓ is computed from that position's output at layer ℓ, rather than from its layer-(ℓ−1) representation. Consequently, a later position t < t′ at layer ℓ attends to a representation at t that already reflects layer ℓ attention and MLP computation. Importantly, Recurrent Transformer performs this recurrence separately within each layer, so each layer maintains its own key–value memory. This differs from the Feedback Transformer (Fan et al., 2020), which uses a shared memory across layers, and this layerwise separation is a key reason why our architecture can be implemented efficiently.

We motivate Recurrent Transformer's design through lenses of representation, optimization and computational efficiency.

*Figure 1: One layer of the Recurrent Transformer mapping input embeddings x₁…x₄ to output embeddings z₁…z_N. Notice how the *persistent* key–value pairs are a function of the layer's output and are used for all subsequent attention computations. The *temporary* key–value pairs are only used at the time they are computed and then discarded. They are only used to avoid ill-defined attention since, for example, a₂ cannot attend to (k₂, v₂) as that indirectly depends on it. This is in contrast to a vanilla Transformer that uses these same key–value pairs for all subsequent attention computation as well.*

#### (i) Representational perspective.

Recurrent Transformers retain per-token key–value memory just like a Transformer, but increase the space of computations that can be expressed within a single layer by allowing later positions to attend to representations that have already undergone attention and MLP processing. Under mild assumptions, Recurrent Transformers can emulate standard Transformer behavior; conversely, by restricting attention to the previous position, they can implement token-to-token recurrent computation. This positions Recurrent Transformer between fully parallel attention and purely recurrent state-space computation, while avoiding a capped-memory bottleneck.

#### (ii) Training Stability.

Viewing the model as a directed computation graph over positions, classical RNNs transmit information from position i to j only through the length-(j−i) chain of intermediate states. The potentially large length of such paths gives rise to vanishing and exploding gradient phenomena (Bengio et al., 1994; Pascanu et al., 2013), making it hard to ensure information flow between distant positions. Recurrent Transformer alleviates this by introducing many additional multi-hop paths, corresponding to repeated attend+MLP applications across positions within a layer, while still permitting direct one-hop attention interactions between any two positions. In practice, we find that this architecture, together with appropriate normalization before key–value computation and standard depth-wise residual scaling (Bordelon et al., 2023; Yang et al., 2023), trains stably. We expand on this view, and on why exploding gradients are not expected to be an issue, in Section 4.

#### (iii) Training-time efficiency.

A naive implementation of Recurrent Transformer training/prefill is sequential in position and appears bandwidth-bound: keys and values are revealed one position at a time, and each query must aggregate over a linearly-growing prefix, leading to a very low effective arithmetic intensity – Θ(1) – under the Roofline model (Williams et al., 2009). We give an *exact* tiling algorithm that preserves the mathematical attention computation while reorganizing memory movement, reducing high-bandwidth memory (HBM) traffic from Θ(N²) to Θ(N log N) and raising effective arithmetic intensity to Θ(N / log N). Our key observation is that, during training/prefill, the full sequence of queries is available in advance even though persistent key–value pairs are revealed causally. This makes it possible to reorganize the computation into a tiled schedule, in the spirit of Flash Inference (Oncescu et al., 2025), that reuses each revealed key–value tile across many future queries before it is evicted from fast memory. The final algorithm interleaves attention blocks and MLP computation while employing the same methodology as Rabe and Staats (2021); Dao et al. (2022) to accumulate attention contribution.

#### (iv) Depth to inference efficiency.

Crucially, the additional effective temporal depth can translate into a better depth–width tradeoff: at fixed parameter count, achieving the same quality with fewer layers reduces the amount of stored key–value state and the corresponding decode-time memory traffic. Our experiments support this regime, with shallower Recurrent Transformer models outperforming deeper Transformer baselines.

#### Contributions.

- In Section 2, we propose the Recurrent Transformer (RT), a layerwise recurrent attention architecture that computes each layer's key–value pairs from that layer's outputs rather than from the previous layer's representations.
- In Section 3, we provide representational arguments showing Recurrent Transformer can emulate standard self-attention behavior and can implement token-to-token recurrent computation via attention concentration under mild assumptions.
- In Section 4, we provide a path-based analysis of training stability in Recurrent Transformer, showing how the architecture combines additional multi-hop computation with direct one-hop attention paths, and giving theoretical evidence in a simplified setting that neither exploding gradients nor vanishing gradients are expected under appropriate scaling.
- In Section 5, we provide an *exact*, IO-aware tiling algorithm for prefill/training that preserves the mathematical attention computation while reducing memory traffic from Θ(N²) to Θ(N log N) and increasing effective arithmetic intensity from Θ(1) to Θ(N / log N).
- In Section 6, we outline various computational challenges and design choices required to make Recurrent Transformer training more efficient and practical.
- In Section 7, we present empirical results on 300M-parameter C4 pretraining showing improved cross-entropy over parameter-matched Transformer baselines and favorable depth–width tradeoffs at fixed parameter count (Figure 2). In particular, Recurrent Transformer with 6 layers performs comparably to 12 layers (fixed parameters), reducing KV cache size by approximately 30% and lowering decode-time memory traffic, thereby improving inference efficiency. Additional results for the 150M-parameter model are provided in Appendix E.3.

*Figure 2: C4 pretraining: loss curves for 300M parameter model trained on C4 dataset.*

*Table 1: C4 pretraining loss for 300M parameter model.*

| Model | Layers | Width | Val CE ↓ |
| --- | --- | --- | --- |
| Transformer | 6 | 2048 | 2.917 |
| Transformer | 12 | 1408 | 2.896 |
| Transformer | 24 | 1024 | 2.892 |
| Recurrent Transformer | 12 | 1408 | 2.867 |
| Recurrent Transformer | 6 | 2048 | 2.86 |

## 2 Architectural overview and notation

#### Architectural overview.

Relative to a standard causal Transformer, the defining change in Recurrent Transformer is where the key–value pairs exposed to future positions come from. In a standard Transformer, the key–value pair at position i is computed from the layer input at that position. In Recurrent Transformer, by contrast, the *persistent* key–value pair at position i is computed from that position's layer output. Consequently, later positions attend to earlier positions whose representations have already undergone same-layer attention and MLP computation, making each layer recurrent along the temporal axis.

This creates a circularity at the current position: because the layer output at position i also attends to the current position, the persistent pair (kᵢ, vᵢ) cannot itself be used while computing that output. To resolve this, Recurrent Transformer distinguishes between two kinds of key–value pairs. A *temporary* pair, computed from the current layer input, is used only when evaluating attention at the current position. A *persistent* pair, computed from the resulting layer output, is then stored and made available to all later positions.

#### Notation.

We present the single-head formulation; multihead attention applies the same construction independently per head and then uses the usual output projection. We assume a sequence length of N and use L for the number of stacked layers. Let D be the embedding dimension and consider a single layer with inputs x₁, …, x_N ∈ ℝ^D. Let MLP: ℝ^D → ℝ^D denote the MLP block and let RMS: ℝ^D → ℝ^D denote Root Mean Square normalization (Zhang and Sennrich, 2019). While in practice we use learnable parameters, as far as presentation and analysis is concerned, we take RMS(x) = √D · x / ‖x‖₂. We use a distinct (magenta) RMS to denote query/key normalization (Dehghani et al., 2023).

The attention operator Attn: (ℝ^D × ℝ^D)* × ℝ^D → ℝ^D maps a sequence of key–value pairs (k₁, v₁), …, (k_ℓ, v_ℓ) and a query q to:

> Attn((k₁, v₁), …, (k_ℓ, v_ℓ), q) = Σᵢ₌₁..ℓ vᵢ · exp(⟨kᵢ, q⟩) / Σⱼ₌₁..ℓ exp(⟨kⱼ, q⟩)

We use projection matrices Q, K, V ∈ ℝ^{D×D} to compute queries, keys and values based off an embedding. Following standard Transformer parameterizations (Bordelon et al., 2023; Yang et al., 2023), we use pre-LN (Xiong et al., 2020) and assume attention and MLP residual updates are initialized/parameterized with an appropriate 1/√L scale so that chaining maps of the form x ↦ x + (1/√L){Attn, MLP}(RMS(x)) is well-behaved.

### 2.1 The Transformer layer

We first recall a standard *causal* decoder-only Transformer layer (Vaswani et al., 2017). Given inputs x₁, …, x_N ∈ ℝ^D, position i forms its query, key, and value from the current layer input:

> qᵢ = RMS[Q · RMS(xᵢ)]
> kᵢ = RMS[K · RMS(xᵢ)]
> vᵢ = V · RMS(xᵢ)

The attention output at position i is then computed by attending over the prefix of key–value pairs available up to that position:

> aᵢ = Attn((k₁, v₁), …, (kᵢ, vᵢ), qᵢ)

Finally, the layer output is obtained by adding the attention and MLP residual branches:

> yᵢ = xᵢ + (1/√L)(aᵢ + MLP[RMS(xᵢ + (1/√L)aᵢ)])

The key structural point is that, in a standard Transformer, the key–value pair stored at position i is computed from the layer input at the same position.

### 2.2 The Recurrent Transformer layer

Recurrent Transformer layers (illustrated in Figure 1) differ from standard Transformer layers only in how the key–value pairs exposed to future positions are formed. At position i, Recurrent Transformer first forms the query together with a *temporary* key–value pair from the current layer input:

> qᵢ = RMS[Q · RMS(xᵢ)]
> kᵢ^temp = RMS[K · RMS(xᵢ)]
> vᵢ^temp = V · RMS(xᵢ)

These definitions are identical to the Transformer's query, key, and value projections at position i. The attention output at position i is then computed using the persistent key–value pairs from earlier positions together with the temporary pair at the current position:

> aᵢ = Attn((k₁, v₁), …, (kᵢ₋₁, vᵢ₋₁), (kᵢ^temp, vᵢ^temp), qᵢ)

We next form the layer output representation:

> zᵢ = xᵢ + (1/√L)(aᵢ + MLP[RMS(xᵢ + (1/√L)aᵢ)])

which is both the representation passed to the next layer and the source from which the persistent key–value pair at position i is computed. We define that persistent pair by projecting from this output:

> kᵢ = RMS[K · RMS(zᵢ)]    (1)
> vᵢ = V · RMS(zᵢ)          (2)

Thus, the temporary pair (kᵢ^temp, vᵢ^temp) is used only locally to resolve attention at the current position, while the persistent pair (kᵢ, vᵢ), derived from the layer output zᵢ, is what later positions attend to — giving each layer its own recurrent, layerwise key–value memory.

---

*Extraction note: The arXiv HTML retrieved for this paper ended within Section 2.2. Sections 3 (Representational Perspective), 4 (Training Stability), 5 (Exact Tiling for Training and Prefill), 6 (Computational challenges), 7 (Experiments), 8 (Related Work), 9 (Discussion and Conclusion), References, and Appendices A–E were listed in the document's table of contents but were not included in the fetched content. See the PDF at https://arxiv.org/pdf/2604.21215 for the complete text.*
