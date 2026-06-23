# OpenMythos — Summary of Summaries

A reading map for the `summary/` folder: the looped/recurrent-depth transformer literature, the mental model that ties it together, and how each piece maps onto OpenMythos. Full-text copies of every paper live in [`../docs/references/`](../docs/references/).

---

## Start here — the lens

- **[structure_review_foundation.md](structure_review_foundation.md)** — the reusable framework for reviewing *any* architecture. Core ideas:
  - A structure is **two geometries**: the representation map (expressivity) and the loss landscape (trainability).
  - **Three evaluation axes**: *expressivity* (a solution exists) · *trainability* (the descent can reach it) · *generalization* (the reached solution transfers) — plus *capacity/cost* as the design dial.
  - **Trainability ≠ global optimization**: it's "is the descent stable and well-conditioned," governed by dynamical isometry (keep the Jacobian's gain ≈ 1).
  - **Three dials**: knowledge ← parameters · compute ← active params/FLOPs · reasoning depth ← steps.
  - **Route work to the right substrate**: knowledge → parameters (MoE), reasoning → depth (loop).

Companion artifacts:

- **[looped_transformer_minimal.py](looped_transformer_minimal.py)** — the whole algorithm in ~130 commented lines (shared block, input injection, step embedding, ACT halting, MoE-FFN).
- **[universal_transformers.svg](universal_transformers.svg)** — the architecture diagram (looped block + MoE internals).

---

## The papers, grouped

### Foundational — the loop primitive
- **[universal_transformers.md](universal_transformers.md)** (Dehghani et al. 2018) — the original looped transformer: one block applied T times = weight-tied deep stack, plus per-token ACT halting; Turing-complete because depth can scale with input. *Established expressivity; left trainability open.*

### Why looping helps reasoning
- **[reasoning_with_latent_thoughts.md](reasoning_with_latent_thoughts.md)** (Saunshi et al. 2025) — reasoning needs *depth not parameters*; `(k⊗L)` looped models match `kL`-deep models on reasoning while trading away memorization; **L loops ≈ T steps of CoT** (looping = internal chain-of-thought).
- **[loop_think_generalize.md](loop_think_generalize.md)** (Kohli et al. 2026) — controlled study: looping enables *systematic generalization* (3-stage grokking) and *depth extrapolation*; the deep takeaway is that **looping changes the access pattern** — it folds knowledge into one re-readable bank and re-queries it across iterations (a CPU with re-queryable memory). Overthinking is the failure mode.

### Making the loop work (the three orthogonal fixes)
- **[parcae.md](parcae.md)** (Prairie et al. 2026) — **stability/trainability**: recast looping as an LTI dynamical system; instability = spectral radius ρ(Ā) ≥ 1; fix by a negative-diagonal parameterization that guarantees ρ < 1 for free. Establishes looping as a predictable **scaling axis** (the scaling story UT lacked).
- **[hyperloop_transformers.md](hyperloop_transformers.md)** (Zeitoun et al. 2026) — **parameter efficiency** for edge: middle-cycle loop + loop-level hyper-connections (matrix-valued residual); beats baselines at ~50% params, survives INT4. A relaxed-weight-tying via loop-specific params.
- **[lt2_linear_time_looped.md](lt2_linear_time_looped.md)** (Deng et al. 2026) — **attention cost**: replace quadratic attention in the loop with linear/sparse mixers; "looping turns compute into context" (rank-1→rank-T memory; window w→T·w receptive field). Convert pretrained loops cheaply via distillation.

### Relaxing strict weight-tying
- **[relaxed_recursive_transformers.md](relaxed_recursive_transformers.md)** (Bae et al. 2024) — convert a pretrained model into a looped one; restore per-layer distinctness with **low-rank LoRA capturing `ΔW = W_layer − W_shared`** (SVD-initialized; rank = the dial between recursive and vanilla). The basis for OpenMythos's `LoRAAdapter`.

### Other axes of recurrence / depth
- **[recurrent_transformer.md](recurrent_transformer.md)** (Oncescu et al. 2026) — recurrence on the **position axis** (not depth): each layer reads earlier positions' *outputs* → multi-hop composition within one layer (distinct weights, *no folding*). Trades depth for width → smaller KV cache. IO-aware tiling makes the sequential recurrence accelerator-friendly.
- **[mixture_of_depths_attention.md](mixture_of_depths_attention.md)** (Zhu et al. 2026) — attention on the **depth axis** (across layers): each layer adaptively reads earlier layers' KV, fixing residual-stream information dilution. Data-dependent retrieval beats blind residual-add across layers.

### Knowledge & latent reasoning (supporting)
- **[deepseekmoe.md](deepseekmoe.md)** (Dai et al. 2024) — the **knowledge store**: fine-grained expert segmentation + shared-expert isolation for specialization. FFN/MoE = stored key→value primitives; the basis for OpenMythos's `MoEFFN`.
- **[coconut_continuous_latent.md](coconut_continuous_latent.md)** (Hao et al. 2024) — reason in **latent space**: feed the last hidden state back directly instead of decoding to a token; avoids the lossy token bottleneck, enabling BFS-like superposed reasoning. The continuous, externalized middle term between token-CoT and the internal loop.

---

## Cross-cutting threads

**1. Recurrence lives on different axes.** The same goal — "more effective depth without more parameters" — is reached on perpendicular axes:
- *depth* (reuse one block T×, weight-shared) → the looped family = **folding**.
- *position* (per-layer left-to-right scan, distinct weights) → Recurrent Transformer = **no folding**.
- *cross-layer* (attend to earlier layers' KV) → MoDA.
They can be combined.

**2. Stability and trainability are the same spectral condition.** Iterating a map raises it to a power; gain > 1 explodes, < 1 vanishes/forgets, ≈ 1 is trainable. Every working design pins the gain near 1: residual + norm, input injection, zero-init (Loop-Think), spectral leash (Parcae ρ<1, OpenMythos `LTIInjection`), per-loop residual gate (LT2). Depth amplifies; the cure is isometry, not less depth.

**3. Stable loops must forget — selectively.** A contractive loop is lossy by construction (it must, or it explodes), so it forgets transients while input injection re-anchors it to the signal. Forgetting is the *mechanism* of refinement, not a side effect.

**4. CoT is a loop.** Chain-of-thought = an explicit loop in token space; a looped transformer = the same loop internally in latent space; Coconut = the explicit loop kept continuous. Decoding to a token is a lossy projection that forces premature commitment.

**5. Knowledge vs reasoning — route to the right substrate.** The loop computes/reasons but can't store much (contractive); knowledge lives in parameters (FFN/MoE). FFN/MoE = stored key→value primitives; attention routes across positions; the residual stream is the workspace; depth/loop is the compose-iteration. Interleaved, not phased.

**6. Three orthogonal costs of looping, three papers.** Stability (Parcae) · parameter memory (Hyperloop) · attention FLOPs/KV cache (LT2). The UT critique — *no scaling story, no stability analysis* — is what these later papers pay off.

---

## Map to OpenMythos

| Component | Paper / idea |
|---|---|
| Recurrent block (prelude → loop → coda) | Universal Transformers; middle-cycle looping |
| `LTIInjection` (ρ(A) < 1) | Parcae (negative-diagonal LTI) |
| `ACTHalting` | Universal Transformers (adaptive computation time) |
| `MoEFFN` (routed + shared experts) | DeepSeekMoE |
| `LoRAAdapter` (per-loop low-rank delta) | Relaxed Recursive Transformers |
| Loop-index / step embedding | "where am I in the loop" (UT, Loop-Think zero-init lineage) |
| Reasoning vs memorization tradeoff | Reasoning with Latent Thoughts |
| Future directions | LT2 (subquadratic in-loop attention), MoDA (cross-layer KV), Recurrent Transformer (position-axis), Coconut (latent CoT) |

---

## One-paragraph synthesis

A looped transformer manufactures **reasoning depth** by reusing one block (folding), which is parameter-efficient but (a) hard to train unless you pin the loop's spectral gain near 1 (Parcae/LTI), (b) limited in what it can store, so knowledge is offloaded to **MoE** (DeepSeekMoE), and (c) expensive in attention/KV at long context unless you go subquadratic (LT2) or memory-light (Hyperloop). Looping helps reasoning specifically because it turns one block into a **re-readable knowledge bank** queried across iterations (Loop-Think), behaving like internal chain-of-thought (Latent-Thoughts) — and CoT/Coconut are the externalized, token- and latent-space versions of the same loop. Recurrence can also be placed on the **position axis** (Recurrent Transformer) or as **cross-layer attention** (MoDA) instead of the depth axis. Reviewing any of these comes down to the three axes — *can it represent, can it be trained stably, does it generalize* — with trainability, not expressivity, being the part novel structures fail first.
