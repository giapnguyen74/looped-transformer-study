# LT2: Linear-Time Looped Transformers — Summary

**Deng, Zhang, Zhu, Xu, Liu, Ng & Chen (Rice / Apple / UCSC / CMU, May 2026)** · [arXiv:2605.20670](https://arxiv.org/abs/2605.20670) · [PDF](https://arxiv.org/pdf/2605.20670) · [code](https://github.com/chili-lab/LT2) · [checkpoint](https://huggingface.co/chili-lab/Ouro-hybrid-1.4B)
Full text (partial): [`../docs/references/lt2-linear-time-looped.md`](../docs/references/lt2-linear-time-looped.md)

This paper fixes a cost the other loop papers ignored: the **attention bill of looping**.

## The problem

A looped Transformer reuses N blocks for T iterations → effective depth T·N with only N params. But each loop re-applies **full quadratic softmax attention over the entire sequence**. So even though *parameters* are shared, **attention FLOPs (training) and KV-cache memory (inference) blow up** — quadratic in sequence length *and* multiplied by the loop count T. Attention becomes the dominant bottleneck: adding loops or context quickly gets impractical.

## The fix

LT2 replaces the quadratic softmax attention inside the looped block with a **subquadratic / linear-time token mixer**, in two families:

- **LT2-linear** — linear attention with a recurrent state `S_t` (O(L·d_k·d_v) FLOPs, *constant* O(d_k·d_v) cache). Supports the modern zoo: Linear Attention, RetNet, Mamba2, GLA, HGRN2, DeltaNet, GDN, KDA.
- **LT2-sparse** — sparse attention (sliding window / NSA / DSA), O(L·w·d) with window `w ≪ L`.

## The elegant insight — "looping turns compute into context"

The interesting part isn't just efficiency; it's that **looping precisely patches the weaknesses of cheap attention**:

- **Loop × linear attention:** T iterations turn the rank-1 state update into a **rank-T update** → the loop progressively *refines the recurrent memory*, recovering expressiveness linear attention normally lacks.
- **Loop × sparse attention:** T iterations turn a window of size `w` into an **effective receptive field of size T·w** → the loop progressively *expands the context* a window can see.

So the loop gives back exactly what you sacrificed by going subquadratic. (Complements the other papers' "what does looping buy?" — Loop-Think: knowledge *access*; Latent-Thoughts: reasoning *depth*; LT2: *context/memory* traded away for speed.)

## Stability detail (the recurring pattern)

Beyond the per-block residual, LT2 adds a **learned per-loop residual gate** `ρ_τ` across iterations (`h^(τ) = h̃^(τ) + ρ_τ ⊙ h^(τ-1)`), **zero-initialized** — identity at the start of training. Same isometry / identity-at-init trick as Loop-Think's zero-init and Parcae's spectral leash.

## LT2-hybrid (headline results)

Mixing attention variants in the looped setting:

- **GDN + DSA** (linear + sparse): *matches* the standard looped Transformer's quality (59.3% avg zero-shot) at **fully linear-time cost** — ~5.7× higher decode throughput at 8k context (125 vs 22 tok/s, batch 8).
- **Full + GDN** (a small fraction of full attention + linear): **beats** the standard looped Transformer by +2.1 points (61.4% vs 59.3%) *and* ~5× higher decode throughput. Best of both.

## Conversion, not retraining

They distill a pretrained looped Transformer (Ouro) into LT2 with only **~1B tokens** of continued training → **Ouro-Hybrid-1.4B** keeps the full-attention teacher's quality with linear-time efficiency, competitive with industry 1B–4B models. Same "convert an existing model cheaply" spirit as Relaxed Recursive — but converting the *attention mechanism* rather than the layer-tying.

## Where it sits in the set

Three orthogonal fixes to the same loop primitive:

- **Parcae** → stability / trainability (ρ(Ā) < 1).
- **Hyperloop** → parameter memory (edge deployment).
- **LT2** → attention FLOPs & KV-cache at long context.

For OpenMythos (which uses switchable MLA/GQA full attention), LT2 is the natural direction if you ever scale loops × context — swap the in-loop mixer for a subquadratic one and let the loop refine its state. Note also the linear-attention mixers (Mamba2, GLA, GDN, KDA) carry diagonal recurrent states — the same diagonal-state-space lineage Parcae draws its stability trick from.

Caveat: the local copy is truncated mid-§2.2; the experiments, distillation details, and proofs are in the [PDF](https://arxiv.org/pdf/2605.20670) (the abstract supplies the headline numbers above).
