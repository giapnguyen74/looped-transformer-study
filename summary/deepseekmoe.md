# DeepSeekMoE: Towards Ultimate Expert Specialization — Summary

**Dai, Deng, Zhao, et al. (DeepSeek-AI / Peking / Tsinghua / Nanjing, Jan 2024)** · [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) · [PDF](https://arxiv.org/pdf/2401.06066) · [code](https://github.com/deepseek-ai/DeepSeek-MoE)
Full text (partial): [`../docs/references/deepseekmoe.md`](../docs/references/deepseekmoe.md)

The **knowledge-store** paper of the set — the basis for OpenMythos's `MoEFFN`, and the complement to all the looping work: where looping buys *reasoning depth* cheaply, MoE buys *knowledge capacity* cheaply.

## MoE background

Replace a Transformer's FFN with N experts (each a small FFN) plus a router that sends each token to its top-K experts. Only K of N run, so you scale *parameters* (knowledge) without scaling *compute per token*.

## The problem with conventional MoE

GShard-style MoE (~8–16 experts, top-1/2) gets poor **expert specialization**, from two failure modes:

1. **Knowledge hybridity** — too few experts means each must cram diverse, unrelated knowledge into its parameters, which it can't use coherently.
2. **Knowledge redundancy** — different experts separately re-learn the *same* common knowledge, wasting parameters.

## DeepSeekMoE's two strategies

1. **Fine-grained expert segmentation.** Split each expert FFN into `m` smaller ones (intermediate dim → 1/m) and activate `m×` more — *same parameters, same compute*. Finer experts decompose knowledge more precisely, and combinatorial flexibility explodes: 16 experts top-2 → C(16,2)=120 combinations; 64 experts pick-8 → **C(64,8) ≈ 4.4 billion**. (The "more combinations → finer, targeted knowledge" intuition.)
2. **Shared expert isolation.** Reserve `Ks` **always-on shared experts** to absorb common knowledge, so routed experts stop re-learning it and are freed to specialize.

## Results

DeepSeekMoE 2B ≈ GShard 2.9B (1.5× its expert params) and nearly reaches the *dense upper bound* at equal total params. Scaled: 16B ≈ DeepSeek 7B / LLaMA2 7B at **~40% compute**; 145B preliminary ≈ DeepSeek 67B at **~28.5%** compute.

## Where knowledge lives — the storage-vs-compute framing

DeepSeekMoE clarifies *what the knowledge dial is*. The precise mental model (two different "key/value" notions — don't fuse them):

- **FFN / MoE key–value = *memory*.** The FFN's first matrix = keys (pattern detectors), second = values (content written to the residual stream). The **stored knowledge primitives**. MoE just makes this store bigger and sparse.
- **Attention's K/V = *addressing*.** Per-token projections that decide which *positions* attend to which — *routing*, not storage.

So the whole machine is a division of labor:

- **FFN / MoE** — store & recall primitives (associative memory). The *content*. *(also computes — it's a nonlinear transform, a memory-that-also-computes.)*
- **Attention** — gather/route the right operands *across positions*.
- **Residual stream** — the shared workspace everything reads from and writes to.
- **Depth / loop** — repeated read → combine → write cycles that compose primitives into higher-order results.

It's **interleaved, not phased**: every block alternates attention (gather across positions) then FFN (recall/transform), accumulating in the residual stream. One-liner: **FFN/MoE = the stored primitives; the rest of the architecture is the routing-and-composition engine over them.**

## Ties to the thread and the repo

This is the **capacity/knowledge dial** in concrete form, implemented by OpenMythos's `MoEFFN` (routed + always-on shared experts: `n_experts`, `n_shared_experts`, `n_experts_per_tok`). In the foundation's terms:

- **Loop** = reasoning/compute depth, parameter-light → the recurrent block.
- **MoE** = knowledge capacity, compute-light → the FFN.

They're complementary dials, and OpenMythos pairs them because the contractive loop can't *store* much (the memorization cost Saunshi measured) — the knowledge it gives up is bought back by DeepSeekMoE-style experts. (Load-balancing — needed to stop routing collapse — is via auxiliary losses in the paper; OpenMythos uses the newer aux-loss-free `router_bias` trick from DeepSeek-V3.)

Caveat: the local copy is full through §4.5 (architecture + specialization analysis); the 16B/145B scaling sections, alignment, related work, and conclusion are omitted — the abstract supplies the headline scaling numbers. See the [PDF](https://arxiv.org/pdf/2401.06066).
