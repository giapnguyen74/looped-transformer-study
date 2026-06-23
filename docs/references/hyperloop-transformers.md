---
title: "Hyperloop Transformers"
arxiv_id: "2604.21254"
url: "https://arxiv.org/abs/2604.21254"
pdf: "https://arxiv.org/pdf/2604.21254"
authors: "Abbas Zeitoun, Lucas Torroba-Hennigen, Yoon Kim (Massachusetts Institute of Technology)"
date: "2026-04-23"
source: "html-fulltext"
---

# Hyperloop Transformers

**Authors:** Abbas Zeitoun, Lucas Torroba-Hennigen, Yoon Kim — Massachusetts Institute of Technology ({zeitoun, lucastor, yoonkim}@mit.edu)
**arXiv:** 2604.21254 — https://arxiv.org/abs/2604.21254
**License:** CC BY 4.0 · arXiv:2604.21254v1 [cs.LG] 23 Apr 2026

> Note: This markdown was extracted from the arXiv HTML (v1). The HTML fetch was truncated partway through Section 4.2 (Post-training Quantization), so the document below contains the abstract through the Main Results, but not the later ablations, analysis, discussion, related work, or conclusion sections. Inline math has been rendered to readable notation.

## Abstract

LLM architecture research generally aims to maximize model quality subject to fixed compute/latency budgets. However, many applications of interest such as edge and on-device deployment are further constrained by the model's memory footprint, thus motivating *parameter-efficient* architectures for language modeling. This paper describes a simple architecture that improves the parameter-efficiency of LLMs. Our architecture makes use of looped Transformers as a core primitive, which reuse Transformer layers across depth and are thus more parameter-efficient than ordinary (depth-matched) Transformers. We organize the looped Transformer into three blocks — begin, middle, and end blocks — where each block itself consists of multiple Transformer layers, and only the middle block is applied recurrently across depth. We augment the looped middle block with *hyper-connections* (Xie et al., 2026), which expand the residual stream into matrix-valued residual streams. Hyper-connections are applied only after each loop, and therefore add minimal new parameters and compute cost. Across various model scales, we find that our *Hyper-Connected Looped Transformer (Hyperloop Transformer)* is able to outperform depth-matched Transformer and mHC Transformer baselines despite using approximately 50% fewer parameters. The outperformance persists through post-training weight quantization, thus positioning Hyperloop Transformers as an attractive architecture for memory-efficient language modeling.

## 1 Introduction

Pushing the Pareto frontier of performance and efficiency is a major goal of modern LLM architecture research. In cloud deployment, efficiency is measured primarily by latency, which depends on both compute and memory movement. Because memory (DRAM) is relatively abundant in such environments, parameter efficiency is typically not the primary concern. This makes parameter-*in*efficient architectures such as mixture-of-experts (MoE; Shazeer et al., 2017) viable for cloud deployment. In contrast, edge and on-device deployments are often constrained not only by compute, but also by the total amount of available memory, which is often orders of magnitude smaller. For example, modern smartphones typically have 8GB–16GB of RAM, and only a portion of this memory is available for deployment of machine learning workloads.

In such settings, a model's memory footprint becomes a major bottleneck, since it directly affects whether a model can be stored and executed at all. Even for cloud deployment, future frontier models may become so large that fitting the full model across a practical number of accelerators becomes challenging, making total parameter memory a key concern. This motivates the study of *parameter-efficient architectures* for language modeling, where the goal is to push the performance-memory frontier for a given compute constraint.

*Looped Transformers* are Transformers that share parameters across depth, and thus enable greater parameter-efficiency than ordinary Transformers. (Other terminology used to describe looped Transformers includes *universal Transformers* (Dehghani et al., 2018; Tan et al., 2023), *recursive Transformers* (Bae et al., 2024; 2025), and *recurrent-depth Transformers* (Geiping et al., 2025; Pappone et al., 2025).) When the number of loops is variable, they have also been shown to overcome certain theoretical limitations of fixed-depth Transformers (Giannou et al., 2023; Yang et al., 2023; Xu and Sato, 2024), and recent empirical work suggests that they can perform particularly well on some real-world reasoning tasks (Geiping et al., 2025; Zhu et al., 2025b). However, when matched for depth, looped Transformers still generally underperform unlooped baselines especially from a perplexity standpoint (Saunshi et al., 2025).

This paper develops a simple looped architecture that outperforms depth-matched Transformer baselines while using approximately half the parameters. Following prior work (Bae et al., 2025), we adopt a "middle cycle" strategy where we organize the Transformer into begin, middle, and end blocks, and only loop the middle block. We then incorporate a variant of *hyper-connections* (Zhu et al., 2025a; Xie et al., 2026), which expand the residual stream into multiple streams, into (only) the looped block. Specifically, we apply hyper-connections at the loop level (i.e., only after each loop iteration) instead of at the layer-level, thus incurring minimal additional parameters and compute. We find that our *Hyper-Connected Looped Transformer (Hyperloop Transformer)* improves the performance-parameter frontier, achieving lower perplexities than depth-matched ordinary Transformers with 240M, 1B, and 2B parameters, despite using 50% fewer parameters. These gains persist through post-training quantization of the model's weights, thus positioning Hyperloop Transformers as an attractive alternative to ordinary Transformers for memory-efficient language modeling.

## 2 Background

### 2.1 Looped Transformers

For a length-T input, a Transformer transforms input representations at layer l, X^(l) ∈ ℝ^(T×C), to obtain the output X^(l+1) ∈ ℝ^(T×C) through an attention layer followed by an MLP layer:

$$\mathbf{H}^{(l)}=\text{Attention}(\mathbf{X}^{(l)};\theta_{\text{attn}}^{(l)})+\mathbf{X}^{(l)},$$
$$\mathbf{X}^{(l+1)}=\text{MLP}(\mathbf{X}^{(l)};\theta_{\text{mlp}}^{(l)})+\mathbf{H}^{(l)}.$$

Here θ^(l)_attn, θ^(l)_MLP are the layer-specific parameters for multiheaded attention and the feedforward layers respectively. (The LayerNorm parameters are absorbed into the attention/MLP layers.) Letting 𝓕_l(·) be the application of a Transformer layer l, an L-layer Transformer then obtains the final output via X^(L) = 𝓕_L(…𝓕_2(𝓕_1(X^(1)))…). Looped Transformers share parameters across depth, e.g., a fully looped model would have X^(L) = 𝓕_1(…𝓕_1(𝓕_1(X^(1)))…). More recent works have shown that a "middle cycle" strategy, which partitions the Transformer layers into beginning, middle, and end blocks (the begin/end layers are also called prelude/coda or encoder/decoder blocks in the literature) and only loops the middle block, is particularly effective (Bae et al., 2025; Saunshi et al., 2025). We also adopt this middle cycle strategy in our architecture.

### 2.2 Hyper-Connected Transformers

As shown above, each layer of a Transformer adds to the C-dimensional *residual stream*. Hyper-connected Transformers (Zhu et al., 2025a) expand the residual stream to an n×C dimensional matrix through "hyper-connections". In the more recent *manifold-constrained hyper-connections* (mHC; Xie et al., 2026), the residual stream at time step t at depth l (given by x_t^(l) ∈ ℝ^C) is expanded by an expansion factor n to yield n parallel residual streams y_t^(l) ∈ ℝ^(n×C). This expanded residual stream is then read from, written to, and mixed using input-dependent projections H_{l,t}^pre, H_{l,t}^post, and H_{l,t}^res. Specifically, the transformations at depth l can be computed as follows:

$$\mathbf{z}_{t}^{(l)}=\operatorname{RMSNorm}(\text{flatten}(\mathbf{y}_{t}^{(l)})),$$
$$\mathbf{H}_{l,t}^{\text{pre}}=\sigma(\alpha_{l}^{\text{pre}}\cdot(\mathbf{W}_{l}^{\text{pre}}\mathbf{z}_{t}^{(l)})+\mathbf{b}_{l}^{\text{pre}}),$$
$$\mathbf{H}_{l,t}^{\text{post}}=2\cdot\sigma(\alpha_{l}^{\text{post}}\cdot(\mathbf{W}_{l}^{\text{post}}\mathbf{z}_{t}^{(l)})+\mathbf{b}_{l}^{\text{post}}),$$
$$\mathbf{H}_{l,t}^{\text{res}}=\text{sinkhorn}(\alpha_{l}^{\text{res}}\cdot\text{reshape}(\mathbf{W}_{l}^{\text{res}}\mathbf{z}_{t}^{(l)})+\mathbf{b}_{l}^{\text{res}}).$$

Here W_l^pre ∈ ℝ^(n×nC), W_l^post ∈ ℝ^(n×nC), W_l^res ∈ ℝ^(n²×nC) are linear projections; α_l^pre, α_l^post, α_l^res ∈ ℝ are learned scalars; b_l^pre ∈ ℝ^n, b_l^post ∈ ℝ^n, b_l^res ∈ ℝ^(n×n) are learned biases; and reshape(·) is an operator that converts an n²-dimensional vector to an n×n matrix. Finally, sinkhorn(·) applies the Sinkhorn-Knopp algorithm, which exponentiates the input and iteratively performs column- and row-normalization, ensuring that H_{l,t}^res is doubly stochastic (i.e., on the Birkhoff polytope) in the limit. Xie et al. (2026) find that 20 Sinkhorn-Knopp iterations are sufficient.

Given the input-dependent matrices H_{l,t}^pre ∈ ℝ^(1×n), H_{l,t}^post ∈ ℝ^(n×1), H_{l,t}^res ∈ ℝ^(n×n) and a sub-layer 𝓕_l ∈ {Attention_l, MLP_l} of a Transformer layer, mHC applies attention/MLP layers in a smaller residual stream of dimension C via (in practice mHC uses different input-dependent matrices for attention and MLP layers):

$$\mathbf{y}^{(l+1)}_{t}=\mathbf{H}_{l,t}^{\text{res}}\mathbf{y}^{(l)}_{t}+\mathbf{H}_{l,t}^{\text{post}}\mathcal{F}_{l}(\mathbf{H}_{l,t}^{\text{pre}}\mathbf{y}_{t}^{(l)}).$$

Thus, mHC Transformers make it possible to work with a larger matrix-valued residual stream without incurring much additional compute (since the compute-heavy attention/MLP layers still work with C-dimensional inputs/outputs).

*Figure 1: (Left) A vanilla middle-cycle looped Transformer architecture with two loops. (Right) A Hyper-Connected Looped Transformer, which uses parallel residual streams that are written to after each loop using hyper-connections.* (Image: https://arxiv.org/html/2604.21254v1/x1.png)

## 3 Hyperloop Transformers

Our architecture, shown in Figure 1, is extremely simple. We partition the Transformer into begin, middle, and end blocks, and then apply (a modification of) hyper-connections at the loop-level when we loop the middle block.

Concretely, let X_begin ∈ ℝ^(T×C) be the residual stream after applying the begin block. We expand this to n parallel streams by simply copying it n times, thus giving Y^(0) ∈ ℝ^(T×n×C), which will serve as input to the hyper-connected looped block. We then compute the input-dependent matrices H_{0,t}^pre, H_{0,t}^post, H_{0,t}^res ∈ ℝ^(n×n) for all {y_t^(0)}_{t=1}^T as above, but using a simpler parameterization of H_{0,t}^res given by:

$$\mathbf{H}_{0,t}^{\text{res}}=\text{diag}(\sigma(\alpha_{0}^{\text{res}}\cdot(\mathbf{W}_{0}^{\text{res}}\mathbf{z}_{t}^{(0)})+\mathbf{b}_{0}^{\text{res}})),$$

where W_0^res is now an n×nC matrix (instead of n²×nC) and b_0^res ∈ ℝ^n.

We use {H_{0,t}^pre}_{t=1}^T on Y^(0) to obtain the C-dimensional input to the middle block, apply the middle block, and then use {H_{0,t}^post}_{t=1}^T to project out into the n×C residual stream. We add a "loop position embedding" e_l ∈ ℝ^C after the middle block, resulting in the recurrence:

$$\mathbf{y}^{(l+1)}_{t}=\mathbf{H}_{l,t}^{\text{res}}\mathbf{y}^{(l)}_{t}+\mathbf{H}_{l,t}^{\text{post}}\left(\mathcal{F}(\mathbf{H}_{l,t}^{\text{pre}}\mathbf{y}_{t}^{(l)})+\mathbf{e}_{l}\right).$$

This process continues for L loops to obtain Y^(L). Finally we average Y^(L) across the parallel streams to obtain X_end ∈ ℝ^(T×C), which is used as input to the end block.

Our approach differs from the original mHC in that (1) we use a simpler parameterization of H_{l,t}^res that substitutes the sinkhorn(·) operator over a dense matrix with a sigmoid over a diagonal matrix (which we found to be sufficient performance-wise while being more efficient), (2) we add a loop position embedding, which, when viewing the architecture as a "depth-wise RNN" with matrix-valued hidden states Y^(0), acts as the input at each time (i.e., loop) step, and (3) we only apply hyper-connections at the loop level, instead of after every attention/MLP layer (so an architecture with 3 loops would have 3 hyper-connections).

Our architecture can also be seen as a more flexible parameterization of looped Transformers, which allows parameters to vary slightly across loop iterations. Concretely, we have loop-specific parameters {W_l^τ, b_l^τ, α_l^τ, e_l} for τ ∈ {pre, post, res} that can vary across loop iterations l. While the number of additional parameters here is still minimal, we posit that this parameterization allows model representations to change in a more flexible manner compared to ordinary looped Transformers which strictly enforce parameter sharing across each loop iteration.

## 4 Empirical Study

### 4.1 Experimental Setup

We train Hyperloop Transformers at various scales along with depth-matched vanilla, looped, and mHC Transformer baselines on the FineWeb-Edu dataset (Lozhkov et al., 2024). All models make use of SwiGLU MLP layers (Shazeer, 2020) and RoPE embeddings (Su et al., 2024). We use 4 parallel residual streams for both the mHC and Hyperloop Transformers. For looped models, we allocate (roughly) 25% of the available parameters to the begin block, 25% of the parameters to the end block, and the remaining 50% to the middle block, which is looped three times. This results in looped models that contain half as many parameters as their depth-matched baselines. We ablate on these choices in our ablation study.

We train models on 2.5× the Chinchilla-optimal token count of the vanilla Transformer corresponding to their size (Hoffmann et al., 2022). We use the Llama-2 tokenizer to tokenize our data and AdamW as our optimizer, with a linear warmup and cosine decay learning rate schedule. Our full hyperparameters can be found in Appendix A. These hyperparameters are generally off-the-shelf hyperparameters that have been found to work well for ordinary Transformers, i.e., we did not do any hyperparameter tuning for our architecture.

### 4.2 Main Results

For perplexity we evaluate our models on a held-out set consisting of 50M tokens from the FineWeb-Edu dataset. These are shown in Table 1. Our results show that while vanilla Looped Transformers can underperform depth-matched Transformer baselines, the Hyperloop Transformer only needs 150–300K extra parameters (compared to the vanilla Looped Transformer) to outperform both looped and non-looped depth-matched baseline models.

While perplexity provides a more robust measure of performance at this scale, we also evaluate our models on downstream tasks. Specifically, we evaluate our models on ARC (Clark et al., 2018), COPA (Gordon et al., 2012), HellaSwag (Zellers et al., 2019), LAMBADA (Paperno et al., 2016), OpenBookQA (Mihaylov et al., 2018), PIQA (Bisk et al., 2020), RACE (Lai et al., 2017), SciQ (Welbl et al., 2017), and WinoGrande (Sakaguchi et al., 2019). Interestingly, we find that the looped Transformer also outperforms the vanilla Transformer on most tasks, despite using 50% fewer parameters and despite underperforming the Transformer model in perplexity terms. This outperformance corroborates similar findings reported in the literature (Saunshi et al., 2025). Hyperloop Transformer outperforms all other baselines overall. Results broken down by task can be found in Appendix B.

**Table 1: Main results of our architecture and baselines pretrained on FineWeb-Edu.** For looped models, [2L → 4L (×3) → 2L] means we have 2 begin layers, 4 middle layers looped 3 times, and 2 end layers. Perplexities are computed with both BF16 and INT4, where we use GPTQ to quantize to INT4. Task accuracies are based on BF16. Training throughput measures tokens/second and is based on eight H100s with NVLink.

| Model | Dim | Unrolled Depth | Train Tokens | Params | PPL (BF16) | PPL (INT4) | Task Acc | Training Toks/s |
|---|---|---|---|---|---|---|---|---|
| Transformer | 1024 | 16 | 12.5B | 238 M | 14.65 | 14.85 | 41.1% | 786K |
| mHC | 1024 | 16 | 12.5B | 241 M | 14.55 | 14.73 | 41.1% | 514K |
| Looped [2L → 4L (×3) → 2L] | 1024 | 16 | 12.5B | 135.5 M | 14.85 | 15.18 | 41.4% | 786K |
| Hyperloop [2L → 4L (×3) → 2L] | 1024 | 16 | 12.5B | 135.7 M | 14.40 | 14.68 | 41.6% | 750K |
| Transformer | 2048 | 18 | 50B | 990.5 M | 10.19 | 10.27 | 48.0% | 367K |
| mHC | 2048 | 18 | 50B | 997.5 M | 10.07 | 10.16 | 48.6% | 237K |
| Looped [3L → 4L (×3) → 3L] | 2048 | 18 | 50B | 579.4 M | 10.02 | 10.24 | 49.2% | 367K |
| Hyperloop [3L → 4L (×3) → 3L] | 2048 | 18 | 50B | 579.7 M | 9.65 | 9.81 | 49.8% | 354K |
| Transformer | 2048 | 38 | 100B | 2018 M | 8.60 | 8.71 | 52.8% | 181K |
| mHC | 2048 | 38 | 100B | 2033 M | 8.57 | 8.62 | 53.7% | 109K |
| Looped [4L → 10L (×3) → 4L] | 2048 | 38 | 100B | 990.5 M | 8.68 | 8.97 | 53.3% | 183K |
| Hyperloop [4L → 10L (×3) → 4L] | 2048 | 38 | 100B | 990.8 M | 8.49 | 8.59 | 54.6% | 180K |

#### Post-training Quantization

Post-training quantization of a model's weights is a standard approach for reducing a model's memory footprint. While looped models are *parameter*-efficient, models that are harder to quantize would be practically *memory*-inefficient. Insofar as models trained with more tokens have been shown to be generally harder to quantize (Huang et al., 2024; Ouyang et al., 2024), it is possible that looped models would also be harder to quantize because the looped layers are trained on "more" inputs. [The arXiv HTML fetch was truncated at this point; the remainder of this paragraph and Sections 4.3–7 (Ablations, Analysis, Discussion, Related Work, Conclusion) and the appendices are not included here. See the full paper at https://arxiv.org/abs/2604.21254 or https://arxiv.org/pdf/2604.21254.]
