---
title: "Reasoning with Latent Thoughts: On the Power of Looped Transformers"
arxiv_id: "2502.17416"
url: "https://arxiv.org/abs/2502.17416"
pdf: "https://arxiv.org/pdf/2502.17416"
authors: "Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, Sashank J. Reddi"
date: "2025-02-24"
source: "html-fulltext"
---

# Reasoning with Latent Thoughts: On the Power of Looped Transformers

**Authors:** Nikunj Saunshi (Google Research), Nishanth Dikkala (Google Research), Zhiyuan Li (Google Research; Toyota Technological Institute at Chicago), Sanjiv Kumar (Google Research), Sashank J. Reddi (Google Research)
**arXiv:** 2502.17416 — https://arxiv.org/abs/2502.17416

## Abstract

Large language models have shown remarkable reasoning abilities and scaling laws suggest that large parameter count, especially along the depth axis, is the primary driver. In this work, we make a stronger claim — many reasoning problems require a large depth but not necessarily many parameters. This unlocks a novel application of looped models for reasoning.

Firstly, we show that for many synthetic reasoning problems like addition, p-hop induction, and math problems, a k-layer transformer looped L times nearly matches the performance of a kL-layer non-looped model, and is significantly better than a k-layer model. This is further corroborated by theoretical results showing that many such reasoning problems can be solved via iterative algorithms, and thus, can be solved effectively using looped models with nearly optimal depth.

Perhaps surprisingly, these benefits also translate to practical settings of language modeling — on many downstream reasoning tasks, a language model with k-layers looped L times can be competitive to, if not better than, a kL-layer language model. In fact, our empirical analysis reveals an intriguing phenomenon: looped and non-looped models exhibit scaling behavior that depends on their effective depth, akin to the inference-time scaling of chain-of-thought (CoT) reasoning.

We further elucidate the connection to CoT reasoning by proving that looped models implicitly generate *latent thoughts* and can simulate T steps of CoT with T loops. Inspired by these findings, we also present an interesting dichotomy between reasoning and memorization, and design a looping-based regularization that is effective on both fronts.

## 1 Introduction

Language models have shown a lot of promise in solving problems that require strong reasoning abilities like math, coding, common sense reasoning and logical puzzles (Brown et al., 2020; Team et al., 2023). This has sparked interest in developing techniques to improve reasoning on harder problems (Wei et al., 2022b) and has inspired theoretical studies on how Transformers are able to perform reasoning (Feng et al., 2024; Sanford et al., 2024a). Reasoning abilities are often emergent in larger language models (Wei et al., 2022a) — this aligns with various scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022; Allen-Zhu & Li, 2024) that show that the performance of language models is very strongly dependent on the model size (i.e., number of parameters) and much lesser on other architectural design choices.

However, recent works have started to question this view. Ye et al. (2024) argue that scaling laws for reasoning are more subtle, and depth is very important in addition to parameter count — at the same parameter count, deeper but shallower models are better. This is a deviation from the conventional scaling law wisdom, but it intuitively makes sense because reasoning problems often require multi-step compositional thinking, and thus, depth can play a crucial role.

In this work, we make a stronger claim — while depth is important, many reasoning problems do not necessarily require a lot of parameters. How does one solve reasoning problems with large depth but few parameters? We argue that looped models are perfectly suited for this, where the same function, parameterized with few parameters, is iteratively applied on the input. This leads us to our first important claim:

> **Claim 1:** Many reasoning problems require depth but not necessarily parameters. That is, they can be solved via looped models.

Looped models have been studied in the literature for parameter efficiency (Lan et al., 2020), adaptive compute (Dehghani et al., 2018), equilibrium models (Bai et al., 2019) and for in-context learning (Yang et al., 2023; Gatmiry et al., 2024a). In this work, we initiate the study of looped models in the context of reasoning. Admittedly, reasoning is not very well-defined and can be of various forms (Sun et al., 2023). Acknowledging this hurdle, we focus on a non-exhaustive list of problems that intuitively require reasoning and that are inspired by reasoning benchmarks.

Throughout the paper, we use the notation **(k ⊗ L)** to denote a k-layer model looped L times (precise definition in Section 2), which has the same number of parameters as a (k ⊗ 1) model and same flops as a (kL ⊗ 1) non-looped model (see Figure 1). As a first step towards connecting looped models and reasoning, we empirically evaluate looped models on several simple reasoning tasks in the literature (Section 2). Perhaps surprisingly, we find that a (k ⊗ L) looped model does almost as well as, if not better than, a non-looped model (kL ⊗ 1) that has the same effective depth but L times more parameters on these reasoning tasks. The looped model is also significantly better than a (k ⊗ 1) model which has the same number of parameters. Our theoretical results on the expressiveness of looped models in representing iterative algorithms with short description further corroborate these empirical findings and provide strong support for our claim. This naturally raises an important question: do looped models benefit language modeling in a similar manner?

**Figure 1:** Illustration of the simple and architecture-agnostic looping mechanism considered. A k-layer block looped L times (middle) is denoted by (k ⊗ L), which can essentially be viewed as a weight-shared model. The iso-param baseline, (k ⊗ 1), is a k-layer model with the same number of *distinct* parameters. The iso-FLOP baseline, (kL ⊗ 1), is a kL-layer model with the same depth but L times more parameters. Middle looping is a strategy inspired by prior works on model stacking (e.g. Saunshi et al., 2024).

> **Claim 2:** For language modeling, looped models have an inductive bias towards good reasoning despite having worse perplexity and memorization compared to an iso-flop non-looped model.

For the above claim, we again train a (k ⊗ L) looped model on causal language modeling and compare it to the iso-param (k ⊗ 1) and iso-flop (kL ⊗ 1) non-looped baselines. While the looped model improves over the iso-param baseline, perhaps unsurprisingly, it ends up with worse perplexity than the iso-flop baseline, since perplexity depends strongly on number of parameters. However, the downstream evaluations reveal an intriguing trend: looped models have a tendency to improve tasks that require reasoning a lot more than memorization tasks. Specifically, the looped model has reasoning performance much closer to the iso-flop baseline, sometimes even exceeding it despite having L times fewer parameters and worse perplexity.

This contrasting behavior between the pretraining and downstream metrics has been a subject of study lately (Saunshi et al., 2022; Liu et al., 2023) and is attributed to the *inductive biases* introduced by different architectures and training algorithms. Our empirical analysis also uncovers an interesting phenomenon: accuracy on downstream tasks scales as a function of the effective depth of the model, in a manner akin to the inference-time scaling of chain-of-thought reasoning.

---

## Paper structure (section outline)

The full paper continues with the following sections:

- **2 Looped models on simple reasoning tasks** — including 2.1 Experiments with simple reasoning problems (p-hop induction; i-GSM synthetic grade-school math problems).
- **3 Language modeling with looped models** — 3.1 Experiments with 1B language modeling; 3.2 Inductive bias towards reasoning; 3.3 Middle looping variant and relationship with gradual stacking; 3.4 Scaling behavior of looping (latent thoughts and connections to CoT reasoning).
- **4 Looping-inspired regularization.**
- **5 Theoretical analysis for looped models** — 5.1 Preliminaries and notations; 5.2 Group composition problem; 5.3 Looped models can simulate non-looped models; 5.4 Looped models can simulate CoT reasoning.
- **6 Related work.**
- **7 Conclusions, limitations and future work.**
- **Appendix A Experiments** — A.1 Simple reasoning setup details (n-ary addition, p-hop induction, i-GSM); A.2 Language modeling setup; A.3 Results for each task group.
- **Appendix B Theoretical results** — B.1 Detailed notations (multi-head self-attention, feed-forward network, finite-precision modeling); B.2 Proofs (B.2.1 Looped models can simulate non-looped models; B.2.2 Group composition); B.3 Connection to chain-of-thought reasoning.

> Note: This markdown was extracted from the arXiv HTML rendering (v1). The abstract and Section 1 (Introduction) are captured in full above. The remaining sections (2–7 and appendices) are summarized by their section outline; for the complete text, figures, tables, and proofs, see the PDF at https://arxiv.org/pdf/2502.17416.
