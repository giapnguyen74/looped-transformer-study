---
title: "Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA"
arxiv_id: "2410.20672"
url: "https://arxiv.org/abs/2410.20672"
pdf: "https://arxiv.org/pdf/2410.20672"
authors: Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, Tal Schuster
date: 2024-10-28
source: "html-fulltext"
---

# Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA

**Authors:** Sangmin Bae (KAIST AI; work done during an internship at Google DeepMind), Adam Fisch (Google DeepMind), Hrayr Harutyunyan (Google Research), Ziwei Ji (Google Research), Seungyeon Kim (Google DeepMind), Tal Schuster (Google DeepMind; corresponding author)
**arXiv:** 2410.20672 — https://arxiv.org/abs/2410.20672

> Note: This markdown was extracted from the arXiv HTML rendering (v1). The HTML body is heavily interleaved with MathML markup; the prose below is the cleanly extractable narrative covering the abstract, introduction, contributions, and the opening of the methods section. Equation-dense passages and the later experimental sections, related work, discussion, and appendices are best read in the original PDF.

## Abstract

Large language models (LLMs) are expensive to deploy. Parameter sharing offers a possible path towards reducing their size and cost, but its effectiveness in modern LLMs remains fairly limited. In this work, we revisit "layer tying" as a form of parameter sharing in Transformers, and introduce novel methods for converting existing LLMs into smaller "Recursive Transformers" that share parameters across layers, with minimal loss of performance. Here, our Recursive Transformers are efficiently initialized from standard pretrained Transformers, but only use a single block of unique layers that is then repeated multiple times in a loop. We further improve performance by introducing Relaxed Recursive Transformers that add flexibility to the layer tying constraint via depth-wise low-rank adaptation (LoRA) modules, yet still preserve the compactness of the overall model. We show that our recursive models (e.g., recursive Gemma 1B) outperform both similar-sized vanilla pretrained models (such as TinyLlama 1.1B and Pythia 1B) and knowledge distillation baselines—and can even recover most of the performance of the original "full-size" model (e.g., Gemma 2B with no shared parameters). Finally, we propose Continuous Depth-wise Batching, a promising new inference paradigm enabled by the Recursive Transformer when paired with early exiting. In a theoretical analysis, we show that this has the potential to lead to significant (2-3×) gains in inference throughput.

## 1 Introduction

Efficient deployment of large language models (LLMs) demands a balance between performance and resources. While larger models with more parameters consistently demonstrate superior performance, their substantial memory and computational demands are expensive. Parameter sharing approaches, wherein weights are reused across model layers, can lower these costs by reducing memory footprint, and thereby allow for the use of fewer (or lower-grade) accelerators, or larger batch sizes for better throughput. While parameter sharing has shown encouraging capabilities in previous work, its application to modern LLMs has yielded limited reported success.

In this work, we revisit parameter sharing for LLMs, and propose novel methodologies to *convert* existing, unshared models into smaller, and more efficient, Recursive Transformers. These models use a single block of unique layers that are recursively reused across multiple loops, yet still achieve impressive performance relative to their reduced size. To mitigate the potential performance degradation associated with parameter sharing, we first initialize the shared block of layers based on the original model's pre-trained parameters, and then finetune the resulting recursive model for a limited number of "uptraining" steps. Importantly, we show that our initialization strategies allow us to achieve strong performance with minimal training time. This is aligned with observations that model compression techniques such as layer skipping, pruning, or nesting can preserve surprisingly high performance—further motivating our approach of compressing models to more compact yet performant architectures (here, repeated layers with low-rank adapters).

We further propose the Relaxed Recursive Transformer, an extension of the Recursive Transformer in which the weight tying across repeated layer blocks is slightly relaxed through the incorporation of multiple layer-specific, low-rank adaptation (LoRA) modules. Despite its simplicity, this strategy offers several non-trivial advantages. First, it allows for low-rank deltas between shared layers, while only adding minimal overhead. Second, the rank of the LoRA matrices can be adjusted to control the degree of relaxation, which directly influences model capacity. Furthermore, since the relaxed model has the same overall shape as the original Transformer, we can efficiently initialize LoRA modules via truncated Singular Value Decomposition on the residual matrices between the original layer weights and the shared layer weights. Hence, the rank values serve as a pivotal hyperparameter, enabling the Relaxed Recursive Transformer to seamlessly transition between the two extremes of the vanilla and Recursive Transformer architectures.

While the primary focus of this paper lies in how to formulate and train Recursive Transformers, we also highlight their potential to achieve significant throughput gains via a new batched inference paradigm, Continuous Depth-wise Batching, that their recursive nature enables. Prior work introduced continuous sequence-wise batching, which leverages the fact that the computation performed to compute a new token is functionally the same (and uses the same model parameters) regardless of the token position within the sequence. This allows new requests to be continuously scheduled when slots within a batch become available. For example, when one response is completed, the start of the next response to be formed can immediately take the finished response's place in the batch, without waiting for the rest of the batch responses that might be longer. In our Recursive Transformer, parameter sharing occurs not only across different timesteps, but also across different depths (loop iterations). This enables an extra dimension of dynamic grouping: jointly computing different iterations of the looped layer blocks per individual responses within the same batch.

*Figure 1: Overview of the conversion from a vanilla N-layer Transformer to a Recursive Transformer with N/K blocks of K shared layers. The Recursive Transformer is obtained by repeating a single block of K layers multiple times, resulting in a looped architecture. The Recursive Transformer can also be converted into a Relaxed Recursive Transformer by adding layer-specific LoRA modules. This preserves many of the advantages of weight sharing, but also allows for better performance.*

Our key contributions are as follows:

- We introduce a framework for initializing and training Relaxed Recursive Transformers and demonstrate strong performance compared to non-recursive models of comparable size. For example, when we uptrained a recursive Gemma 1B model converted from a pretrained Gemma 2B, we observed up to 13.5 absolute accuracy improvement (22% error reduction) on few-shot tasks compared to a non-recursive Gemma 1B model (pretrained from scratch). Furthermore, we show that by incorporating knowledge distillation, our recursive Gemma model, uptrained on 60 billion tokens, achieves performance on par with the full-size Gemma model trained on a massive 3 trillion token corpus (see §3.3).

- Based on our Relaxed Recursive Transformer, we also evaluate a key use case for continuous depth-wise batching with early-exiting, which opportunistically makes predictions for samples with high confidence at earlier stages. From our simulation, Early Exits reveal a substantial throughput improvement of up to 2-3× compared to a vanilla Transformer with the same architecture. Notably, the recursive Gemma model, which outperforms the vanilla Pythia model, can theoretically achieve a nearly 4× increase in throughput (see §3.8).

## 2 Effective Model Compression with Recursive Patterns

In this section, we present the main details of our method for converting a vanilla Transformer model into a parameter-shared model that outperforms models of equivalent size. We first provide a short overview of the Transformer architecture (§2.1). Then, we introduce the Recursive Transformer and present effective techniques to initialize its looped layers by leveraging the weights of the original pretrained model (§2.2). In §2.3, we relax the parameter-sharing constraint in the model design, and add a limited set of layer-specific parameters to further improve the model's accuracy while maintaining compact representations. Finally, we show how, beyond reduced memory, Recursive Transformers readily support further throughput optimizations via a novel inference paradigm (§2.4).

### 2.1 Basic Transformer Architecture

Large language models typically leverage the Transformer architecture. A Transformer consists of L layers, where the hidden states at each time step t are computed by running through the series of layers:

> h_t^ℓ = f(h_t^(ℓ−1); Φ_ℓ), ℓ ∈ [1, L],   (Eq. 1)

with h_t^0 representing the embedding of the token y_(t−1) from the previous time step, and Φ_ℓ denoting the trainable parameters of the ℓ-th layer.

Each layer has two core components: a multi-head attention (MHA) mechanism and a feed-forward network (FFN). MHA employs multiple attention heads to capture diverse relationships within the input sequence via linear attention weights and scaled dot-product attention mechanisms. The FFN structure typically consists of two linear transformations, but different models exhibit distinct structural variations. (See Appendix A for further details.)

### 2.2 Recursive Transformer: Looped Layer Tying

In this work, we revisit parameter sharing in the context of LLMs and propose the Recursive Transformer architecture. Among various looping strategies (refer to Appendix B), we specifically adopt the CYCLE strategy (Takase and Kiyono, 2023) for Recursive Transformers, wherein a single block of unique layers is recursively reused. This inherent design aligns seamlessly with early-exiting mechanisms, potentially offering substantial speedup. The model's hidden states are computed as:

> h_t^ℓ = f(h_t^(ℓ−1); Φ′_(((ℓ−1) mod L/B) + 1)), ℓ ∈ [1, L],   (Eq. 2)

where the looped block of layers is reused across the depth of the model.

---

*The remaining content of the paper — the full methods detail (§2.3 Relaxed Recursive Transformer via Multi-LoRA Layers, §2.4 Continuous Depth-wise Batching and Early-Exiting), all of §3 Experiments, §4 Related Work, §5 Discussion and Future Work, and Appendices A–K — is equation- and table-heavy and was not cleanly extractable from the arXiv HTML rendering. Refer to the original PDF for those sections: https://arxiv.org/pdf/2410.20672*
