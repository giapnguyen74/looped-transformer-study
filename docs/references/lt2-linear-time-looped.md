---
title: "LT2: Linear-Time Looped Transformers"
arxiv_id: "2605.20670"
url: "https://arxiv.org/abs/2605.20670"
pdf: "https://arxiv.org/pdf/2605.20670"
authors: "Chunyuan Deng, Yizhe Zhang, Rui-jie Zhu, Yuanyuan Xu, Jiarui Liu, T. S. Eugene Ng, Hanjie Chen"
date: "2026-05-20"
source: "html-fulltext"
---

# LT2: Linear-Time Looped Transformers

**Authors:** Chunyuan Deng (Rice University), Yizhe Zhang (Apple), Rui-jie Zhu (UC Santa Cruz), Yuanyuan Xu (Rice University), Jiarui Liu (Carnegie Mellon University), T. S. Eugene Ng (Rice University), Hanjie Chen (Rice University)
**arXiv:** 2605.20670 — https://arxiv.org/abs/2605.20670

- Codebase: https://github.com/chili-lab/LT2
- Huggingface Checkpoints: https://huggingface.co/chili-lab/Ouro-hybrid-1.4B

> **Note:** The arXiv HTML source captured for this document was truncated by the fetch and ends mid-equation in Section 2.2 (after equation 4). Sections 2.2 (remainder), 3 (Experiments), 4 (Distillation), 5 (Related Work), and 6 (Conclusion), along with the References and Appendices, were not present in the captured HTML and are therefore not reproduced below. See the arXiv links above for the complete paper.

## Abstract

Looped Transformers (LT) have emerged as a powerful architecture by iterating their layers multiple times before decoding the final token. However, their pairing with full attention retains quadratic complexity making it computationally expensive and slow. We introduce LT2 (Linear-Time Looped Transformers), a family of looped architectures that replace quadratic softmax attention with subquadratic attention with linear-time complexity. We study two variants: LT2-linear with linear attention and LT2-sparse with sparse attention. We find looping uniquely synergizes with these variants: it enables iterative memory refinement in linear attention and progressively expands the effective receptive field in sparse attention. We formalize these benefits theoretically and demonstrate consistent empirical gains across controlled recall, state-tracking, and language modeling tasks. We then explore LT2-hybrid, a hybrid architecture that combines different attention variants in a looped setting. We find two architectural variants promising: (1) LT2-hybrid (GDN+DSA), which interleaves linear and sparse attention to maximize efficiency, matching the standard looped transformer's quality at fully linear-time cost; and (2) LT2-hybrid (Full+GDN), which interleaves GDN with a small fraction of full attention layers to maximize quality, surpassing the standard looped transformer in both performance and efficiency. Furthermore, we also show how to turn a pre-trained LT into an LT2-hybrid model. With only about 1B tokens of training, our converted model (Ouro-hybrid-1.4B) outperforms industry-level 1B models and is competitive with industry-level 4B models while keeping the speed benefits of linear-time attention. Together, these two directions show a clear path to making looped transformers a more scalable architecture for language modeling and advancing the development of efficient, capable small language models.

![Figure 1](https://arxiv.org/html/2605.20670v1/x1.png)

*Figure 1: (Left) New parameter-efficiency frontier introduced by LT2. (Right) Converted LT2-Hybrid outperforms similarly sized industry-level 1B while matching 4B ones.*

## 1 Introduction

Scaling neural language models along the parameter axis has driven much of modern NLP's progress ([brown2020languagemodelsfewshotlearning]; [kaplan2020scalinglawsneurallanguage]; [hoffmann2022trainingcomputeoptimallargelanguage]). A complementary axis—scaling depth via weight-shared recurrence—has recently emerged as a promising alternative. These architectures, often called looped transformers (LT, originally Universal Transformers [dehghani2019universaltransformers]), reuse the same weights across multiple steps before decoding the final prediction token [giannou2023looped]; [yanglooped]; [zhu2025scalinglatentreasoninglooped]. In effect, repeated computation becomes effective depth: the model performs several rounds of latent computation while keeping the unique parameter count fixed, making looped transformers an appealing approach to parameter-efficient reasoning.

![Figure 2](https://arxiv.org/html/2605.20670v1/x2.png)

*Figure 2: Attention FLOPs and inference cache memory vs. sequence length for a $1.3\text{B}$ model.*

However, current looped transformers scale poorly because each loop has to re-apply quadratic full attention over the entire sequence repeatedly. Its cost and inference-time storage therefore grow with sequence length, and compound with each loop iteration. As a result, even though parameters are reused, training-time attention FLOPs and inference-time KV-cache usage scale poorly with the number of loops, making attention the dominant bottleneck in scaling looped transformers [tay-etal-2023-scaling]; [zhu2025scalinglatentreasoninglooped]. As Figure 2 shows, processing every token through attention for $T$ iterations causes both training-time attention FLOPs and inference-time KV-cache memory to grow substantially. At long contexts the quadratic attention term dominates, and adding loop steps quickly becomes impractical [zhu2025scalinglatentreasoninglooped].

We introduce LT2 (Linear-Time Looped Transformers), a family of looped architectures that replace quadratic softmax attention with subquadratic token-mixing primitives. We primarily study two distinct variants, LT2-linear and LT2-sparse, which replace the quadratic attention with linear attention [katharopoulos2020transformers]; [yang2024parallelizing]; [kimiteam2025kimilinearexpressiveefficient] and sparse attention [xiao2024efficientstreaminglanguagemodels]; [deepseekai2025deepseekv32pushingfrontieropen], respectively. We show that looped operation can turn compute into context: it enables finer-grained control over recurrent memory in linear attention and enlarges the receptive field in sparse attention; we provide intuition in § 2.2 and a detailed theoretical analysis in Appendix B.1. Furthermore, we explore LT2-hybrid, a hybrid architecture that pushes the performance–efficiency frontier to a new level by mixing different attention variants in the looped setting. We demonstrate that LT2-hybrid (GDN ([yang2024gated]) + DSA ([deepseekai2025deepseekv32pushingfrontieropen]))—which combines linear and sparse attention within a looped setting—matches the standard looped transformer's quality (59.3% avg. zero-shot) while delivering $\sim$5.7$\times$ higher decode throughput at 8k context (125 vs. 22 tokens/s, batch size 8), entirely without quadratic attention. LT2-hybrid (Full + GDN), which interleaves GDN with a small fraction of full-attention layers, goes further: it improves average zero-shot performance by +2.1 points over the standard looped transformer (61.4% vs. 59.3%) while still achieving $\sim$5$\times$ higher decode throughput at the same setting, and consistently outperforms the standard looped transformer across language modeling, recall, state-tracking, and efficiency benchmarks (§ 3).

Finally, we explore distilling a pretrained looped transformer (specifically, Ouro ([zhu2025scalinglatentreasoninglooped])) into an LT2 model. As shown in Figure 1 (right), with only $\sim$1B tokens of continued training, our converted Ouro-Hybrid-1.4B retains the quality of its full-attention teacher while inheriting LT2's linear-time efficiency. The resulting model is competitive with industry-level open-source models in the 1B–4B parameter range across standard zero-shot benchmarks, matching or exceeding 1B-class baselines and approaching 3B–4B models on several tasks. This demonstrates that practitioners need not retrain from scratch: existing looped transformers can be efficiently converted into linear-time variants, lowering the cost barrier to adopting the LT2 family models.

## 2 LT2: Linear-Time Looped Transformer

### 2.1 Architecture

**Looped Transformer (LT).** Let $L$ denote sequence length and $d$ the hidden dimension; we write the hidden-state sequence as $\mathbf{h}\in\mathbb{R}^{L\times d}$ and the state at position $t$ as $\mathbf{h}_{t}\in\mathbb{R}^{d}$. A standard Transformer of depth $N$ stacks $N$ independently-parameterized blocks $\{\mathcal{F}_{\ell}\}_{\ell=1}^{N}$, each consisting of a token mixer and a position-wise FFN with residual connections:

$$\mathcal{F}_{\ell}(\mathbf{h})=\mathbf{h}^{\prime}+\mathrm{FFN}_{\ell}(\mathbf{h}^{\prime}),\qquad\mathbf{h}^{\prime}=\mathbf{h}+\mathrm{MHA}_{\ell}(\mathbf{h}),$$ (1)

where $\mathrm{MHA}_{\ell}$ is multi-head self-attention (we omit pre-norm for brevity). A *Looped Transformer* (LT) reuses these $N$ shared blocks for $T$ iterations:

$$\mathbf{h}^{(0)}=\mathrm{Emb}(\mathbf{x}),\quad\mathbf{h}^{(\tau)}=\bigl(\mathcal{F}_{N}\circ\cdots\circ\mathcal{F}_{1}\bigr)\!\bigl(\mathbf{h}^{(\tau-1)}\bigr),\quad\tau=1,\dots,T,\quad\hat{\mathbf{y}}=\mathrm{Dec}\!\bigl(\mathbf{h}^{(T)}\bigr),$$ (2)

yielding effective depth $T\cdot N$ with only $N$ unique parameter sets—a $T\times$ parameter reduction over a Transformer of equivalent depth. Following Ouro ([zhu2025scalinglatentreasoninglooped]), we use a fixed $T$ throughout pre-training and we discuss adaptive computation time in the Appendix A.

**LT2.** LT2 simply replaces the MHA sub-layer in Eq. (1) with a subquadratic token mixer, so each shared block becomes

$$\mathcal{F}_{\ell}(\mathbf{h})=\mathbf{h}^{\prime}+\mathrm{FFN}_{\ell}(\mathbf{h}^{\prime}),\qquad\mathbf{h}^{\prime}=\mathbf{h}+\mathrm{LinearMixer}_{\ell}(\mathbf{h}),$$ (3)

where $\mathrm{LinearMixer}_{\ell}$ is any linear- or sparse-attention primitive in Table 1. Throughout, $\mathbf{q}_{t},\mathbf{k}_{t}\!\in\!\mathbb{R}^{d_{k}}$ and $\mathbf{v}_{t}\!\in\!\mathbb{R}^{d_{v}}$ denote the query/key/value projections of $\mathbf{h}_{t}$; $\mathbf{S}_{t}\!\in\!\mathbb{R}^{d_{k}\times d_{v}}$ is the recurrent state of a linear-attention mixer. We additionally insert a zero-initialized, per-channel learned gate $\boldsymbol{\rho}_{\tau}\!\in\!\mathbb{R}^{d}$ as a residual across loop iterations, $\mathbf{h}^{(\tau)}=\widetilde{\mathbf{h}}^{(\tau)}+\boldsymbol{\rho}_{\tau}\odot\mathbf{h}^{(\tau-1)}$, where $\widetilde{\mathbf{h}}^{(\tau)}$ is the output of the looped block stack at iteration $\tau$ (i.e., $\widetilde{\mathbf{h}}^{(\tau)}=(\mathcal{F}_{N}\circ\cdots\circ\mathcal{F}_{1})(\mathbf{h}^{(\tau-1)})$). Thus our setup includes two levels of residual connections: a traditional per-block identity residual connection and a learned per-loop residual.

*Table 1: Token mixers supported by LT2. Train FLOPs are reported per layer for a sequence of length $L$; cache/state memory is per layer at inference. $w$ denotes the sparse-attention window/budget size with $w\ll L$.*

| Family | Mixer | State update rule | Train FLOPs | Cache / State mem. |
| --- | --- | --- | --- | --- |
| Full attn. | Softmax MHA | $(\mathbf{K}_{t},\mathbf{V}_{t})=\bigl([\mathbf{K}_{t-1};\mathbf{k}_{t}],\,[\mathbf{V}_{t-1};\mathbf{v}_{t}]\bigr)$ | $\mathcal{O}(L^{2}d)$ | $\mathcal{O}(Ld)$ |
| Linear attn. (LT2-LA) | LA ([katharopoulos2020transformers]) | $\mathbf{S}_{t}=\mathbf{S}_{t-1}+\mathbf{k}_{t}\mathbf{v}_{t}^{\!\top}$ | $\mathcal{O}(L\,d_{k}d_{v})$ | $\mathcal{O}(d_{k}d_{v})$ |
| | RetNet ([sun2023retentive]) | $\mathbf{S}_{t}=\gamma\,\mathbf{S}_{t-1}+\mathbf{k}_{t}\mathbf{v}_{t}^{\!\top}$ | $\mathcal{O}(L\,d_{k}d_{v})$ | $\mathcal{O}(d_{k}d_{v})$ |
| | Mamba2 ([dao2024transformersssmsgeneralizedmodels]) | $\mathbf{S}_{t}=\alpha_{t}\,\mathbf{S}_{t-1}+\mathbf{k}_{t}\mathbf{v}_{t}^{\!\top}$ | $\mathcal{O}(L\,d_{k}d_{v})$ | $\mathcal{O}(d_{k}d_{v})$ |
| | GLA ([yanggated]) | $\mathbf{S}_{t}=\mathrm{Diag}(\boldsymbol{\alpha}_{t})\mathbf{S}_{t-1}+\mathbf{k}_{t}\mathbf{v}_{t}^{\!\top}$ | $\mathcal{O}(L\,d_{k}d_{v})$ | $\mathcal{O}(d_{k}d_{v})$ |
| | HGRN2 ([qin2024hgrn2]) | $\mathbf{S}_{t}=\mathrm{Diag}(\boldsymbol{\alpha}_{t})\mathbf{S}_{t-1}+\bigl(\mathbf{1}-\boldsymbol{\alpha}_{t}\bigr)\mathbf{v}_{t}^{\!\top}$ | $\mathcal{O}(L\,d_{k}d_{v})$ | $\mathcal{O}(d_{k}d_{v})$ |
| | DeltaNet ([schlag2021linear]; [yang2024parallelizing]) | $\mathbf{S}_{t}=\bigl(\mathbf{I}-\beta_{t}\mathbf{k}_{t}\mathbf{k}_{t}^{\!\top}\bigr)\mathbf{S}_{t-1}+\beta_{t}\mathbf{k}_{t}\mathbf{v}_{t}^{\!\top}$ | $\mathcal{O}(L\,d_{k}d_{v})$ | $\mathcal{O}(d_{k}d_{v})$ |
| | GDN ([yang2024gated]) | $\mathbf{S}_{t}=\alpha_{t}\,\bigl(\mathbf{I}-\beta_{t}\mathbf{k}_{t}\mathbf{k}_{t}^{\!\top}\bigr)\mathbf{S}_{t-1}+\beta_{t}\mathbf{k}_{t}\mathbf{v}_{t}^{\!\top}$ | $\mathcal{O}(L\,d_{k}d_{v})$ | $\mathcal{O}(d_{k}d_{v})$ |
| | KDA ([kimiteam2025kimilinearexpressiveefficient]) | $\mathbf{S}_{t}=\bigl(\mathbf{I}-\beta_{t}\mathbf{k}_{t}\mathbf{k}_{t}^{\!\top}\bigr)\mathrm{Diag}(\boldsymbol{\alpha}_{t})\mathbf{S}_{t-1}+\beta_{t}\mathbf{k}_{t}\mathbf{v}_{t}^{\!\top}$ | $\mathcal{O}(L\,d_{k}d_{v})$ | $\mathcal{O}(d_{k}d_{v})$ |
| Sparse attn. (LT2-SA) | Window | $(\mathbf{K}_{t},\mathbf{V}_{t})=(\mathbf{K}_{[t-w:t]},\mathbf{V}_{[t-w:t]})$ (sliding cache) | $\mathcal{O}(L\,w\,d)$ | $\mathcal{O}(w\,d)$ |
| | NSA ([yuan2025nativesparseattentionhardwarealigned]) | KV cache + compressed blocks; $\mathcal{I}_{t}$: top-$w$ selected indices | $\mathcal{O}(L\,w\,d)$ | $\mathcal{O}(L\,d)$ |
| | DSA ([deepseekai2025deepseekv32pushingfrontieropen]) | KV cache; $\mathcal{I}_{t}$: top-$w$ via lightning indexer | $\mathcal{O}(L\,w\,d)$ | $\mathcal{O}(L\,d)$ |

### 2.2 Beyond Efficiency: Benefits of Looping

Subquadratic attention provides clear efficiency gains. A more interesting question is what looping adds to these attention variants. We make two claims: with $T$ loop iterations, a diagonal-plus-low-rank (DPLR) linear-attention block turns its rank-$1$ state update into a rank-$T$ update, and a sliding-window block turns its window of size $w$ into an effective receptive field of size $Tw$.

#### Loop $\times$ DPLR linear attention: rank-$T$ update on recurrent memory.

Frontier linear-attention architectures now use DPLR mixers, e.g. GDN ([yang2024gated]), KDA ([kimiteam2025kimilinearexpressiveefficient]), and RWKV7 ([peng2025rwkv7gooseexpressivedynamic]). We take KDA as our running example, which maintains a recurrent state $\mathbf{S}_{t}\!\in\!\mathbb{R}^{d_{k}\times d_{v}}$ at sequence position $t$ via

$$\mathbf{S}_{t}=\mathbf{A}_{t}\,\mathbf{S}_{t-1}+\beta_{t}\,\mathbf{k}_{t}\mathbf{v}_{t}^{\top},\qquad\mathbf{A}_{t}=\mathrm{Diag}(\boldsymbol{\alpha}_{t})\bigl(\mathbf{I}-\beta_{t}\,\mathbf{k}_{t}\mathbf{k}_{t}^{\top}\bigr),$$ (4)

<!-- The arXiv HTML capture ends here, mid-Section 2.2. Remaining sections (2.2 cont., 3 Experiments, 4 Distillation, 5 Related Work, 6 Conclusion), References, and Appendices were not present in the fetched source. See https://arxiv.org/abs/2605.20670 for the full paper. -->
