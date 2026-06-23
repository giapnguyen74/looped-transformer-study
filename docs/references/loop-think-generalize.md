---
title: "Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers"
arxiv_id: "2604.07822"
url: "https://arxiv.org/abs/2604.07822"
pdf: "https://arxiv.org/pdf/2604.07822"
authors: Harsh Kohli, Srinivasan Parthasarathy, Huan Sun, Yuekun Yao (The Ohio State University)
date: 2026-04-09
source: "html-fulltext"
---

# Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers

**Authors:** Harsh Kohli, Srinivasan Parthasarathy, Huan Sun, Yuekun Yao — The Ohio State University
**arXiv:** 2604.07822 — https://arxiv.org/abs/2604.07822
**Code:** https://github.com/OSU-NLP-Group/Loop-Think-Generalize
**License:** CC BY 4.0

## Abstract

We study implicit reasoning, i.e. the ability to combine knowledge or rules within a single forward pass. While transformer-based large language models store substantial factual knowledge and rules, they often fail to compose this knowledge for implicit multi-hop reasoning, suggesting a lack of compositional generalization over their parametric knowledge. To address this limitation, we study recurrent-depth transformers, which enables iterative computation over the same transformer layers. We investigate two compositional generalization challenges under the implicit reasoning scenario: systematic generalization, i.e. combining knowledge that is never used for compositions during training, and depth extrapolation, i.e. generalizing from limited reasoning depth (e.g. training on up to 5-hop) to deeper compositions (e.g. 10-hop). Through controlled studies with models trained from scratch, we show that while vanilla transformers struggle with both generalization challenges, recurrent-depth transformers can effectively make such generalization. For systematic generalization, we find that this ability emerges through a three-stage grokking process, transitioning from memorization to in-distribution generalization and finally to systematic generalization, supported by mechanistic analysis. For depth extrapolation, we show that generalization beyond training depth can be unlocked by scaling inference-time recurrence, with more iterations enabling deeper reasoning. We further study how training strategies affect extrapolation, providing guidance on training recurrent-depth transformers, and identify a key limitation, overthinking, where excessive recurrence degrades predictions and limits generalization to very deep compositions.

## 1 Introduction

Large language models (LLMs) (Brown et al., 2020) are known to acquire substantial factual knowledge during pretraining, storing it in their parameters (Geva et al., 2023). However, how effectively this knowledge can be composed for reasoning remains less understood (Dziri et al., 2023; Press et al., 2023). In particular, recent work shows that transformer-based LLMs struggle under implicit reasoning, i.e. reasoning within a single forward pass without explicit chain-of-thought (CoT) (Wei et al., 2022). Such failures reveal a fundamental limitation of transformers: despite storing rich knowledge, they are often unable to flexibly combine it to solve novel questions. This limitation has important implications for generalization, as many tasks require composing multiple pieces of seen knowledge in novel ways not observed during training (Lake and Baroni, 2018; Berglund et al., 2023).

Why do transformers struggle to combine their parametric knowledge in implicit reasoning? Consider a query such as "The spouse of the performer of Imagine is". Previous work shows that transformers solve this by chaining two facts: first retrieving that the performer of Imagine is John Lennon in shallow layers, and then that the spouse of John Lennon is Yoko Ono in deeper layers (Biran et al., 2024; Wang et al., 2024a; Yang et al., 2024b). However, since knowledge is distributed across different layers of the transformer, there is no guarantee that the fact required for a particular query can be accessed correctly. For example, if the fact the spouse of John Lennon is Yoko Ono is only stored in shallow layers, deeper layers cannot access it because parameters are not shared across layers. While transformers can be trained to learn to combine such knowledge properly (Wang et al., 2024a; Yao et al., 2025), they fail to compositionally generalize to unfamiliar combinations or deeper recursive combinations.

To address this limitation, we introduce depth-recurrence into transformers, allowing the same set of layers to be applied iteratively. The input sequence is processed multiple times by a shared transformer block, where the output of each iteration serves as input to the next. In contrast to vanilla transformers, where knowledge is tied to specific layers, recurrence enables more flexible access to and composition of parametric knowledge within a single forward process. Such models, known as recurrent-depth transformers or looped transformers, have recently gained attention as a promising architecture (Dehghani et al., 2019; Geiping et al., 2025; Zhu et al., 2025). While prior work has shown that recurrent-depth transformers improve length generalization (Bansal et al., 2022; Fan et al., 2025), it remains unclear whether they can overcome compositional generalization limitations when reasoning over parametric knowledge.

In this paper, we systematically study whether recurrent-depth transformers can compositionally combine their parametric knowledge implicitly. By constructing synthetic datasets, we train models to learn implicit reasoning from scratch. Unlike LLMs trained on vast, opaque web-scale corpora, this setup provides control over the data and mitigates confounding biases introduced during pretraining. Specifically, we characterize two challenges: systematic generalization (combining knowledge not used in any composition during training) and depth extrapolation (e.g., training on 5-hop reasoning and evaluating on 10-hop).

Our main findings are two-fold. First, recurrent-depth transformers exhibit strong systematic generalization, while vanilla transformers fail to do so. We show that this ability emerges through a sharp three-stage grokking process, that transitions from memorization to in-distribution generalization, and finally to systematic generalization. We also support this with evidence from the internal activations of models across different training stages.

Second, recurrent-depth transformers enable depth extrapolation, generalizing to reasoning depths beyond those observed during training, as inference-time compute (i.e., recurrent iterations) increases. We further find that the training-time recurrence strategy plays a critical role in extrapolation performance, with dynamic recurrence achieving the strongest generalization. Despite these gains, we identify a key limitation: recurrent-depth transformers suffer from overthinking (Bansal et al., 2022), which degrades performance and limits generalization to extremely deep recursions.

> **Figure 1:** Recurrent depth model architecture. The transformer block is repeated R times. The embedding layer and language model head (LM Head) have tied weights. In our experiments, we use a simple looped transformer similar to Saunshi et al. (2025) without design elements such as input injection, gated halting, and middle looping.

## 2 Related Work

Several small-scale studies pretrain looped or recurrent-depth transformers on synthetic tasks to better understand their behavior in a controlled setting. Our work best aligns with such studies where we are able to cleanly attribute differences in performance and generalization to specific architectural choices and model design decisions. Yang et al. (2024a) demonstrate how "looping" a transformer block helps to better emulate learning algorithms such as gradient descent for in-context linear regressions, 2-layer neural networks, and decision trees. Fan et al. (2025) show that such looped transformers offer superior length generalization on algorithmic tasks such as parity and binary addition. Saunshi et al. (2025) conduct a larger-scale pretraining with the 250B tokens of the Pile dataset (Gao et al., 2020) and find that looped versions of transformer models of the same effective depth have a greater inductive bias towards reasoning at the cost of memorization and perplexity. Based on these results, they propose a regularization term that encourages certain layers to be closer to each other, thus improving the tradeoff between reasoning and fact recall.

Relative to other works, our targeted setting yields unique insights on training dynamics and model behavior. We demonstrate how weight sharing through recurrence can solve systematic composition where vanilla transformers are known to struggle and extrapolation in multi-hop composition is possible with increased recurrence at inference-time. While Fan et al. (2025) propose looped architectures for length generalization, they assume an oracle number of training iterations based on sample complexity. We believe that our setup is closer to real-world scenarios where task complexity cannot be easily estimated through heuristics (such as input length). Without the assumption of task complexity a priori, we face distinct challenges in training our models. We analyze how best to apply methods like recurrent-depth and common pitfalls to avoid, which can help inform more robust implicit reasoning models in the future. We discuss other related work in Appendix D.

## 3 Task Formulation

We formally define our implicit reasoning setup using a synthetic multi-hop reasoning task, and categorize three generalization challenges under this formulation: in-distribution generalization, systematic generalization, and depth extrapolation (Figure 2). The latter two can be viewed as out-of-distribution (OOD) generalization. Such tasks have been shown to be difficult for vanilla transformers to learn (Yao et al., 2025), highlighting their limitations in composing parametric knowledge for reasoning (Allen-Zhu and Li, 2023; Yang et al., 2024b).

### 3.1 Task Definition

> **Figure 2:** Illustration of systematic and extrapolation generalization tasks with a sample dataset.

Our implicit reasoning task relies on a directed knowledge graph (KG) where nodes represent a set of entities E = {e_i} and edges represent a set of relations R = {r_j}. The KG is composed of atomic (1-hop) facts, each taking the form of a triplet (h, r, t), where h, t ∈ E and r ∈ R (here h and t imply the head and tail entities, respectively).

A k-hop inferred fact is defined as a chain of k atomic facts connecting a head entity h to a final tail entity t via a sequence of k−1 intermediate entities (i_1, ..., i_{k-1}):

> (h, r_1, i_1), (i_1, r_2, i_2), ..., (i_{k-1}, r_k, t)

Given the head entity h and the sequence of relations r_1, ..., r_k, we use an auto-regressive decoder-only model to predict the final tail entity t. The input prefix is `<e_h><r_1><r_2>...<r_k>`, and the target is `<e_t>`. Ideally, the model must implicitly perform the k-hop traversal, successively retrieving each intermediate entity (i_1, ..., i_{k-1}) until it can resolve the final tail entity t.

### 3.2 Generalization Challenges

Given a generated knowledge graph, we first define the complete atomic fact set as C = {(h, r, t)}, and the induced set of k-hop inferred facts from C as

> I_k(C) = {(h, r_1, ..., r_k, t) | ∃ i_1, ..., i_{k-1}, (h, r_1, i_1), ..., (i_{k-1}, r_k, t) ∈ C}.

#### Training set.

The training set includes two parts: all possible atomic facts C together with a set of inferred facts I_train up to a maximum depth k_train (e.g. k-hop facts with k ∈ [2, k_train]). To characterize different generalization challenges, we partition the atomic fact set into two disjoint subsets C = C_ID ∪ C_OOD. The training inferred facts I_train can then be defined as

> I_train = {(h, r_1, ..., r_k, t) | (h, r_1, ..., r_k, t) ∈ I_k(C_ID), k ≤ k_train}.

We then define three generalization challenges:

#### In-distribution generalization.

The model is evaluated on inferred facts (h, r_1, ..., r_k, t) ∈ I_k(C_ID) that are not observed during training, equivalent to randomly sample inferred facts from I_k(C_ID) as held-out test set. Despite its simplicity, previous work shows that vanilla transformers can only learn such tasks through extended training (Wang et al., 2024a).

#### Systematic generalization.

The model is evaluated on inferred facts (h, r_1, ..., r_k, t) ∈ I_k(C_OOD), which are induced from atomic facts that are never used in compositions in the training data. This requires the learner to systematically combine its learned knowledge, without having seen combinations of it in training. This setting simulates scenarios where knowledge appears only as plain text in pretraining data (e.g. long-tail knowledge), but never forms answers to reasoning queries during training. Previous work (Wang et al., 2024a) shows that vanilla transformers completely fail on this generalization challenge.

#### Depth extrapolation.

The model is evaluated on inferred facts of greater depth than those included in the training dataset, i.e., (h, r_1, ..., r_k, t) ∈ I_k(C_ID) with k larger than k_train. Solving this requires the learner to infer the underlying rules of the task and iteratively apply them at depths far beyond those observed during training. This setting simulates scenarios where the complexity (i.e. reasoning depth) of training data is limited due to budget constraints, yet we expect the model to generalize beyond training. Although related to length generalization, depth extrapolation is conceptually distinct: it measures the depth to which a model can repeatedly apply learned rules over its parametric knowledge beyond training.

## 4 Recurrent-Depth Transformer

#### Model architecture.

Across all experiments we use a decoder-only transformer with a *recurrent-depth* design illustrated in Figure 1. Concretely, we instantiate a GPT-2 style block with L layers and reuse this for R recurrent iterations, yielding an effective rolled-out depth of D = L × R layers. At each recurrent iteration the same stack of layers is applied to the current hidden states. This allows the model to allocate more computation (increasing R) at inference time without changing the architecture or re-training the parameters. We exploit this property in our inference-time scaling experiments described in Section 6. Formally, let f_θ denote the L transformer layers with shared parameters θ, and let h^(0) be the input sequence after the initial embedding layer. The model computes

> h^(r+1) = f_θ(h^(r); m),  r = 0, ..., R−1,

where m denotes the causal attention and padding masks. The final representation h^(R) is passed through a final layer normalization and a tied output projection to produce logits over the vocabulary at each position. We only supervise the next-token distribution at the final position corresponding to the tail entity t. Each entity and relation is represented by a dedicated token (`<e_i>`, `<r_j>`), and the query prefix is mapped to token embeddings.

We adopt a zero-initialization strategy to stabilize training under repeated application of shared weights. Specifically, we initialize the output projection matrices (c_proj) of both the multi-head attention and feed-forward blocks to zero, so that each recurrent block is an exact identity mapping at initialization. This ensures that the input-output Jacobian remains stable even when the model is unrolled to a large number of recurrent iterations. This design is motivated by the known instability of deep networks with shared parameters (Agarwala and Schoenholz, 2022), which becomes particularly pronounced in recurrent-depth transformers (Saunshi et al., 2025). Following Zhang et al. (2019), this initialization supports stable optimization under unbounded unrolling of the recurrent iterations.

#### Stopping strategies of the recurrent iterations.

Training the looped transformer requires a stopping strategy to determine the number of recurrent iterations in the forward pass on the input. We consider two stopping strategies: fixed iteration and dynamic iteration. Fixed iteration determines the recurrent iterations to be the same fixed value for all training instances. The dynamic iteration strategy samples the number of recurrent iterations independently for each training batch. Concretely, for the dynamic model we sample

> R ∼ clip(Poisson(λ), R_min, R_max),

where R_min and R_max are hyperparameters. Such strategies have been shown to be effective in realistic pretraining (Geiping et al., 2025; Zhu et al., 2025), and here we adopt a simple Poisson distribution to control the sampling distribution. Importantly, our strategies contrast with prior studies on the generalization ability of looped transformers, where the iteration number is matched to the complexity of each training instance, assuming oracle access to such complexity. Instead, our setup reflects practical scenarios (Geiping et al., 2025), where the complexity is unknown and computation cannot be allocated precisely in advance.

## 5 Systematic Generalization

In this section, we study systematic generalization, i.e. whether models can combine parametric knowledge not composed during training for multi-hop tasks. We focus on 2-hop, as Wang et al. (2024a) shows that vanilla transformers already struggle with this simple task.

### 5.1 Experiment Setup

#### Dataset.

We construct the dataset by instantiating a knowledge graph with |E| = 2000 and |R| = 200, where each entity has average out-degree 20. We then include all 40k atomic facts, randomly partitioned with 95% C_ID, and 5% C_OOD, together with 273.6k inferred facts for training. Our in-distribution set includes 3k held-out two-hop inferred facts composed from C_ID, and OOD set includes nearly 2k two-hop inferred facts composed from C_OOD.

#### Model.

We train our looped transformer with L = 4 layers, and use fixed training recurrence with R ∈ {1, 2, 4, 8}. The model with R = 1 is equivalent to a 4-layer vanilla transformer. We evaluate the accuracy of the predicted token against the gold answer. Absolute position embeddings (APE) (Vaswani et al., 2017) are used as positional embeddings in this setup. We do not use dynamic recurrence in this setting, as systematic generalization in the 2-hop task already emerges from weight sharing under fixed recurrence. In the multi-hop setting, however, it improves extrapolation to more complex samples at inference time and helps alleviate latent overthinking, as discussed in Section 6.

### 5.2 Results

> **Figure 3:** Accuracy curves for recurrent-depth models across training epochs and wall-clock time. Left: Test OOD accuracy for models trained with R ∈ {1, 2, 4, 8}, plotted against training epochs. Curves are smoothed with a 100-epoch rolling mean, with shading indicating standard deviation. Middle: Test OOD accuracy for the same models, plotted against training wall-clock time (hours). Right: Accuracy of the R = 4 model on the examples from training, ID, and OOD test splits, plotted against training epochs.

#### Recurrent-depth transformers perform systematic generalization, while vanilla transformers do not.

On the left of Figure 3, we plot OOD accuracy as a function of training epochs. We find that the vanilla transformer (i.e. R = 1) completely fails when the task requires combining unfamiliar atomic facts, while even the simplest R = 2 recurrence achieves non-trivial generalization performance. Increasing training iterations further accelerates the convergence, e.g. R = 4 converges with 2k epoch, while R = 2 takes 7k. This acceleration is not only in terms of training steps, but also in absolute wall-clock time (Figure 3, middle).

#### Systematic generalization emerges through a three-stage grokking dynamic.

We further analyze the training dynamics of the R = 4 model (right panel of Figure 3) to understand how systematic generalization emerges. We observe a three-stage dynamic: In the first stage, the model overfits the training set, with only training accuracy improving. In the second stage, in-distribution generalization emerges after prolonged training beyond memorization, a phenomenon referred to as grokking. In the final stage, systematic generalization arises only after the model achieves near-perfect in-distribution accuracy, occurring at a much later point than training overfitting (e.g., 10^4 vs. 10^2 epochs).

#### Analyzing model internals with logit lens.

We use the logit lens technique (nostalgebraist, 2020) to examine how models represent the bridge entity and the final target during different stages of training. After each layer and recurrent iteration, we project the intermediate hidden states through the final layer norm and language modeling head to obtain logits over the output vocabulary. For 2-hop inputs of the form (h, r_1, r_2), where h is the head entity and r_1, r_2 are the two relations, we measure at each effective depth the accuracy of predicting the bridge entity at the r_1 position and the target entity at the r_2 position. Figure 4 shows logits lens for an R = 2 recurrent-depth model on Training, Test ID, and Test OOD splits, across checkpoints corresponding to the three training stages. We compare against an iso-FLOP 8-layer vanilla transformer with matched effective depth. The vanilla model exhibits only two training stages and fails to achieve non-zero systematic generalization regardless of training time, consistent with Wang et al. (2024a).

> **Figure 4:** Accuracy of predicting bridge and target entities using logit lens at corresponding token positions for the recurrent-depth (R = 2) and the 8-layer vanilla transformer.

#### Grokking marks a transition from memorization to systematic generalization.

We first focus on the recurrent-depth model (Figure 4 left panels), which exhibits distinct mechanisms across the three stages. In Stage 1, the model predicts targets without reliably decoding the bridge, indicating memorization. In Stage 2, the bridge becomes decodable, followed by correct target prediction for in-distribution data at deeper effective depths. Only in Stage 3 does the model succeed on OOD inputs, marking a transition from rote learning to systematic composition. In contrast, although vanilla transformers (right panels) can recover the bridge entity on Test OOD inputs, they fail to perform the second-hop reasoning, as they lack incentives to encode OOD facts in deeper layers.

## 6 Depth Extrapolation

In this section, we study depth extrapolation, i.e., whether the model can perform deeper recursions when combining its parametric knowledge than those observed during training.

> **Figure 5:** Accuracy of recurrent-depth models on multi-hop composition, trained under various fixed and dynamic recurrence setups. The violet dash-dotted horizontal line indicates the training recurrence (or maximum in the dynamic setting), while the teal dashed vertical line marks the maximum ID generalization achieved by each model. The x-axis shows hop complexity and the y-axis shows inference-time recurrence (r* denotes adaptive halting).

### 6.1 Experiment Setup

#### Curriculum training.

Different from 2-hop scenarios, learning k-hop tasks generally requires training the model with an easy-to-hard curriculum over hop depth (k) as suggested in Yao et al. (2025). Specifically, we start with training on atomic and 2-hop facts until an accuracy threshold (95%) is reached on a held-out 2-hop test split. Next, 3-hop data is included in the training for the next stage of our curriculum until 95% is achieved on the held-out 3-hop test split. This process is repeated for each hop level k ∈ {2, ..., N_max}. To prevent forgetting, at each stage we jointly train on all previously introduced facts (e.g., at stage k = 5 we train on atomic and 2-hop through 5-hop facts).

Due to this threshold-based curriculum, training facts beyond the model's capability are never exposed. That is, if the model fails to achieve above-threshold accuracy on k-hop queries, training terminates and (k+1)-hop data is never introduced. We define the largest such k as the learnable recursion depth of the model.

#### Dataset.

We construct the dataset by instantiating a knowledge graph with |E| = 200 entities and |R| = 10 relations, where each entity has an average out-degree of 10. We additionally impose a permutation constraint on the knowledge graph to avoid learning shortcut solutions, for which we provide details in Appendix B. We pre-generate 2k atomic facts and 15k k-hop inferred facts for each k ∈ [2, K_max], with K_max = 40. During training, these facts are progressively introduced following the curriculum described above.

For each model, we evaluate on 750 held-out k-hop facts. Facts with k up to the model's learnable recursion depth form the in-distribution test set, while those beyond it constitute the extrapolation test set. This split is model-dependent, reflecting each model's maximum achievable reasoning depth.

#### Model.

We follow the model setups in Section 5.1, except that we use R ∈ {1, 2, 3, 4, 5, 6, 7, 8} for fixed iteration. Here we use no positional embeddings (NoPE) (Kazemnejad et al., 2023; Wang et al., 2024b), which shows better generalization in pilot studies. In addition to results with L = 4, we present results with varying model size and dynamic train recurrence in Appendix F, and across random seed initializations in Appendix G.

### 6.2 In-Distribution Generalization

#### Scaling up training-time iteration increases the learnable recursion depth of looped transformers.

Looking at the blue area of Figure 5, we find that increasing training recurrent iterations accordingly improves the ID generalization. This is consistent with previous findings (Wang et al., 2024a; Yao et al., 2025) that the learnable recursion depth of a transformer is bounded by the depth of its layers, and we demonstrate that for looped transformers, scaling up training recurrent iterations can also increase its "effective depth", without relying on additional parameters. Training with dynamic iteration further increases the learnable recursion depth over the fixed iteration, suggesting that the fixed iteration is not an optimal design choice. Interestingly, more recurrent iterations do not always translate into larger learnable depth (e.g., both R = 7 and R = 8 learns up to 16-hop task).

> **Figure 6:** Cumulative gradient updates required to first generalize to each hop complexity.

#### Phase transition from prolonged training to rapid learning.

We observe that models require a prolonged training phase to acquire low-hop tasks, after which they rapidly generalize to much more complex samples (Figure 6). We illustrate this for the model trained with dynamic recurrence by plotting training steps against the compositional complexity achieved on in-distribution data. This suggests that the main difficulty lies in discovering the underlying compositional rule. Once such a rule is internalized, the model can quickly extend it to samples of much higher complexity. In Figure 6 we observe how the model required over 1.3 million steps for grokking up to 4-hop train samples but was quickly able to learn up to 19-hop samples very few additional steps. Beyond that, for even more complex samples, while each new hop requires additional training steps to cross the 95% threshold, the model still achieves strong generalization (>90%) on hops 20, 21, and 22 within fewer than 8k extra steps per hop which is commensurate with the steps required for each additional hop from 4 through 19. By loading a trained checkpoint (20, say) and continuing training exclusively on 21-hop samples instead of the data mix consisting of samples from all previous stages in our curriculum, generalization over 90% on the new split can be achieved in as little as 50 additional steps of training.

### 6.3 Depth Extrapolation

#### Scaling inference-time iterations unlocks depth extrapolation.

In Figure 5, we observe that when using the same number of recurrent iterations as in training, all models struggle to generalize to tasks of higher complexity than those seen during training. However, this limitation is immediately alleviated when we increase the number of inference-time iterations, with more iterations enabling generalization to progressively harder tasks. Notably, this scaling effect only emerges for R > 4, suggesting that sufficient training-time iterations are a prerequisite for benefiting from increased inference-time computation.

#### The effect of training iteration strategy on extrapolation.

The above results characterize the maximum reasoning depth each model can achieve under the curriculum setting, but they do not disentangle whether the differences (e.g. R = 6 generalizing to 17-hop vs. R = 8 to 24-hop) are due to more training iterations or exposure to more complex training data (e.g. R = 6 is trained up to 13-hop, while R = 8 up to 16-hop). To isolate the effect of the training iteration strategy, we train all models on the same data (up to 12-hop) and evaluate extrapolation on 12-hop [...]

---

*Note: This markdown was extracted from the arXiv HTML full-text version (v1). The source fetch was truncated within Section 6.3; the remainder of Section 6.3 (including "Scaling inference-time iterations is limited by overthinking" and "Adaptive halting improves inference efficiency"), Section 7 (Conclusion), References, and the Appendices (A–H) were not captured. All MathML markup was cleaned into plain notation. Figures are referenced but not embedded.*
