---
title: "Parcae: Scaling Laws For Stable Looped Language Models"
arxiv_id: "2604.12946"
url: "https://arxiv.org/abs/2604.12946"
pdf: "https://arxiv.org/pdf/2604.12946"
authors: Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, Daniel Y. Fu
date: 2026-04-14
source: "html-fulltext"
---

# Parcae: Scaling Laws For Stable Looped Language Models

**Authors:** Hayden Prairie (University of California, San Diego; Together AI), Zachary Novack (University of California, San Diego), Taylor Berg-Kirkpatrick (University of California, San Diego), Daniel Y. Fu (University of California, San Diego; Together AI)
**arXiv:** 2604.12946 — https://arxiv.org/abs/2604.12946
**License:** CC BY 4.0 — arXiv:2604.12946v1 [cs.LG] 14 Apr 2026

## Abstract

Traditional fixed-depth architectures scale quality by increasing training FLOPs, typically through increased parameterization, at the expense of a higher memory footprint, or data. A potential alternative is *looped architectures*, which instead increase FLOPs by sending activations through a block of layers in a loop. While promising, existing recipes for training looped architectures can be unstable, suffering from residual explosion and loss spikes. We address these challenges by recasting looping as a nonlinear time-variant dynamical system over the residual stream. Via a linear approximation to this system, we find that instability occurs in existing looped architectures as a result of large spectral norms in their injection parameters. To address these instability issues, we propose *Parcae*, a novel *stable*, looped architecture that constrains the spectral norm of the injection parameters via discretization of a negative diagonal parameterization. As a result, Parcae achieves up to 6.3% lower validation perplexity over prior large-scale looped models. Using our stable looped architecture, we investigate the scaling properties of looping as a medium to improve quality by increasing FLOPs in training and test-time. For training, we derive predictable power laws to scale FLOPs while keeping parameter count fixed. Our initial scaling laws suggest that looping and data should be increased in tandem, given a fixed FLOP budget. At test-time, we find that Parcae can use looping to scale compute, following a predictable, saturating exponential decay. When scaled up to 1.3B parameters, we find that Parcae improves CORE and Core-Extended quality by 2.99 and 1.18 points when compared to strong Transformer baselines under a fixed parameter and data budget, achieving a relative quality of up to 87.5% a Transformer twice the size.

Contact: {hprairie, znovack, tberg, danfu}@ucsd.edu

## 1 Introduction

Scaling laws have established that model performance improves predictably with increased FLOPs, typically by increasing parameter count or training data. These scaling laws suggest that FLOP-optimal training increases parameters and training data in tandem following empirical power laws. As a result, the depth and width of state-of-the-art models have grown in an effort to scale with data, subsequently inflating the memory footprint to deploy these models.

However, as inference deployments take on an increasingly large portion of compute, and deployments begin to move to the edge, there is increasing interest in scaling model quality without increasing parameters. One mechanism to do this is layer-looped models, such as looped transformers, which iteratively loop activations through a block of layers. Initial results have been encouraging, with looped models matching the quality of larger fixed-depth architectures. Moreover, they show potential for latent reasoning and per-token adaptive compute.

Unfortunately, prior research and our work observe these models' training to be unstable, exhibiting residual state explosion and loss spikes. Since these models loop the layers of complex non-linear architectures (e.g., transformer blocks), the source of instability in looped models can be difficult to understand analytically. As a result, training requires sensitive hyperparameter selection and residual normalization (e.g., Post-Norm) to correct this instability. Furthermore, even in convergent training runs, we observe loss spikes as looped models train on stochastic amounts of depth to induce stronger test-time scaling. In this paper, we study this instability and ask whether stabilizing these models can unlock looping as a predictable, orthogonal axis for scaling compute.

To analyze instability, we observe that prior looped architectures can be recast as a nonlinear time-variant dynamical system over the residual stream, taking the form:

> h_{t+1} = Ā·h_t + B̄·e + R̄(h_t, e)    (1)

where for an input e, the hidden state h across the depth of an architecture is modulated by Ā, controlling the balance between prior and current residual states; B̄, conditioning the residual on the input e; and a non-linear operator R̄, which subsumes the original transformer modules (e.g., Attention, MLPs). By linearizing this framework (e.g., removing R̄), we observe that Equation 1 resolves to a linear time invariant (LTI) system from which classic control theory can be used to infer divergence conditions on the residual stream based on the spectral norm of Ā. We observe that prior looped architectures can learn unstable parameterizations of Ā, which we empirically find to induce residual stream explosion.

To address these issues, we propose *Parcae*, a novel looped transformer that corrects the parameter instability conditions of Equation 1 and uses algorithmic fixes to reduce loss spikes during training. *Parcae* explicitly uses discretization on a continuous representation A of Equation 1 and parametrizes A as a negative diagonal matrix, constraining the spectral norm to prevent residual explosion in looped layers. Additionally, Parcae introduces a normalization on e, which empirically prevents loss spikes in late stages of training. Finally, Parcae modifies the training algorithm (which aims to minimize the expected loss over variable depths) by enabling intra-batch per-sequence depth sampling to further reduce loss spikes.

We evaluate Parcae on end-to-end quality, training FLOP scaling, and test-time scaling:

- **End-to-End Quality.** We compare Parcae against parameter- and data-matched RDMs and Transformers. Against RDMs, Parcae reduces val. PPL by 6.3%. When scaled up to 1.3B parameters and 100B tokens, Parcae outperforms parameter-matched Transformers by up to 2.99 and 1.18 points on Core and Core-Extended benchmarks, respectively — matching Transformers up to twice the size.
- **Training FLOP Scaling.** To evaluate FLOP training scaling, we study scaling laws for looping in a parameter-matched isoFLOP setting (i.e., whether to scale FLOPs with increased data or looping). We find that looping introduces an orthogonal scaling axis, similar to parameters and data. Specifically, FLOP-optimal training increases looping and data following empirical power laws.
- **Test-Time Scaling.** We study looping as a mechanism to scale test-time compute, observing that recurrence follows predictable exponential decay with an irreducible loss. We further combine both test-time and training power laws to create a single unifying scaling law for looping in Parcae models.

**Figure 1: Parcae and the Scaling Laws of Looping.** (*Left*) Parcae constrains the spectral norm of Ā and normalizes the input injection, stabilizing the residual stream h_t across loops. (*Right*) We observe looping to be an orthogonal axis of scaling compute which follows a power law.

## 2 Background

We first provide a brief background on looped models (Section 2.1), LTI systems (Section 2.2), and modeling scaling laws (Section 2.3). Prior work has studied looped architectures along several design axes: loop placement (pre-, mid-, or post-looping), halting mechanism (explicit routers vs. implicit stochastic depth), topology (single block or hierarchical) and differentiation (explicit or implicit backpropagation). Our work focuses on implicit-halting middle-looped architectures using explicit differentiation; an extended review is in Appendix B.

### 2.1 Existing Middle-Looped Architectures

In this paper, we focus on middle-looped architectures. Middle-looped recurrent depth architecture contains three units: an initial prelude unit P, a middle recurrent unit R, and a final coda unit C. Formally, given an input s ∈ V^n, where V is vocabulary and n is sequence dimension, the outputs p ∈ R^{n×|V|} can be computed by the following update rule:

> e = P(s),    h_{t+1} = R(h_t, e),    p = C(h_T),

where h_0 ~ N(0, σ²I_{d×d}) and d the embedding dimension. Intuitively, P embeds inputs into the latent space, conditioning R as it recursively updates the hidden state h_t ∈ R^{n×d} for T iterations, which C uses to generate p. Within R, prior work inject e using addition h_{t+1} = R(h_t + e) or concatenation with projection h_{t+1} = R(W[h_t; e]), where W ∈ R^{d×2d}.

While looped models can be viewed as weight-sharing layers, modern variants allow for variable depth. During training, depth T is sampled per micro-batch from Λ (e.g., Poisson with mean μ_rec), exposing the model to variable depths for stronger test-time scaling. The training objective thus minimizes the expectation over the dataset and Λ. Lastly, truncated backpropagation through depth, analogous to BPTT, limits the backward pass to a constant μ_bwd.

#### Stability.

Geiping et al. found looped models unstable at scale and adopted a block pattern, combining Pre- and Post-Norm to normalize the residual:

> x̄^(ℓ) = LN(MHA(LN(x^(ℓ−1))) + x^(ℓ−1)),    x^(ℓ) = LN(FFN(LN(x̄^(ℓ))) + x̄^(ℓ))

where LN(·) denotes layer normalization, MHA(·) multi-head attention, and FFN(·) feed-forward networks. We later show that residual normalization is unnecessary when stability is properly controlled.

### 2.2 Linear Time-Invariant Dynamical Systems

To study the instability of looped models, we will use an LTI dynamical system as a tractable linear surrogate for complex non-linear looped models. In control theory, LTI systems are formalized through first-order differential equations

> ḣ(t) = A·h(t) + B·e(t),    y(t) = C·h(t)

that describe the evolution of a hidden state h(t) ∈ R^{d_h} given an input signal e(t) ∈ R^{d_e}, where A ∈ R^{d_h×d_h} governs the dynamics of the system, B ∈ R^{d_h×d_e} controls how external inputs influence the state, and C ∈ R^{d_e×d_h} projects the hidden state to the output y(t) ∈ R^{d_e}. The continuous system can be discretized to obtain

> h_t = Ā·h_{t−1} + B̄·e_t,    y_t = C·h_t

using a step size Δ; for instance, zero-order hold (ZOH) would yield Ā = exp(ΔA) and B̄ = (ΔA)^{−1}(exp(ΔA) − I)·ΔB.

LTI systems fall into three regimes: *stable* (bounded and convergent), *marginally stable* (oscillatory), and *unstable* (explosive and divergent). A fundamental property of LTI systems is that their *stability* is determined by the eigenvalues of A. Continuous LTI systems require negative eigenvalues of A; Discrete LTI systems requires ρ(Ā) < 1, where ρ computes the spectral norm, with unstable systems having ρ(Ā) > 1.

### 2.3 Modeling Scaling Laws

We follow Hoffmann et al., which modeled scaling law behaviors via parabolic and parametric fits for varying model sizes and training tokens with a fixed FLOP budget. For parabolic fits, a quadratic is fit to several FLOP budgets to estimate the loss-optimal model size or number of training tokens. For parametric fits, a function form of L̂(N, D) = E + X·N^{−x} + Y·D^{−y} is fit using the Huber loss between the predicted and empirical log loss values for varying parameters N and tokens D, using L-BFGS to minimize.

## 3 Understanding Instability in Looped Architectures

**Figure 2: Training Instability of Looped Architectures.** (*left*) Pre-Norm looped models diverge, while residual norm. and Parcae converge. (*right*) Instability stems from an exploding recurrent state norm ‖h_T‖₂, the hidden embedding norm after T recurrences.

In this section, we study the instability of looped architectures. Using an LTI view over the residual, we find that instability stems from an unconstrained residual state explosion (Figure 2; Table 2 [*Baseline*]; Appendix F). While residual normalization helps mitigate this issue, it requires sensitive hyperparameter tuning (Table 2 [*Res. Norm*]), similar to fixed-depth transformers. Using this LTI framework, we derive stability conditions for the eigenvalues of Ā. We find that prior work does not satisfy these conditions for Ā, which we empirically verify creates major state explosion (Table 2).

#### Dynamical System over Residual Stream.

Our key insight is to recast the forward pass as a dynamical system over the residual stream. Consider a transformer-based looped model as defined in Section 2.1 for language modeling, where P is an embedding layer that maps a sequence of tokens s ∈ V^n into embedding space e ∈ R^{n×d_h}, C is a projection head that maps into probability space g: d_h → |V|, and R is parameterized with L transformer blocks. While several methods of input injection could condition R on e, building on prior work, we focus on linear methods of injection (e.g., R(h_t, e) = R(W_1·h_t + W_2·e), where W_1 ∈ R^{d_h×d_h} and W_2 ∈ R^{d_h×d_e}). Both addition and concatenation fall under this framework.

Recall that R denotes the full recurrent update h_{t+1} = R(h_t, e), encompassing all transformer operations, including residual connections. The recurrent update can be exactly formulated as a non-linear time-variant dynamical system of the form h_t = Ā·h_{t−1} + B̄·e + R̄(h_{t−1}, e), y_t = C·h_t, where C ∈ R^{d_c×d_h} decouples the C and R embedding dimension (i.e. p = C(C(h_T))). This derivation is shown in Appendix C. Though this formulation does not immediately elucidate instability, linearizing of this system (i.e., dropping R̄) yields a discrete LTI system of the form:

> h_{t+1} = Ā·h_t + B̄·e    (2)

**Table 1: Comparison of Prior Update Rule Stability based on LTI Representation.**

| Method | Ā | B̄ | ρ(Ā) | LTI Stability |
| --- | --- | --- | --- | --- |
| Addition | I | I | ρ(Ā) = 1 | *marginally-stable* |
| Concatenation | R^{d_h×d_h} | R^{d_h×d_e} | ρ(Ā) ∈ R | *unstable* |
| Parcae (ours) | ZOH(Diag(−exp(R^{d_h}))) | Euler(R^{d_h×d_e}) | ρ(Ā) < 1 | *stable* |

**Table 2: Hyperparameter Instability.** Convergence across learning rates for baseline RDMs, Res. Norm RDMs, and Parcae. Parcae is more robust to hyperparameter selection. Full logs are in Appendix F.

| LR | Base | Res. Norm | Parcae |
| --- | --- | --- | --- |
| 2e-4 | ✓ | ✓ | ✓ |
| 4e-4 | ✗ | ✓ | ✓ |
| 6e-4 | ✗ | ✗ | ✓ |
| 8e-4 | ✗ | ✗ | ✓ |
| 1e-3 | ✗ | ✗ | ✓ |

**Figure 3: Spectral Radius of Unconstrained Ā.** For a Pre-Norm RDM, we plot the ρ(Ā) throughout training using different learning rates, observing divergent runs learn ρ(Ā) > 1. The state explosion in Figure 2 is thus directly linked to Ā.

#### State Explosion from Unconstrained Ā and B̄.

Analyzing the stability of Equation 2 identifies ρ(Ā) as a critical factor governing instability. As shown in Table 1, prior work chooses parameterizations of Ā such that ρ(Ā) = 1 or ρ(Ā) is unconstrained. Critically, these are *marginally-stable* or *unstable parameterizations*.

Table 2 confirms this empirically: divergent runs learn a spectral radius of ρ(Ā) ≥ 1, with convergent runs maintaining ρ(Ā) < 1, affirming that LTI stability constraints are necessary. Finally, at scale, we observe loss spikes late in training (e.g., after 170k steps), which we address by normalizing the input to B̄ (see Appendix J for ablation).

## 4 Parcae: A Stable Looped Architecture

Using our dynamical systems framework, we create *Parcae*, a looped architecture that explicitly satisfies the stability constraints (Section 4.1). Additionally, we propose a per-sequence depth sampling method to stabilize variance introduced by variable depth (Section 4.2).

### 4.1 Block Design and Stable Parameterization of Parcae

We parameterize A and B in continuous form, and discretize using a learned Δ ∈ R^{d_h} with ZOH and Euler schemes (i.e., Ā = exp(ΔA) and B̄ = ΔB), following prior sequence modeling work. To achieve our target stability conditions by constraining the eigenvalues of A to be negative, we parameterize A := Diag(−exp(log_A)) as a negative diagonal matrix, where Diag(−exp(·)) of a vector enforces negativity and log_A ∈ R^{d_h} is our learnable vector. While many formulations of A would work, ensuring negative eigenvalues in the diagonal case is simple and cheap. B is left unconstrained; however, we introduce a normalization layer to the input e to further stabilize training (see Appendix J for ablation). With this, our update rule, given an input sequence s, becomes

> e = LN(P(s)),    h_{t+1} = Ā·h_t + B̄·e + R̄(h_t, e),    p = C(C·h_T),    (3)

where h_0 ~ N(0, σI_{d_h×d_h}) and T is the number of loops.

We parameterize P, R̄, and C using L_P, L_R and L_C transformer blocks respectively. For exact block architecture, we match two different architectural setups: one for prior RDMs and one for strong Transformer baselines. Parcae's architecture matches RDMs, differing only in residual normalization and the dynamical systems parameters (e.g., A, B, C, Δ). Against Transformers, we follow a simplified nanochat setup, where we match exact architecture, except we loop the middle third layers and include our dynamical systems parameters and a prelude norm. Exact model definitions and a forward pass can be found in Appendix P and Appendix E, respectively.

### 4.2 Stable Training Algorithms for Parcae

We further stabilize Parcae by adjusting the training objective. Specifically, looped models' training objective is

> θ* = arg min_θ E_{(x,y)~D, T~Λ}[ℓ(f_θ(x; T), y)],

implying that more depths should be sampled per global batch to more faithfully model the expectation over Λ. Thus, we introduce a per-sequence depth sampling algorithm within a micro-batch, which we empirically observe to reduce loss spikes (ablation in Appendix G). Additionally, unlike prior work, we parameterize Λ based on μ_rec alone, as we find that truncating based on μ_bwd significantly hurts extrapolation to both lower and higher recurrences (ablation in Appendix H). Finally, we choose μ_bwd = ⌈μ_rec/2⌉ throughout (see Appendix I for ablation). A detailed training algorithm is in Appendix E.

## 5 Results

We evaluate Parcae on end-to-end quality (Section 5.1), training FLOP scaling (Section 5.2), and test-time scaling (Section 5.3). We find that Parcae outperforms both parameter- and data-matched RDMs and Transformers, optimal looping and data follow predictable power laws, and test-time looping follows a saturating exponential decay.

**Table 3: Zero-Shot and Perplexity Results Trained on RDM Setup.** Comparison of Parcae and RDM on a variety of open source benchmarks and perplexity held-out validation set and Wikitext. Best results are bolded.

| | Model | T | Val. | WikiText | Hellaswag | ARC-c | ARC-e | PIQA | BoolQ | SciQ | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100M | RDM | 16 | 14.23 | 63.27 | 27.16 | 17.66 | 42.38 | 59.14 | 51.35 | 72.50 | 45.03 |
| | Parcae | 16 | 13.59 | 60.33 | 27.18 | 18.09 | 43.10 | 59.30 | 61.83 | 71.50 | 46.83 |
| 350M | RDM | 8 | 10.76 | 41.31 | 28.55 | 20.90 | 47.26 | 61.75 | 61.53 | 76.70 | 49.45 |
| | Parcae | 8 | 10.09 | 37.53 | 29.23 | 21.08 | 48.78 | 62.08 | 60.73 | 78.80 | 50.12 |

**Table 4: Stability Results Trained on Transformer Setup.** To illustrate stability, we retrofit a baseline 140M Transformer into a RDM and then sequentially add our stability improvements.

| Configuration | Val Loss T=1 | T=4 | T=8 | Core T=1 | T=4 | T=8 | Core Ext T=1 | T=4 | T=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RDM | Divergent Training | | | Divergent Training | | | Divergent Training | | |
| + Constrained Ā | 8.99 | 3.15 | 2.97 | −2.0±0.1 | 11.0±0.1 | 13.2±0.2 | 0.5±0.1 | 7.8±0.0 | 9.1±0.5 |
| + Per-Seq. Sampling | 3.38 | 3.01 | 2.98 | 7.6±0.2 | 13.4±0.2 | 14.0±0.2 | 5.9±0.4 | 9.3±0.2 | 9.9±0.2 |
| + Prelude Norm | 3.28 | 2.97 | 2.95 | 7.5±0.3 | 13.5±0.0 | 14.0±0.2 | 5.8±0.3 | 9.4±0.1 | 9.7±0.3 |

### 5.1 Parcae Improves End-to-End Quality

We compare Parcae against parameter- and data-matched RDMs and Transformers, finding that Parcae is more stable than prior looped models and that it outperforms both in quality.

#### Setup.

For RDMs, we follow Geiping et al., using the Huginn dataset and tokenizer for training. For transformers, we follow Karpathy and train on FineWeb-Edu. For both RDM and Transformer setups, we perform hyperparameter sweeps for both RDMs and Transformers, and then use them for Parcae (i.e., we perform no hyperparameter sweeps for Parcae models). Extended model definitions, hyperparameter selection, and evaluation setup can be found in Appendix P, Appendix Q, and Appendix M, respectively.

**Comparison against RDMs.** Table 3 shows that Parcae reduces perplexity by up to 6.2% and 9.1% on a held-out validation set and WikiText against prior RDMs, while additionally performing up to 1.8 points better on the average of several downstream benchmarks. Table 4 ablates that each modification of Parcae contributes: constraining Ā enables convergence at high T (e.g., μ_rec = T = 8), per-sequence sampling stabilizes lower test-time depths, and the prelude norm further improves quality across all T (and late stage stability, Appendix J).

**Table 5: Comparing Parcae to Fixed-Depth Transformers.** We pretrain Transformers and Parcae with a nanochat setup at several scales, evaluating on a held-out validation set, Lambada, Core, and Core-Extended. Best results are bolded.

| | Model | T | Val. PPL (↓) | Lambada PPL (↓) | Core (↑) | Core-Extended (↑) |
| --- | --- | --- | --- | --- | --- | --- |
| 140M | Transformer | – | 21.48 | 127.39 | 13.00 ± 0.15 | 8.80 ± 0.21 |
| | Parcae | 8 | 19.06 | 80.64 | 14.04 ± 0.20 | 9.67 ± 0.28 |
| 370M | Transformer | – | 15.79 | 40.77 | 17.46 ± 0.03 | 11.71 ± 0.22 |
| | Parcae | 8 | 14.49 | 32.74 | 20.00 ± 0.06 | 12.75 ± 0.31 |
| 770M | Transformer | – | 13.08 | 22.37 | 22.42 ± 0.20 | 14.20 ± 0.63 |
| | Parcae | 8 | 12.49 | 19.71 | 25.07 ± 0.33 | 15.19 ± 0.43 |
| 1.3B | Transformer | – | 11.95 | 17.26 | 25.45 ± 0.08 | 15.90 ± 0.23 |
| | Parcae | 8 | 11.42 | 14.71 | 28.44 ± 0.28 | 17.08 ± 0.09 |

**Comparison Against Transformers.** Table 5 shows that Parcae reduces validation perplexity by 4.3–9.2% and improves Core and Core-Extended Scores by up to 2.99 and 1.18 points, respectively. We find that our 770M Parcae model achieves quality comparable to the 1.3B Transformer on Core with roughly half the parameters. Measured as a fraction of the quality gap to the next larger Transformer (e.g., for 140M Core-Extended: (9.67−8.80)/(11.71−8.80)·100 ≈ 29.9%), Parcae achieves a *23.3-87.5% and 29.9-58.2%* better parameter efficiency for Core and Core-Extended, respectively.

**Figure 4: Looping Scales Training Compute Optimally.** (*Left*) Parametric isoLoss contours over μ_rec and data. The efficient frontier traces the lowest FLOP budget required to achieve each loss level, showing that optimal training requires increased looping. (*Right*) Parabolic isoFLOP fits for 140M and 370M models reveal a clear optimum μ_rec at each FLOP budget, indicating that looping is an orthogonal scaling axis to data.

### 5.2 Looping as an Orthogonal Scaling Axis in Training

In this section, we explore the FLOP efficiency of looping under a fixed FLOP and parameter budgets. We find that looping introduces an orthogonal axis for scaling compute, where compute-optimal training increases μ_rec and data in tandem following empirical power laws.

#### Setup.

We train 140M and 370M Parcae models under fixed FLOP and parameter budgets, varying training tokens and mean recursion μ_rec using the nanochat setup. Additional training details and FLOP estimates can be found in Appendix O and Appendix D, respectively.

**Modeling Scaling Laws of Looping.** At 140M and 370M scales, isoFLOP curves show that increasing μ_rec while proportionally reducing tokens yields lower validation loss than training at low recurrence (Figure 4 [*right*]). Using a parabolic fit, we extract the optimal μ_rec and token budget at each FLOP level, finding that both follow predictable power laws (Figure 5) with consistent exponents (γ_μ ≈ 0.40, γ_D ≈ 0.78). We also fit a parametric function L̂(μ_rec, D) = E + X·N(μ_rec)^{−x} + Y·D^{−y} over the effective parameterization N(μ_rec) (i.e., parameters of unrolling the looped model) and tokens D (Figure 4 [*left*]; details in Appendix K), enabling predictable extrapolation of loss to unseen budgets. To verify, we predict the validation loss of held-out models in Section 5.1, achieving 1.3% and 0.8% error at 140M and 370M, respectively.

**Figure 5: Optimal μ_rec and Tokens Follows Predictable Power Laws.** We fit a parabola to each isoFLOP budget for both 140M and 370M Parcae models, using its minima to approximate the optimal μ_rec and token budget at each scale. We observe that optimal recurrence and tokens follow a predictable power law with similar coefficients at both scales.

**Figure 6: Pareto Frontier of Looping.** We observe that looping has a stricter IsoFLOP optimal loss frontier over fixed-depth, non-looped models. Dots are empirical points.

**Table 6: Core Scores Comparison of Looping Optimal Frontier over Purely Scaling Data.** We evaluate the downstream quality of fixed-depth (μ_rec = 1) and looped Parcae models trained with fixed parameters and FLOP budgets. At both scales, using the optimal μ_rec results in better Core and Core-Extended scores at extended FLOP budgets. Expanded results in Appendix N.

| | FLOPs (×10^18) | Optimal μ_rec* | Core | Core Ext. | Fixed-Depth Core | Fixed-Depth Core Ext. |
| --- | --- | --- | --- | --- | --- | --- |
| 140M | 11 | 2 | 7.6 | 5.7 | **7.9** | **6.1** |
| | 22 | 2 | 9.0 | 6.2 | **10.5** | **6.4** |
| | 44 | 4 | **11.2** | **8.4** | 10.7 | 8.1 |
| | 88 | 6 | 10.5 | **7.8** | **11.8** | 7.7 |
| | 1616 | 8 | **14.6** | **9.8** | 13.0 | 8.8 |
| | 6464 | 10 | **16.2** | **11.0** | 15.0 | 9.5 |
| 370M | 3232 | 4 | 15.2 | 10.1 | **16.8** | **11.2** |
| | 6464 | 6 | **18.1** | 11.6 | **18.1** | **12.1** |
| | 128128 | 6 | **20.1** | **13.0** | 18.1 | 12.0 |

**IsoFLOP comparison of Looping with Fixed-Depth.** Table 6 shows fixed-depth Parcae models without looping at each FLOP budget. The optimal curve achieves a strictly lower loss, which translates to 1.2-2.0 points higher Core scores (Table 6).

### 5.3 Test-Time Scaling Laws of Parcae

We study looping as a mechanism for scaling test-time compute. We find the test-time compute follows a predictable saturating exponential decay, which can be unified with Section 5.2, connecting both training and test-time scaling laws.

#### Setup.

We train 140M and 370M Parcae models under a fixed data budget with μ_rec ∈ {2, 4, 6, 8, 10, 12} following our nanochat setup, evaluating up to T = 24. We additionally evaluate models from Section 5.2 for the unified scaling laws. See Appendix O for details.

**Saturation of Test-Time Compute.** While prior works observed test-time generalization in small synthetic tasks, we find quality to be bounded in large-scale language modeling. Evaluating models from Section 5.1 at 2× μ_rec across all four scales (Figure 7), we observe that gains plateau near μ_rec, suggesting training depth determines the test-time scaling ceiling.

**Figure 7: Test-Time Scaling of Parcae.** When evaluating Parcae models from Table 5, we observe test-time looping follows a predictable saturating trend, consistent across model sizes.

**Figure 8: Scaling Test-Time Compute follows a Predictable Power Laws.** We plot the validation loss with different μ_rec as a function of test-time recurrence T, and find the fitted exponential decay tightly captures the test-time performance of looping.

**Modeling Scaling Laws of Test-Time Looping.** We find that the test-time scaling curves are well-described by a saturating exponential decay of the form: L(T) = L_∞ + Z·e^{−z·T}. This form tightly captures the saturation dynamics for each model (Figure 8; see Appendix L for details), achieving an average Huber loss of 2.5×10^{−7} and 1.8×10^{−7} for 140M and 370M, respectively.

**Unifying Training and Test-Time Scaling Laws.** From the learned fits in Figure 8, we observe that L_∞ matches the training law prediction at T = μ_rec (Section 5.2), and that the per-curve decay rate scales inversely with training depth as z/μ_rec (see Appendix L for details). These observations motivate a unified scaling law that connects training and test-time compute:

> L̂_unified(T | μ_rec, D) = [E + X·N(μ_rec)^{−x} + Y·D^{−y}]  (Training Law Floor L̂_train(μ_rec, D)) + [Z·exp(−z·T·μ_rec^{−1})]  (Test-Time Decay)    (4)

where L̂_train(μ_rec, D) is the training law in Section 5.2, and (Z, z) are two fitted parameters governing the test-time scaling. The training law sets the irreducible floor, while the decay rate −z·T/μ_rec captures how quickly additional recurrences approach it. On held-out 140M and 370M Parcae models (Section 5.1), the unified fit predicts test-time loss within 0.85-1.31% average error, dropping further to 0.1-0.17% average error when the empirical loss at T = μ_rec is used. This confirms that Equation 4 captures saturation dynamics, with residual error attributable to the training law's ~1% extrapolation gap (see Appendix L for extended details).

## 6 Discussion and Future Work

In this section, we briefly discuss limitations and future directions.

#### Looped Architectures.

While several design choices around looped architectures have been guided by small-scale empirical results, a deep investigation of loop-unit placement, composition (e.g., number of parameters in the recurrent unit and usage of different architectures), and extreme looping (e.g., increasing mean recurrence to deeper depths) at a larger scale is warranted. Within our dynamical systems framework, the use of different discretizations, full-rank parameterizations, and recurrent update rules warrants investigation to enable recurrence at larger depths.

#### Scaling.

While we find Parcae to induce predictable, optimal scaling laws for layer looping, our observations are limited to small architectures. It remains to be seen if Parcae compares favorably when scaling these observations to large FLOP budgets and parameterizations. We are also interested in the interplay of parameters, data, and recurrence as orthogonal axes, and how they should be efficiently scaled together. Finally, one limitation of looping is that, as μ_rec increases, the number of test-time steps required to achieve equivalent quality increases. An investigation of techniques that maintain quality with fewer inference time steps is an interesting future direction.

## 7 Conclusion

In this work, we study the stability of looped models through a dynamical systems framework and propose Parcae, a stable looped architecture that prevents residual explosion by constraining the spectral norm of the injection parameters. Parcae outperforms data- and parameter-matched prior looped models and baseline Transformers, matching downstream quality of models up to twice its size. We further establish scaling laws for looping: FLOP-optimal training increases looping and data in tandem following predictable power laws, while test-time looping follows a saturating exponential decay law, yielding a unified scaling law connecting training and inference compute.

## References

Full reference list available in the source HTML at https://arxiv.org/html/2604.12946v1 (References section) and in the PDF. Appendices A–R (glossary, extended literature review, derivations, FLOP estimates, training algorithms, additional ablations, model definitions, hyperparameters, and tokenizer training) are also available in the full paper.
