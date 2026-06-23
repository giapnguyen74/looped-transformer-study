# A Foundation for Reviewing Any Model Architecture

A reusable lens distilled from the looped-transformer discussion. The aim is to be able to pick up *any* architecture paper or design and ask the right questions — separating what's genuinely hard from what's just loudly claimed.

The central thesis: **a structure is two geometries at once, and most reviews only look at one.**

---

## 1. The one idea: a structure is two geometries

**Geometry A — the representation map.** The architecture is a family of functions `f_θ`; it projects data into a feature space. *Expressivity / universality* is a statement about this — which functions are **reachable** as θ varies (the image of the family).

**Geometry B — the loss landscape.** Over *parameter* space sits the loss surface `L(θ)`. Optimization does not navigate the data map; it walks downhill on `L(θ)`. **Trainability is a property of Geometry B.**

The link that matters: **the architecture (A) determines the shape of `L(θ)` (B).** Same data, different structure → different landscape. A review fails when it evaluates A (can it represent the answer?) and silently *assumes* B (can we find it?).

---

## 2. The questions every structure must answer

Three **evaluation** axes — and they are *independent* of one another:

1. **Expressivity** — can *any* θ represent the target? (a solution exists in reach)
2. **Trainability** — can *local* search actually *reach* a low-*training*-loss θ, *stably*? (landscape geometry)
3. **Generalization** — does the θ you reach also do well on *unseen* data? (this is where overfitting lives)

Plus one **design** concern that shapes all three:

4. **Capacity & cost** — where does knowledge live, what does it compute, how does it scale? (how it allocates parameters / FLOPs / depth)

Most architecture papers answer **1** loudly, **2** weakly, **3** with a benchmark table, and **4** partially. The real risk is almost always 2 and 4.

**Don't fuse the axes — especially #2 and #3.** "How many solutions exist" is not a count of points but the *size, connectivity, and conditioning* of the low-loss region; over-parameterization makes it bigger and smoother → *easier* to train. Overfitting is a **separate** axis, governed by capacity-vs-data and inductive bias. The classical intuition "many solutions → overfit" is exactly what modern over-parameterized networks falsified: more parameters usually train *easier* **and** generalize *better* (double descent; SGD's implicit bias toward flat, simple minima). Conversely, a tightly constrained structure (few params, weight-tied) is *harder* to train but — because the constraint acts as a regularizer — often generalizes *better*, provided a solution still exists at all. Looped / weight-tied models are precisely this trade: give up some trainability to gain generalization and cheap depth.

---

## 3. Trainability: the usable theory

There is no complete theory of neural-net trainability — but there are sharp, usable pieces:

- **SGD is local and greedy.** It only sees the gradient, so it succeeds only when `L(θ)` is *navigable by local steps*: well-conditioned (no needle-thin ravines), gradients that neither explode nor vanish, and enough good minima to fall into.
- **Signal propagation / dynamical isometry** (Saxe, Pennington, Xiao) — the master condition. A network trains well when its input-output **Jacobian has singular values near 1** (neither amplifying nor shrinking signal or gradient). This one condition explains most trainability successes and failures.
- **Over-parameterization** (NTK / lazy regime, mode connectivity) — more parameters than strictly needed makes minima abundant and connected, smoothing descent.
- **Conditioning & sharpness** — balanced curvature gives stable steps; sharp ravines give instability and poor generalization.
- **Stability *is* trainability.** The *same* spectral condition governs the forward signal and the backward gradient. If the forward pass stays bounded (isometric), the gradients usually do too. So "is it stable?" and "is it trainable?" are one question.

---

## 4. The levers that shape the landscape (inspect these in any design)

- **Depth & composition.** Each layer folds the representation space — depth buys exponential expressivity, but composes Jacobians, so it compounds whatever conditioning each layer has.
- **Weight sharing / recurrence.** Saves parameters and adds expressive depth, **but** couples gradients across depth (one θ sets the function at every step, signals can disagree) and raises the Jacobian to a power → roughens and ill-conditions the landscape. Demands a spectral leash.
- **Residual connections + normalization.** Keep each map near identity → condition the landscape toward isometry. Effectively mandatory once there is depth.
- **Input re-injection / anchoring.** In any iterated map, re-supplying the input prevents the state from drifting and stabilizes both forward and backward passes.
- **Sparsity / routing (MoE).** Adds parameter capacity *without* adding compute or depth; watch for routing collapse and load-balancing.
- **Initialization / normalization scheme.** Sets the operating point — ideally at or near dynamical isometry.

---

## 5. The three dials: capacity vs compute vs depth

- **Knowledge / capacity** ← parameters (FFN / experts). Where stored facts live.
- **Compute per token** ← active parameters / FLOPs (e.g., top-k routing).
- **Reasoning depth** ← sequential steps / layers.

A good structure lets you move these **independently**; a confused one couples them. For any paper, ask: *which dial does it actually turn, and what does it claim from turning it?* (A weight-tied loop turns "depth" cheaply but couples it to a harder landscape; MoE turns "capacity" without paying compute; etc.)

---

## 6. The review checklist

**Expressivity**

- What is the reachable function class? Is universality claimed, and under what assumptions (precision, depth-scaling with input)?
- Does the theoretical claim transfer to a *trained, finite* instance, or is it vacuous in deployment?

**Trainability**

- Is the input-output Jacobian near isometry at init *and* during training? Is there a mechanism (normalization, residual scaling, spectral constraint) that enforces it?
- Does the structure iterate or share a map? If so, where is the leash against Jacobian-to-a-power blow-up?
- What training-stability evidence is reported: LR sensitivity, loss spikes, max depth before failure, gradient norms?
- How over-parameterized is it for the task? Any landscape sharpness / Hessian-spectrum evidence?

**Generalization** (separate from trainability!)

- Is overfitting controlled by data and inductive bias, rather than assumed away?
- Does the structure's inductive bias (weight-tying, sparsity, locality) buy out-of-distribution or length-extrapolation gains?
- Is performance shown in the regime that *stresses* generalization, not just in-distribution?

**Capacity & cost**

- Where do facts live, and how is knowledge addressed/retrieved?
- Are comparisons **compute-matched**, not just parameter-matched?
- Is there a scaling story — do gains persist with size and depth? Any scaling law?

**Evidence quality**

- Are the wins in the regime where the hard part actually bites, or only on small/structured tasks?
- Does the flagship mechanism help on the realistic, large-scale benchmark — or quietly hurt and get hand-waved?

---

## 7. Red flags

- Proves expressivity, silent on optimization ("can represent" ≠ "can train").
- Universality that needs unbounded precision or unbounded steps — true but vacuous for any deployed model.
- Parameter-matched comparisons that hide extra FLOPs.
- No scaling story.
- An iterated or weight-shared structure with **no** stability / signal-propagation analysis.
- The flagship mechanism failing on the only large-scale test, explained away in one line.

---

## 8. Worked lens: looped / recurrent-depth transformers

Applying the foundation to the case we studied:

- **Expressivity:** strong — the class is Turing-complete (depth can scale with input). *But* this needs unbounded precision/steps, so it doesn't transfer to a finite trained model.
- **Trainability:** the open part. A naive loop *violates* dynamical isometry by construction (one map raised to the Tth power → singular values run away → residual explosion, loss spikes). It becomes trainable only by restoring isometry: residual + norm, input injection, and a spectral leash (LTI injection with ρ < 1; Parcae's negative-diagonal parameterization).
- **Capacity & cost:** the contractive loop *computes* but can't *store*, so knowledge is offloaded to an **MoE** (capacity), with top-k routing (compute) — the three dials cleanly separated.

The Universal Transformer paper answers question 1 and is largely silent on 2 — which is exactly the gap a structure review should surface. (See `critique` notes and `looped_transformer_minimal.py` for the concrete instance.)

---

## 9. Heterogeneous structure: route each sub-problem to the substrate that fits it

A single uniform stack is being asked to do incompatible jobs. Separate them and send each to the structure whose strength matches it:

- **Knowledge / style = content → parameters.** Wide FFN / MoE memory, content-addressed by routing. Shallow, capacity-heavy. (This is the memorization a looped model gives up.)
- **Reasoning = process → depth / iteration.** Carried by the loop, or by chain-of-thought. Parameter-light, depth-heavy.

This "routing" happens at two levels: *inside* the model, the MoE router sends each token to the relevant knowledge experts; *at the architecture level*, the prelude → loop → coda layout sends encoding/decoding/content to distinct non-shared layers and reasoning to the shared recurrent core.

**CoT and looping are the same reasoning primitive in different spaces.** Chain-of-thought is an *explicit* loop in **token space** (interpretable, expensive, discrete state); a looped transformer is an *implicit* loop in **latent space** (cheap, opaque, continuous "latent thoughts"). `L` loops ≈ `T` CoT steps (Saunshi et al., 2025). CoT is externalized looping; looping is internal CoT.

**Keep the functional win and the trainability requirement separate.** Splitting knowledge out makes the recurrent core *leaner* and gives it a reasoning-friendly inductive bias — a *functional* gain. It does **not** by itself fix the deep loop's gradient stability; that still needs residual + norm + spectral leash + input injection — an *optimization* requirement. Route work for function; keep isometry for trainability. You need both.

**Review question:** not only "is it expressive / trainable / general?" but "**does it use the right substrate for each kind of work, or force one structure to do everything?**"

## 10. The one-line foundation

> **Expressivity** says a solution exists within the structure's reach; **trainability** says the landscape can lead you there; **generalization** says the solution you land on also transfers; **capacity** says where it keeps what it knows. The axes are independent — *many solutions ease training without forcing overfitting*. And remember that a single spectral condition, *Jacobian ≈ isometry*, quietly governs stability and trainability at the same time.

---

*Related notes:* [`universal_transformers.md`](universal_transformers.md) (worked summary + intuitions) · [`looped_transformer_minimal.py`](looped_transformer_minimal.py) (the algorithm in code) · [`../docs/references/`](../docs/references/) (the cited papers).
