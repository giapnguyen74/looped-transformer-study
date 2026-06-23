# Parcae: Scaling Laws for Stable Looped Language Models — Summary

**Prairie, Novack, Berg-Kirkpatrick & Fu (UC San Diego / Together AI, Apr 2026)** · [arXiv:2604.12946](https://arxiv.org/abs/2604.12946) · [PDF](https://arxiv.org/pdf/2604.12946)
Full text: [`../docs/references/parcae.md`](../docs/references/parcae.md)

The paper that formalizes loop stability — it turns the hand-wavy "keep the gain near 1" into control theory, and it is the direct basis for OpenMythos's `LTIInjection`.

## The problem

Looped models scale FLOPs without adding parameters, but training is **unstable**: residual-stream explosion and loss spikes. Prior fixes were band-aids — sensitive hyperparameter tuning and residual normalization.

## The key insight: recast looping as a dynamical system

The looped forward pass over the residual stream is exactly:

> `h_{t+1} = Ā·h_t + B̄·e + R̄(h_t, e)`

where `Ā` balances old vs new state, `B̄` injects the input `e`, and `R̄` is the nonlinear transformer machinery. **Linearize** (drop `R̄`) and you get a discrete **Linear Time-Invariant (LTI) system** `h_{t+1} = Ā·h_t + B̄·e`. Control theory then says: stable iff the **spectral radius ρ(Ā) < 1** — the same `xⁿ` gain condition from the trainability thread, now named and proven. (`ρ = 1` is only *marginally* stable; the nonlinear part can tip it over, so Parcae wants strict contraction.)

## The diagnosis

Prior architectures sit at `ρ(Ā) = 1` (additive injection → marginally stable) or unconstrained (concatenation → unstable). Empirically, divergent runs *learn* `ρ(Ā) > 1`, which directly causes residual explosion (shown via spectral-radius-over-training plots).

## Why constraining ρ is hard — and Parcae's trick to make it free

Enforcing `ρ(Ā) < 1` on a *free/general* matrix is genuinely hard: eigenvalues cost an `O(d³)` decomposition every step; there's no cheap projection back into `{ρ < 1}`; soft penalties give no guarantee (and left free, training *learns* `ρ > 1`); spectral-normalization constrains singular values, not eigenvalues, and still needs power iteration.

**Parcae's move: don't enforce it — parameterize so it's stable by construction.**

- Make `A` **diagonal** → eigenvalues are just the diagonal entries (no decomposition).
- Write each entry as `−exp(log_A)` → always negative → continuous system stable.
- Discretize `Ā = exp(ΔA)` (zero-order hold) → `exp` of negatives lands in `(0,1)` → **ρ(Ā) < 1 guaranteed, for free, every step, gradient-friendly.**

This converts a hard constraint into a trivially-satisfied parameterization — the same idea as storing a log + exponentiating to force positivity, or softmax to force a simplex. **Cost:** a diagonal `A` is restrictive (per-channel scaling, no cross-channel mixing in the linear term); Parcae pushes mixing into `B̄` and the nonlinear `R̄`, keeping only the stability-critical linear backbone diagonal and contractive. Full-rank stable parameterizations are future work.

Two more algorithmic fixes: **normalize the input injection `e`** (stops late-stage loss spikes) and **per-sequence depth sampling** within a micro-batch (reduces variance from variable-depth training).

## Does heavily constraining Ā reduce expressivity?

A fair worry — and the answer is "less than it looks," for three reasons.

- **Yes, the matrix itself loses expressivity.** A diagonal `Ā` can only scale each channel independently — no cross-channel mixing, no rotation. The *linear state-transition term* genuinely gives something up.
- **But the constraint is surgical.** It sits only on `Ā`, the stability-critical linear backbone that governs how the residual stream *decays/retains* across loops — essentially a per-channel memory gate, not the engine of computation. All the real computation and cross-channel mixing lives in the **unconstrained** parts: the nonlinear `R̄` (full attention + MLP blocks) and the input injection `B̄`. You constrain *stability*, you leave *expressivity* free — the "route work to the right substrate" principle.
- **Empirically quality goes up, not down.** If the constraint were crippling, Parcae would lose to the unconstrained baselines; instead it beats them. For looped models, *instability was a bigger cost than the lost matrix freedom* — the baselines burned capacity being unstable. Removing the pathology, plus the mild regularization a constraint provides, more than pays for the diagonal restriction.

**Precedent:** this is the same choice as state-space models (S4, Mamba) — diagonal, negative-real-part state matrices with ZOH discretization. Diagonal SSMs nearly match full ones because the nonlinearity and input/output projections recover the expressivity the diagonal gives up.

**Honest caveat:** not free lunch in the limit. A diagonal `Ā` can't mix channels in the linear memory term, and the authors flag **full-rank stable parameterizations as future work** — an admission that the diagonal leaves something on the table, which may matter at extreme looping depths. Principle: expressivity vs stability trade off, but *not uniformly* — constrain the small stability-critical part, leave the expressive part free.

## Results

Residual normalization becomes *unnecessary*, and training is far more robust to learning rate (converges at LRs where baseline and res-norm RDMs diverge). Quality: 6.3% lower validation PPL than prior recurrent-depth models; at 1.3B parameters, beats parameter-matched Transformers by up to 2.99 / 1.18 points on Core / Core-Extended — **matching Transformers up to twice its size** (770M Parcae ≈ 1.3B Transformer).

## The headline: looping is an orthogonal scaling axis

This is exactly the *scaling story* the Universal Transformer critique said was missing.

- **Training:** under fixed FLOPs/params, compute-optimal training increases **looping (μ_rec) and data in tandem** following predictable power laws (exponents ≈0.40 and ≈0.78). Looping is a third axis alongside parameters and data.
- **Test-time:** scaling inference recurrence follows a **saturating exponential decay** `L(T) = L_∞ + Z·e^{−zT}`. Gains **plateau near μ_rec — training depth sets the test-time ceiling.**
- A **unified scaling law** connects the two: the training law sets the irreducible floor; the decay term says how fast extra recurrences approach it.

## Two contrasts with the other papers

- **vs Loop-Think-Generalize:** that paper (small synthetic tasks) found inference-time recurrence keeps unlocking deeper reasoning; Parcae finds that at *language-modeling scale*, test-time scaling **saturates** — you can't loop far past your training depth. A sobering correction to the depth-extrapolation optimism.
- **Stability approach:** Loop-Think-Generalize used zero-init (identity-at-init); Parcae uses a principled spectral constraint on the injection. Two roads to the same isometry goal (`ρ ≤ 1`).

## Tie to the repo and the foundation

This is the literal source of OpenMythos's spectral-leash design: `LTIInjection` with `ρ(A) < 1` (the `get_A()` / "must be < 1" check in `example.py`) *is* Parcae's negative-diagonal LTI constraint. In structure-review terms, Parcae is the clean case study of fixing **trainability** (make the descent well-conditioned via `ρ < 1`) so that an otherwise-unstable structure becomes not just trainable but *predictably scalable*.
