# Problem statement — does looping buy test-time reasoning depth?

Write the problem down before building. This file defines the question, why our current
experiment doesn't answer it, and what a clean experiment must look like. No solution here —
just the spec to build against.

## 1. The question

> If we train a weight-tied looped transformer to solve problems that need up to **T** internal
> iterations, can it solve **harder problems that need more iterations** at test time **just by
> running the loop more times** — with no extra parameters and no retraining?

This is the core claim of the looped / recurrent-depth literature (Universal Transformers,
Loop-Think, Parcae): reasoning depth is a *test-time dial* (number of iterations), separate from
knowledge (parameters). We want a controlled experiment that confirms or falsifies it.

## 2. The distinction that everything hinges on: two "depth" axes

A model can realize a k-step computation in two different places:

- **Token axis (external CoT).** Emit each intermediate step as text; depth = sequence length.
  Extrapolating to harder problems = generating a *longer sequence* than seen in training.
- **Loop axis (internal recurrence).** Re-apply one block across iterations; depth = number of
  loops. Extrapolating = running *more iterations* on a same-length input.

These are different questions. The loop-axis one is the paper's claim. The token-axis one is
"length generalization," a separate and notoriously hard problem.

## 3. Why our k-step experiment does NOT answer the question

`experiments/kstep_skill.py` writes the full reasoning trace as tokens. So "deeper problem"
became "longer output sequence," i.e. we accidentally tested the **token axis**. Result
(see `LESSONS.md`): in-distribution learning is solid (k≤4 ≈ 0.95+), but extrapolation to k≥5
fails for every positional encoding, because that's length generalization, not looping.

Conclusion: to test the real claim, the per-example **input/output length must stay fixed**
while the **required number of iterations varies**. The work has to move off the token axis and
onto the loop axis.

## 4. Design constraints a valid experiment must satisfy

1. **Fixed I/O size across difficulty.** Input length and output length must NOT grow with the
   number of reasoning steps — otherwise we're back to length generalization.
2. **Difficulty = iteration count.** There must be a task knob `k` that provably requires ≈k
   sequential steps, independent of input size.
3. **Test-time depth is the only thing changed at eval.** Same weights; vary only the loop
   budget T (and/or let a halting mechanism choose it).
4. **A real control.** Compare against (a) a fixed-depth model and (b) the same model evaluated
   at its training depth, so any gain is attributable to *more loops*, not to the architecture
   in general.
5. **Exact, automatic verification.** Programmatic gold, as we already do.

## 5. Task — adopt Loop-Think's design, don't reinvent

The `/summary` papers already contain the right experiment. **Loop, Think, & Generalize**
(Kohli 2026, arXiv:2604.07822) reasons in a **single forward pass on a fixed-size input** and
tests exactly our question (train ≤k hops, test deeper). Adopt its setup:

- **Task:** synthetic **multi-hop knowledge-graph queries** — compose stored facts
  (hop1 → hop2 → …) in one forward pass, *no token CoT*. Difficulty = number of hops = required
  iterations; input and output length are fixed regardless of hops → satisfies constraint #1 by
  construction. (Their mechanism: looping folds knowledge into one re-readable bank and re-queries
  it each cycle, so hop *t* happens on iteration *t*.)
- **What to reproduce/extend:** recurrence enables **systematic generalization** (a three-stage
  grokking curve) and **depth extrapolation** — inference-time recurrence solves deeper-than-
  trained hops, *but only if* training used **enough recurrence (R > 4)** and **dynamic
  recurrence** (`R ~ clip(Poisson(λ), R_min, R_max)`).

Fallback fixed-size tasks for a second datapoint: iterated permutation/automaton applied k times,
or a pointer-chase repeated k times. (k-step *arithmetic-with-emitted-trace* is ruled out — it
grows with k; that was our mistake.)

## 6. Protocol (once the task is fixed)

- **Train:** dynamic loop depth T sampled per example up to `T_train_max`, plus ACT halting so
  the model learns to stop when done. **Supervise the iterations, not just the answer**
  (deep / per-iteration supervision): decode the loop state after iteration *t* and train it
  toward the *t-th intermediate result*, so each iteration learns the *same* one reusable step.
  Supervise only up to `T_train_max` iterations.
- **Eval:** hold weights fixed; for each difficulty `k` (including `k > T_train_max`), measure
  accuracy at loop budgets T = 1 … several× `T_train_max`. No intermediate supervision at test.
- **Headline plot:** accuracy vs (k, T). The claim predicts that harder `k` becomes solvable as
  T increases, *past* the training depth.

### 6.1 The supervision principle — why outcome-only loss fails

Teaching the loop is like teaching a child to add a list: you don't just show the final total,
you show the **method** — "adding three numbers = add the first two, then add that running total
to the third," and so on. The invariant is a single step applied to the *previous result*.

- **Deep supervision** shows the model each running total (68, then 61, …), so iteration *t*
  is explicitly trained to perform that one reusable step. Learn the invariant step → repeating
  it handles any *k*, and more loops at test time = more steps. This is what makes the loop
  actually *repeat the calculation*.
- **Outcome-only loss** (`input → answer`) rewards the result but never the procedure. Nothing
  ties "one loop" to "one step," so the model learns length-bounded shortcuts instead of the
  invariant — exactly the "stop after ~4" failure we already observed. (Earlier draft of this
  doc said "loss on the answer only"; that was the trap.)

(Exception: if the answer is a **fixed point** the loop converges to — Deep Equilibrium style —
outcome-only can work, since extra iterations just converge further. A genuine *k-step
sequential* computation is not a fixed point, so it needs the per-iteration signal.)

### 6.2 Concrete recipe knobs from the /summary papers

Don't tune blind — these are already pinned down in the literature:

- **Recurrence schedule (Loop-Think, 2604.07822):** train with **dynamic** recurrence and
  **R_max > 4**. Inference-time depth extrapolation does *not* unlock if trained shallow — our
  k-step run used 1–4, which is borderline/too low. Go higher.
- **Latent-reasoning curriculum (Coconut, 2412.06769):** the deep supervision in §6.1 can be
  staged Coconut-style — start with explicit per-step targets, then **progressively replace them
  with continuous latent thoughts** (feed the hidden state back instead of decoding a token), so
  reasoning migrates into the loop's latent space and dodges the token bottleneck. Plain
  cross-entropy, fully differentiable, no RL needed.
- **Halting (Universal Transformer ACT):** add per-token adaptive halting. Loop-Think shows
  **overthinking** degrades very-deep generalization, so a learned stop is required (this is H3).
- **Stability:** Parcae negative-diagonal ρ<1 injection (what we already have) **or** Loop-Think's
  **zero-init** of attention/FFN output projections (identity-at-init) — a cheaper alternative.
- **Theory anchor (Saunshi, 2502.17416):** L loops ≈ T steps of CoT; expect downstream accuracy
  to scale with *effective depth*, and expect a memorization↔reasoning tradeoff vs an iso-FLOP
  (more-parameter) baseline.

## 7. Hypotheses & falsifiable predictions

- **H1 (depth = iterations):** for k ≤ T, accuracy is high; for k > T, accuracy rises as eval T
  increases — i.e. you can "buy" correctness with more test-time loops. *Falsified if* accuracy
  saturates at `T_train_max` regardless of extra eval loops (Parcae's "training depth is the
  ceiling" — itself an interesting, publishable negative).
- **H2 (stability):** with the Parcae ρ<1 injection, running extra loops does not blow up the
  residual; accuracy degrades gracefully rather than diverging. *Falsified if* extra loops cause
  instability/garbage.
- **H3 (halting):** ACT learns to use ≈k loops for difficulty k. *Falsified if* the halpting
  step count is uncorrelated with k.

## 8. Confounds to rule out

- More eval loops helping merely because of *train/test loop-count mismatch* (fix: dynamic-depth
  training so a range of T is in-distribution).
- The task secretly solvable in O(1) by a shortcut (fix: verify a fixed-depth/shallow baseline
  fails at high k).
- Length creeping back in (fix: assert input & output length are independent of k).
- **Training recurrence too shallow** so extrapolation never had a chance (fix: R_max > 4 with
  dynamic recurrence, per Loop-Think — the likely reason a naive 1–4 setup would fail).

## 9. What success looks like

A single figure: accuracy held high for k beyond `T_train_max` *when and only when* eval loops
are increased, with a fixed-depth baseline flat. That — not the k-step token experiment — is the
evidence that looping is a genuine test-time reasoning-depth axis.

## 10. Decision: task settled → first build step

**Resolved** (was the open question): adopt **Loop-Think's multi-hop knowledge-graph** task
(single forward pass, fixed I/O length, difficulty = hops). It satisfies constraint #1 by
construction and has reference code (github.com/OSU-NLP-Group/Loop-Think-Generalize).

So the first thing to build is **the data generator for multi-hop KG queries** (atomic facts +
compositional queries, with held-out compositions and deeper-than-trained hops for the
extrapolation split) — not another arithmetic variant. Everything else (dynamic R>4 training,
deep/Coconut-style supervision, ACT, fixed-depth baseline) follows §6.

## References (local summaries → arXiv)

- Loop, Think, & Generalize — [`summary/loop_think_generalize.md`] · arXiv:2604.07822
- Coconut (continuous latent reasoning) — [`summary/coconut_continuous_latent.md`] · arXiv:2412.06769
- Reasoning with Latent Thoughts — [`summary/reasoning_with_latent_thoughts.md`] · arXiv:2502.17416
- Universal Transformers (ACT halting) — [`summary/universal_transformers.md`] · arXiv:1807.03819
- Parcae (stability substrate, what we tested) — [`summary/parcae.md`] · arXiv:2604.12946
