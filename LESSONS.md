# Lessons — debugging the looped-transformer experiments

A running log of what went wrong and how we fixed it, in the order we hit each problem.
Most of these came from the k-step arithmetic experiment (`experiments/kstep_skill.py`),
which is exactly the point of a controlled toy: it surfaces failures one at a time.

Status legend: ✅ confirmed by a run · ⏳ fix in place, not yet verified.

| # | Symptom | Root cause | Fix | Status |
|---|---------|-----------|-----|--------|
| 1 | Data stage couldn't be tested; torch unavailable in sandbox | Heavy deps not installable where we iterate | Keep data generation torch-free; verify data offline, compile-check trainers, run training on the user's machine | ✅ |
| 2 | Depth-1 test set was empty | Tiny low-depth problem space got fully consumed by the train set | Enlarge the space (2-digit start) **and** build the test set first, then exclude it from train | ✅ |
| 3 | Accuracy ~6% even at k=1 | Numbers emitted big-endian → multi-digit arithmetic is unlearnable autoregressively (need the carry before the high digit) | Digit-**reverse** all predicted numbers (least-significant first); un-reverse at eval | ✅ |
| 4 | Still ~6%, loss floored ~0.7 | Loss computed over the whole sequence, incl. the **random prompt operands** that are irreducible noise → gradient diluted | Mask the loss to the **completion only** (trace + answer) | ✅ |
| 5 | More loops at test time hurt (T=8 < T=4) | Fixed loop depth in training → "training depth is the ceiling" (Parcae) | Train with **dynamic depth** (loop count sampled per step) | ⏳ |
| 6 | k=1 perfect but k≥2 at chance | "Running-values" format put the first value in a different position than the rest → model learned a **positional shortcut**, not the recurrence | **Interleaved** format: each step is a local `X op d = Y`, reusing the k=1 circuit with the previous result as the next operand | ✅ (k≤4: .99/.97/.93/.91) |
| 7 | In-dist great, k≥5 collapses to ~0 | **Learned absolute position embeddings** don't cover sequences longer than training; layout disintegrates | **NoPE** (no token positional encoding) | ⚠️ partial |
| 8 | NoPE: step **layout now generalizes** to k=5–8, but **arithmetic breaks everywhere** (even k=1 ~10%) | Removing all position signal also removed the **digit alignment** reversed addition needs (units-with-units); absolute position helps arithmetic but blocks length gen | Use **relative / rotary (RoPE)** positions — local alignment *and* translation-invariant length gen | ⏳ |
| 8b | Confirmation: `--pos sinusoidal` restored arithmetic (k≤4: .97/.99/.96/.91) but k≥5 still ~0 — and now it **skips operators / stops after ~4 steps** | Sinusoidal is still **absolute** → anchors "stop near depth 4"; doesn't length-generalize | Same conclusion: RoPE | ✅ confirmed (abs PE fails length) |
| 9 | `--pos rope`: arithmetic ✓ (k≤4: .99/.98/.96/.92) but k≥5 **still stops after ~3 steps** | RoPE fixes digit alignment, but ANY positional handle (abs *or* relative) lets the model learn a depth-bounded "stop after ~4" shortcut. Only **NoPE** generalized step count — because with no position the only learnable continue/stop rule is content-based ("any operator left?") | PE alone is insufficient. The token-CoT framing turned *depth* into *sequence-length generalization* (a separate hard problem). Real fix is elsewhere → see below | 🔚 PE avenue exhausted |

---

## The throughline

Almost every bug was a **representation or training-signal** problem, not a model-capacity
problem. The looped architecture was never the bottleneck; how we fed and scored it was.

A few principles that kept paying off:

**Reach for the controlled toy.** k-step arithmetic with an explicit depth dial let each
failure show up in isolation — arithmetic (3), signal (4), recurrence (6), length (7) — instead
of as one undifferentiated "it doesn't work."

**Read the samples, not just the metric.** The aggregate accuracy said "broken" four different
ways. Printing actual generations (`--show-samples`) is what distinguished *can't compute* from
*computes but won't chain* from *chains but can't count past 4*. Each diagnosis pointed to a
different fix.

**Match the data format to the circuit you want.** The single biggest jump (6) came from making
every step look identical and local, so the model could reuse one learned operation instead of a
position-dependent shortcut. Format choices are load-bearing:
- digit-reversal makes carry tractable left-to-right,
- completion-masking concentrates the gradient on what's deterministic,
- interleaving makes the recurrence a repeat of a known local op,
- NoPE removes the absolute-position anchor that blocks length generalization.

**Errors compound multiplicatively.** A k-step task needs every step right, so accuracy ≈
(per-step accuracy)^k. "k=1 is perfect, k=2 is at chance" is a signature of a step that *doesn't
transfer*, not of a slightly-noisy step.

**Two skills, two requirements — don't fix one and break the other.** This task needs both
*structure* (copy the next operator, chain k steps, stop) and *arithmetic* (add a digit with
carry). Structure wants **no absolute position** (so it length-generalizes); arithmetic wants
**local position** (to align digits). Learned PE serves the second and sabotages the first;
NoPE does the reverse. The lesson: when a fix swings a metric hard in both directions, you've
found a representation serving two needs that pull apart — the answer is usually a *relative*
encoding that satisfies both, not picking a side.

## Honest gaps

- Torch can't run in our iteration sandbox, so fixes (5) and (7) are in the code but **not yet
  confirmed by a run**. Each prior unverified fix that "should work" sometimes only half-worked
  (reversal + dynamic depth fixed k=1 but not k≥2) — so treat ⏳ rows as hypotheses until the
  table says otherwise.
- Test-time loop scaling (5) is a separate axis from depth extrapolation (7); don't read the
  `acc@T=8` column as the extrapolation result.

## The big reframe (after exhausting positional encodings)

We tried all four PE schemes. None gave depth extrapolation, and the reason is structural, not
a missing hyperparameter:

**Externalizing each reasoning step as tokens turns "reasoning depth" into "sequence-length
generalization."** That is a hard, well-known open problem in transformers, and it's *not* the
claim the looped-transformer literature actually makes. The papers (Universal Transformers,
Loop-Think, Parcae) put depth on the **loop axis**: iterate one block T times, decide when to
stop with a halting mechanism (ACT), and **add more loops at test time** to think longer. Test-
time depth extrapolation there means "run more iterations," not "emit a longer token sequence."

So the honest status of the k-step experiment:
- ✅ It cleanly demonstrates *learning a controlled k-step skill in-distribution* (k≤4 ≈ 0.95+),
  and it was a superb microscope for the representation/training-signal bugs (1–8).
- 🔧 It does **not** demonstrate depth extrapolation, because the CoT-as-tokens design tests the
  wrong axis for that question.

### Two ways forward
1. **Loop-axis depth (the real test of the paper's claim):** drop the token scratchpad; have the
   model emit just the answer after running T internal loops, train with dynamic T + ACT halting,
   then evaluate with **more loops** than trained. This directly asks "does more iteration =
   more reasoning depth?"
2. **Keep token-CoT but attack length generalization head-on:** depth curriculum (train k=1→6),
   index hints, or randomized/position-coupling tricks. Useful, but it's studying length
   generalization, not the loop.

---

## Conceptual conclusions — structure vs behavior

The bigger lessons from working through what the experiments mean:

1. **Looped = a weight-tied deep stack with a test-time loop dial.** Unroll the loop and you get
   a deep network whose layers share one set of weights, plus input re-injection, a loop-index
   embedding, and a contractive carry. The only thing it has beyond a tied deep stack is that you
   can change the number of loops at inference.

2. **Two memories, two axes — don't confuse them.** The loop's hidden state is a *fixed-size,
   contractive working register* (refines step by step and forgets, per position). The *full
   sequence* is remembered by attention over the per-token states / KV — not by the loop. So
   "lossy loop" does not mean "can't remember the sequence"; the contraction is along the
   depth/iteration axis, not the token axis.

3. **We tested the wrong axis.** k-step CoT put each step in a token, so "harder" meant "longer
   sequence" = **length generalization**, a separate hard problem (positions + composition), not
   the loop's claim. The loop's actual claim is depth on the **iteration axis** at fixed I/O
   length. "Learn 4 / test 8" is the right idea — but the 4 and 8 must be *loop counts on a
   constant-length input*, not step-counts in the output.

4. **Length generalization is its own beast — and out of scope.** It needs dedicated methods
   (abacus/index embeddings, position coupling, NoPE, randomized positions) and is best handled
   at the **data / post-training** layer (CoT, instruction tuning from a base checkpoint), not by
   the architecture.

5. **Outcome-only loss doesn't teach repetition.** Supervising only `input → answer` rewards the
   result, never the procedure, so nothing ties "one loop" to "one step" and the model learns
   length-bounded shortcuts. You must **supervise the iterations** (deep / per-iteration
   supervision) — like a teacher showing each running total, not just the final answer:
   "add 3 numbers = add 2, then add that result to the third." Teach the invariant step → it
   repeats. (Exception: fixed-point/DEQ tasks, where extra iterations just converge.)

6. **Structure gives capability, not behavior.** Under a vanilla next-token objective the looped
   model is just **autocomplete, like base GPT** — same loss, same behavior as a vanilla
   transformer. Long reasoning does *not* emerge for free from the structure. It must be
   *installed* by post-training: **SFT on reasoning traces** (the `gen_math_*` pipeline) and/or
   **RL with verifiable rewards**. Even using test-time depth is a *learned* behavior, not a
   freebie.

7. **So what the looped transformer actually buys:** parity with a vanilla transformer on
   reasoning/algorithmic tasks at **fewer parameters** (depth via tied iteration), trading away
   some memorization capacity — roughly what our in-distribution k-step result already shows. Its
   *unique* pitch (think-longer-at-test-time) is the still-untested, deferred experiment in
   `PROBLEM.md`.

8. **The smoke test's two trainings are not a two-phase pipeline.** `train_addition.py` and
   `train_sft.py` are independent from-scratch runs with different vocabularies; neither loads the
   other's checkpoint. The real recipe is *chained on one model*: pretrain base (autocomplete) →
   SFT-from-checkpoint on traces → RL. Building that faithfully needs a shared vocab + checkpoint
   loading (deferred).

**One-line takeaway:** the architecture is a cheaper, depth-capable *substrate*; the reasoning is
*post-training*. Same as base-GPT → o1/R1 — structure changes the cost and adds a test-time knob,
it doesn't change the fact that behavior is trained.

---

## Phase B — first multi-hop KG results (looped reasoner)

Setup: small world (~3k facts, memorizable), looped reasoner with Parcae injection + deep
(per-iteration) supervision, dynamic depth R_max=10, 20k steps. Fair eval uses matched **T=k**.

| Capability | Result | Read |
|---|---|---|
| Fact memorization | ✅ k=1 = 1.00 (incl. sys facts never queried) | world must be sized to model capacity (an early 80k-fact world failed outright) |
| In-distribution composition | ⚠️ works but decays fast: k=2 .90, k=3 .58, k=4 .27, k=5 .13 | the loop chains *practiced* transitions; later hops less reliable |
| Systematic generalization | ❌ k≥2 ≈ 0 | composes familiar facts, not novel ones |
| Depth extrapolation | ❌ k=6–10 ≈ 0 at matched T=k | the **"training depth is the ceiling"** negative — falsifies H1 *for this config* |

**Why systematic/extrapolation fail (the mechanism).** Hop 1 (`head + r₁`) uses an *input-token*
key → it's the memorized 1-hop lookup, works everywhere. Hop ≥2 (`eₜ + r`) must key the lookup on
an **internally-derived** entity. For `iid` that exact transition was traversed in training, so
it's learned path-specifically; for `systematic` the fact was memorized only with the entity as an
*input token*, and the model never learned that its **internal** representation of that entity is
the *same key*. So composition ≠ a general entity-keyed lookup — it's memorized transitions.

The needed property — "an entity looks the same whether given to you or derived by you" — is the
representational alignment Loop-Think reports emerging only after a long **grokking** phase. 20k
steps is almost certainly far short, and we diverged from their recipe (we kept the Parcae input
injection, which re-anchors to the original query each loop and can fight the state evolving into
"current entity"; they used a **bare** loop, no injection, zero-init).

**Open levers (not yet tried):** (a) train *much* longer (grokking is delayed); (b) bare config —
injection off + zero-init, to match the setup that achieved it; (c) ACT halting for the
overthinking seen when loops ≫ k. The substrate and memorization are sound; the missing piece is
the alignment that makes composition general.
