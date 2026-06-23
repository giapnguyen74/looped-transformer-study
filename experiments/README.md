# Loop Scaling Lab

A tiny, self-contained lab to *experience* the scaling laws of looped / recurrent-depth
transformers — the phenomena summarized in [`../summary/`](../summary/). It trains a
minimal looped transformer on a synthetic **addition** task and runs four micro-experiments.

## Setup

```bash
pip install -r ../requirements.txt   # torch, sympy, matplotlib, requests
# or minimally for just the lab:  pip install torch matplotlib
```

Single GPU recommended; defaults run in minutes. Everything is in one file:
[`loop_scaling_lab.py`](loop_scaling_lab.py). Results (CSV + PNG) land in `experiments/results/`.

## Run

```bash
python loop_scaling_lab.py stability     # Exp 1
python loop_scaling_lab.py iso           # Exp 2
python loop_scaling_lab.py testtime      # Exp 3
python loop_scaling_lab.py isoflop       # Exp 4
python loop_scaling_lab.py all           # all four

# knobs
python loop_scaling_lab.py all --dim 256 --heads 4 --loops 4 --nd 3 --steps 2000 --lr 3e-4
```

## What each experiment shows

**1. `stability`** — trains the same loop with two state-carry rules:
- `free`: `h ← h + B·e + R̄` (carry `Ā = I`, ρ = 1) — residual norm grows, loss destabilizes.
- `parcae`: `h ← Ā·h + B·e + R̄` with `Ā = exp(softplus(Δ)·(−exp(log_A)))` diagonal, ρ < 1 — bounded.

*Expected:* in `stability.png`, the `free` residual-norm curve climbs (often orders of
magnitude / diverges) while `parcae` stays flat; `parcae` reaches lower, smoother loss.
This is the most visceral plot — you watch ρ ≥ 1 explode.

**2. `iso`** — three models at fixed hidden size:
- `shallow` (k unique layers, T=1), `looped` (k layers, T=L), `deep` (kL layers, T=1).

*Expected:* `looped` (few params) ≈ `deep` (L× params) ≫ `shallow` (few params). Depth via
looping buys most of what extra parameters would, at the shallow model's parameter count.

**3. `testtime`** — train with dynamic depth (T ~ Poisson(mean)), then evaluate at
`T = 1, 2, …, 4×` train mean.

*Expected:* `testtime.png` shows loss dropping then **saturating** near the training depth —
training depth sets the test-time ceiling (Parcae's finding; contrast with the unbounded
extrapolation seen on tiny synthetic tasks in Loop-Think — try `--nd 2` vs `--nd 5`).

**4. `isoflop`** — fixed compute budget (`loops × steps` held constant), vary loop count T
(so more loops ⇒ fewer steps ⇒ less data).

*Expected:* a U-shape / optimum in `isoflop.csv` — at fixed compute there is a best loop
count; too few loops underuses depth, too many starves data. That optimum existing at all is
the point: **looping is an orthogonal scaling axis** alongside data.

## Notes & knobs

- Primary metric is **validation cross-entropy on the answer tokens** (what scaling-law plots
  use); answer-token and exact-sequence accuracy are reported too.
- Eval is **teacher-forced** (fast). For a stricter number, swap in autoregressive decoding.
- To make the `free` instability more dramatic: raise `--loops` (deeper unroll compounds the
  ρ=1 carry) or `--lr`. To make addition harder (more reasoning depth needed): raise `--nd`.
- This mirrors OpenMythos's real components: the `parcae` injection here is the same
  negative-diagonal LTI trick as `open_mythos/main.py:LTIInjection`; `loop_index_embedding`
  matches the repo's; the block is a stripped TransformerBlock (no MoE/MLA, for speed).

## Suggested first run

```bash
python loop_scaling_lab.py stability --loops 8     # see the explosion vs the leash
python loop_scaling_lab.py iso                      # see looping ≈ depth
```

---

# Data pipeline — `gen_math_problems.py` + `gen_math_transcripts.py`

The *data* side of training a looped model, in two separate scripts coupled only by a
`problems.jsonl` file. Where the lab studies the architecture, these build the reasoning
corpus you'd SFT it on: **Stage A** authors verifiable problems, **Stage B** turns them
into verified solution traces (the STaR / rejection-sampling recipe).

```
 STAGE A — gen_math_problems.py             STAGE B — gen_math_transcripts.py
 ┌─────────────────────────────┐           ┌────────────────────────────────────────┐
 │ programmatic: exact SymPy    │           │ [1] SAMPLE k candidates (high temp)      │
 │   gold (trustworthy, offline)│  ──────▶  │ [2] VERIFY \boxed{} vs gold (SymPy eq:    │
 │ teacher: model proposes,     │ problems  │     1/2 == 0.5 == 50%)                    │
 │   gold left empty            │  .jsonl   │ [3] FILTER dedupe·length·degenerate      │
 │ mock: offline, no gold       │           │ [4] CURATE K diverse traces · tag DEPTH  │
 │                              │           │     → suggested loops                     │
 └─────────────────────────────┘           └────────────────────────────────────────┘
                                                          → transcripts.jsonl
```

**Two ways to get verifiable problems** (Stage A). *Programmatic* templates (arithmetic,
linear equations, fraction sums, gcd, percent) compute gold in code — trustworthy by
construction. *Teacher-proposed* problems come from a model, so their gold can't be trusted
and is left empty; Stage B then fills it by **self-consistency** (solve each k times, take
the majority answer, drop problems where agreement < `--consensus`). Caveat: if the same
model proposes and solves, a confidently-wrong consensus can mislabel — prefer programmatic
gold, or solve with a different model than the proposer.

Why this shape: math answers are **verifiable**, so the keep/reject signal is exact —
the reason reasoning pipelines anchor on math. And because *the loop is internal CoT*,
each transcript is tagged with a depth bucket (`easy/medium/hard` → `T~2/4/8`) so the
trainer can pair hard problems with more loops (dynamic / Poisson depth). This is the
data-side complement to the `testtime` experiment's finding that **training depth sets
the test-time ceiling** — keep the hard, long-chain tail or the loop never learns to use
its iterations.

## Run

```bash
pip install sympy            # required (verification);  requests only for the OpenRouter teacher

# offline, end-to-end demo (given-gold AND self-consistency paths, no keys)
python gen_math_transcripts.py demo

# --- real run ---
export OPENROUTER_API_KEY=sk-or-...                 # required for the OpenRouter teacher
# optional: OPENROUTER_BASE_URL, OPENROUTER_REFERER, OPENROUTER_TITLE (app ranking)

# Stage A — make problems (pick one)
python gen_math_problems.py --source programmatic \
    --n 500 --kinds arith,linear,fraction,gcd,percent --out problems.jsonl
python gen_math_problems.py --source teacher --model openai/gpt-5.5 \
    --n 200 --topic "algebra word problems" --difficulty hard --out problems.jsonl

# Stage B — turn problems into verified transcripts
python gen_math_transcripts.py generate \
    --problems problems.jsonl --out transcripts.jsonl \
    --teacher openrouter --model openai/gpt-5.5 --samples 16 --temp 0.8 --keep 2 --consensus 0.5
```

Or skip Stage A and bring your own problems — JSONL with `question` and `gold` (or `answer`);
optional `id`, `source`. **Missing gold is allowed** — it's filled by self-consistency. Swap in
GSM8K / MATH / NuminaMath by dumping them to that format. The teacher is pluggable
(`MockTeacher` offline, `OpenRouterTeacher` for any model on [OpenRouter](https://openrouter.ai)
— `openai/gpt-5.5`, `deepseek/deepseek-r1`, `qwen/qwen3-235b`, …). Because `n>1` isn't uniformly
supported across OpenRouter's upstream providers, `--samples` is drawn as that many independent
requests at `--temp`.

## Output

Each line is one transcript: `question`, `gold`, `solution` (CoT ending in `\boxed{}`),
`steps` (split for **Coconut-style latent compression** — delete the first *k* steps,
replace with *k* loop iterations), `answer`, `n_steps`, `depth_bucket`, `suggested_loops`.
The run prints a funnel (sampled → verified → filtered) and a `sample→verified` keep rate;
a low rate means the problems are too hard for the teacher, a near-100% rate means too easy.

## How it feeds training

1. **Explicit-CoT SFT** — train on `solution` directly, using `suggested_loops` /
   `depth_bucket` to set per-example loop depth (or sample depth ~ Poisson(suggested)).
2. **Latent compression** (optional) — Coconut/CODI curriculum over `steps`: progressively
   move reasoning out of tokens and into the loop's continuous state.
3. **RL** (optional) — the same SymPy verifier here becomes the reward checker for
   GRPO-style training (and, per the looped-LM literature, reward the latent trajectory,
   not just the outcome).

---

# Controlled skill — `kstep_skill.py` (depth extrapolation)

The first "real" experiment after the smoke test: can a looped transformer learn a
**k-step arithmetic** skill and **extrapolate to depths it never trained on**? `k` (the number
of sequential +/− updates) is an explicit dial for required reasoning depth, so this isolates
the headline looped-model claim from every other confound.

The task is **exact and self-generated** (no teacher / rejection sampling). The default
`--format steps` writes an **interleaved** chain-of-thought where every step is a *local
single operation*, and all predicted numbers are **digit-reversed** (least-significant first):

```
prompt:  S 36 + 5 - 7 - 8 - 4 =                              (start 63 reversed, then k=4 ops)
CoT:                          36 + 5 = 86 - 7 = 16 - 8 = 35 - 4 = 94   (each '= val' computed)
answer:                                                              # 94 ;   (= 49 reversed)
```

Two design choices, both load-bearing:

- **Digit-reversed numbers** — emitting a multi-digit result most-significant-first is nearly
  unlearnable autoregressively (you'd need the carry before the high digit); the lab's
  addition task reverses for the same reason. Eval un-reverses before checking exact gold.
- **Interleaved local steps** (`steps`, default) — each step appears as `X op d = Y`, exactly
  the pattern the model masters at k=1, with the previous result as the next left operand (no
  un-reversing, since everything is reversed-consistent). This is what lets composition and
  depth-extrapolation work. The earlier `--format values` (just the running values
  `86 16 35 94`) made the first value sit in a different position from the rest, so the model
  learned a positional shortcut that nailed k=1 but collapsed at k≥2 — kept as an option so you
  can reproduce that failure.
- **Positional encoding = the length-generalization lever** (`--pos`, default `rope`).
  Empirically on this task: `learned` absolute → arithmetic fine in-distribution but k>4
  collapses (layout disintegrates past trained positions); `sinusoidal` → arithmetic fine but
  still skips/stops after ~4 steps (still absolute); `none` (NoPE) → step layout generalizes to
  any depth but per-digit arithmetic breaks (lost digit alignment). The two skills want opposite
  things, so the fix is **`rope`** (rotary/relative): q·k depends on *relative* position, giving
  local digit alignment for the arithmetic **and** translation-invariance for length
  generalization.

Training is SFT with the **loss masked to the completion** (trace + answer) only — the prompt's
operands are random, so scoring them would swamp the gradient with irreducible noise and starve
the deterministic computation we actually want learned (same idea as the lab's masked
answer-token loss).

The eval prompt is everything up to `=`; the model must **generate** the running values and the
answer. Default split: train `k=1–4`, test `k=1–8` (the `k>4` rows are pure extrapolation).
Training uses **dynamic depth** by default (loop count sampled per step in `[--min-loops,
--loops]`) so the model learns to use a *variable* number of iterations — this is what lets
extra loops help at test time. At eval it varies the **loop count** (e.g. T=4 vs T=8) to probe
**test-time loop scaling**: a looped model can in principle solve a deeper problem by iterating
its one block more. (`--fixed-loops` trains at a single depth — expect test-time scaling to
flatten or hurt, the Parcae "training depth is the ceiling" effect.)

## Run

```bash
pip install -r ../requirements.txt
python kstep_skill.py            # gen + train + eval (auto-selects cuda/mps/cpu)
# stages can run separately:
python kstep_skill.py gen        # writes kstep_data/{train,test}.jsonl (gen needs only python, no torch)
python kstep_skill.py train      # trains + prints the extrapolation table
# knobs:
python kstep_skill.py --n-train 40000 --steps 8000 --loops 4 --kmax-test 10
```

Defaults: ~20k train problems (depths 1–4), 200 test problems per depth (1–8), a tiny looped
LM (dim 128, 1 block, 4 loops), ~4000 steps. Runtime is a few minutes on a GPU/MPS, longer on
CPU — heavier than the smoke test. Results print as a table and save to
`results/kstep_extrapolation.csv`.

## Reading the result

```
  depth   acc@T=4   acc@T=8
   k=1      0.99      0.99
   ...
   k=4      0.95      0.96      <- edge of training distribution
 ■ k=5      0.71      0.78      <- extrapolation begins (k > train max)
 ■ k=8      0.30      0.41
```

High in-distribution accuracy (k≤4) means the skill was learned. Accuracy on the `■` rows is
**depth extrapolation**; if it degrades gracefully rather than collapsing, the loop generalized
its computation. Accuracy *rising with more loops* on the deep rows (acc@T=8 > acc@T=4) is
**test-time loop scaling** — evidence the model is using the extra iterations as extra reasoning
depth, exactly the Loop-Think / Parcae story the `summary/` notes describe.
