# Phase A — Multi-hop KG data generator

The first concrete build step from [`PROBLEM.md`](PROBLEM.md) §10. The goal of this phase is
**data, not training**: produce a verifiable, fixed-I/O-length multi-hop KG corpus that the
Phase B trainer can consume to test the real claim ("more loops at test time = more reasoning
depth"). No model code lives here.

## 1. Why this task

The k-step arithmetic experiment failed the spec by externalizing each step as a token, which
turned "deeper problem" into "longer sequence" — length generalization, not looping
(`LESSONS.md` §"big reframe"). Multi-hop KG queries fix this by construction: difficulty (hop
count) is independent of input/output length. The query is a *fixed-shape* prompt and the
answer is a *single entity token*; only the **number of internal iterations needed** changes
with k. This is the Loop-Think setup (`summary/loop_think_generalize.md`), which we adopt
rather than reinvent.

## 2. What Phase A produces

Three artifacts on disk, all consumable by Phase B without further preprocessing:

1. `data/kg/facts.jsonl` — the atomic fact bank. One JSON object per fact:
   `{"head": int, "rel": int, "tail": int}`. This is the model's "knowledge" — it will be
   shown these facts during training (interleaved with queries or as a memorization phase;
   that's a Phase B knob).
2. `data/kg/queries.jsonl` — labeled queries. One JSON per query:
   `{"head": int, "rels": [int, int, ...], "answer": int, "k": int, "split": str}`.
   `split` ∈ `{train, iid_test, systematic, extrapolation}` (see §4).
3. `data/kg/meta.json` — vocab + split sizes + generation seed + the exact knobs used. Enough
   to reproduce the corpus from the seed alone.

Token IDs (entity, relation) are dense integer ranges. Token-to-string mapping is **not
stored** — Phase A operates entirely in integer space. (We're testing the loop, not language.)

## 3. The graph and the queries

**Entities** `E`: a set of integer IDs, size `|E|`.
**Relations** `R`: a set of integer IDs, size `|R|`. Relations are *functional* — for each
`(head, rel)` there is exactly one `tail`. (Functional relations are required for queries to
have a unique answer.)

**Atomic facts.** Sample `N_facts` triples `(h, r, t)` such that no `(h, r)` collides. The
underlying graph is a forest of functional edges; chains of length k are well-defined paths.

**k-hop query.** Pick a head entity and a length-k sequence of relations
`r_1, r_2, …, r_k`; the answer is the unique entity reached by `r_k(…r_2(r_1(h))…)`. Encode
as a fixed-shape prompt:

```
prompt:  [BOS] [HEAD=h] [REL=r_1] [REL=r_2] … [REL=r_K_max] [QUERY]
answer:  [ENT=t]
```

For k < K_max, pad the unused relation slots with a dedicated `[PAD_REL]` token (or
equivalently, define an identity relation that the generator never assigns a fact to and
which the model can learn to skip — Phase B decides which is cleaner). **Either way, the
sequence length is exactly `K_max + 3` for every query, regardless of k.** That is the
constraint that makes this a clean test of loop-axis depth.

## 4. Splits — and what each one tests

Four splits, generated jointly so they're disjoint by construction:

- **`train`** — queries with k ∈ `[1, K_train_max]` whose constituent facts may overlap
  freely with other train queries. The model's main supervision diet.
- **`iid_test`** — held-out queries from the same distribution (same k range, same fact
  reuse pattern). Confirms in-distribution learning works.
- **`systematic`** — queries whose **atomic facts were never used inside any train query**
  (the facts appear in `facts.jsonl`, so the model has had a chance to memorize them, but no
  compositional query in `train` ever traversed them). Tests systematic generalization —
  Loop-Think's first headline finding. Hop count in this split stays ≤ `K_train_max`.
- **`extrapolation`** — queries with k ∈ `(K_train_max, K_eval_max]`. Tests depth
  extrapolation — Loop-Think's second headline finding. Facts may be shared with train.

The generator must guarantee these are disjoint as sets of `(head, rels)` tuples and that the
`systematic` split's facts truly never appear in any `train` query path. Verifier is a small
post-generation sanity script (see §6).

## 5. Knobs (defaults set so a single laptop run is feasible)

| Knob | Default | Notes |
|---|---|---|
| `|E|` (entities) | 1000 | Loop-Think uses ~10³–10⁴; start small for fast iteration. |
| `|R|` (relations) | 20 | Few enough that each relation gets dense coverage. |
| `N_facts` | 5000 | Bounded by `|E|·|R|` since relations are functional. |
| `K_train_max` | 5 | Strictly > 4 per Loop-Think (R > 4 is the threshold for extrapolation). |
| `K_eval_max` | 10 | Test up to 2× the training depth. |
| `K_max` (prompt slots) | 10 | Equals `K_eval_max`; sets the fixed prompt length. |
| `N_train` | 50000 | |
| `N_iid_test` | 2000 | |
| `N_systematic` | 2000 | |
| `N_extrapolation` | 2000 per k | i.e. 2000 each for k = K_train_max+1 … K_eval_max. |
| `seed` | 0 | Single source of randomness; written to `meta.json`. |

## 6. Sanity checks (run by the generator before writing)

These are cheap, deterministic, and catch the silent failures that bit us in the k-step
experiment:

1. **Functionality.** No `(head, rel)` appears twice in `facts.jsonl`.
2. **Fixed length.** Every prompt tokenizes to exactly `K_max + 3` tokens; every answer to 1.
3. **Answerability.** For every query, the path of `k` hops exists in `facts.jsonl` and ends
   at the labeled answer. (Re-run the chain; assert equality.)
4. **Split disjointness.** No `(head, rels)` tuple appears in two splits.
5. **Systematic-split purity.** For each query in `systematic`, every fact on its path is
   *absent* from every query path in `train`. (This is the expensive check; do it once.)
6. **Length-vs-k independence.** Sanity assertion that no token in the prompt sequence
   encodes `k` — i.e. the only signal of difficulty is the count of non-`PAD_REL` relation
   slots, which the model must *infer*, not read off a position.

Any failed check aborts generation with a non-zero exit code. We are not shipping a corpus
that violates constraint #1 of `PROBLEM.md` §4 by accident.

## 7. Out of scope for Phase A

To keep this phase tight, the following live in later phases:

- **Model code**, including the recurrent block, the loop-index embedding, and the ACT
  halting head. (`src/loop_scaling_lab.py` has a starter looped block; Phase B will adapt.)
- **Deep / per-iteration supervision** signals (PROBLEM.md §6.1). Phase A emits only
  `answer`; Phase B decides whether to *also* supervise intermediate hop targets by re-deriving
  them from `(head, rels[:t])` at training time.
- **Dynamic loop schedule** (`R ~ clip(Poisson(λ), R_min, R_max)`) — a Phase B knob.
- **The fixed-depth baseline** — Phase B trains it from the same data.
- **The headline plot** (accuracy vs (k, T)) — Phase C eval.

If a question comes up that touches any of the above, the answer for Phase A is "defer."

## 8. Done criteria

Phase A is done when:

1. `data/kg/{facts,queries,meta}.jsonl` exist with the four splits at the sizes in §5.
2. All six §6 sanity checks pass.
3. A 20-line "load and print 5 samples per split" script demonstrates the format end-to-end.
4. The generator is **torch-free** (per LESSONS.md #1 — data must run in any sandbox).

No model has been trained at this point. That's correct. Phase B starts from these files.

## 9. Where it goes in the repo

```
experiments/
  gen_kg_queries.py        # Phase A entrypoint — single file, stdlib + nothing else
data/
  kg/
    facts.jsonl
    queries.jsonl
    meta.json
```

`gen_kg_queries.py` exposes a `main(seed, out_dir, **knobs)` so Phase B (or a notebook) can
regenerate from code without shelling out. CLI is a thin wrapper.

## References

- The task and splits: `summary/loop_think_generalize.md`
- Why fixed I/O length is non-negotiable: `PROBLEM.md` §4
- Why a torch-free generator: `LESSONS.md` row 1
- Why we are not extending k-step arithmetic: `LESSONS.md` "big reframe"

---

## Implementation status (built & verified)

Implemented in **`phase_a/`** (not `experiments/`, to keep the phase self-contained):
`gen_kg_queries.py` (generator, importable `main(seed, out_dir, **knobs)`) and `inspect_kg.py`
(sample printer). Output goes to `phase_a/data/kg/` (gitignored).

Two spec changes from the review, both load-bearing:

1. **Constructive forward walks** — paths are built by walking forward and choosing relations
   that are defined or creatable, never by sampling a relation sequence and hoping it resolves
   (which made deep chains ~1e-6 likely). Deep k=6–10 chains now generate reliably.
2. **Reserved edges for a pure systematic split** — a per-head fraction (`sys_reserve_frac`)
   of `(head, rel)` edges is reserved for the sys pool; train may only create on non-reserved
   slots, sys only on reserved. Without this, train queries saturate `E*R` and starve the
   systematic split.

3. **World size scaled to the model (Phase B finding).** The model must memorize every fact in
   its parameters. An initial large world (E=5000,R=20 → ~80k facts) was far beyond a small
   looped model's associative-memory capacity — even 1-hop recall failed (loss ≈ random). The
   defaults are now **E=300, R=10 → ~3k facts**, which a dim-256 block can store. Lesson: scale
   the world *with* the model, not independently.

Verified run (`seed=0`, defaults): 2,937 facts (2,100 train-owned / 837 sys-owned); splits at
target (train 20000, iid_test 1000, extrapolation 2500 = 500 each for k=6..10, systematic 1000);
all six §6 checks pass in ~1s. Independent cross-checks (at the earlier larger world): systematic
vs train query paths share **0 edges**; most-common answer ≪1% of queries (no guess-the-mode
shortcut).

Regenerate: `python phase_a/gen_kg_queries.py` · inspect: `python phase_a/inspect_kg.py`.
