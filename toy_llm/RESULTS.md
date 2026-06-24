# Toy LLM — results

Parameter-matched comparison of vanilla vs looped char/byte-LMs, Parcae §5.1 protocol (sweep the
vanilla baseline, transfer hyperparameters to all models, no per-model tuning).

## Setup

- **Data:** enwik8 (byte-level, vocab 256, 90M train / 5M val). Large enough that no model
  overfits — `train ≈ val` and `final ≈ best` throughout (the Tiny-Shakespeare run overfit the
  deep baseline; that's why we switched).
- **Models** (all dim=256, heads=4, ctx=128, dropout=0.1): `vanilla-2L`, `parcae-2x4`,
  `bare-2x4` (block ×loops = effective depth 8), `vanilla-8L`.
- **Hparams:** lr=1e-3 (swept on `vanilla-8L`, transferred to all), wd=0.01, 20k steps,
  AdamW, single seed, CUDA. Best-val (early-stopping-style) reported.

## Result

| model | params | best val | ppl | best@ | final | time |
|---|--:|--:|--:|--:|--:|--:|
| vanilla-2L | 1,742,848 | 1.3370 | 3.81 | 18000 | 1.3390 | 446s |
| **parcae-2x4** | 1,808,896 | **1.2428** | **3.47** | 20000 | 1.2428 | 1834s |
| bare-2x4 | 1,742,848 | 1.2476 | 3.48 | 20000 | 1.2476 | 1744s |
| vanilla-8L | 6,478,336 | 1.1779 | 3.25 | 20000 | 1.1779 | 1697s |

(All `final ≈ best` → no overfitting. All except vanilla-2L were still improving at 20k → not
fully converged.)

## Reading

| comparison | numbers | verdict |
|---|---|---|
| **iso-param** (looped vs shallow, ~1.74M) | parcae 3.47 vs vanilla-2L 3.81 (Δ≈0.09 nats) | ✅ looping buys depth cheaply |
| **iso-depth** (looped vs deep, eff-depth 8) | parcae 3.47 vs vanilla-8L 3.25 (Δ≈0.065), deep has 3.7× params | ❌ looped does *not* fully match deep |
| **injection** (parcae vs bare) | 3.47 vs 3.48 (Δ≈0.005) | ➖ neutral at this scale/T (see below) |
| **compute** | parcae 1834s ≈ vanilla-8L 1697s ≫ vanilla-2L 446s | looped = deep's FLOPs, shallow's params |

## Conclusion

At matched parameters, looping yields a large perplexity gain over a shallow transformer
(3.47 vs 3.81); it captures **most but not all** of a 4×-larger deep model's quality (3.47 vs
3.25), trading parameters for compute (~4× the FLOPs of the shallow model). So on **general LM
perplexity**, depth-via-looping is a *favorable trade*, not a free substitute for parameters —
perplexity tracks both params and effective depth, and the param-rich deep model still wins.

This matches `summary/reasoning_with_latent_thoughts.md`: looped models **underperform iso-FLOP
(more-param) baselines on LM perplexity** but are competitive where the task is **depth-bound
reasoning**. So the regime where looped should *match* deep is a reasoning task (the i-GSM /
multi-hop KG game in `phase_b/`), not raw LM perplexity. The two experiments are complementary.

And the **Parcae injection's role is clear from the depth sweep below**: at low loop counts it's
neutral, but it's what *enables deep looping at all* — a bare loop collapses past eff-depth ~8,
while Parcae keeps converging to ~24. So the full statement is: looping is parameter-efficient
depth, and the ρ<1 injection is what lets you actually push that depth without the loop blowing up.

## Depth sweep — where the Parcae injection earns its keep

In the main comparison parcae ≈ bare, because at **T=4** the bare loop is shallow enough that
RMSNorm + residual keep it stable — there's no instability for the ρ<1 leash to fix. The injection
is a *stability* mechanism; it only matters at **high loop counts**. Sweeping T (model size and
data fixed, `--depth-list 2,4,8,12,16`, 8000 steps, enwik8) shows exactly that:

| T | eff_depth | parcae (best val) | bare (best val) | bare − parcae |
|--:|--:|--:|--:|--:|
| 2 | 4 | 1.345 | 1.374 | +0.03 |
| 4 | 8 | 1.310 | 1.322 | +0.01 |
| 8 | 16 | **1.293** | **3.503** | **+2.21** |
| 12 | 24 | **1.282** | 3.503 | +2.22 |
| 16 | 32 | 1.286 | 3.503 | +2.22 |

- **Parcae** improves with depth and stays stable throughout (best at T=12 / eff-depth 24), then
  gently saturates — looping is a usable depth axis *with* the leash.
- **Bare** is stable to T=4, then **falls off a cliff** between eff-depth 8 and 16 and sticks at
  ~3.50 (ppl ~33) for T=8/12/16 — it collapsed to a degenerate fixed point and never recovers
  (finite, not NaN, so `bare_div=False`).

This is a **threshold/cliff**, not a gradual gap: below it parcae≈bare, above it the difference is
~2.2 nats — the spectral radius crossing ~1 under more unrolling flips the bare loop to
instability, while Parcae's ρ<1 parameterization keeps it bounded. With the **same** swept
hyperparameters, Parcae converges where the bare loop dies — robustness, not tuning. This is
exactly Parcae Table 4's "constraining Ā enables convergence at high T."

(`results/depth_sweep.csv` has the raw numbers.)

**Takeaway:** the Parcae injection's value is *enabling deep looping at all*. The bare loop caps
out around eff-depth ~8; Parcae keeps converging to ~24, which is what lets you actually spend the
depth axis. Still open: a *test-time* depth probe (train at T, eval at >T) — Parcae should degrade
gracefully where bare overthinks (as seen in `phase_b/`); not yet wired here.

## Caveats

- **Single seed.** Big gaps (0.065–0.09 nats) are likely real; the parcae-vs-bare tie (0.005) is
  within seed noise — needs 2–3 seeds before any claim about the injection at this scale.
- **Not converged** at 20k (val still dropping); gaps could shift, though the ordering is unlikely
  to flip.
- **T=4 doesn't stress Parcae** — see above; this run isn't evidence for/against the injection.

## Reproduce

```bash
python train.py sweep   --dataset enwik8 --steps 20000 --sweep-steps 3000 --dropout 0.1
python train.py compare --dataset enwik8 --steps 20000 --dropout 0.1
```
