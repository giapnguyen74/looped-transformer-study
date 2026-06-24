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

## Why does Parcae ≈ bare here?

The Parcae injection is a **stability** mechanism (ρ<1 leash), and stability only matters when the
loop is at risk of becoming unstable — at **high loop counts T** and under **test-time depth
changes**. Parcae's own ablation (Table 4) frames the Ā constraint as "enables convergence at
**high T** (e.g. μ_rec=T=8)." We ran **T=4, fixed depth**, where RMSNorm + residual already keep
the bare loop well-behaved, so the leash has nothing to fix (the tiny edge parcae shows is ~its
66k extra injection params). It's a *regime* issue, not a *scale* issue.

To actually surface Parcae's benefit, stress the regime it targets:
1. **Higher loop count** (`--loops 8`/`12`) — bare should start to destabilize/degrade while
   Parcae stays bounded.
2. **Test-time depth** (train at T, eval at >T) — Parcae's ρ<1 stays bounded; bare drifts
   (the "overthinking" we observed in `phase_b/`). Not yet wired into this harness.

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
