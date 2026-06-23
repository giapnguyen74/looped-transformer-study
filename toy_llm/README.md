# Toy LLM — vanilla vs Parcae-looped, parameter-matched

A small char-level language model (Tiny Shakespeare, Karpathy-style) for a **fair, parameter-
matched** comparison of a vanilla Transformer against a **looped** transformer (Parcae and a
bare prior-RDM variant). It mirrors the methodology of [Parcae §5.1](https://arxiv.org/html/2604.12946v1#S5)
on a single-GPU toy scale: match params/data, **sweep the baseline's hyperparameters, then
transfer them to the looped model with no extra tuning**, and compare quality (val loss /
perplexity).

The question it answers: *does looping buy reasoning depth cheaply* — i.e. does a k-layer block
looped L times (params ≈ k layers, effective depth ≈ kL) **match a kL-deep vanilla model** and
**beat a k-layer shallow one at the same parameter count**?

## Setup

```bash
pip install -r ../requirements.txt        # torch (+ stdlib)
```

Tiny Shakespeare (~1 MB) auto-downloads to `toy_llm/input.txt` on first run, or pass
`--data <textfile>`. Auto-selects CUDA / MPS / CPU.

## End-to-end experiment

Two steps — this *is* the protocol:

```bash
# 1) sweep optimization hyperparams (lr) on the VANILLA baseline; logs best_hparams.json
python train.py sweep

# 2) train the matched set with those SAME hyperparams for every model; prints the table
python train.py compare
```

`sweep` tries an lr grid (`--sweep-lr`, default `1e-3,5e-4,3e-4,1e-4`) on `vanilla-deep` at a
reduced step count (`--sweep-steps`, default 1500), keeps the best by validation loss, and writes
`best_hparams.json`. `compare` reads that file and applies the **same lr/weight-decay to all four
models** — no per-model tuning (that's the fairness rule; only *optimization* hparams are swept,
never architecture, so parameter counts stay matched).

You can also train a single model:

```bash
python train.py train --model parcae          # or vanilla / bare
python train.py train --model vanilla --layers 8
```

## The compare set

All share `dim`, `heads`, vocab, context, steps, and the swept hparams. With defaults
`--k 2 --loops 4` (effective depth 8):

| Model | What | Role |
|---|---|---|
| `vanilla-2L` | 2 distinct layers | shallow baseline — **iso-param** with looped |
| `parcae-2x4` | 2-layer block looped 4× (Parcae ρ<1 injection) | the model |
| `bare-2x4` | same, no injection (prior-RDM style) | looped baseline |
| `vanilla-8L` | 8 distinct layers | deep baseline — **iso-depth** with looped |

## Reading the result

`compare` prints params / val_loss / perplexity / time per model. The claim looks like:

- **`parcae-2x4` ≈ `vanilla-8L`** in perplexity, but at roughly **`vanilla-2L`'s parameter
  count** → looping reaches deep-model quality cheaply (depth without parameters).
- **`parcae-2x4` < `vanilla-2L`** (lower perplexity at equal params) → at a fixed parameter
  budget, depth-via-looping beats more distinct shallow layers.
- **`parcae-2x4` vs `bare-2x4`** → whether the Parcae injection actually helps/stabilizes at this
  scale (Parcae's own ablation found it does, especially at higher loop counts).

If instead `parcae ≈ vanilla-2L` and both ≪ `vanilla-8L`, looping didn't convert into usable
depth here — itself a clean result, pointing at scale/recipe.

## Knobs

```bash
python train.py compare --dim 384 --k 2 --loops 6 --steps 6000      # bigger / deeper loop
python train.py sweep   --sweep-lr 2e-3,1e-3,5e-4 --sweep-steps 2000
python train.py compare --device cpu                                 # force device
```

`--k`/`--loops` set the looped block size and loop count (effective depth = k·loops, which also
sets the deep baseline's layer count); `--dim`, `--heads`, `--block` (context), `--bsz`,
`--steps` are shared across all models.

## Files

- `models.py` — `VanillaLM` (distinct layers) and `LoopedLM` (Parcae injection or bare), causal.
- `data.py` — Tiny Shakespeare char dataset (auto-download), train/val split.
- `train.py` — `sweep` / `compare` / `train`.
- `best_hparams.json` — written by `sweep`, read by `compare` (the logged baseline hparams).

## Notes & caveats

- **What this tests:** the *parameter-efficiency* claim (the robust one looping actually buys).
- **Test-time depth** (run a trained looped model at *more* loops than trained — the unique
  looped knob a vanilla model can't match) is a separate axis, not in this harness yet; it can be
  added by evaluating `LoopedLM` at varied `n_loops`.
- **Fairness scope:** we sweep the vanilla baseline and transfer to all. A stricter version would
  also sweep the bare RDM separately (Parcae tuned RDM baselines too) — easy to add if needed.
- Same seed across models; differences are architecture, not initialization luck.
