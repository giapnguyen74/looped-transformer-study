# End-to-end smoke test

A single command that exercises the **whole stack** — synthetic data generation, the
verified-transcript pipeline, and training a tiny looped transformer two ways — and fails
loudly if anything is broken. It's a *does-it-all-run-and-train* check, not a benchmark.

```bash
pip install -r ../requirements.txt    # torch + sympy (matplotlib/requests not needed here)
bash run_smoke.sh
```

Everything is tiny and **offline** — no API key needed. The training steps auto-select the
best device (**CUDA → MPS (Apple Silicon) → CPU**); CPU runtime is roughly 3–5 minutes and a
GPU/MPS is faster. Override with `--device cuda|mps|cpu` on either training script.

## What it runs

```
STAGE A  ../experiments/gen_math_problems.py      → problems.jsonl     (arith, exact gold)
STAGE B  ../experiments/gen_math_transcripts.py   → transcripts.jsonl  (verified CoT, mock teacher)
TRAIN 1  train_addition.py   from-scratch looped model on multi-digit addition (the depth task)
TRAIN 2  train_sft.py        next-token SFT of a looped model on the transcripts
```

Stages A+B use the real pipeline scripts, so a regression in either breaks the smoke test.
Training uses the compact looped model in [`model.py`](model.py) (Parcae-style contractive
injection, ρ<1), fed by [`data.py`](data.py).

## Pass / fail

Each step exits non-zero on failure and `run_smoke.sh` aborts on the first failure. The two
training scripts assert (lenient, smoke-level):

- **TRAIN 1 (addition):** loop stayed stable (`‖h_T‖` finite & bounded — the ρ<1 leash),
  loss fell to ≤ 0.6× its start, and answer-token accuracy improved.
- **TRAIN 2 (SFT):** loss stayed finite and fell to ≤ 0.7× its start.

A green run prints `SMOKE TEST PASSED ✅`.

## Why these two tasks

The from-scratch **addition** task is the one that actually validates the *loop*: it's a
depth-scaling carry chain (depth ∝ digits), so a model that loops can learn it — single-step
problems wouldn't tell you anything (see the discussion in `../experiments/README.md`). The
**SFT** pass instead validates that the data pipeline's output is well-formed and trainable.
Together they cover data → transcripts → training.

## Knobs

```bash
python3 train_addition.py --nd 3 --loops 4 --steps 800   # harder task, deeper loop, longer
python3 train_sft.py --data transcripts.jsonl --steps 800 --block 96
python3 train_addition.py --device cpu                    # force a device (default: auto)
```

Raising `--nd` makes addition need more reasoning depth; raising `--loops` gives the model
more iterations to use. To regenerate the data by hand, see the two commands under
"What it runs" (or just re-run `run_smoke.sh`).

## Note

PyTorch is required for the training steps. The data-generation stages need only `sympy`,
so if you just want to check the pipeline you can run STAGE A and STAGE B directly.
