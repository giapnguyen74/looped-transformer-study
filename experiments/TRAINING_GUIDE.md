# Training Guide — Looped-Transformer Scaling Lab

A complete, practical guide to *running* the experiments in
[`loop_scaling_lab.py`](loop_scaling_lab.py) and actually **experiencing** the scaling
laws of looped / recurrent-depth transformers. It assumes the conceptual background in
[`../summary/`](../summary/) (especially `parcae.md`, `reasoning_with_latent_thoughts.md`,
`structure_review_foundation.md`).

---

## 0. TL;DR — the 15-minute path

```bash
pip install torch matplotlib
cd experiments
python loop_scaling_lab.py stability --loops 8     # watch free explode, parcae stay bounded
python loop_scaling_lab.py iso                       # looped(k params) ≈ deep(kL params) ≫ shallow
python loop_scaling_lab.py testtime                  # train T, eval >T → saturation
python loop_scaling_lab.py isoflop                   # fixed compute, find the optimal loop count
```

Outputs (CSV + PNG) appear in `experiments/results/`. Read the plots; that's the experience.

---

## 1. Prerequisites

- **Python 3.10+**, **PyTorch** (`pip install torch`). `matplotlib` optional (plots; CSV is
  always written).
- **Hardware:** a single GPU is ideal (each run = minutes). CPU works for the smallest
  settings (`--dim 128 --nd 2 --steps 1000`) but is slow.
- No dataset download is needed for the default (synthetic) task — data is generated on the fly.

Sanity check before a long run:

```bash
python loop_scaling_lab.py iso --dim 128 --steps 300   # ~1–2 min, confirms the pipeline trains
```

If `seq_acc` for the `looped`/`deep` configs climbs above ~0.5 in this quick run, the loop is
learning and you're good to scale up.

---

## 2. What you're training

- **Task:** procedurally generated **integer addition**, `a + b = c`, answer digits reversed
  (least-significant first, so carries flow left→right). Difficulty set by `--nd` (digits).
  It's *structured and learnable* (not random noise); carries make it a genuine multi-step
  problem, so loop depth helps.
- **Model:** a minimal looped transformer — `n_unique` shared blocks looped `n_loops` times.
  Two state-carry rules (the heart of the stability experiment):
  - `free`: `h ← h + B·e + R̄` (carry `Ā = I`, ρ = 1) — can explode.
  - `parcae`: `h ← Ā·h + B·e + R̄`, `Ā = exp(softplus(Δ)·(−exp(log_A)))` diagonal, ρ < 1 —
    stable by construction (same trick as `open_mythos/main.py:LTIInjection`).
- **Metric:** validation cross-entropy on the **answer tokens** (the quantity scaling-law
  plots use). Answer-token accuracy and exact-sequence accuracy are reported too. Eval is
  teacher-forced (fast).

---

## 3. The four experiments — run, monitor, interpret

### 3.1 `stability` — does the loop blow up?

```bash
python loop_scaling_lab.py stability --loops 8 --steps 2000
```

- **Monitor:** the printed `||h_T||` (residual norm) and `loss` per model; `results/stability.png`.
- **Expect:** `free` residual norm climbs (often by orders of magnitude, sometimes to
  `inf`/`nan` = "diverged"), and its loss is erratic or stuck. `parcae` residual norm stays
  flat/bounded **and** its loss decreases smoothly to a low value.
- **Interpret:** this is ρ ≥ 1 vs ρ < 1 in front of you. Stability *and* learning both matter —
  a zero-output model is also stable but useless; `parcae` is stable **and** solves addition.
- **Make it more dramatic:** raise `--loops` (deeper unroll compounds the ρ=1 carry) or `--lr`.

### 3.2 `iso` — does looping buy depth's benefit cheaply?

```bash
python loop_scaling_lab.py iso --loops 4 --steps 2000
```

- Trains three models at equal hidden size: `shallow` (k, T=1), `looped` (k, T=4),
  `deep` (4k, T=1). See `results/iso.csv`.
- **Expect:** `looped` (few params) ≈ `deep` (4× params) ≫ `shallow` (few params), by
  `val_loss` and `seq_acc`.
- **Interpret:** depth-via-looping recovers most of what extra parameters would buy, at the
  shallow model's parameter count — the core looped-transformer value proposition.

### 3.3 `testtime` — can you "think longer" at inference?

```bash
python loop_scaling_lab.py testtime --loops 4 --steps 2500
```

- Trains with **dynamic depth** (T ~ Poisson(mean 4)), then evaluates at T = 1, 2, 4, 8, 12, 16.
  See `results/testtime.png`.
- **Expect:** loss falls as eval-T rises, then **saturates** near the training depth.
- **Interpret:** training depth sets the test-time ceiling (Parcae's finding). To see how
  task difficulty changes the ceiling, compare `--nd 2` (easy, saturates early) vs `--nd 5`
  (harder, benefits from more loops).

### 3.4 `isoflop` — is looping an orthogonal scaling axis?

```bash
python loop_scaling_lab.py isoflop --loops 4 --steps 2000
```

- Holds compute (`loops × steps`) fixed and sweeps T ∈ {1,2,4,8} (more loops ⇒ fewer steps).
  See `results/isoflop.csv`.
- **Expect:** a U-shape / interior optimum — a best loop count at fixed compute.
- **Caveat (important):** on the *synthetic* task data is effectively infinite, so the "data"
  axis is weak and the U-shape may be shallow or noisy. For a faithful **data↔looping power
  law** (Parcae §5.2), use a finite real corpus — see §6.

---

## 4. Reading the outputs

- `results/*.csv` — raw numbers (config, params, eff_depth, val_loss, accuracies).
- `results/*.png` — the plots (need matplotlib).
- Console — per-step `loss` and `||h_T||`; watch for `(diverged)` flags in `stability`.

The headline numbers to compare:
- `val_loss` (lower = better; the scaling-law metric),
- `seq_acc` (exact-match on the full answer; the human-readable "did it solve it"),
- `resid_norm` (stability; should be bounded for `parcae`).

---

## 5. Hyperparameter guide

| Flag | Default | What it does | When to change |
|---|---|---|---|
| `--dim` | 256 | hidden size | ↑ for capacity (slower); 128 for CPU |
| `--heads` | 4 | attention heads | keep `dim/heads` ≥ 32 |
| `--loops` | 4 | base loop count T | ↑ to stress depth & instability |
| `--nd` | 3 | digits per operand | ↑ = harder, more reasoning depth needed |
| `--steps` | 2000 | training steps | ↑ for cleaner curves / harder tasks |
| `--lr` | 3e-4 | AdamW LR | ↑ stresses stability; ↓ if `parcae` is noisy |

Rules of thumb:
- If **nothing learns** (loss flat, acc ~ chance): lower `--lr` (try 1e-4), or raise `--steps`,
  or reduce `--nd`.
- If **even parcae is unstable**: lower `--lr`; gradient clipping is already on at 1.0.
- If **the free model won't diverge** (you want to *see* it): raise `--loops` to 8–16 and/or
  `--lr` to 1e-3.
- **OOM:** lower `--dim`, or reduce batch size (edit `bsz` defaults), or `--nd`.

---

## 6. Going further — real data for a true scaling law

The synthetic task is best for stability / iso / test-time. For the **IsoFLOP data↔looping
power law**, switch to a finite real corpus so "data" is a meaningful axis:

- **text8 / enwik8** (char-level, one ~100MB file, no tokenizer) — least friction; drops into
  the lab's existing char-vocab design. **Recommended.**
- **TinyStories** — tiny, very learnable for small models (needs a tokenizer or char-level).
- **WikiText-103 / FineWeb-Edu slice** — closest to the papers; more setup.

Sketch to adapt the lab for text8:
1. Download `text8`, build a char vocab, hold out the last ~5M chars as validation.
2. Replace `make_batch` with a random-window sampler over the train split; drop `ans_mask`
   (score *all* next-token positions); validation loss = perplexity.
3. For `isoflop`, fix a **token budget** (not just steps) and trade it between loop count and
   tokens-seen; fit `L(μ_rec, D) = E + X·N(μ_rec)^−x + Y·D^−y` (Parcae's form).

For a sharper **reasoning / depth-extrapolation** story, add a **multi-hop** task (follow a
k-link chain to the answer) — difficulty scales linearly with hops, so loop depth maps
directly to task difficulty, making `testtime` extrapolation dramatic.

*(Both the `text8` loader and the `multihop` task are small additions — ask and I'll wire
them into the lab.)*

---

## 7. A suggested end-to-end protocol

To actually *feel* the laws in one sitting (~30–60 min on a GPU):

1. **Confirm the leash matters.** `stability --loops 8` → free explodes, parcae bounded.
   *Takeaway: ρ(Ā) < 1 is what makes deep looping trainable.*
2. **Confirm looping ≈ depth.** `iso` → looped ≈ deep ≫ shallow.
   *Takeaway: depth via reuse buys quality at low parameter count.*
3. **Confirm think-longer + its ceiling.** `testtime` at `--nd 2` and `--nd 5`.
   *Takeaway: inference loops help up to the training depth; harder tasks raise the ceiling.*
4. **Confirm the orthogonal axis.** `isoflop` (ideally on text8) → optimal loop count exists.
   *Takeaway: looping trades against data at fixed compute — a real scaling dimension.*

Each step writes a CSV/PNG you can keep as your own reproduction of the corresponding figure.

---

## 8. How this maps to the papers

| Experiment | Confirms | Paper |
|---|---|---|
| `stability` | ρ(Ā) < 1 prevents residual explosion | Parcae |
| `iso` | looped ≈ deep ≫ shallow (depth not params) | Reasoning with Latent Thoughts |
| `testtime` | inference-loop scaling + saturation ceiling | Parcae §5.3 / Loop-Think |
| `isoflop` | looping is an orthogonal scaling axis | Parcae §5.2 |

---

## 9. Caveats

- Eval is teacher-forced (optimistic vs autoregressive); fine for relative comparisons and
  loss-based scaling, swap to AR decoding for a strict accuracy number.
- Results are at tiny scale — they reproduce the *qualitative* laws, not frontier magnitudes.
  As discussed, some advantages attenuate at large scale; treat these as intuition-builders.
- The lab strips the real repo down (no MoE/MLA, simplified injection) for speed; the
  `parcae` injection and loop-index embedding match `open_mythos/main.py` so the intuition
  transfers. To exercise the full stack (MLA/GQA, MoE, LTIInjection, ACT), build a training
  loop around the real `open_mythos` modules instead — `../example.py` shows how to
  instantiate `OpenMythos` and run a forward/generate pass.
