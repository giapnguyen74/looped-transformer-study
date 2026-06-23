"""
toy_llm/train.py — parameter-matched comparison of vanilla vs Parcae-looped char-LMs,
following Parcae §5.1's fairness protocol:

  1. SWEEP optimization hyperparams (lr, wd) on the VANILLA baseline.
  2. LOG the best to best_hparams.json.
  3. COMPARE: apply those SAME hyperparams to every model (vanilla / parcae / bare) — no
     extra sweep for the looped models. Architecture (dim/depth) is fixed by the matched-param
     design and is NOT swept.

  python train.py sweep                    # sweep lr on vanilla-deep, save best_hparams.json
  python train.py compare                  # uses best_hparams.json (if present) for ALL models
  python train.py train --model parcae     # train one model

Compare set (same dim/heads/data/steps/hparams for all):
  vanilla-k   shallow baseline (k layers)        — iso-PARAM vs looped
  parcae-kxL  k-block looped L times (Parcae)    — the model
  bare-kxL    same, no injection (prior-RDM)
  vanilla-kL  deep baseline (k*loops layers)     — iso-DEPTH vs looped

Needs torch. Auto-selects cuda/mps/cpu.
"""

import argparse
import json
import math
import os
import time

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("[toy_llm] PyTorch required: pip install -r ../requirements.txt")

from models import build, n_params, pick_device
from data import CharData

HERE = os.path.dirname(os.path.abspath(__file__))
HPARAMS_FILE = os.path.join(HERE, "best_hparams.json")


@torch.no_grad()
def val_loss(model, data, bsz, n_batches=40):
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        x, y = data.batch("val", bsz)
        logits = model(x)
        tot += F.cross_entropy(logits.reshape(-1, data.vocab), y.reshape(-1)).item()
    model.train()
    return tot / n_batches


def train_one(kind, data, args, device, lr, wd, steps, layers=None, quiet=False):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    model, label = build(kind, data.vocab, args.dim, args.heads, args.block,
                         layers=layers if layers is not None else args.layers,
                         k=args.k, loops=args.loops, dropout=args.dropout)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    val_every = args.val_every or max(1, steps // 10)
    best_val, best_step, last_val = float("inf"), 0, float("nan")
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = data.batch("train", args.bsz, generator=g)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, data.vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % val_every == 0 or step == steps:
            last_val = val_loss(model, data, args.bsz)
            if last_val < best_val:
                best_val, best_step = last_val, step
            if not quiet:
                print(f"    [{label}] step {step:5d}  train {loss.item():.3f}  val {last_val:.3f}"
                      f"  best {best_val:.3f}@{best_step}  ({time.time()-t0:.0f}s)")
    return {"label": label, "params": n_params(model), "best_val": best_val,
            "best_step": best_step, "final_val": last_val, "ppl": math.exp(best_val),
            "secs": time.time() - t0}


def cmd_sweep(data, args, device):
    eff = args.k * args.loops
    grid = [float(x) for x in args.sweep_lr.split(",")]
    print(f"  sweeping lr {grid} on vanilla-{eff}L (wd={args.weight_decay}, "
          f"{args.sweep_steps} steps each)")
    best, results = None, []
    for lr in grid:
        r = train_one("vanilla", data, args, device, lr=lr, wd=args.weight_decay,
                      steps=args.sweep_steps, layers=eff, quiet=True)
        results.append((lr, r["best_val"]))
        print(f"    lr={lr:.1e}  best_val={r['best_val']:.4f}  ppl={r['ppl']:.2f}")
        if best is None or r["best_val"] < best[1]:
            best = (lr, r["best_val"])
    hp = {"lr": best[0], "weight_decay": args.weight_decay, "swept_on": f"vanilla-{eff}L",
          "sweep_steps": args.sweep_steps, "val_loss": best[1], "grid": results}
    json.dump(hp, open(HPARAMS_FILE, "w"), indent=2)
    print(f"\n  best lr={best[0]:.1e} (val_loss={best[1]:.4f}) -> saved {HPARAMS_FILE}")
    print("  now run:  python train.py compare   (applies these to ALL models, parcae included)")


def cmd_compare(data, args, device):
    if os.path.exists(HPARAMS_FILE):
        hp = json.load(open(HPARAMS_FILE))
        lr, wd = hp["lr"], hp["weight_decay"]
        print(f"  using swept hparams (from {hp['swept_on']}): lr={lr:.1e} wd={wd}  "
              f"— transferred to ALL models, no per-model sweep")
    else:
        lr, wd = args.lr, args.weight_decay
        print(f"  [warn] no {os.path.basename(HPARAMS_FILE)}; using defaults lr={lr:.1e} wd={wd}. "
              f"Run `train.py sweep` first for the fair protocol.")

    eff = args.k * args.loops
    plan = [("vanilla", args.k), ("parcae", None), ("bare", None), ("vanilla", eff)]
    runs = []
    for kind, layers in plan:
        print(f"\n--- training {kind} (layers={layers if layers else f'{args.k}x{args.loops}'}) ---")
        runs.append(train_one(kind, data, args, device, lr=lr, wd=wd, steps=args.steps, layers=layers))

    print(f"\n  model           params    best_val  ppl    best@   final   time   (lr={lr:.1e})")
    for r in runs:
        print(f"  {r['label']:14s} {r['params']:>9,}  {r['best_val']:.4f}  {r['ppl']:5.2f}  "
              f"{r['best_step']:>6}  {r['final_val']:.4f}  {r['secs']:5.0f}s")
    print("\n  Read: compare BEST val (fair under different overfit rates). final≫best = overfit.")
    print("        parcae ≈ vanilla-deep at far fewer params + parcae < vanilla-shallow = looping")
    print("        buys depth cheaply; parcae vs bare = does the injection help.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["sweep", "compare", "train"])
    p.add_argument("--model", choices=["vanilla", "parcae", "bare"], default="parcae")
    p.add_argument("--layers", type=int, default=8, help="distinct layers (vanilla / single train)")
    p.add_argument("--k", type=int, default=2, help="block layers (looped)")
    p.add_argument("--loops", type=int, default=4, help="loop count (looped)")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--block", type=int, default=128, help="context length")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bsz", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4, help="used by `train`; sweep overrides for compare")
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.0, help="regularization (curbs overfit)")
    p.add_argument("--dataset", choices=["shakespeare", "enwik8"], default="shakespeare",
                   help="enwik8 (~100MB) so the deep baseline is data-bound, not overfitting")
    p.add_argument("--val-every", dest="val_every", type=int, default=0,
                   help="validate every N steps for best-val tracking; 0 => steps//10")
    p.add_argument("--sweep-lr", dest="sweep_lr", default="1e-3,5e-4,3e-4,1e-4")
    p.add_argument("--sweep-steps", dest="sweep_steps", type=int, default=1500)
    p.add_argument("--data", default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    data = CharData(args.dataset, args.block, device, path=args.data)
    print(f"== toy LLM ({args.dataset})  device={device}  vocab={data.vocab} "
          f"dim={args.dim} ctx={args.block} dropout={args.dropout} ==")

    if args.command == "sweep":
        cmd_sweep(data, args, device)
    elif args.command == "compare":
        cmd_compare(data, args, device)
    else:
        r = train_one(args.model, data, args, device, lr=args.lr, wd=args.weight_decay,
                      steps=args.steps)
        print(f"\n  {r['label']}: params={r['params']:,}  best_val={r['best_val']:.4f}"
              f"@{r['best_step']}  ppl={r['ppl']:.2f}  final={r['final_val']:.4f}  ({r['secs']:.0f}s)")


if __name__ == "__main__":
    main()
