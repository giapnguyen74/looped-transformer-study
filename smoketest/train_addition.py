"""
train_addition.py — SMOKE TEST 1: train a tiny looped transformer FROM SCRATCH on the
synthetic multi-digit addition task (the depth-scaling carry chain that actually exercises
the loop). This is the part that validates "the looped model trains and stays stable".

PASS criteria (lenient, this is a smoke check not a benchmark):
  * training stayed STABLE      — residual norm finite and bounded (Parcae rho < 1)
  * loss DROPPED meaningfully   — final loss <= 0.6 * initial loss
  * answer accuracy IMPROVED    — final answer-token accuracy > initial

Exits 0 on PASS, 1 on FAIL. Runs on CPU in ~1-2 min at defaults.
"""

import argparse
import math
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("[FAIL] PyTorch not installed. `pip install -r ../requirements.txt`")

from model import LoopedLM, n_params, pick_device
from data import make_addition_batch, ADD_V


def masked_ce(logits, tgt, ans_mask):
    loss = F.cross_entropy(logits.reshape(-1, ADD_V), tgt.reshape(-1), reduction="none")
    return (loss * ans_mask.reshape(-1)).sum() / ans_mask.sum().clamp(min=1)


@torch.no_grad()
def answer_accuracy(model, nd, device="cpu", n_batches=10, bsz=128):
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        inp, tgt, am = make_addition_batch(bsz, nd, device=device)
        pred = model(inp).argmax(-1)
        ans = am.bool()
        correct += ((pred == tgt) & ans).sum().item()
        total += ans.sum().item()
    model.train()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nd", type=int, default=2, help="digits per operand")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--loops", type=int, default=3)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--bsz", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    args = p.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    print(f"== SMOKE 1: addition (nd={args.nd}, dim={args.dim}, loops={args.loops}, "
          f"steps={args.steps}, device={device}) ==")
    model = LoopedLM(ADD_V, dim=args.dim, heads=args.heads, n_unique=1,
                     n_loops=args.loops, max_len=64, inj="parcae").to(device)
    print(f"  params={n_params(model):,}  rho(A)={model.inj.rho():.3f}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    first_loss = None
    last_loss = max_resid = float("nan")
    acc0 = answer_accuracy(model, args.nd, device=device)
    for step in range(1, args.steps + 1):
        inp, tgt, am = make_addition_batch(args.bsz, args.nd, device=device)
        logits, resid = model(inp, return_resid=True)
        loss = masked_ce(logits, tgt, am)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        max_resid = resid if math.isnan(max_resid) else max(max_resid, resid)
        if step % 100 == 0 or step == 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}  ||h_T|| {resid:8.2f}")
    acc1 = answer_accuracy(model, args.nd, device=device)

    # ---- assertions ----
    stable = math.isfinite(max_resid) and max_resid < 1e4
    dropped = math.isfinite(last_loss) and last_loss <= 0.6 * first_loss
    improved = acc1 > acc0
    print(f"\n  initial loss {first_loss:.4f} -> final {last_loss:.4f}")
    print(f"  answer acc   {acc0:.3f} -> {acc1:.3f}")
    print(f"  max ||h_T||  {max_resid:.2f}  (stable={stable})")

    ok = stable and dropped and improved
    if not ok:
        print(f"[FAIL] addition smoke (stable={stable}, loss_dropped={dropped}, "
              f"acc_improved={improved})")
        sys.exit(1)
    print("[PASS] addition smoke")


if __name__ == "__main__":
    main()
