"""
train_sft.py — SMOKE TEST 2: next-token SFT of a tiny looped transformer on the verified
transcripts produced by the data pipeline (gen_math_problems -> gen_math_transcripts).
This validates that the *pipeline output* is well-formed and trainable end to end.

Char-level LM over "question \n solution" strings. PASS criteria (lenient):
  * training stayed STABLE     — loss finite throughout
  * loss DROPPED meaningfully  — final loss <= 0.7 * initial loss

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
from data import load_transcript_texts, build_char_vocab, CharLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="transcripts.jsonl")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--loops", type=int, default=3)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--bsz", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    args = p.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    texts = load_transcript_texts(args.data)
    if len(texts) < 5:
        sys.exit(f"[FAIL] only {len(texts)} transcripts in {args.data} — run the data "
                 f"pipeline first (see run_smoke.sh)")
    stoi, V = build_char_vocab(texts)
    ds = CharLM(texts, stoi, block_size=args.block, device=device)
    if ds.data.numel() < args.block * 4:
        sys.exit(f"[FAIL] transcript corpus too small ({ds.data.numel()} chars)")

    print(f"== SMOKE 2: transcript SFT ({len(texts)} transcripts, vocab={V}, "
          f"{ds.data.numel()} chars, loops={args.loops}, steps={args.steps}, "
          f"device={device}) ==")
    model = LoopedLM(V, dim=args.dim, heads=args.heads, n_unique=1,
                     n_loops=args.loops, max_len=args.block, inj="parcae").to(device)
    print(f"  params={n_params(model):,}  rho(A)={model.inj.rho():.3f}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    first_loss = last_loss = None
    finite = True
    for step in range(1, args.steps + 1):
        x, y = ds.batch(args.bsz)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not math.isfinite(loss.item()):
            finite = False
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        if step % 100 == 0 or step == 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")

    dropped = finite and math.isfinite(last_loss) and last_loss <= 0.7 * first_loss
    print(f"\n  initial loss {first_loss:.4f} -> final {last_loss:.4f}  (finite={finite})")
    if not dropped:
        print(f"[FAIL] SFT smoke (finite={finite}, loss_dropped={dropped})")
        sys.exit(1)
    print("[PASS] SFT smoke")


if __name__ == "__main__":
    main()
