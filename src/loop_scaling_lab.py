"""
loop_scaling_lab.py — a tiny, self-contained lab for *experiencing* the scaling
laws of looped / recurrent-depth transformers (Universal Transformer / Parcae style).

It trains a minimal looped transformer on a synthetic ADDITION task and runs four
micro-experiments, each isolating one phenomenon from the papers in ../summary/:

  1. stability   — free injection (rho(A)~1, explodes) vs Parcae diagonal A (rho<1, stable)
  2. iso         — iso-param vs iso-FLOP: a k-layer block looped L times vs a kL-layer
                   model vs a k-layer model (looping ~ depth, beats shallow at equal params)
  3. testtime    — train at T loops, evaluate at >T loops -> saturating curve
  4. isoflop     — fixed compute, vary (loops x data) -> looping is an orthogonal axis

Dependencies: torch (required), matplotlib (optional; plots are skipped if missing).
Designed for a single GPU; defaults run in minutes. Scale up via CLI flags.

Examples:
  python loop_scaling_lab.py stability
  python loop_scaling_lab.py iso
  python loop_scaling_lab.py testtime
  python loop_scaling_lab.py isoflop
  python loop_scaling_lab.py all --steps 3000 --dim 256

Primary metric is validation cross-entropy on the ANSWER tokens (the quantity
scaling-law plots use); answer-token accuracy is reported as a secondary signal.
Eval is teacher-forced (fast); switch to autoregressive if you want a stricter number.
"""

import argparse
import math
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Synthetic task: integer addition with reversed-digit answers.
# Sequence layout for n digits:  a(n) '+' b(n) '=' c(n+1, reversed)
# We train next-token LM and only score the answer positions.
# --------------------------------------------------------------------------- #
VOCAB = {c: i for i, c in enumerate("0123456789+=P")}  # P = pad (unused for fixed nd)
IVOCAB = {i: c for c, i in VOCAB.items()}
V = len(VOCAB)


def make_batch(bsz, nd, device=DEVICE):
    """Return (inp, tgt, ans_mask) for bsz addition problems with nd-digit operands."""
    a = torch.randint(0, 10 ** nd, (bsz,))
    b = torch.randint(0, 10 ** nd, (bsz,))
    s = a + b
    seqs = []
    for ai, bi, si in zip(a.tolist(), b.tolist(), s.tolist()):
        astr = str(ai).zfill(nd)
        bstr = str(bi).zfill(nd)
        cstr = str(si).zfill(nd + 1)[::-1]  # reversed answer, fixed width nd+1
        seqs.append([VOCAB[ch] for ch in (astr + "+" + bstr + "=" + cstr)])
    full = torch.tensor(seqs, dtype=torch.long)             # (B, L) , L = 3nd+3
    inp = full[:, :-1].to(device)                           # (B, L-1)
    tgt = full[:, 1:].to(device)                            # (B, L-1)
    # answer targets are the last (nd+1) positions of tgt
    ans_mask = torch.zeros_like(tgt, dtype=torch.float)
    ans_mask[:, -(nd + 1):] = 1.0
    return inp, tgt, ans_mask


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


class Block(nn.Module):
    """One transformer layer: attention + FFN, pre-norm + residual (this is R-bar)."""
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)

    def forward(self, x, attn_mask):
        a = self.n1(x)
        x = x + self.attn(a, a, a, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.ffn(self.n2(x))
        return x


class Injection(nn.Module):
    """State-carry rule between loop iterations:  h_{t+1} = carry(h_t) + B(e) + block_out

    mode='free'   : carry = identity  (A-bar = I, rho = 1)  -> can explode
    mode='parcae' : carry = A-bar * h  with A-bar = exp(softplus(dt) * (-exp(log_A)))
                    diagonal, guaranteed in (0,1) -> rho < 1, stable by construction.
    """
    def __init__(self, dim, mode):
        super().__init__()
        self.mode = mode
        self.proj_e = nn.Linear(dim, dim, bias=False)
        if mode == "parcae":
            self.log_A = nn.Parameter(torch.zeros(dim))   # free, learnable
            self.dt = nn.Parameter(torch.zeros(dim))      # step size (kept positive via softplus)

    def a_bar(self):
        A = -torch.exp(self.log_A)                        # negative diagonal
        return torch.exp(F.softplus(self.dt) * A)         # in (0,1)

    def forward(self, h, e, block_out):
        if self.mode == "free":
            return h + self.proj_e(e) + block_out
        return self.a_bar() * h + self.proj_e(e) + block_out

    def rho(self):
        if self.mode == "free":
            return 1.0
        return float(self.a_bar().max().item())


def loop_index_embedding(h, t, loop_dim, theta=10000.0):
    """Sinusoidal loop-index signal added to the first loop_dim channels (RoPE-over-depth)."""
    dev, dt = h.device, h.dtype
    freqs = 1.0 / (theta ** (torch.arange(0, loop_dim, 2, device=dev, dtype=dt) / loop_dim))
    ang = t * freqs
    emb = torch.cat([ang.sin(), ang.cos()], dim=-1)[:loop_dim]
    full = torch.zeros(h.shape[-1], device=dev, dtype=dt)
    full[:loop_dim] = emb
    return h + full.view(1, 1, -1)


class LoopedModel(nn.Module):
    """n_unique shared blocks, looped n_loops times. Effective depth = n_unique * n_loops.

      looped  : n_unique = k,   n_loops = L      (params = k blocks)
      deep    : n_unique = k*L, n_loops = 1       (params = kL blocks)
      shallow : n_unique = k,   n_loops = 1       (params = k blocks)
    """
    def __init__(self, dim=256, heads=4, n_unique=2, n_loops=4, max_len=64, inj="parcae"):
        super().__init__()
        self.dim, self.n_loops = dim, n_loops
        self.tok = nn.Embedding(V, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(n_unique)])
        self.inj = Injection(dim, inj)
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, V)
        self.loop_dim = max(2, dim // 8)
        self._mask = {}

    def causal_mask(self, T, device):
        if T not in self._mask:
            m = torch.full((T, T), float("-inf"), device=device)
            self._mask[T] = torch.triu(m, diagonal=1)
        return self._mask[T]

    def forward(self, ids, n_loops=None, return_resid=False):
        n_loops = n_loops or self.n_loops
        B, T = ids.shape
        mask = self.causal_mask(T, ids.device)
        e = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))[None]
        h = e
        for t in range(n_loops):
            x = loop_index_embedding(self.norm(h + e), t, self.loop_dim)
            for blk in self.blocks:
                x = blk(x, mask)
            h = self.inj(h, e, x)
        logits = self.head(self.norm(h))
        if return_resid:
            return logits, float(h.norm(dim=-1).mean().item())
        return logits


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def masked_ce(logits, tgt, ans_mask):
    loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none")
    loss = (loss * ans_mask.reshape(-1)).sum() / ans_mask.sum().clamp(min=1)
    return loss


@torch.no_grad()
def evaluate(model, nd, eval_loops=None, n_batches=20, bsz=256):
    model.eval()
    tot_loss, tot_tok, correct_tok, correct_seq, n_seq = 0.0, 0, 0, 0, 0
    for _ in range(n_batches):
        inp, tgt, am = make_batch(bsz, nd)
        logits = model(inp, n_loops=eval_loops)
        tot_loss += masked_ce(logits, tgt, am).item()
        pred = logits.argmax(-1)
        ans = am.bool()
        correct_tok += ((pred == tgt) & ans).sum().item()
        tot_tok += ans.sum().item()
        seq_ok = ((pred == tgt) | ~ans).all(dim=1)
        correct_seq += seq_ok.sum().item()
        n_seq += inp.size(0)
    model.train()
    return tot_loss / n_batches, correct_tok / max(tot_tok, 1), correct_seq / max(n_seq, 1)


def train(model, nd, steps=2000, bsz=256, lr=3e-4, train_loops=None,
          poisson_mean=None, log_every=200, label=""):
    """Train; train_loops=int for fixed depth, poisson_mean set for dynamic depth.
    Returns history dict with step/loss/resid lists."""
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    hist = {"step": [], "loss": [], "resid": []}
    for step in range(1, steps + 1):
        if poisson_mean is not None:
            nl = int(max(1, torch.poisson(torch.tensor(float(poisson_mean))).item()))
        else:
            nl = train_loops
        inp, tgt, am = make_batch(bsz, nd)
        logits, resid = model(inp, n_loops=nl, return_resid=True)
        loss = masked_ce(logits, tgt, am)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            hist["step"].append(step)
            hist["loss"].append(loss.item())
            hist["resid"].append(resid)
            print(f"[{label}] step {step:5d}  loss {loss.item():.4f}  ||h_T|| {resid:9.2f}"
                  + ("  (diverged)" if not math.isfinite(resid) or resid > 1e4 else ""))
    return hist


# --------------------------------------------------------------------------- #
# Plotting (optional)
# --------------------------------------------------------------------------- #
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        print("[plot] matplotlib not available — skipping plots, CSV still saved.")
        return None


def save_csv(name, rows, header):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[csv] wrote {path}")


# --------------------------------------------------------------------------- #
# Experiment 1: stability
# --------------------------------------------------------------------------- #
def exp_stability(args):
    print("\n=== Experiment 1: stability (free rho~1 vs Parcae rho<1) ===")
    nd = args.nd
    results = {}
    for mode in ["free", "parcae"]:
        torch.manual_seed(0)
        m = LoopedModel(args.dim, args.heads, n_unique=2, n_loops=args.loops, inj=mode)
        print(f"\n-- injection={mode}  rho(A)={m.inj.rho():.3f}  params={n_params(m):,}")
        results[mode] = train(m, nd, steps=args.steps, lr=args.lr,
                              train_loops=args.loops, label=mode)
    plt = _plt()
    rows = []
    for mode in results:
        for s, l, r in zip(results[mode]["step"], results[mode]["loss"], results[mode]["resid"]):
            rows.append([mode, s, l, r])
    save_csv("stability.csv", rows, ["mode", "step", "loss", "resid_norm"])
    if plt:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for mode in results:
            ax[0].plot(results[mode]["step"], results[mode]["loss"], marker="o", label=mode)
            ax[1].plot(results[mode]["step"], results[mode]["resid"], marker="o", label=mode)
        ax[0].set(title="Answer loss", xlabel="step", ylabel="CE")
        ax[1].set(title="Residual norm ||h_T|| (log)", xlabel="step", ylabel="||h||", yscale="log")
        for a in ax:
            a.legend(); a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "stability.png"), dpi=120)
        print(f"[plot] {os.path.join(OUT_DIR, 'stability.png')}")


# --------------------------------------------------------------------------- #
# Experiment 2: iso-param vs iso-FLOP
# --------------------------------------------------------------------------- #
def exp_iso(args):
    print("\n=== Experiment 2: iso-param vs iso-FLOP ===")
    nd, k, L = args.nd, 2, args.loops
    configs = {
        f"shallow (k={k}, T=1)":   dict(n_unique=k,     n_loops=1),
        f"looped  (k={k}, T={L})": dict(n_unique=k,     n_loops=L),
        f"deep    (k={k*L}, T=1)": dict(n_unique=k * L, n_loops=1),
    }
    rows = []
    for name, cfg in configs.items():
        torch.manual_seed(0)
        m = LoopedModel(args.dim, args.heads, inj="parcae", **cfg)
        train(m, nd, steps=args.steps, lr=args.lr, train_loops=cfg["n_loops"], label=name)
        loss, tok_acc, seq_acc = evaluate(m, nd, eval_loops=cfg["n_loops"])
        eff_depth = cfg["n_unique"] * cfg["n_loops"]
        rows.append([name, n_params(m), eff_depth, round(loss, 4), round(tok_acc, 4), round(seq_acc, 4)])
        print(f"  -> {name}: params={n_params(m):,} eff_depth={eff_depth} "
              f"val_loss={loss:.4f} tok_acc={tok_acc:.3f} seq_acc={seq_acc:.3f}")
    save_csv("iso.csv", rows, ["config", "params", "eff_depth", "val_loss", "tok_acc", "seq_acc"])
    print("\nExpect: looped (few params) ~ deep (many params) >> shallow (few params).")


# --------------------------------------------------------------------------- #
# Experiment 3: test-time scaling
# --------------------------------------------------------------------------- #
def exp_testtime(args):
    print("\n=== Experiment 3: test-time scaling (train T, eval >T) ===")
    nd, train_T = args.nd, args.loops
    torch.manual_seed(0)
    # dynamic depth in training helps extrapolation (Parcae/Loop-Think)
    m = LoopedModel(args.dim, args.heads, n_unique=2, n_loops=train_T, inj="parcae")
    train(m, nd, steps=args.steps, lr=args.lr, poisson_mean=train_T,
          label=f"dyn~Poisson({train_T})")
    rows = []
    print(f"\n  eval at varying loop counts (train mean T={train_T}):")
    for T in [1, 2, train_T, train_T * 2, train_T * 3, train_T * 4]:
        loss, tok_acc, seq_acc = evaluate(m, nd, eval_loops=T)
        rows.append([T, round(loss, 4), round(tok_acc, 4), round(seq_acc, 4)])
        print(f"    T={T:3d}  val_loss={loss:.4f}  tok_acc={tok_acc:.3f}  seq_acc={seq_acc:.3f}")
    save_csv("testtime.csv", rows, ["eval_loops", "val_loss", "tok_acc", "seq_acc"])
    plt = _plt()
    if plt:
        Ts = [r[0] for r in rows]; ls = [r[1] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(Ts, ls, marker="o")
        ax.axvline(train_T, ls="--", c="gray", label=f"train T={train_T}")
        ax.set(title="Test-time scaling (saturating)", xlabel="eval loops T", ylabel="val loss")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "testtime.png"), dpi=120)
        print(f"[plot] {os.path.join(OUT_DIR, 'testtime.png')}")
    print("\nExpect: loss drops then plateaus near train T (training depth = ceiling).")


# --------------------------------------------------------------------------- #
# Experiment 4: IsoFLOP scaling law (looping vs data at fixed compute)
# --------------------------------------------------------------------------- #
def exp_isoflop(args):
    print("\n=== Experiment 4: IsoFLOP — looping is an orthogonal axis ===")
    # FLOPs per step ~ (effective depth) * (batch). Hold compute = loops * steps constant.
    nd = args.nd
    budget = args.loops * args.steps  # (loop-steps) proxy for fixed FLOP budget
    print(f"  fixed budget loops*steps = {budget}; varying loop count T")
    rows = []
    for T in [1, 2, 4, 8]:
        steps_T = max(200, budget // T)   # more loops -> fewer steps (less data) at fixed compute
        torch.manual_seed(0)
        m = LoopedModel(args.dim, args.heads, n_unique=2, n_loops=T, inj="parcae")
        train(m, nd, steps=steps_T, lr=args.lr, train_loops=T, label=f"T={T},steps={steps_T}")
        loss, tok_acc, seq_acc = evaluate(m, nd, eval_loops=T)
        rows.append([T, steps_T, T * steps_T, round(loss, 4), round(tok_acc, 4)])
        print(f"  -> T={T}: steps={steps_T} (compute={T*steps_T}) val_loss={loss:.4f} tok_acc={tok_acc:.3f}")
    save_csv("isoflop.csv", rows, ["loops", "steps", "compute", "val_loss", "tok_acc"])
    plt = _plt()
    if plt:
        Ts = [r[0] for r in rows]; ls = [r[3] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(Ts, ls, marker="o")
        ax.set(title="IsoFLOP: val loss vs loop count (fixed compute)",
               xlabel="train loops T", ylabel="val loss", xscale="log")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "isoflop.png"), dpi=120)
        print(f"[plot] {os.path.join(OUT_DIR, 'isoflop.png')}")
    print("\nExpect a U-shape / optimum: at fixed compute there's a best loop count;\n"
          "too few loops underuses depth, too many starves data (Parcae's tradeoff).")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Looped-transformer scaling lab")
    p.add_argument("experiment",
                   choices=["stability", "iso", "testtime", "isoflop", "all"])
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--loops", type=int, default=4, help="base loop count T")
    p.add_argument("--nd", type=int, default=3, help="number of digits in operands")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    print(f"device={DEVICE}  out_dir={OUT_DIR}")
    runners = {"stability": exp_stability, "iso": exp_iso,
               "testtime": exp_testtime, "isoflop": exp_isoflop}
    if args.experiment == "all":
        for fn in runners.values():
            fn(args)
    else:
        runners[args.experiment](args)


if __name__ == "__main__":
    main()
