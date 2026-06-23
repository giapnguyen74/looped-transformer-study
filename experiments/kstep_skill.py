"""
kstep_skill.py — the first "real" controlled experiment: can a looped transformer learn a
k-STEP ARITHMETIC skill and EXTRAPOLATE to depths it never trained on?

THE TASK (controlled toy, on purpose).
A problem is a start value followed by k sequential +/- updates; the model must track the
running value across all k steps. k = the reasoning DEPTH. We use a compact grammar so the
"chain of thought" is exact and self-generated (no teacher, no rejection sampling needed):

    problem:  S 5 + 3 - 7 + 2 =          (start 5, then k=3 ops)
    trace:                      8 1 3     (running value after each step  <- the exact CoT)
    answer:                           # 3 ;

So one training line is:   "S 5 + 3 - 7 + 2 = 8 1 3 # 3 ;"
The eval prompt is everything up to "="; the model must GENERATE the running values and the
answer. Because the trace is the running value after each op, it is correct by construction.

WHY THIS DESIGN.
k is an explicit dial for required depth. Train on shallow problems (k=1..4) and test on
deeper ones (k=1..8) to measure DEPTH EXTRAPOLATION — the headline claim for looped models
(Loop-Think / Parcae). We also vary the number of LOOPS at eval time (test-time scaling):
a looped model can, in principle, solve a deeper problem by iterating its one block more.

STAGES (like loop_scaling_lab.py):
    gen    — write kstep_data/train.jsonl (k=1..4) and test.jsonl (k=1..8, disjoint)
    train  — char-level SFT of a looped LM on the traces, then depth-stratified eval
    all    — gen then train  (default)

Torch is imported lazily so `gen` (pure data) runs anywhere; `train` needs torch.
Defaults are sized for a single GPU/MPS in a few minutes; scale via flags.
"""

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "kstep_data")
OUT_DIR = os.path.join(HERE, "results")


# --------------------------------------------------------------------------- #
# Task generation (no torch) — exact gold + exact CoT trace.
# --------------------------------------------------------------------------- #
def rev_num(n):
    """Render an int with its DIGITS REVERSED (least-significant first), keeping the sign:
    68 -> '86', -28 -> '-82'. Big-endian emission makes multi-digit arithmetic nearly
    unlearnable autoregressively (you'd need the carry before you emit the high digit); the
    lab's addition task reverses for the same reason. Only PREDICTED numbers are reversed;
    the given prompt operands stay normal."""
    s = str(abs(n))[::-1]
    return "-" + s if n < 0 else s


def make_example(k, rng, fmt="steps"):
    """Return (prompt, full, answer). Two chain-of-thought formats:

      values : prompt 'S 63 + 5 - 7 =', completion = the reversed running values
               '86 16 ...' then '# ans ;'. Compact, but the first running value sits in a
               different position than later ones, so the model learns a positional shortcut
               that doesn't compose (k=1 can be perfect while k>=2 is at chance).

      steps  : prompt 'S 36 + 5 - 7 =' (start reversed), completion re-echoes each step as a
               LOCAL single op  'Xrev op d = Yrev', chained:
                   S 36 + 5 = 86 - 7 = 16 - 8 = 35 - 4 = 94 # 94 ;
               Now EVERY arithmetic step matches the 'X op d =' circuit the model already
               nails at k=1, the left operand of each step is the previous result (no
               un-reversing needed since everything is reversed-consistent), and the same
               local map repeats at every depth -> far better composition / extrapolation.

    Predicted numbers are digit-reversed; the eval answer is parsed after '#'. The prompt has a
    FIXED length per depth (start is 2 chars) so eval can batch-decode by depth."""
    v = rng.randint(10, 99)
    ops, runs, cur = [], [], v
    for _ in range(k):
        op = rng.choice("+-")
        d = rng.randint(1, 9)
        cur = cur + d if op == "+" else cur - d
        ops.append((op, d)); runs.append(cur)
    answer_tag = f" # {rev_num(runs[-1])} ;"
    if fmt == "values":
        prompt = "S " + str(v) + "".join(f" {op} {d}" for op, d in ops) + " ="
        comp = "".join(f" {rev_num(r)}" for r in runs) + answer_tag
    else:  # steps (interleaved, the default)
        prompt = "S " + rev_num(v) + "".join(f" {op} {d}" for op, d in ops) + " ="
        comp = " " + rev_num(v) + "".join(f" {op} {d} = {rev_num(r)}"
                                          for (op, d), r in zip(ops, runs)) + answer_tag
    return prompt, prompt + comp, runs[-1]


def build_split(n, kmin, kmax, seed, avoid=None, fmt="steps"):
    """n distinct problems with depth uniform in [kmin, kmax]. Returns (rows, prompt_set)."""
    rng = random.Random(seed)
    avoid = avoid or set()
    rows, seen = [], set()
    tries = 0
    while len(rows) < n and tries < n * 50:
        tries += 1
        k = rng.randint(kmin, kmax)
        prompt, full, ans = make_example(k, rng, fmt)
        if prompt in seen or prompt in avoid:
            continue
        seen.add(prompt)
        rows.append({"k": k, "prompt": prompt, "full": full, "answer": ans})
    return rows, seen


def _data_paths(fmt):
    return (os.path.join(DATA_DIR, f"train_{fmt}.jsonl"),
            os.path.join(DATA_DIR, f"test_{fmt}.jsonl"))


def gen(args):
    os.makedirs(DATA_DIR, exist_ok=True)
    fmt = args.format
    # Build TEST first (fixed count per depth, 1..kmax_test), then TRAIN avoiding test
    # prompts — guarantees the held-out set exists even for small low-depth spaces.
    test_rows, test_prompts = [], set()
    for k in range(1, args.kmax_test + 1):
        rows, ps = build_split(args.n_eval, k, k, seed=1000 + k, avoid=test_prompts, fmt=fmt)
        test_rows.extend(rows); test_prompts |= ps
    train_rows, _ = build_split(args.n_train, args.kmin, args.kmax_train, seed=0,
                                avoid=test_prompts, fmt=fmt)

    train_path, test_path = _data_paths(fmt)
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(test_path, "w") as f:
        for r in test_rows:
            f.write(json.dumps({"k": r["k"], "prompt": r["prompt"], "answer": r["answer"]}) + "\n")

    by_depth = defaultdict(int)
    for r in train_rows:
        by_depth[r["k"]] += 1
    print(f"[gen] format={fmt}")
    print(f"[gen] train: {len(train_rows)} problems (depths {args.kmin}-{args.kmax_train}) "
          f"{dict(sorted(by_depth.items()))}")
    print(f"[gen] test : {len(test_rows)} problems ({args.n_eval}/depth, depths 1-{args.kmax_test})")
    print(f"[gen] in-distribution depths: 1-{args.kmax_train}   "
          f"extrapolation depths: {args.kmax_train + 1}-{args.kmax_test}")
    print(f"[gen] wrote -> {train_path}, {test_path}")
    print(f"      example: {train_rows[0]['full']!r}")


# --------------------------------------------------------------------------- #
# Model (lazy torch) — compact looped LM, same skeleton as loop_scaling_lab.
# --------------------------------------------------------------------------- #
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except Exception:
    _TORCH = False

if _TORCH:
    def pick_device(prefer="auto"):
        if prefer != "auto":
            return prefer
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    class RMSNorm(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.g = nn.Parameter(torch.ones(d))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g

    def _rope_cache(T, dh, device, dtype, theta=10000.0):
        inv = 1.0 / (theta ** (torch.arange(0, dh, 2, device=device, dtype=torch.float) / dh))
        ang = torch.outer(torch.arange(T, device=device, dtype=torch.float), inv)  # (T, dh/2)
        emb = torch.cat([ang, ang], dim=-1)                                        # (T, dh)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    class RoPEAttention(nn.Module):
        """Multi-head self-attention with ROTARY positions applied to q,k. Encodes position
        RELATIVELY (q·k depends on i-j, not absolute i), which gives local digit alignment for
        the arithmetic AND translation-invariance so it generalizes to longer sequences."""
        def __init__(self, dim, heads):
            super().__init__()
            self.h, self.dh = heads, dim // heads
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
            self.o = nn.Linear(dim, dim)

        def forward(self, x, attn_mask):
            B, T, D = x.shape
            q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
            cos, sin = _rope_cache(T, self.dh, x.device, x.dtype)
            cos, sin = cos[None, None], sin[None, None]
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            att = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5) + attn_mask
            out = att.softmax(-1) @ v                                  # (B, H, T, dh)
            return self.o(out.transpose(1, 2).reshape(B, T, D))

    class Block(nn.Module):
        def __init__(self, dim, heads, use_rope=False):
            super().__init__()
            self.use_rope = use_rope
            self.attn = (RoPEAttention(dim, heads) if use_rope
                         else nn.MultiheadAttention(dim, heads, batch_first=True))
            self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
            self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)

        def forward(self, x, attn_mask):
            a = self.n1(x)
            if self.use_rope:
                x = x + self.attn(a, attn_mask)
            else:
                x = x + self.attn(a, a, a, attn_mask=attn_mask, need_weights=False)[0]
            x = x + self.ffn(self.n2(x))
            return x

    class Injection(nn.Module):
        """Parcae contractive carry: diagonal A-bar in (0,1) -> rho < 1, stable by design."""
        def __init__(self, dim):
            super().__init__()
            self.proj_e = nn.Linear(dim, dim, bias=False)
            self.log_A = nn.Parameter(torch.zeros(dim))
            self.dt = nn.Parameter(torch.zeros(dim))

        def a_bar(self):
            return torch.exp(F.softplus(self.dt) * (-torch.exp(self.log_A)))

        def forward(self, h, e, block_out):
            return self.a_bar() * h + self.proj_e(e) + block_out

        def rho(self):
            return float(self.a_bar().max().item())

    def loop_index_embedding(h, t, loop_dim, theta=10000.0):
        dev, dt = h.device, h.dtype
        freqs = 1.0 / (theta ** (torch.arange(0, loop_dim, 2, device=dev, dtype=dt) / loop_dim))
        ang = t * freqs
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)[:loop_dim]
        full = torch.zeros(h.shape[-1], device=dev, dtype=dt)
        full[:loop_dim] = emb
        return h + full.view(1, 1, -1)

    def _sinusoid(max_len, dim):
        pos = torch.arange(max_len).float()[:, None]
        i = torch.arange(0, dim, 2).float()[None]
        ang = pos / (10000.0 ** (i / dim))
        out = torch.zeros(max_len, dim)
        out[:, 0::2] = ang.sin(); out[:, 1::2] = ang.cos()
        return out

    class LoopedLM(nn.Module):
        def __init__(self, vocab, dim=128, heads=4, n_unique=1, n_loops=4, max_len=256, pos="none"):
            super().__init__()
            self.dim, self.n_loops, self.pos_mode = dim, n_loops, pos
            self.tok = nn.Embedding(vocab, dim)
            # Token positional encoding governs LENGTH GENERALIZATION:
            #   learned     — absolute, does NOT extrapolate past trained positions
            #   sinusoidal  — fixed absolute; arithmetic ok but still stops/miscounts past depth
            #   none (NoPE) — structure length-generalizes, but loses digit alignment -> bad math
            #   rope        — RELATIVE (rotary); local alignment AND length generalization (both)
            if pos == "learned":
                self.pos = nn.Embedding(max_len, dim)
            elif pos == "sinusoidal":
                self.register_buffer("pos_table", _sinusoid(max_len, dim), persistent=False)
            # pos == "rope": positions handled inside attention; pos == "none": no positions
            self.blocks = nn.ModuleList([Block(dim, heads, use_rope=(pos == "rope"))
                                         for _ in range(n_unique)])
            self.inj = Injection(dim)
            self.norm = RMSNorm(dim)
            self.head = nn.Linear(dim, vocab)
            self.loop_dim = max(2, dim // 8)
            self._mask = {}

        def causal_mask(self, T, device):
            if T not in self._mask:
                m = torch.full((T, T), float("-inf"), device=device)
                self._mask[T] = torch.triu(m, diagonal=1)
            return self._mask[T]

        def forward(self, ids, n_loops=None):
            n_loops = n_loops or self.n_loops
            B, T = ids.shape
            mask = self.causal_mask(T, ids.device)
            e = self.tok(ids)
            if self.pos_mode == "learned":
                e = e + self.pos(torch.arange(T, device=ids.device))[None]
            elif self.pos_mode == "sinusoidal":
                e = e + self.pos_table[:T].to(e.dtype)[None]
            h = e
            for t in range(n_loops):
                x = loop_index_embedding(self.norm(h + e), t, self.loop_dim)
                for blk in self.blocks:
                    x = blk(x, mask)
                h = self.inj(h, e, x)
            return self.head(self.norm(h))


# --------------------------------------------------------------------------- #
# Train + depth-stratified eval
# --------------------------------------------------------------------------- #
def _load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_vocab(strings):
    chars = sorted(set("".join(strings)))
    stoi = {c: i for i, c in enumerate(chars)}
    return stoi, {i: c for c, i in stoi.items()}


@torch.no_grad() if _TORCH else (lambda f: f)
def eval_depths(model, test_rows, stoi, itos, device, n_loops, fmt="steps", max_chunk=256, collect=0):
    """Autoregressive greedy decode, grouped by depth (fixed prompt length per depth).
    If collect>0, also return up to `collect` sample (prompt, generation, pred, gold) per
    depth for inspection."""
    model.eval()
    by_k = defaultdict(list)
    for r in test_rows:
        by_k[r["k"]].append(r)
    acc, samples = {}, {}
    for k, rows in sorted(by_k.items()):
        prompts = [r["prompt"] for r in rows]
        golds = [r["answer"] for r in rows]
        L = len(prompts[0])                      # same length for all depth-k prompts
        # steps format echoes op+result each step, so it needs more room than values
        max_new = (10 * k + 16) if fmt == "steps" else (4 * k + 12)
        correct = 0
        for s in range(0, len(prompts), max_chunk):
            chunk = prompts[s:s + max_chunk]
            x = torch.tensor([[stoi[c] for c in p] for p in chunk],
                             dtype=torch.long, device=device)
            for _ in range(max_new):
                nxt = model(x, n_loops=n_loops)[:, -1].argmax(-1, keepdim=True)
                x = torch.cat([x, nxt], dim=1)
            for j in range(len(chunk)):
                gen_str = "".join(itos[int(t)] for t in x[j, L:])
                pred = _parse_answer(gen_str)
                if pred is not None and pred == golds[s + j]:
                    correct += 1
                if collect and len(samples.get(k, [])) < collect:
                    samples.setdefault(k, []).append(
                        (prompts[s + j], gen_str.split(";")[0] + ";", pred, golds[s + j]))
        acc[k] = correct / len(rows)
    model.train()
    return acc, samples


def _parse_answer(gen_str):
    """Pull the (reversed) integer after '#' in a generated continuation and un-reverse it."""
    if "#" not in gen_str:
        return None
    tail = gen_str.split("#", 1)[1].strip()
    tok = ""
    for ch in tail:
        if ch == "-" and not tok:
            tok = "-"
        elif ch.isdigit():
            tok += ch
        elif tok not in ("", "-"):
            break
    neg = tok.startswith("-")
    digs = tok[1:] if neg else tok
    if not digs.isdigit():
        return None
    val = int(digs[::-1])                                 # un-reverse the digits
    return -val if neg else val


def train(args):
    if not _TORCH:
        raise SystemExit("[train] PyTorch not installed. `pip install -r ../requirements.txt`")
    train_path, test_path = _data_paths(args.format)
    if not os.path.exists(train_path):
        gen(args)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)
    device = pick_device(args.device)

    train_rows = _load(train_path)
    test_rows = _load(test_path)
    fulls = [r["full"] for r in train_rows]
    stoi, itos = build_vocab(fulls + [r["prompt"] for r in test_rows])
    PAD = len(stoi)
    V = len(stoi) + 1                                # +1 for the pad id
    itos = dict(itos); itos[PAD] = ""

    # SFT examples: (ids, prompt_len). CRUCIAL: loss is masked to the COMPLETION
    # (trace + answer) only. The prompt operands are random, so scoring them would dominate
    # the gradient with irreducible noise and starve the deterministic computation we want
    # the model to learn — exactly the failure of a plain packed-stream LM here. This mirrors
    # the lab's masked answer-token loss.
    examples = [([stoi[c] for c in r["full"]], len(r["prompt"])) for r in train_rows]
    maxL = max(len(ids) for ids, _ in examples)

    depth_mode = "fixed" if args.fixed_loops else f"dynamic[{args.min_loops}-{args.loops}]"
    print(f"== k-step skill: train depths {args.kmin}-{args.kmax_train}, "
          f"test depths 1-{args.kmax_test}, device={device} ==")
    print(f"  vocab={len(stoi)}  examples={len(examples):,}  loops={depth_mode}  (loss on completion only)")
    model = LoopedLM(V, dim=args.dim, heads=args.heads, n_unique=1, n_loops=args.loops,
                     max_len=max(args.max_len, maxL + 8), pos=args.pos).to(device)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}  rho(A)={model.inj.rho():.3f}"
          f"  pos={args.pos}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    rng = random.Random(0)

    def make_batch(bsz):
        picks = [examples[rng.randrange(len(examples))] for _ in range(bsz)]
        m = max(len(ids) for ids, _ in picks)
        X = torch.full((bsz, m), PAD, dtype=torch.long)
        M = torch.zeros((bsz, m - 1), dtype=torch.float)   # mask aligned to targets y=X[:,1:]
        for r, (ids, pl) in enumerate(picks):
            X[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            for t in range(m - 1):                          # y[t]=X[t+1] is a completion token?
                if pl <= t + 1 < len(ids):
                    M[r, t] = 1.0
        return X.to(device), M.to(device)

    for step in range(1, args.steps + 1):
        X, M = make_batch(args.bsz)
        x, y = X[:, :-1], X[:, 1:]
        # dynamic depth: vary loop count per step so the model learns to USE a variable
        # number of iterations -> enables test-time loop scaling & depth extrapolation
        # (Parcae/Loop-Think). Fixed depth caps the test-time benefit at the train depth.
        nl = args.loops if args.fixed_loops else rng.randint(args.min_loops, args.loops)
        logits = model(x, n_loops=nl)
        ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none").view_as(y)
        loss = (ce * M).sum() / M.sum().clamp(min=1)        # completion-only loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            print(f"  step {step:5d}  loss {loss.item():.4f}")

    # depth-stratified accuracy at train loops and at extra loops (test-time scaling)
    eval_loops = sorted(set([args.loops, 2 * args.loops]))
    accs, samples = {}, {}
    for T in eval_loops:
        a, s = eval_depths(model, test_rows, stoi, itos, device, n_loops=T, fmt=args.format,
                           collect=(args.show_samples if T == eval_loops[0] else 0))
        accs[T] = a
        if s:
            samples = s

    if args.show_samples and samples:
        print(f"\n  sample generations @T={eval_loops[0]} (prompt -> generation | pred vs gold):")
        for k in sorted(samples):
            if k in (1, 2, args.kmax_train, args.kmax_train + 1, args.kmax_test):
                for prompt, gen, pred, gold in samples[k]:
                    ok = "ok" if pred == gold else "XX"
                    print(f"   k={k} {ok}  {prompt}{gen}  | pred={pred} gold={gold}")

    print("\n  answer accuracy by depth (■ = extrapolation, k > train max):")
    header = "  depth " + "".join(f"  acc@T={T}" for T in eval_loops)
    print(header)
    rows_csv = []
    for k in range(1, args.kmax_test + 1):
        mark = " " if k <= args.kmax_train else "■"
        line = f"  {mark}k={k:<2d}" + "".join(f"   {accs[T].get(k, float('nan')):.3f} " for T in eval_loops)
        print(line)
        rows_csv.append([k, k <= args.kmax_train] + [round(accs[T].get(k, float("nan")), 4) for T in eval_loops])

    csv_path = os.path.join(OUT_DIR, "kstep_extrapolation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth", "in_distribution"] + [f"acc_T{T}" for T in eval_loops])
        w.writerows(rows_csv)
    print(f"\n[csv] wrote {csv_path}")
    print("  Read: high in-distribution acc = skill learned; acc on ■ rows = depth"
          " extrapolation;\n  acc rising with more loops on deep k = test-time loop scaling.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="k-step arithmetic: depth extrapolation for looped LMs")
    p.add_argument("stage", choices=["gen", "train", "all"], nargs="?", default="all")
    p.add_argument("--kmin", type=int, default=1)
    p.add_argument("--kmax-train", dest="kmax_train", type=int, default=4)
    p.add_argument("--kmax-test", dest="kmax_test", type=int, default=8)
    p.add_argument("--n-train", dest="n_train", type=int, default=20000)
    p.add_argument("--n-eval", dest="n_eval", type=int, default=200, help="eval problems per depth")
    p.add_argument("--format", choices=["steps", "values"], default="steps",
                   help="steps = interleaved local single-op CoT (default); values = running-values CoT")
    p.add_argument("--pos", choices=["rope", "none", "sinusoidal", "learned"], default="rope",
                   help="positions: rope=rotary/relative (default, best of both), "
                        "none=NoPE, sinusoidal, learned absolute")
    # model / training
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--loops", type=int, default=4, help="max train loops (and base eval loops)")
    p.add_argument("--min-loops", dest="min_loops", type=int, default=1,
                   help="min train loops when dynamic depth is on")
    p.add_argument("--fixed-loops", dest="fixed_loops", action="store_true",
                   help="disable dynamic depth; train at exactly --loops every step")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--bsz", type=int, default=64)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--max-len", dest="max_len", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--show-samples", dest="show_samples", type=int, default=3,
                   help="print this many decoded samples per depth for inspection")
    args = p.parse_args()

    if args.stage == "gen":
        gen(args)
    elif args.stage == "train":
        train(args)
    else:
        gen(args)
        train(args)


if __name__ == "__main__":
    main()
