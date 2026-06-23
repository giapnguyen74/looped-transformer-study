"""
phase_b/train.py — train the looped transformer on the KG game (Phase B).

Reads the Phase A corpus (respecting `meta.json`), trains on a MIX of:
  * facts as 1-hop prompts  -> memorization (knowledge into parameters; includes sys facts)
  * train queries (k=1..K_train_max) -> composition (the reasoning behavior)
answering from parametric memory (facts are never in the prompt).

Key recipe (PROBLEM.md §6 / summary):
  * dynamic loop depth per step, R ~ randint(1, R_max), R_max > 4.
  * deep / per-iteration supervision: after loop t the readout is trained toward the entity
    reached in min(t, k) hops, so each loop learns ONE hop and extra loops HOLD the answer
    (idempotent past k -> lets test-time loops extrapolate to deeper k).

Eval sweeps the loop budget T per split to test the real claim: do more test-time loops solve
deeper-than-trained queries?  (iid_test, systematic, extrapolation by depth.)

Needs torch (training). Auto-selects cuda/mps/cpu.

  python train.py                       # uses ../phase_a/data/kg
  python train.py --steps 8000 --r-max 8 --dim 128
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("[phase_b] PyTorch required: pip install -r ../requirements.txt")

from model import LoopedKGReasoner, n_params, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_corpus(data_dir):
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    facts = [json.loads(l) for l in open(os.path.join(data_dir, "facts.jsonl"))]
    queries = [json.loads(l) for l in open(os.path.join(data_dir, "queries.jsonl"))]
    tail = {(f["head"], f["rel"]): f["tail"] for f in facts}
    return meta, facts, queries, tail


def encode_prompt(head, rels, meta):
    """[BOS, head, rel_ids..., PAD_REL*pad, QUERY] — fixed length = prompt_len."""
    t = meta["tokens"]
    E = meta["config"]["E"]
    K_max = meta["config"]["K_max"]
    rel_ids = [E + r for r in rels] + [t["PAD_REL"]] * (K_max - len(rels))
    return [t["BOS"], head] + rel_ids + [t["QUERY"]]


def chain_of(head, rels, tail):
    """Running entity after each hop: [head, e1, ..., ek]. None if a hop is missing."""
    out = [head]
    cur = head
    for r in rels:
        if (cur, r) not in tail:
            return None
        cur = tail[(cur, r)]
        out.append(cur)
    return out


def build_examples(meta, facts, queries, tail):
    """Each example: (prompt_ids, chain). Facts become 1-hop examples; train queries compose."""
    train_ex, eval_ex = [], defaultdict(list)
    # memorization: every fact (incl sys-owned) as a 1-hop prompt
    for f in facts:
        train_ex.append((encode_prompt(f["head"], [f["rel"]], meta), [f["head"], f["tail"]]))
    # composition + eval splits
    for q in queries:
        ch = chain_of(q["head"], q["rels"], tail)
        if ch is None or ch[-1] != q["answer"]:
            continue
        ex = (encode_prompt(q["head"], q["rels"], meta), ch)
        if q["split"] == "train":
            train_ex.append(ex)
        else:
            eval_ex[q["split"]].append((ex, q["k"]))
    return train_ex, eval_ex


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def deep_sup_targets(chains, T):
    """Target entity after each of T loops: chain[min(t, k)] (hold the answer past k)."""
    tgt = torch.empty(len(chains), T, dtype=torch.long)
    for i, ch in enumerate(chains):
        k = len(ch) - 1
        for t in range(1, T + 1):
            tgt[i, t - 1] = ch[min(t, k)]
    return tgt


@torch.no_grad()
def evaluate(model, examples_k, meta, device, n_loops, n_max=None):
    """examples_k: list of ((prompt, chain), k). Accuracy of final-loop answer (argmax over
    the entity range only). n_max=None evaluates all (avoids dropping deep-k items)."""
    model.eval()
    E = meta["config"]["E"]
    by_k = defaultdict(lambda: [0, 0])
    items = examples_k if n_max is None else examples_k[:n_max]
    for s in range(0, len(items), 512):
        chunk = items[s:s + 512]
        ids = torch.tensor([e[0][0] for e in chunk], dtype=torch.long, device=device)
        logits = model(ids, n_loops=n_loops)[:, :E]       # restrict to entity tokens
        pred = logits.argmax(-1)
        for j, ((_, ch), k) in enumerate(chunk):
            ok = int(pred[j].item() == ch[-1])
            by_k[k][0] += ok; by_k[k][1] += 1
    model.train()
    return {k: c / max(n, 1) for k, (c, n) in by_k.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(HERE, "..", "phase_a", "data", "kg"))
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--n-unique", dest="n_unique", type=int, default=1)
    p.add_argument("--r-max", dest="r_max", type=int, default=0,
                   help="max train loops (R_max>4); 0 => use K_eval_max so eval depths are in-range")
    p.add_argument("--zero-init", dest="zero_init", action="store_true",
                   help="zero-init block outputs (Loop-Think); off by default")
    p.add_argument("--no-inject", dest="inject", action="store_false",
                   help="bare loop (Loop-Think): drop the Parcae input injection")
    p.add_argument("--no-deep-sup", dest="deep_sup", action="store_false",
                   help="outcome-only loss (final answer only); default = per-iteration deep supervision")
    p.add_argument("--eval-every", dest="eval_every", type=int, default=0,
                   help="probe eval (iid k=3, sys k=2, ext k=6 at matched T) every N steps; 0=off — "
                        "the right instrument for watching grokking")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--bsz", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--fact-ratio", dest="fact_ratio", type=float, default=0.5,
                   help="fraction of each batch drawn from facts (memorization) vs queries")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    meta, facts, queries, tail = load_corpus(args.data)
    V, L = meta["vocab_size"], meta["prompt_len"]
    Kt, Ke = meta["config"]["K_train_max"], meta["config"]["K_eval_max"]
    r_max = args.r_max or Ke           # train with enough loops that eval depths are in-range
    train_ex, eval_ex = build_examples(meta, facts, queries, tail)

    # split the train pool into facts vs composition queries for ratio sampling
    n_facts = len(facts)
    fact_pool = train_ex[:n_facts]
    query_pool = train_ex[n_facts:]
    print(f"== Phase B: KG looped reasoner  device={device} ==")
    print(f"  vocab={V} prompt_len={L} train_depths=1-{Kt} eval_depths=1-{Ke}  R_max={r_max}")
    print(f"  memorization facts={len(fact_pool):,}  composition queries={len(query_pool):,}")

    model = LoopedKGReasoner(V, L, dim=args.dim, heads=args.heads, n_unique=args.n_unique,
                             n_loops=r_max, zero_init=args.zero_init, inject=args.inject).to(device)
    cfg_str = f"inject={args.inject} zero_init={args.zero_init} deep_sup={args.deep_sup}"
    rho_str = f"rho(A)={model.inj.rho():.3f}" if model.inj is not None else "bare-loop"
    print(f"  params={n_params(model):,}  {rho_str}  {cfg_str}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)

    # probe subsets for periodic grokking watch (cheap)
    def probe(split, k, n=300):
        return [e for e in eval_ex[split] if e[1] == k][:n]
    probes = {"iid_k3": (probe("iid_test", 3), 3),
              "sys_k2": (probe("systematic", 2), 2),
              "ext_k6": (probe("extrapolation", 6), 6)}

    def sample_batch():
        n_f = int(round(args.bsz * args.fact_ratio))
        picks = ([fact_pool[int(i)] for i in torch.randint(0, len(fact_pool), (n_f,), generator=g)]
                 + [query_pool[int(i)] for i in torch.randint(0, len(query_pool), (args.bsz - n_f,), generator=g)])
        ids = torch.tensor([e[0] for e in picks], dtype=torch.long, device=device)
        chains = [e[1] for e in picks]
        return ids, chains

    t_start = time.time()
    for step in range(1, args.steps + 1):
        ids, chains = sample_batch()
        T = int(torch.randint(1, r_max + 1, (1,), generator=g).item())        # dynamic depth
        if args.deep_sup:
            per = model(ids, n_loops=T, per_iter=True)                        # list[T] of (B,V)
            tgt = deep_sup_targets(chains, T).to(device)                      # (B,T)
            loss = sum(F.cross_entropy(per[t], tgt[:, t]) for t in range(T)) / T
        else:
            logits = model(ids, n_loops=T)                                   # final readout only
            final = torch.tensor([c[-1] for c in chains], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, final)                            # outcome-only
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            el = time.time() - t_start
            print(f"  step {step:5d}  loss {loss.item():.4f}  [{el:6.1f}s  {step/el:5.1f} it/s]")
        if args.eval_every and step % args.eval_every == 0:
            msg = []
            for name, (items, k) in probes.items():
                a = evaluate(model, items, meta, device, n_loops=k).get(k, float('nan')) if items else float('nan')
                msg.append(f"{name}={a:.3f}")
            print(f"    [grok-watch @ {step}] " + "  ".join(msg))
    train_secs = time.time() - t_start
    print(f"  training done in {train_secs:.1f}s "
          f"({train_secs/60:.1f} min, {args.steps/train_secs:.1f} it/s avg)")

    # ---- eval: sweep loop budget T (within trained range) + a MATCHED T=k column ----
    # T=k isolates "can it do k hops when given exactly k loops" from overthinking (too many
    # loops drifting). All T <= r_max so loop count itself is in training distribution.
    eval_loops = sorted({Kt, r_max})
    nan = float("nan")

    def matched(items):
        by_k = defaultdict(list)
        for ex_k in items:
            by_k[ex_k[1]].append(ex_k)
        return {k: evaluate(model, v, meta, device, n_loops=max(1, k))[k] for k, v in by_k.items()}

    print("\n  accuracy by split / depth — swept T (<=R_max) and matched T=k:")
    t_eval = time.time()
    rows = []
    for split in ["iid_test", "systematic", "extrapolation"]:
        accs = {T: evaluate(model, eval_ex[split], meta, device, n_loops=T) for T in eval_loops}
        m = matched(eval_ex[split])
        ks = sorted(set(m) | {k for T in eval_loops for k in accs[T]})
        for k in ks:
            swept = "".join(f"  T={T}:{accs[T].get(k, nan):.3f}" for T in eval_loops)
            print(f"   {split:13s} k={k:<2d}{swept}  T=k:{m.get(k, nan):.3f}")
            rows.append([split, k] + [round(accs[T].get(k, nan), 4) for T in eval_loops]
                        + [round(m.get(k, nan), 4)])

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "phase_b_eval.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "k"] + [f"acc_T{T}" for T in eval_loops] + ["acc_T_eq_k"])
        w.writerows(rows)
    print(f"\n[csv] {csv_path}")
    print(f"  timing: train {train_secs:.1f}s + eval {time.time()-t_eval:.1f}s "
          f"= {(train_secs + time.time()-t_eval)/60:.1f} min total")
    print("  Read: matched T=k is the fair test. extrapolation (k>train max) > 0 at T=k = the claim;")
    print("        flat-zero at T=k = 'training depth is the ceiling' (an interesting negative).")


if __name__ == "__main__":
    main()
