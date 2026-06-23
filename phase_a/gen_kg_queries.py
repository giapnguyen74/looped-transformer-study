"""
phase_a/gen_kg_queries.py — Phase A: multi-hop knowledge-graph data generator.

Implements the spec in ../PHASE_A.md with the two feasibility fixes from review:

  FIX 1 — constructive forward walks. Deep chains are BUILT by walking forward and only
  choosing relations that are defined (or creating them on the fly), never by sampling a
  random relation sequence and hoping the path exists. (Random-sequence sampling makes a
  k=10 chain ~0.25^9 ≈ 1e-6 likely — infeasible.)

  FIX 2 — reserved edges for a pure `systematic` split. Every edge (head, rel) is OWNED by
  exactly one pool: `train` or `sys`. Train/iid/extrapolation queries only traverse train-
  owned edges; systematic queries only traverse sys-owned edges. So their edge sets are
  disjoint *by construction* — the systematic facts are memorizable (they're in facts.jsonl)
  but never appear in any train query path, which is otherwise impossible once 50k train
  queries saturate a small fact bank.

Also: self-loops and within-path revisits are avoided, so each query path is a simple path
of exactly k distinct hops (no trivial fixed-point / cycle shortcuts). Difficulty k is never
written as a token — it is only recoverable by counting non-PAD relation slots.

Stdlib only (torch-free, per LESSONS.md #1). Importable: `main(seed=..., out_dir=..., **knobs)`.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # NOTE: world size is deliberately SMALL. The model must memorize every fact in its
    # parameters; a large world (e.g. E=5000,R=20 → ~80k facts) is far beyond a small looped
    # model's associative-memory capacity and 1-hop recall fails. E=300,R=10 → ~3k facts,
    # which a dim-256 block can store. Scale the world WITH the model, not independently.
    E: int = 300             # entities (ids 0..E-1)
    R: int = 10              # relations
    K_train_max: int = 5     # max hops in train (>4 per Loop-Think: R>4 unlocks extrapolation)
    K_eval_max: int = 10     # deepest hops tested (extrapolation)
    K_max: int = 10          # prompt relation slots = fixed prompt length driver (== K_eval_max)
    N_train: int = 20000
    N_iid: int = 1000
    N_sys: int = 1000
    N_extrap_per_k: int = 500    # per depth in (K_train_max, K_eval_max]
    sys_reserve_frac: float = 0.3   # fraction of (head,rel) edges reserved for the sys pool
    seed: int = 0

    def special(self):
        """Token-id layout (dense integers). entities < relations < specials."""
        PAD_REL = self.E + self.R
        BOS = self.E + self.R + 1
        QUERY = self.E + self.R + 2
        return PAD_REL, BOS, QUERY, BOS  # last unused; keep tuple stable

    def vocab_size(self):
        return self.E + self.R + 3


def prompt_tokens(head, rels, cfg: Config):
    """Fixed-length prompt: [BOS, head, rel_1..rel_k, PAD_REL*(K_max-k), QUERY].
    Length is always K_max + 3, regardless of k."""
    PAD_REL = cfg.E + cfg.R
    BOS = cfg.E + cfg.R + 1
    QUERY = cfg.E + cfg.R + 2
    rel_ids = [cfg.E + r for r in rels] + [PAD_REL] * (cfg.K_max - len(rels))
    return [BOS, head] + rel_ids + [QUERY]


# --------------------------------------------------------------------------- #
# The knowledge graph (functional, edge-owned)
# --------------------------------------------------------------------------- #
class KG:
    def __init__(self, cfg: Config, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.tail = {}    # (head, rel) -> tail        (functional: one tail per (head,rel))
        self.owner = {}   # (head, rel) -> "train"|"sys"
        # Reserve a per-head subset of relations for the sys pool so the systematic split has
        # guaranteed edge capacity (otherwise train saturates E*R and starves sys). Train may
        # only create on non-reserved (head,rel); sys only on reserved -> edge sets disjoint.
        n_res = max(1, round(cfg.sys_reserve_frac * cfg.R))
        self.reserved = [set(rng.sample(range(cfg.R), n_res)) for _ in range(cfg.E)]

    def _usable_relations(self, cur, pool, visited):
        """Relations we may take from `cur` in `pool` without cycling, honoring reservation:
        existing same-pool edges to an unvisited node, plus creatable edges in this pool's
        eligible (reserved-vs-not) slots."""
        out = []
        reserved_here = self.reserved[cur]
        for r in range(self.cfg.R):
            eligible = (r in reserved_here) if pool == "sys" else (r not in reserved_here)
            if not eligible:
                continue
            key = (cur, r)
            if key in self.tail:
                if self.owner[key] == pool and self.tail[key] not in visited:
                    out.append(r)
            else:
                out.append(r)             # creatable in this pool's slot
        return out

    def _step(self, cur, r, pool, visited):
        key = (cur, r)
        if key in self.tail:
            return self.tail[key]
        # create a fresh edge to a node that keeps the path simple (no self-loop / revisit)
        while True:
            t = self.rng.randrange(self.cfg.E)
            if t != cur and t not in visited:
                break
        self.tail[key] = t
        self.owner[key] = pool
        return t

    def make_query(self, k, pool, used_tuples, tries=400):
        """Forward-walk a simple length-k path in `pool`. Returns (head, rels, answer) or None."""
        for _ in range(tries):
            h = self.rng.randrange(self.cfg.E)
            cur, visited, rels, ok = h, {h}, [], True
            for _i in range(k):
                cands = self._usable_relations(cur, pool, visited)
                if not cands:
                    ok = False
                    break
                r = self.rng.choice(cands)
                t = self._step(cur, r, pool, visited)
                rels.append(r)
                visited.add(t)
                cur = t
            if not ok:
                continue
            tup = (h, tuple(rels))
            if tup in used_tuples:
                continue
            used_tuples.add(tup)
            return h, rels, cur
        return None

    def follow(self, head, rels):
        """Deterministically follow a relation chain; returns final entity or None if broken."""
        cur = head
        for r in rels:
            key = (cur, r)
            if key not in self.tail:
                return None
            cur = self.tail[key]
        return cur


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate(cfg: Config):
    rng = random.Random(cfg.seed)
    kg = KG(cfg, rng)
    used = set()              # global (head, rels) tuples -> split disjointness
    queries = []

    def batch(n, kmin, kmax, pool, split):
        made = 0
        for _ in range(n * 3):              # headroom for retries/dupes
            if made >= n:
                break
            k = rng.randint(kmin, kmax)
            q = kg.make_query(k, pool, used)
            if q is None:
                continue
            h, rels, ans = q
            queries.append({"head": h, "rels": rels, "answer": ans, "k": k, "split": split})
            made += 1
        return made

    # Order matters: train first establishes the train-owned subgraph; iid/extrapolation
    # reuse it; systematic builds its own reserved subgraph.
    n_train = batch(cfg.N_train, 1, cfg.K_train_max, "train", "train")
    n_iid = batch(cfg.N_iid, 1, cfg.K_train_max, "train", "iid_test")
    n_extr = 0
    for k in range(cfg.K_train_max + 1, cfg.K_eval_max + 1):
        n_extr += batch(cfg.N_extrap_per_k, k, k, "train", "extrapolation")
    n_sys = batch(cfg.N_sys, 1, cfg.K_train_max, "sys", "systematic")

    facts = [{"head": h, "rel": r, "tail": t} for (h, r), t in kg.tail.items()]
    counts = {"train": n_train, "iid_test": n_iid, "extrapolation": n_extr, "systematic": n_sys}
    return kg, facts, queries, counts


# --------------------------------------------------------------------------- #
# Sanity checks (PHASE_A.md §6) — abort on any failure
# --------------------------------------------------------------------------- #
def sanity_checks(cfg, kg, facts, queries):
    errs = []

    # 1. Functionality: one tail per (head, rel)
    seen = set()
    for f in facts:
        key = (f["head"], f["rel"])
        if key in seen:
            errs.append(f"functionality: duplicate {key}")
        seen.add(key)

    # 2. Fixed length: every prompt is K_max+3 tokens, answer is 1 token
    L = cfg.K_max + 3
    for q in queries:
        if len(prompt_tokens(q["head"], q["rels"], cfg)) != L:
            errs.append(f"length: prompt != {L} for {q['head'],q['rels']}")
            break

    # 3. Answerability: re-derive every answer by following the chain
    for q in queries:
        if kg.follow(q["head"], q["rels"]) != q["answer"]:
            errs.append(f"answerability: {q['head'],q['rels']} -> {q['answer']} mismatch")
            break

    # 4. Split disjointness: no (head, rels) tuple in two splits
    tuples = [(q["head"], tuple(q["rels"])) for q in queries]
    if len(tuples) != len(set(tuples)):
        errs.append("disjointness: duplicate (head, rels) across queries")

    # 5. Systematic purity: every edge on a systematic path is sys-owned (never train)
    for q in queries:
        if q["split"] != "systematic":
            continue
        cur = q["head"]
        for r in q["rels"]:
            if kg.owner.get((cur, r)) != "sys":
                errs.append(f"systematic purity: train edge {(cur,r)} on sys path")
                break
            cur = kg.tail[(cur, r)]
        if errs and errs[-1].startswith("systematic"):
            break

    # 6. Length-vs-k independence: k recoverable ONLY by counting non-PAD relation slots,
    #    and k is never emitted as its own token.
    PAD_REL = cfg.E + cfg.R
    for q in queries:
        toks = prompt_tokens(q["head"], q["rels"], cfg)
        nonpad = sum(1 for t in toks[2:2 + cfg.K_max] if t != PAD_REL)
        if nonpad != q["k"]:
            errs.append(f"length-vs-k: non-PAD count {nonpad} != k {q['k']}")
            break
    # also: no standalone "difficulty token" exists in the layout (only ents/rels/specials)
    # (structural — the layout in prompt_tokens contains no k field; asserted by construction)

    return errs


# --------------------------------------------------------------------------- #
# Write + main
# --------------------------------------------------------------------------- #
def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(seed=0, out_dir=None, **knobs):
    cfg = Config(seed=seed, **knobs)
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = out_dir or os.path.join(here, "data", "kg")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[gen] config: E={cfg.E} R={cfg.R} K_train_max={cfg.K_train_max} "
          f"K_eval_max={cfg.K_eval_max} seed={cfg.seed}")
    kg, facts, queries, counts = generate(cfg)

    print(f"[gen] facts created: {len(facts):,}  "
          f"(train-owned {sum(o=='train' for o in kg.owner.values()):,}, "
          f"sys-owned {sum(o=='sys' for o in kg.owner.values()):,})")
    print(f"[gen] queries by split: {counts}")

    errs = sanity_checks(cfg, kg, facts, queries)
    if errs:
        for e in errs:
            print(f"[FAIL] {e}")
        raise SystemExit(1)
    print("[gen] all 6 sanity checks passed")

    _write_jsonl(os.path.join(out_dir, "facts.jsonl"), facts)
    _write_jsonl(os.path.join(out_dir, "queries.jsonl"), queries)
    PAD_REL, BOS, QUERY, _ = cfg.special()
    meta = {
        "config": asdict(cfg),
        "vocab_size": cfg.vocab_size(),
        "tokens": {"entity_range": [0, cfg.E], "relation_range": [cfg.E, cfg.E + cfg.R],
                   "PAD_REL": PAD_REL, "BOS": BOS, "QUERY": QUERY},
        "prompt_len": cfg.K_max + 3,
        "n_facts": len(facts),
        "facts_by_owner": {"train": sum(o == "train" for o in kg.owner.values()),
                           "sys": sum(o == "sys" for o in kg.owner.values())},
        "split_counts": counts,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[gen] wrote facts.jsonl, queries.jsonl, meta.json -> {out_dir}")
    return out_dir


def _cli():
    p = argparse.ArgumentParser(description="Phase A — multi-hop KG data generator")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", dest="out_dir", default=None)
    for fld, val in asdict(Config()).items():
        if fld == "seed":
            continue
        p.add_argument(f"--{fld.replace('_', '-')}", dest=fld, type=type(val), default=val)
    a = vars(p.parse_args())
    seed = a.pop("seed"); out_dir = a.pop("out_dir")
    main(seed=seed, out_dir=out_dir, **a)


if __name__ == "__main__":
    _cli()
