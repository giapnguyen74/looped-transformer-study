"""
phase_a/inspect_kg.py — load the generated KG corpus and print a few samples per split,
showing the fixed-length prompt token layout. Demonstrates the format end-to-end (PHASE_A.md
§8.3). Stdlib only.

  python inspect_kg.py            # uses ./data/kg
  python inspect_kg.py --n 3
"""

import argparse
import json
import os
from collections import defaultdict

from gen_kg_queries import Config, prompt_tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "data", "kg"))
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()

    meta = json.load(open(os.path.join(args.dir, "meta.json")))
    cfg = Config(**meta["config"])
    print(f"vocab={meta['vocab_size']}  prompt_len={meta['prompt_len']}  "
          f"n_facts={meta['n_facts']} {meta['facts_by_owner']}")
    print(f"split_counts={meta['split_counts']}\n")

    by_split = defaultdict(list)
    for line in open(os.path.join(args.dir, "queries.jsonl")):
        q = json.loads(line)
        if len(by_split[q["split"]]) < args.n:
            by_split[q["split"]].append(q)

    for split in ["train", "iid_test", "systematic", "extrapolation"]:
        print(f"== {split} ==")
        for q in by_split[split]:
            toks = prompt_tokens(q["head"], q["rels"], cfg)
            print(f"  k={q['k']}  head={q['head']} rels={q['rels']} -> answer={q['answer']}")
            print(f"     prompt_tokens({len(toks)}): {toks}")
        print()


if __name__ == "__main__":
    main()
