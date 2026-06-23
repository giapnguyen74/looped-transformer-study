"""
gen_math_problems.py — make verifiable math PROBLEMS to feed gen_math_transcripts.py.

This is STAGE A of the data pipeline. It only writes problems; turning them into verified
solution traces is STAGE B (gen_math_transcripts.py). The two scripts are coupled ONLY by
the JSONL format below, so each stays self-contained.

Three sources:
  programmatic : templated problems with EXACT, code-computed gold (SymPy). Trustworthy by
                 construction, offline, no model. kinds: arith, linear, fraction, gcd, percent.
  teacher      : a model on OpenRouter PROPOSES problems; gold is left EMPTY on purpose.
                 A model's own answer isn't trustworthy, so gold is filled later by
                 SELF-CONSISTENCY in gen_math_transcripts.py (majority vote across k
                 solutions; low-agreement problems are dropped). Caveat: if the same model
                 proposes and solves, a confidently-wrong consensus can mislabel — prefer
                 programmatic gold, or solve with a different model than the proposer.
  mock         : offline arithmetic proposer (no key, no gold) — for testing the teacher path.

Output: problems.jsonl — one JSON object per line:
    {"id": "...", "question": "...", "gold": "...", "source": "..."}
  `gold` is present only for the programmatic route; teacher/mock problems omit it so the
  transcript stage knows to derive it by self-consistency.

Dependencies: sympy (required). `requests` only for the teacher route.

Examples:
  python gen_math_problems.py --source programmatic --n 500 \\
        --kinds arith,linear,fraction,gcd,percent --out problems.jsonl

  export OPENROUTER_API_KEY=sk-or-...
  python gen_math_problems.py --source teacher --model openai/gpt-5.5 \\
        --n 200 --topic "algebra word problems" --difficulty hard --out problems.jsonl
"""

import argparse
import json
import os
import re
import math
import random
from dataclasses import dataclass
from typing import List, Optional

import sympy


@dataclass
class Problem:
    id: str
    question: str
    gold: Optional[str] = None     # None => transcript stage derives it by self-consistency
    source: str = "unknown"


# --------------------------------------------------------------------------- #
# Programmatic generator — EXACT gold computed in code (SymPy).
# --------------------------------------------------------------------------- #
def gen_problems_programmatic(n: int, kinds: List[str], seed: int = 0) -> List[Problem]:
    rng = random.Random(seed)
    probs = []
    for i in range(n):
        kind = rng.choice(kinds)
        if kind == "arith":
            op = rng.choice(["+", "-", "*"])
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            gold = str(a + b if op == "+" else a - b if op == "-" else a * b)
            q = f"What is {a} {op} {b}?"
        elif kind == "linear":
            x = rng.randint(-9, 9); a = rng.randint(2, 9); b = rng.randint(-9, 9)
            c = a * x + b
            term = "" if b == 0 else f" {'+' if b > 0 else '-'} {abs(b)}"
            q = f"Solve for x: {a}x{term} = {c}."
            gold = str(x)
        elif kind == "fraction":
            a, b, c, d = (rng.randint(1, 9) for _ in range(4))
            gold = str(sympy.Rational(a, b) + sympy.Rational(c, d))
            q = f"Compute {a}/{b} + {c}/{d}. Give the answer as a reduced fraction."
        elif kind == "gcd":
            a, b = rng.randint(10, 200), rng.randint(10, 200)
            gold = str(math.gcd(a, b))
            q = f"What is the greatest common divisor of {a} and {b}?"
        elif kind == "percent":
            p, nn = rng.choice([5, 10, 12, 20, 25, 50, 75]), rng.randint(20, 400)
            gold = str(sympy.Rational(p * nn, 100))
            q = f"What is {p}% of {nn}?"
        else:
            raise ValueError(f"unknown kind: {kind}")
        probs.append(Problem(id=f"prog{i}", question=q, gold=gold,
                             source=f"programmatic:{kind}"))
    return probs


# --------------------------------------------------------------------------- #
# Teacher proposer (OpenRouter) — proposes problems, leaves gold empty.
# --------------------------------------------------------------------------- #
WRITE_SYS = ("You are a math problem author. Output ONLY a JSON array of objects, each "
             "{\"question\": \"...\"}. Every problem must have ONE unambiguous, closed-form "
             "numeric answer. Do NOT include solutions or any other text.")


def _parse_problem_list(content: str) -> List[str]:
    """Extract question strings from a model response (JSON array preferred)."""
    s = content.strip()
    a, b = s.find("["), s.rfind("]")
    if a != -1 and b != -1 and b > a:
        try:
            arr = json.loads(s[a:b + 1])
            qs = [str(x.get("question", "")).strip() if isinstance(x, dict) else str(x).strip()
                  for x in arr]
            return [q for q in qs if q]
        except Exception:
            pass
    lines = [re.sub(r"^\s*\d+[\.\)]\s*", "", ln).strip() for ln in s.splitlines()]
    return [ln for ln in lines if len(ln) > 8]


def _openrouter_chat(system: str, user: str, temperature: float, model: str) -> str:
    """One OpenRouter chat call. Env: OPENROUTER_API_KEY (required), OPENROUTER_BASE_URL,
    OPENROUTER_REFERER / OPENROUTER_TITLE (optional ranking metadata)."""
    import requests
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    key = os.environ["OPENROUTER_API_KEY"]
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if os.environ.get("OPENROUTER_REFERER"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_REFERER"]
    if os.environ.get("OPENROUTER_TITLE"):
        headers["X-Title"] = os.environ["OPENROUTER_TITLE"]
    r = requests.post(
        f"{base}/chat/completions",
        headers=headers,
        json={"model": model, "temperature": temperature,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def propose_via_model(n: int, topic: str, difficulty: str, model: str) -> List[str]:
    """Ask the model for problems in batches; parse JSON, fall back to line parsing."""
    out, batch = [], 25
    while len(out) < n:
        k = min(batch, n - len(out))
        user = (f"Write {k} {difficulty} math problems about {topic}. "
                f"Each must have a single unambiguous closed-form answer.")
        qs = _parse_problem_list(_openrouter_chat(WRITE_SYS, user, 1.0, model))
        if not qs:                                   # avoid infinite loop on bad output
            break
        out.extend(qs)
    return out[:n]


def propose_mock(n: int, seed: int = 0) -> List[str]:
    """Offline arithmetic proposer (no gold) — exercises the teacher path without a key."""
    rng = random.Random(seed)
    return [f"What is {rng.randint(2,99)} {rng.choice(['+','-','*'])} {rng.randint(2,99)}?"
            for _ in range(n)]


# --------------------------------------------------------------------------- #
# Save + CLI
# --------------------------------------------------------------------------- #
def save_problems(probs: List[Problem], path: str):
    with open(path, "w") as f:
        for p in probs:
            d = {"id": p.id, "question": p.question, "source": p.source}
            if p.gold is not None:
                d["gold"] = p.gold
            f.write(json.dumps(d) + "\n")
    labeled = sum(p.gold is not None for p in probs)
    print(f"[done] wrote {len(probs)} problems -> {path}\n"
          f"  {labeled} with exact gold, {len(probs) - labeled} for self-consistency "
          f"(solve these with gen_math_transcripts.py)")


def main():
    p = argparse.ArgumentParser(description="Stage A — make verifiable math problems")
    p.add_argument("--source", choices=["programmatic", "teacher", "mock"],
                   default="programmatic")
    p.add_argument("--n", type=int, default=200, help="number of problems to generate")
    p.add_argument("--out", default="problems.jsonl")
    # programmatic
    p.add_argument("--kinds", default="arith,linear,fraction,gcd,percent",
                   help="programmatic problem kinds (comma-separated)")
    # teacher
    p.add_argument("--model", default="openai/gpt-5.5", help="OpenRouter model id (teacher)")
    p.add_argument("--topic", default="grade-school and algebra", help="teacher problem topic")
    p.add_argument("--difficulty", default="medium", help="teacher problem difficulty")
    args = p.parse_args()

    if args.source == "programmatic":
        probs = gen_problems_programmatic(args.n, args.kinds.split(","))
    else:
        if args.source == "mock":
            qs, src = propose_mock(args.n), "mock"
        else:
            qs, src = propose_via_model(args.n, args.topic, args.difficulty, args.model), \
                      f"teacher:{args.model}"
        probs = [Problem(id=f"gen{i}", question=q, gold=None, source=src)
                 for i, q in enumerate(qs)]
    save_problems(probs, args.out)


if __name__ == "__main__":
    main()
