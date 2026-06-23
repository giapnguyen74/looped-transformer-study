"""
gen_math_transcripts.py — a verified math reasoning *transcript factory* for
training looped / recurrent-depth transformers (OpenMythos style).

The pipeline implements the STaR / rejection-sampling recipe, specialized for a
looped model where "the loop IS internal chain-of-thought" (see ../summary/):

    problems (checkable answers)
        |
        v
    [1] SAMPLE     teacher generates many candidate solutions  (high temp)
        |
        v
    [2] VERIFY     keep only traces whose \\boxed{} matches gold (SymPy-equivalence,
        |          not string match -> 1/2 == 0.5 == 0.50)
        v
    [3] FILTER     dedupe, length bounds, drop degenerate / lucky-guess traces;
        |          self-consistency (majority answer) flags mislabeled golds
        v
    [4] CURATE     keep up to K *diverse* correct traces per problem; tag reasoning
        |          DEPTH (#steps) -> bucket -> suggested loop count  (looped-specific)
        v
    transcripts.jsonl   (ready for explicit-CoT SFT and Coconut-style latent compression)

WHY MATH: the final answer is *verifiable*, so the reward signal is exact. That is
the entire reason reasoning-data pipelines anchor on math/code.

WHY DEPTH TAGGING: your `loop_scaling_lab.py testtime` experiment shows training
depth sets the test-time ceiling. So we keep the hard, long-chain tail and tag each
transcript with a depth bucket, so the trainer can pair hard problems with more loops
(dynamic / Poisson depth).

The teacher is PLUGGABLE:
  * MockTeacher       — built-in, offline, deterministic. Generates real (and some
                        deliberately wrong) arithmetic CoT so the whole pipeline runs
                        with zero dependencies beyond sympy. Used by `demo`.
  * OpenRouterTeacher — any model served by OpenRouter (https://openrouter.ai), e.g.
                        deepseek/deepseek-r1, qwen/qwen3-235b, openai/gpt-4o-mini.
                        This is where you plug a strong reasoner.

Dependencies: sympy (required for verification). `requests` only for OpenRouterTeacher.

Examples:
  python gen_math_transcripts.py demo                       # offline, end-to-end
  export OPENROUTER_API_KEY=sk-or-...
  python gen_math_transcripts.py generate \\
        --problems problems.jsonl --out transcripts.jsonl \\
        --teacher openrouter --model deepseek/deepseek-r1 --samples 16 --temp 0.8
"""

import argparse
import json
import os
import re
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import sympy
from sympy.parsing.sympy_parser import parse_expr


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Problem:
    id: str
    question: str
    gold: str            # ground-truth final answer (string; verified symbolically)
    source: str = "unknown"


@dataclass
class Transcript:
    id: str
    question: str
    gold: str
    solution: str        # full chain-of-thought ending in \boxed{answer}
    steps: List[str]     # solution split into reasoning steps (for latent curriculum)
    answer: str          # extracted final answer
    n_steps: int
    depth_bucket: str    # easy / medium / hard  -> suggests loop budget
    suggested_loops: int
    source: str


# --------------------------------------------------------------------------- #
# [2] VERIFY — extract the answer and check SYMBOLIC equivalence, not strings.
# --------------------------------------------------------------------------- #
_BOXED = re.compile(r"\\boxed\{")


def extract_boxed(text: str) -> Optional[str]:
    """Return the content of the LAST \\boxed{...} (brace-balanced)."""
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    i = starts[-1]
    depth, out = 1, []
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip() if depth == 0 else None


def _clean(s: str) -> str:
    s = s.strip().strip("$").replace(r"\!", "").replace(r"\,", "").replace(" ", "")
    s = s.replace(",", "")                       # 1,000 -> 1000
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\dfrac|\\tfrac", r"\\frac", s)
    m = re.fullmatch(r"\\frac\{(.+?)\}\{(.+?)\}", s)   # \frac{a}{b} -> (a)/(b)
    if m:
        s = f"({m.group(1)})/({m.group(2)})"
    s = s.replace(r"\%", "/100").replace("%", "/100")   # \% before bare %
    return s


def answers_equivalent(pred: str, gold: str) -> bool:
    """True if pred and gold are the same number/expression (SymPy)."""
    if pred is None or gold is None:
        return False
    p, g = _clean(pred), _clean(gold)
    if p == g:
        return True
    try:
        diff = sympy.simplify(parse_expr(p) - parse_expr(g))
        return diff == 0
    except Exception:
        return p == g


# --------------------------------------------------------------------------- #
# [3]/[4] step splitting + degeneracy + depth tagging
# --------------------------------------------------------------------------- #
def split_steps(solution: str) -> List[str]:
    """Split a CoT into reasoning steps. Coconut-style latent compression deletes
    the first k of these and replaces them with k latent loop iterations."""
    body = solution.split(r"\boxed")[0]
    parts = re.split(r"\n+|(?<=[.])\s+(?=[A-Z0-9])", body.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


def is_degenerate(solution: str, steps: List[str]) -> bool:
    """Reject lucky-guess / pathological traces: no reasoning, or heavy repetition."""
    if len(steps) < 1 or len(solution) < 8:
        return True
    toks = solution.split()
    if toks and len(set(toks)) / len(toks) < 0.25:    # >75% repeated tokens
        return True
    return False


def depth_of(n_steps: int) -> (str, int):
    """Map reasoning depth -> bucket -> suggested loop count for dynamic-depth training."""
    if n_steps <= 2:
        return "easy", 2
    if n_steps <= 5:
        return "medium", 4
    return "hard", 8


# --------------------------------------------------------------------------- #
# [1] SAMPLE — teacher interface + two implementations
# --------------------------------------------------------------------------- #
class Teacher:
    def sample(self, problem: Problem, n: int, temperature: float) -> List[str]:
        raise NotImplementedError


class MockTeacher(Teacher):
    """Offline teacher for `demo`. Understands 'a OP b' questions, writes a short CoT,
    and (with prob p_wrong) emits a wrong answer so the verifier/filters have work."""
    def __init__(self, p_wrong=0.35, seed=0):
        self.p_wrong = p_wrong
        self.rng = random.Random(seed)

    def sample(self, problem, n, temperature):
        out = []
        m = re.search(r"(-?\d+)\s*([\+\-\*x×])\s*(-?\d+)", problem.question)
        for _ in range(n):
            if not m:
                out.append(r"I am not sure. \boxed{0}")
                continue
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            true = a + b if op == "+" else a - b if op == "-" else a * b
            val = true
            if self.rng.random() < self.p_wrong:        # plausible mistake
                val = true + self.rng.choice([-2, -1, 1, 2, 10])
            opname = {"+": "add", "-": "subtract"}.get(op, "multiply")
            sol = (f"We need to {opname} {a} and {b}.\n"
                   f"Set up the operation: {a} {op} {b}.\n"
                   f"Computing carefully gives {val}.\n"
                   f"So the answer is \\boxed{{{val}}}.")
            out.append(sol)
        return out


class APITeacher(Teacher):
    """OpenAI-compatible chat endpoint. Plug your strong reasoner here.
    Env: OPENAI_API_KEY (required), OPENAI_BASE_URL (default api.openai.com)."""
    SYS = ("Solve the math problem. Think step by step, one step per line, "
           "then give the final answer as \\boxed{...}.")

    def __init__(self, model: str):
        self.model = model

    def sample(self, problem, n, temperature):
        import requests
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        key = os.environ["OPENAI_API_KEY"]
        r = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": self.model, "n": n, "temperature": temperature,
                  "messages": [{"role": "system", "content": self.SYS},
                               {"role": "user", "content": problem.question}]},
            timeout=120,
        )
        r.raise_for_status()
        return [c["message"]["content"] for c in r.json()["choices"]]


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
def process_problem(problem, teacher, samples, temperature, keep_per_problem,
                    min_chars, max_chars, stats):
    """Run [1]->[4] for ONE problem; return up to keep_per_problem Transcripts."""
    cands = teacher.sample(problem, samples, temperature)
    stats["sampled"] += len(cands)

    verified, answers = [], []
    for sol in cands:
        ans = extract_boxed(sol)
        answers.append(_clean(ans) if ans else None)
        if not answers_equivalent(ans, problem.gold):
            continue                                  # [2] wrong / unparsable -> drop
        steps = split_steps(sol)
        if is_degenerate(sol, steps):                 # [3] degenerate -> drop
            stats["degenerate"] += 1
            continue
        if not (min_chars <= len(sol) <= max_chars):  # [3] length bounds
            stats["len_filtered"] += 1
            continue
        verified.append((sol, ans, steps))
    stats["verified"] += len(verified)

    # [3] self-consistency: most common answer among ALL samples. If it disagrees
    # with gold, the gold label is suspect -> flag the whole problem.
    nonnull = [a for a in answers if a is not None]
    if nonnull:
        maj, _ = Counter(nonnull).most_common(1)[0]
        if not answers_equivalent(maj, problem.gold):
            stats["gold_suspect"] += 1

    if not verified:
        stats["no_correct"] += 1
        return []

    # [4] diversity: dedupe identical traces, then keep the K most distinct by length.
    seen, uniq = set(), []
    for sol, ans, steps in verified:
        h = hashlib.md5(re.sub(r"\s+", " ", sol).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        uniq.append((sol, ans, steps))
    uniq.sort(key=lambda t: len(t[2]))                # spread of reasoning depths
    if len(uniq) > keep_per_problem:
        idx = [round(i * (len(uniq) - 1) / (keep_per_problem - 1))
               for i in range(keep_per_problem)] if keep_per_problem > 1 else [0]
        uniq = [uniq[i] for i in sorted(set(idx))]

    transcripts = []
    for j, (sol, ans, steps) in enumerate(uniq):
        bucket, loops = depth_of(len(steps))
        transcripts.append(Transcript(
            id=f"{problem.id}#{j}", question=problem.question, gold=problem.gold,
            solution=sol, steps=steps, answer=ans, n_steps=len(steps),
            depth_bucket=bucket, suggested_loops=loops, source=problem.source))
    return transcripts


def run_pipeline(problems, teacher, args):
    stats = defaultdict(int)
    out_path = args.out
    n_written, buckets = 0, Counter()
    with open(out_path, "w") as f:
        for prob in problems:
            for t in process_problem(prob, teacher, args.samples, args.temp,
                                     args.keep, args.min_chars, args.max_chars, stats):
                f.write(json.dumps(asdict(t)) + "\n")
                n_written += 1
                buckets[t.depth_bucket] += 1
    stats["written"] = n_written

    print(f"\n[done] wrote {n_written} transcripts -> {out_path}")
    print("  pipeline funnel:")
    print(f"    sampled        {stats['sampled']}")
    print(f"    verified       {stats['verified']}  (passed SymPy answer check)")
    print(f"    dropped degen  {stats['degenerate']}")
    print(f"    dropped length {stats['len_filtered']}")
    print(f"    problems w/ 0 correct  {stats['no_correct']}")
    print(f"    gold labels flagged suspect (self-consistency)  {stats['gold_suspect']}")
    print(f"  depth buckets written: {dict(buckets)}")
    keep_rate = (stats['verified'] / stats['sampled'] * 100) if stats['sampled'] else 0
    print(f"  sample->verified keep rate: {keep_rate:.1f}%")
    return stats


# --------------------------------------------------------------------------- #
# Problem loading
# --------------------------------------------------------------------------- #
def load_problems(path) -> List[Problem]:
    """JSONL with fields: id?, question, gold (a.k.a. answer), source?."""
    probs = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            probs.append(Problem(
                id=str(d.get("id", i)),
                question=d["question"],
                gold=str(d.get("gold", d.get("answer"))),
                source=d.get("source", os.path.basename(path))))
    return probs


def synthetic_problems(n=40, seed=0) -> List[Problem]:
    """Built-in checkable problem set for `demo` (no download needed)."""
    rng = random.Random(seed)
    probs = []
    for i in range(n):
        op = rng.choice(["+", "-", "*"])
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        gold = a + b if op == "+" else a - b if op == "-" else a * b
        q = f"What is {a} {op} {b}?"
        probs.append(Problem(id=f"syn{i}", question=q, gold=str(gold), source="synthetic"))
    return probs


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Verified math reasoning transcript factory")
    p.add_argument("command", choices=["demo", "generate"])
    p.add_argument("--problems", help="input problems JSONL (generate)")
    p.add_argument("--out", default="transcripts.jsonl")
    p.add_argument("--teacher", choices=["mock", "api"], default="mock")
    p.add_argument("--model", default="gpt-4o-mini", help="model name for --teacher api")
    p.add_argument("--samples", type=int, default=8, help="candidates sampled per problem")
    p.add_argument("--temp", type=float, default=0.8, help="sampling temperature")
    p.add_argument("--keep", type=int, default=2, help="diverse correct traces kept per problem")
    p.add_argument("--min-chars", dest="min_chars", type=int, default=20)
    p.add_argument("--max-chars", dest="max_chars", type=int, default=8000)
    args = p.parse_args()

    if args.command == "demo":
        args.out = args.out if args.out != "transcripts.jsonl" else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", "transcripts_demo.jsonl")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        print("=== DEMO: synthetic problems + MockTeacher (offline) ===")
        problems = synthetic_problems(n=40)
        teacher = MockTeacher(p_wrong=0.35)
        run_pipeline(problems, teacher, args)
        # show a couple of finished transcripts
        print("\n  sample transcripts:")
        with open(args.out) as f:
            for line in list(f)[:2]:
                d = json.loads(line)
                print(f"    [{d['depth_bucket']}/T~{d['suggested_loops']}] "
                      f"{d['question']} -> {d['answer']}  ({d['n_steps']} steps)")
        return

    # generate
    if not args.problems:
        p.error("--problems is required for `generate`")
    problems = load_problems(args.problems)
    print(f"loaded {len(problems)} problems from {args.problems}")
    teacher = MockTeacher() if args.teacher == "mock" else APITeacher(args.model)
    run_pipeline(problems, teacher, args)


if __name__ == "__main__":
    main()
