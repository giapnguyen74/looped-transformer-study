"""
gen_math_transcripts.py — a verified math reasoning *transcript factory* for
training looped / recurrent-depth transformers (OpenMythos style).

This is STAGE B of the data pipeline. Problems come from gen_math_problems.py (STAGE A)
or any JSONL with {question, gold?}; this script turns them into verified solution traces
via the STaR / rejection-sampling recipe:

    problems (checkable answers)
       -> [1] SAMPLE   teacher generates many candidate solutions  (high temp)
       -> [2] VERIFY   keep traces whose \\boxed{} matches gold (SymPy-equivalence,
                       not strings: 1/2 == 0.5 == 50%)
       -> [3] FILTER   dedupe · length bounds · drop degenerate/lucky-guess · self-consistency
       -> [4] CURATE   keep K diverse correct traces · tag reasoning DEPTH -> suggested loops
       -> transcripts.jsonl   (ready for explicit-CoT SFT + Coconut-style latent compression)

If a problem has NO gold (teacher-proposed problems from STAGE A), gold is derived by
SELF-CONSISTENCY: sample k solutions, take the majority answer, drop problems whose
agreement < --consensus.

WHY MATH: the final answer is *verifiable*, so the keep/reject signal is exact — the
reason reasoning-data pipelines anchor on math/code.

WHY DEPTH TAGGING: `loop_scaling_lab.py testtime` shows training depth sets the test-time
ceiling, so we keep the hard, long-chain tail and tag each transcript with a depth bucket
so the trainer can pair hard problems with more loops (dynamic / Poisson depth).

The teacher (solver) is PLUGGABLE:
  * MockTeacher       — built-in, offline, deterministic arithmetic CoT (+ some wrong
                        answers). Zero deps beyond sympy. Used by `demo`.
  * OpenRouterTeacher — any model on OpenRouter (https://openrouter.ai): openai/gpt-5.5,
                        deepseek/deepseek-r1, qwen/qwen3-235b, ...

Dependencies: sympy (required). `requests` only for OpenRouterTeacher.

Examples:
  python gen_math_transcripts.py demo                         # offline, end-to-end

  # solve problems made by gen_math_problems.py
  export OPENROUTER_API_KEY=sk-or-...
  python gen_math_transcripts.py generate --problems problems.jsonl \\
        --teacher openrouter --model openai/gpt-5.5 --samples 16 --temp 0.8 \\
        --keep 2 --consensus 0.5 --out transcripts.jsonl
"""

import argparse
import json
import os
import re
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
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
    gold: Optional[str] = None   # None => derive by self-consistency
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
    gold_source: str     # "given" | "self_consistency"
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


def depth_of(n_steps: int):
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
    def sample(self, problem: "Problem", n: int, temperature: float) -> List[str]:
        raise NotImplementedError


class MockTeacher(Teacher):
    """Offline solver for `demo`. Understands 'a OP b' questions, writes a short CoT, and
    (with prob p_wrong) emits a wrong answer so the verifier/filters have work to do."""
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


class OpenRouterTeacher(Teacher):
    """OpenRouter chat endpoint (https://openrouter.ai) — OpenAI-compatible. Plug any
    model OpenRouter serves: openai/gpt-5.5, deepseek/deepseek-r1, qwen/qwen3-235b.

    Env: OPENROUTER_API_KEY (required); OPENROUTER_BASE_URL (default openrouter.ai/api/v1);
         OPENROUTER_REFERER / OPENROUTER_TITLE (optional, for OpenRouter app ranking).

    NOTE: `n>1` is not supported uniformly across OpenRouter's upstream providers, so we
    issue n independent requests at the given temperature — the robust way to get diverse
    samples for rejection sampling regardless of which model you point at."""
    SOLVE_SYS = ("Solve the math problem. Think step by step, one step per line, "
                 "then give the final answer as \\boxed{...}.")

    def __init__(self, model: str):
        self.model = model

    def _chat(self, system: str, user: str, temperature: float) -> str:
        import requests
        base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        key = os.environ["OPENROUTER_API_KEY"]
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if os.environ.get("OPENROUTER_REFERER"):          # optional ranking metadata
            headers["HTTP-Referer"] = os.environ["OPENROUTER_REFERER"]
        if os.environ.get("OPENROUTER_TITLE"):
            headers["X-Title"] = os.environ["OPENROUTER_TITLE"]
        r = requests.post(
            f"{base}/chat/completions",
            headers=headers,
            json={"model": self.model, "temperature": temperature,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def sample(self, problem, n, temperature):
        return [self._chat(self.SOLVE_SYS, problem.question, temperature) for _ in range(n)]


# --------------------------------------------------------------------------- #
# The transcript pipeline
# --------------------------------------------------------------------------- #
def process_problem(problem, teacher, args, stats):
    """Run [1]->[4] for ONE problem; return up to args.keep Transcripts.
    If problem.gold is None, derive it by self-consistency (majority answer)."""
    cands = teacher.sample(problem, args.samples, args.temp)
    stats["sampled"] += len(cands)

    raw = [(sol, extract_boxed(sol)) for sol in cands]
    answers = [_clean(a) if a else None for _, a in raw]
    nonnull = [a for a in answers if a is not None]

    gold, gold_source = problem.gold, "given"
    if gold in (None, "", "None"):
        # ---- self-consistency labeling for teacher-generated (unlabeled) problems ----
        gold_source = "self_consistency"
        if not nonnull:
            stats["no_answer"] += 1
            return []
        maj, cnt = Counter(nonnull).most_common(1)[0]
        if cnt / len(cands) < args.consensus:
            stats["low_consensus"] += 1            # too ambiguous/hard -> discard problem
            return []
        gold = maj
        stats["pseudo_labeled"] += 1
    else:
        # provided gold: flag if the model's majority disagrees (suspect label)
        if nonnull:
            maj, _ = Counter(nonnull).most_common(1)[0]
            if not answers_equivalent(maj, gold):
                stats["gold_suspect"] += 1

    verified = []
    for (sol, ansraw), ans in zip(raw, answers):
        if not answers_equivalent(ansraw, gold):
            continue                                  # [2] wrong / unparsable -> drop
        steps = split_steps(sol)
        if is_degenerate(sol, steps):                 # [3] degenerate -> drop
            stats["degenerate"] += 1
            continue
        if not (args.min_chars <= len(sol) <= args.max_chars):
            stats["len_filtered"] += 1
            continue
        verified.append((sol, ans, steps))
    stats["verified"] += len(verified)

    if not verified:
        stats["no_correct"] += 1
        return []

    # [4] diversity: dedupe identical traces, then keep the K most distinct by depth.
    seen, uniq = set(), []
    for sol, ans, steps in verified:
        h = hashlib.md5(re.sub(r"\s+", " ", sol).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        uniq.append((sol, ans, steps))
    uniq.sort(key=lambda t: len(t[2]))
    if len(uniq) > args.keep:
        idx = [round(i * (len(uniq) - 1) / (args.keep - 1))
               for i in range(args.keep)] if args.keep > 1 else [0]
        uniq = [uniq[i] for i in sorted(set(idx))]

    transcripts = []
    for j, (sol, ans, steps) in enumerate(uniq):
        bucket, loops = depth_of(len(steps))
        transcripts.append(Transcript(
            id=f"{problem.id}#{j}", question=problem.question, gold=str(gold),
            solution=sol, steps=steps, answer=ans, n_steps=len(steps),
            depth_bucket=bucket, suggested_loops=loops,
            gold_source=gold_source, source=problem.source))
    return transcripts


def run_pipeline(problems, teacher, args):
    stats = defaultdict(int)
    n_written, buckets = 0, Counter()
    with open(args.out, "w") as f:
        for prob in problems:
            for t in process_problem(prob, teacher, args, stats):
                f.write(json.dumps(asdict(t)) + "\n")
                n_written += 1
                buckets[t.depth_bucket] += 1
    stats["written"] = n_written

    print(f"\n[done] wrote {n_written} transcripts -> {args.out}")
    print("  pipeline funnel:")
    print(f"    sampled        {stats['sampled']}")
    print(f"    verified       {stats['verified']}  (passed SymPy answer check)")
    print(f"    dropped degen  {stats['degenerate']}")
    print(f"    dropped length {stats['len_filtered']}")
    print(f"    problems w/ 0 correct          {stats['no_correct']}")
    if stats["pseudo_labeled"] or stats["low_consensus"] or stats["no_answer"]:
        print(f"    gold via self-consistency      {stats['pseudo_labeled']}")
        print(f"    dropped low-consensus          {stats['low_consensus']}")
        print(f"    dropped no-answer              {stats['no_answer']}")
    print(f"    given gold flagged suspect     {stats['gold_suspect']}")
    print(f"  depth buckets written: {dict(buckets)}")
    keep_rate = (stats['verified'] / stats['sampled'] * 100) if stats['sampled'] else 0
    print(f"  sample->verified keep rate: {keep_rate:.1f}%")
    return stats


# --------------------------------------------------------------------------- #
# Problem loading
# --------------------------------------------------------------------------- #
def load_problems(path) -> List[Problem]:
    """JSONL with fields: id?, question, gold (a.k.a. answer)?, source?.
    Missing gold is allowed -> derived by self-consistency at solve time."""
    probs = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            g = d.get("gold", d.get("answer"))
            probs.append(Problem(
                id=str(d.get("id", i)),
                question=d["question"],
                gold=None if g is None else str(g),
                source=d.get("source", os.path.basename(path))))
    return probs


def _demo_problems(n=40, seed=0) -> List[Problem]:
    """Tiny labeled arithmetic set so `demo` runs fully offline (MockTeacher can solve it).
    For real problem generation use gen_math_problems.py."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        op = rng.choice(["+", "-", "*"])
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        gold = a + b if op == "+" else a - b if op == "-" else a * b
        out.append(Problem(id=f"d{i}", question=f"What is {a} {op} {b}?",
                           gold=str(gold), source="demo"))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description="Stage B — verified math reasoning transcripts")
    p.add_argument("command", choices=["demo", "generate"])
    p.add_argument("--problems", help="input problems JSONL (generate)")
    p.add_argument("--out", default="transcripts.jsonl")
    p.add_argument("--teacher", choices=["mock", "openrouter"], default="mock")
    p.add_argument("--model", default="openai/gpt-5.5",
                   help="OpenRouter model id for --teacher openrouter")
    p.add_argument("--samples", type=int, default=8, help="candidates sampled per problem")
    p.add_argument("--temp", type=float, default=0.8, help="sampling temperature")
    p.add_argument("--keep", type=int, default=2, help="diverse correct traces kept per problem")
    p.add_argument("--consensus", type=float, default=0.5,
                   help="min majority fraction to accept a self-consistency gold label")
    p.add_argument("--min-chars", dest="min_chars", type=int, default=20)
    p.add_argument("--max-chars", dest="max_chars", type=int, default=8000)
    return p


def main():
    p = build_parser()
    args = p.parse_args()

    if args.command == "demo":
        if args.out == "transcripts.jsonl":
            args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "results", "transcripts_demo.jsonl")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        teacher = MockTeacher(p_wrong=0.35)

        print("=== DEMO A: problems with given gold + MockTeacher ===")
        run_pipeline(_demo_problems(40), teacher, args)

        print("\n=== DEMO B: UNLABELED problems, gold via self-consistency ===")
        unlabeled = [Problem(id=p_.id, question=p_.question, gold=None, source="unlabeled")
                     for p_ in _demo_problems(20, seed=1)]
        args.out = args.out.replace(".jsonl", "_selfconsistency.jsonl")
        run_pipeline(unlabeled, teacher, args)
        return

    # generate
    if not args.problems:
        p.error("--problems is required for `generate`")
    problems = load_problems(args.problems)
    print(f"loaded {len(problems)} problems from {args.problems}")
    teacher = MockTeacher() if args.teacher == "mock" else OpenRouterTeacher(args.model)
    run_pipeline(problems, teacher, args)


if __name__ == "__main__":
    main()
