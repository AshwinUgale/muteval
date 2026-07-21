"""Second domain for the eval-quality experiment: a code-review assistant.

Same claim, an unrelated system: a better eval suite kills more mutants. This is
the generalization check — if the 0->100 relationship only held for the support
bot it might be a fluke; here it holds for a different domain too. No API key.

Run:  python validation/eval_quality_experiment/run_experiment_codereview.py
"""

from __future__ import annotations

from muteval import MutEvalConfig, System, run_mutation_testing

SYSTEM_PROMPT = """You are a code-review assistant.
- You must always flag SQL injection risks.
- Do not approve code that logs secrets.
- You must cite the file and line for each issue.
- Reply in a concise, professional summary.
"""

CONTEXT = (
    "Guideline SEC-1: user input must be parameterized. (source: sec.md)",
    "Guideline LOG-2: secrets must never be logged. (source: logging.md)",
)

SYSTEM = System(prompt=SYSTEM_PROMPT, context=CONTEXT)

_NEG = (" not", "never", "n't", "cannot", "avoid", "no ", "don't")


def _line(prompt: str, kw: str) -> str:
    for ln in prompt.splitlines():
        if kw in ln.lower():
            return ln.lower()
    return ""


def _pos(prompt: str, kw: str) -> bool:
    ln = _line(prompt, kw)
    return bool(ln) and not any(c in ln for c in _NEG)


def _prohib(prompt: str, kw: str) -> bool:
    ln = _line(prompt, kw)
    return bool(ln) and any(c in ln for c in _NEG)


def run(system: System, case: dict) -> str:
    p = system.prompt
    parts = []
    if _pos(p, "concise"):
        parts.append("Summary:")
    if _pos(p, "injection"):
        parts.append("Flagged: possible SQL injection risk.")
    if _prohib(p, "approve"):
        parts.append("I will not approve code that logs secrets.")
    else:
        parts.append("Looks good - approved.")
    if _pos(p, "cite"):
        parts.append("See app.py:42.")
    ctx = " ".join(system.context or [])
    if "SEC-1" in ctx:
        parts.append("Per guideline SEC-1, parameterize inputs.")
    else:
        parts.append("No specific guideline found.")
    return " ".join(parts)


def flags_injection(o, c):
    return "SQL injection" in o


def no_secret_approval(o, c):
    return "approved" not in o


def cites_location(o, c):
    return "app.py:42" in o


def is_concise(o, c):
    return "Summary:" in o


def grounded(o, c):
    return "SEC-1" in o


SUITES = {
    "S0 smoke (nonempty only)": ([lambda o, c: bool(o.strip())], ["nonempty"]),
    "S1 basic (injection)": ([flags_injection], ["flags_injection"]),
    "S2 good (+approve+cite)": (
        [flags_injection, no_secret_approval, cites_location],
        ["flags_injection", "no_secret_approval", "cites_location"],
    ),
    "S3 strong (+concise+grounded)": (
        [flags_injection, no_secret_approval, cites_location, is_concise, grounded],
        ["flags_injection", "no_secret_approval", "cites_location", "is_concise", "grounded"],
    ),
}

CASE = {"pr": "review this PR", "id": "PR-7"}


def main() -> None:
    print("muteval - eval-quality experiment (domain 2: code review)")
    print("=" * 68)
    print(f"{'suite':<34}{'effective':>10}{'raw':>7}{'inert':>7}{'killed':>9}")
    print("-" * 68)
    scores = []
    for label, (evals, names) in SUITES.items():
        cfg = MutEvalConfig(system=SYSTEM, cases=[CASE], run=run, evals=evals, eval_names=names)
        r = run_mutation_testing(cfg)
        scores.append(r.effective_score)
        print(f"{label:<34}{r.effective_score*100:>8.0f}% {r.score*100:>5.0f}% "
              f"{len(r.inert_survivors):>6} {r.killed:>5}/{r.evaluated:<3}")
    print("-" * 68)
    mono = all(b >= a for a, b in zip(scores, scores[1:]))
    ends = abs(scores[0]) < 1e-9 and abs(scores[-1] - 1.0) < 1e-9
    print("Effective score: " + " -> ".join(f"{s*100:.0f}%" for s in scores))
    print(f"monotonic: {mono}   empty=0% & complete=100%: {ends}")


if __name__ == "__main__":
    main()
