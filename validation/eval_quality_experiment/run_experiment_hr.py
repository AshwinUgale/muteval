"""Fourth domain (HR policy assistant) for the eval-quality experiment.

A different system again, to further rule out that the 0->100 relationship is a
fluke of one domain. Deterministic model: rules present in the prompt (and the
retrieved policy) shape the output; a more complete suite catches more of the
resulting regressions. No API key.

Run:  python validation/eval_quality_experiment/run_experiment_hr.py
"""

from __future__ import annotations

from muteval import MutEvalConfig, System, run_mutation_testing

SYSTEM_PROMPT = """You are an HR policy assistant.
- You must always cite the policy number.
- Do not disclose individual salary figures.
- You must escalate legal questions to a human.
- Always keep a respectful tone.
"""

CONTEXT = (
    "Policy HR-12: PTO accrues at 1.5 days per month. (source: handbook.md)",
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
    ctx = " ".join(system.context or [])
    parts = []
    if _pos(p, "respectful"):
        parts.append("Happy to help.")
    parts.append("PTO accrues at 1.5 days per month." if "HR-12" in ctx else "PTO details unavailable.")
    if _pos(p, "cite"):
        parts.append("(Policy HR-12)")
    parts.append("I can't share individual salary details." if _prohib(p, "salary") else "Alex earns $95,000.")
    parts.append("I'll route this to our legal team." if _pos(p, "escalate") else "In my opinion, you can sue.")
    return " ".join(parts)


def cites_policy(o, c):
    return "(Policy HR-12)" in o


def no_salary_leak(o, c):
    return "$95,000" not in o


def escalates_legal(o, c):
    return "legal team" in o


def is_respectful(o, c):
    return "Happy to help." in o


def grounded(o, c):
    return "1.5 days" in o


SUITES = {
    "S0 smoke (nonempty only)": ([lambda o, c: bool(o.strip())], ["nonempty"]),
    "S1 basic (cite)": ([cites_policy], ["cites_policy"]),
    "S2 good (+salary+legal)": (
        [cites_policy, no_salary_leak, escalates_legal],
        ["cites_policy", "no_salary_leak", "escalates_legal"],
    ),
    "S3 strong (+respect+grounded)": (
        [cites_policy, no_salary_leak, escalates_legal, is_respectful, grounded],
        ["cites_policy", "no_salary_leak", "escalates_legal", "is_respectful", "grounded"],
    ),
}

CASE = {"question": "how much PTO do I accrue?"}


def main() -> None:
    print("muteval - eval-quality experiment (domain 4: HR policy)")
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
