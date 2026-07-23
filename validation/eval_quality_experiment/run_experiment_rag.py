"""Third domain (RAG / context-grounding) for the eval-quality experiment.

Same claim, a retrieval-heavy system: a better eval suite kills more mutants.
This domain leans on CONTEXT mutation — the deterministic 'model' answers the
port from the retrieved docs, so dropping/corrupting/clearing the context makes
the answer wrong, exactly as a real RAG regression would. A grounding-aware suite
catches it; a smoke suite doesn't. No API key.

Run:  python validation/eval_quality_experiment/run_experiment_rag.py
"""

from __future__ import annotations

from muteval import MutEvalConfig, System, run_mutation_testing

SYSTEM_PROMPT = """You are a documentation assistant.
- You must always cite the source file.
- Do not answer from outside the provided context.
- Always answer in one concise sentence.
"""

CONTEXT = (
    "The API server listens on port 8080 by default. (source: config.md)",
    "Set the PORT environment variable to override the default. (source: config.md)",
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
    if _pos(p, "concise"):
        parts.append("In short:")
    # Grounding: the port comes from the retrieved context. Lose/corrupt the
    # context and the model answers from stale memory (wrong).
    parts.append("the default port is 8080." if "8080" in ctx else "the default port is 3000.")
    if _pos(p, "cite"):
        parts.append("(source: config.md)")
    if _prohib(p, "outside"):
        parts.append("I only used the provided docs.")
    else:
        parts.append("I also recalled it from elsewhere.")
    return " ".join(parts)


def grounded(o, c):
    return "8080" in o          # catches context drop/corrupt/clear/swap


def cites_source(o, c):
    return "(source: config.md)" in o   # catches the cite rule being dropped


def uses_only_docs(o, c):
    return "provided docs" in o          # catches the 'no outside' rule being lost


def is_concise(o, c):
    return "In short:" in o              # catches the concise rule being dropped


SUITES = {
    "S0 smoke (nonempty only)": ([lambda o, c: bool(o.strip())], ["nonempty"]),
    "S1 basic (grounded)": ([grounded], ["grounded"]),
    "S2 good (+cite+onlydocs)": (
        [grounded, cites_source, uses_only_docs],
        ["grounded", "cites_source", "uses_only_docs"],
    ),
    "S3 strong (+concise)": (
        [grounded, cites_source, uses_only_docs, is_concise],
        ["grounded", "cites_source", "uses_only_docs", "is_concise"],
    ),
}

CASE = {"question": "what port does the server use by default?"}


def main() -> None:
    print("muteval - eval-quality experiment (domain 3: RAG grounding)")
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
