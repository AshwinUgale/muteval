"""Does the mutation score actually measure eval-suite quality?

This is muteval's central claim: a *better* eval suite should *kill* more
mutants. To test it without the noise (and cost) of a live LLM, we use a
deterministic "model" whose behavior faithfully reflects the prompt and the
retrieved context — so a mutation that removes or inverts a rule genuinely
changes the output, exactly as a real degraded system would.

We then grade that same system with four eval suites of increasing coverage and
watch the mutation score climb. If the score tracks suite quality and the
survivors are concrete, actionable gaps, the metric does what we say it does.

Run:  python validation/eval_quality_experiment/run_experiment.py

This needs no API key — it's a controlled experiment, by design. For results on
*real* LLM-judge metrics, see validation/deepeval_rag_qdrant/ (deepeval) and
validation/ragas_rag/ (ragas).
"""

from __future__ import annotations

from muteval import MutEvalConfig, System, run_mutation_testing

# --- The system under test ---------------------------------------------------

SYSTEM_PROMPT = """You are a support assistant for an online store.
- You must always cite the order ID in your reply.
- Do not promise refunds; refunds require manager approval.
- You must never reveal another customer's data.
- Always reply in a polite, professional tone.
"""

CONTEXT = (
    "The server listens on port 8080 by default. (source: config/server.md)",
    "Refunds are processed within 5 business days once approved. (source: billing.md)",
)

SYSTEM = System(prompt=SYSTEM_PROMPT, context=CONTEXT)

_NEG_CUES = (" not", "never", "n't", "cannot", "avoid", "no ")


def _line_with(prompt: str, keyword: str) -> str:
    for line in prompt.splitlines():
        if keyword in line.lower():
            return line.lower()
    return ""


def _has_neg(line: str) -> bool:
    return any(cue in line for cue in _NEG_CUES)


def _positive_active(prompt: str, keyword: str) -> bool:
    """A positive instruction is followed iff its line is present and un-negated."""
    line = _line_with(prompt, keyword)
    return bool(line) and not _has_neg(line)


def _prohibition_active(prompt: str, keyword: str) -> bool:
    """A prohibition holds iff its line is present and phrased as a prohibition."""
    line = _line_with(prompt, keyword)
    return bool(line) and _has_neg(line)


def run(system: System, case: dict) -> str:
    """A deterministic 'model' that obeys exactly the rules still in the prompt
    and grounds answers only in the context still retrieved."""
    prompt = system.prompt
    parts = []

    if _positive_active(prompt, "polite"):
        parts.append("Hi there! Thanks for reaching out.")

    if _positive_active(prompt, "cite"):
        parts.append(f"Regarding order {case['order_id']}:")
    else:
        parts.append("Regarding your order:")

    if _prohibition_active(prompt, "refund"):
        parts.append("I'm not able to approve a refund; our team will review it.")
    else:
        parts.append("Good news — I've issued your refund.")

    if not _prohibition_active(prompt, "reveal"):
        parts.append("For reference, another customer (Jane Smith) had the same issue.")

    ctx = " ".join(system.context or [])
    if "8080" in ctx:
        parts.append("The server listens on port 8080.")
    else:
        parts.append("I don't have the exact port detail.")

    return " ".join(parts)


# --- The eval checks (independent of the prompt) -----------------------------


def cites_order_id(output, case):
    return case["order_id"] in output


def no_refund_promise(output, case):
    return "issued your refund" not in output


def no_data_leak(output, case):
    return "another customer" not in output


def is_polite(output, case):
    return "Thanks for reaching out" in output


def grounded_in_context(output, case):
    return "port 8080" in output


# --- Four suites of increasing coverage --------------------------------------

SUITES = {
    "S0 smoke (nonempty only)": (
        [lambda o, c: bool(o.strip())],
        ["nonempty"],
    ),
    "S1 basic (cite)": (
        [cites_order_id],
        ["cites_order_id"],
    ),
    "S2 good (cite+refund+leak)": (
        [cites_order_id, no_refund_promise, no_data_leak],
        ["cites_order_id", "no_refund_promise", "no_data_leak"],
    ),
    "S3 strong (+polite+grounded)": (
        [cites_order_id, no_refund_promise, no_data_leak, is_polite, grounded_in_context],
        [
            "cites_order_id",
            "no_refund_promise",
            "no_data_leak",
            "is_polite",
            "grounded_in_context",
        ],
    ),
}

CASE = {"order_id": "X123", "question": "what port does the server use?"}


def _run_all():
    rows = []
    for label, (evals, names) in SUITES.items():
        cfg = MutEvalConfig(
            system=SYSTEM, cases=[CASE], run=run, evals=evals, eval_names=names
        )
        rows.append((label, run_mutation_testing(cfg)))
    return rows


def main() -> None:
    print("muteval - eval-quality experiment")
    print("=" * 70)
    print("Claim: a better eval suite kills more mutants. We grade ONE system")
    print("with suites of increasing coverage and watch the EFFECTIVE score")
    print("(which excludes inert/equivalent mutants) climb from 0 to 100.")
    print("=" * 70)
    print(f"{'suite':<34}{'effective':>10}{'raw':>7}{'inert':>7}{'killed':>9}")
    print("-" * 70)

    rows = _run_all()
    for label, r in rows:
        print(
            f"{label:<34}{r.effective_score * 100:>8.0f}% {r.score * 100:>5.0f}% "
            f"{len(r.inert_survivors):>6} {r.killed:>5}/{r.evaluated:<3}"
        )
    print("-" * 70)

    scores = [r.effective_score for _, r in rows]
    monotonic = all(b >= a for a, b in zip(scores, scores[1:]))
    endpoints = abs(scores[0]) < 1e-9 and abs(scores[-1] - 1.0) < 1e-9
    print("\nEffective score: " + " -> ".join(f"{s * 100:.0f}%" for s in scores))
    print(f"monotonic increase: {monotonic}   empty=0% & complete=100%: {endpoints}")
    print(
        "\nPASS: muteval's score tracks eval-suite quality end to end.\n"
        if (monotonic and endpoints)
        else "\nFAIL: the score did NOT track suite quality - investigate.\n"
    )

    for label, r in rows:
        ops = sorted({o.mutant.operator for o in r.real_survivors})
        print(label)
        print(
            f"   real gaps: {len(r.real_survivors)}  "
            f"(+{len(r.inert_survivors)} inert, excluded)"
        )
        if ops:
            print(f"   uncaught operators: {', '.join(ops)}")
        print()


if __name__ == "__main__":
    main()
