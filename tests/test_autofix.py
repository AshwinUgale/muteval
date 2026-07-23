"""v0.6: the verify loop — a suggested fix must PROVABLY close the gap."""

from __future__ import annotations

from muteval import EvalOutcome, MutEvalConfig, System, run_mutation_testing
from muteval.autofix import suggest_and_verify, verify_fix

SYSTEM = System(
    prompt=(
        "Be a support agent.\n"
        "- Do not promise refunds; a manager must approve.\n"
        "- Cite the source.\n"
        "- Greet politely."
    ),
    model="gpt-4o-mini",
)
CASES = [{"q": "can I get a refund?"}]


def _run(system, case):
    refuse = "do not promise refunds" in system.prompt.lower()
    return "I cannot promise a refund." if refuse else "Sure, refund coming right up!"


def _weak(output, case):
    # WEAK on purpose: only checks the word 'refund' appears (passes either way).
    return EvalOutcome(passed="refund" in output.lower(), name="weak")


def _config():
    return MutEvalConfig(
        system=SYSTEM, cases=CASES, run=_run, evals=[_weak], eval_names=["weak"]
    )


def _a_survivor():
    survivors = run_mutation_testing(_config()).real_survivors
    assert survivors, "expected at least one real survivor"
    return survivors[0]


# --- candidate fixes ---------------------------------------------------------

def good_fix(output, case):
    # passes on baseline ("I cannot promise a refund"), fails on mutant.
    return "sure, refund" not in output.lower()


def breaks_baseline(output, case):
    # fails on the baseline output -> not a valid fix.
    return "sure" in output.lower()


def noop(output, case):
    return True  # never fails -> catches nothing


def test_only_the_good_candidate_is_verified():
    verified = suggest_and_verify(_config(), _a_survivor(), [good_fix, breaks_baseline, noop])
    names = {v.name for v in verified}
    assert "good_fix" in names
    assert "breaks_baseline" not in names  # would redden the baseline
    assert "noop" not in names  # catches nothing


def test_verify_fix_reports_both_conditions():
    mutant_sys = _a_survivor().mutant.system

    kills, keeps = verify_fix(_config(), mutant_sys, good_fix)
    assert kills and keeps

    _, keeps_bad = verify_fix(_config(), mutant_sys, breaks_baseline)
    assert keeps_bad is False

    kills_noop, keeps_noop = verify_fix(_config(), mutant_sys, noop)
    assert kills_noop is False and keeps_noop is True


def test_verified_fix_carries_the_catching_case():
    verified = suggest_and_verify(_config(), _a_survivor(), [good_fix])
    assert len(verified) == 1
    assert verified[0].killed_case in CASES
