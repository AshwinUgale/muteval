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


# --- LLM candidate generation (offline, injected chat) -----------------------

def test_parse_specs_builds_checks_from_vocabulary():
    from muteval.autofix import parse_specs

    raw = '[{"type": "not_contains", "value": "Sure, refund"}, {"type": "contains", "value": "cannot"}]'
    evals = parse_specs(raw)
    assert len(evals) == 2
    # they behave like the real checks
    assert bool(evals[0]("I cannot promise a refund.", {})) is True   # not_contains passes
    assert bool(evals[0]("Sure, refund now!", {})) is False


def test_parse_specs_tolerates_fences_and_drops_unknown():
    from muteval.autofix import parse_specs

    raw = '```json\n[{"type": "contains", "value": "x"}, {"type": "os_system", "value": "rm -rf /"}]\n```'
    evals = parse_specs(raw)
    assert len(evals) == 1  # the bogus/unsafe type is dropped, no code executed


def test_parse_specs_handles_garbage():
    from muteval.autofix import parse_specs

    assert parse_specs("not json at all") == []
    assert parse_specs('{"not": "a list"}') == []


def test_autofix_end_to_end_with_injected_llm():
    from muteval.autofix import autofix

    # A fake LLM that proposes a good check + a useless one.
    def fake_chat(prompt, model):
        assert "regression" in prompt.lower()
        return '[{"type": "not_contains", "value": "Sure, refund"}, {"type": "contains", "value": "xyz-never"}]'

    verified = autofix(_config(), _a_survivor(), chat=fake_chat)
    # Only the not_contains("Sure, refund") check is verified: it passes on the
    # baseline ("I cannot promise a refund") and fails on the mutant.
    names = [v.name for v in verified]
    assert "suggested_not_contains" in names
    assert "suggested_contains" not in names  # 'xyz-never' fails the baseline too
