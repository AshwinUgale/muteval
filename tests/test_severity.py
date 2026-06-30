"""Tests for severity ranking of mutants."""

from types import SimpleNamespace

from muteval import MutEvalConfig, run_mutation_testing
from muteval.severity import HIGH, LOW, MEDIUM, severity_of, severity_rank


def _m(operator, description=""):
    # severity_of only reads .operator and .description.
    return SimpleNamespace(operator=operator, description=description)


def test_operator_base_severity():
    assert severity_of(_m("flip_negation", "inverted a -> b")) == HIGH
    assert severity_of(_m("weaken_modals", "weakened could")) == MEDIUM
    assert severity_of(_m("remove_emphasis", "removed bold markup")) == LOW


def test_critical_content_escalates_one_level():
    # MEDIUM base, but the change touches a refund / "do not" rule -> HIGH.
    assert severity_of(_m("weaken_modals", 'weakened "do not" near refund')) == HIGH
    # LOW base touching 'password' -> MEDIUM.
    assert severity_of(_m("remove_emphasis", "removed emphasis near password")) == MEDIUM


def test_high_never_exceeds_high():
    assert severity_of(_m("flip_negation", "inverted never reveal customer data")) == HIGH


def test_non_critical_stays_at_base():
    assert severity_of(_m("weaken_modals", "weakened ideally use json")) == MEDIUM


def test_rank_orders_high_first():
    assert severity_rank(HIGH) < severity_rank(MEDIUM) < severity_rank(LOW)


def test_high_severity_survivors_surface_real_gaps():
    # Eval only checks citation, so refund-rule mutations survive — and those
    # are HIGH severity (they touch a refund / "do not" rule).
    def run(p, c):
        out = [f"order {c['order_id']}"]
        out.append("no refund" if "do not promise refunds" in p.lower() else "refund ok")
        return " ".join(out)

    cfg = MutEvalConfig(
        prompt="- You must cite the order ID.\n"
        "- Do not promise refunds.\n"
        "- Use a friendly tone.",
        cases=[{"order_id": "X1"}],
        run=run,
        evals=[lambda o, c: c["order_id"] in o],
    )
    result = run_mutation_testing(cfg)
    assert all(o.severity in (HIGH, MEDIUM, LOW) for o in result.real_survivors)
    assert len(result.high_severity_survivors) >= 1
