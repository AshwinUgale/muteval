"""--json / --badge serialization for CI + the eval-coverage badge.

v0.4 adds: a pinned top-level key-set + schema_version (drift guard) and secret
redaction (no API key may leak into emitted JSON/logs).
"""

from muteval import MutEvalConfig, run_mutation_testing
from muteval.report import RESULT_SCHEMA_VERSION, _redact, badge_dict, result_to_dict


def _run():
    cfg = MutEvalConfig(
        prompt="- You must cite the order ID.\n- Do not promise refunds.",
        cases=[{"order_id": "X1"}],
        run=lambda p, c: ("order X1 " + ("no refund" if "do not promise refunds"
                          in p.lower() else "refund ok")),
        evals=[lambda o, c: c["order_id"] in o],  # only checks citation
    )
    return run_mutation_testing(cfg)


def test_result_to_dict_shape():
    d = result_to_dict(_run())
    for key in ("effective_score", "score_ci", "killed", "evaluated",
                "high_severity_survivors", "survivors"):
        assert key in d
    if d["survivors"]:
        s = d["survivors"][0]
        assert set(s) == {
            "id", "operator", "description", "severity", "fix",
            "baseline_output", "mutant_output",
        }


def test_badge_dict_is_shields_endpoint():
    b = badge_dict(_run())
    assert b["schemaVersion"] == 1
    assert b["label"] == "eval coverage"
    assert b["message"].endswith("%")
    assert b["color"] in ("brightgreen", "yellow", "red")


# The exact top-level key-set the JSON contract promises. Any change is a
# schema_version bump; this catches accidental drift.
EXPECTED_KEYS = {
    "schema_version", "status", "baseline_passed", "baseline_error", "score",
    "effective_score", "score_ci", "effective_score_ci", "killed", "evaluated",
    "total", "errored", "error_rate", "inert", "high_severity_survivors",
    "survivors",
}


def test_result_dict_keyset_is_pinned():
    d = result_to_dict(_run())
    assert set(d.keys()) == EXPECTED_KEYS
    assert d["schema_version"] == RESULT_SCHEMA_VERSION


def test_redaction_scrubs_secret_patterns():
    leaky = {
        "baseline_error": "boom: OPENAI_API_KEY=sk-abc123DEF456ghi789 rejected",
        "survivors": [{"description": "prompt leaked Authorization: Bearer gsk_livesecret999xyz"}],
        "nested": ["AIzaSyA1234567890abcdefghij_KLMNOPqrst"],
    }
    blob = str(_redact(leaky))
    assert "sk-abc123DEF456ghi789" not in blob
    assert "gsk_livesecret999xyz" not in blob
    assert "AIzaSyA1234567890abcdefghij_KLMNOPqrst" not in blob
    assert "[REDACTED]" in blob


def test_real_run_json_has_no_secret_patterns():
    import re

    blob = str(result_to_dict(_run()))
    assert not re.search(r"sk-[A-Za-z0-9]{8,}|gsk_[A-Za-z0-9]{8,}", blob)
