"""--json / --badge serialization for CI + the eval-coverage badge."""

from muteval import MutEvalConfig, run_mutation_testing
from muteval.report import badge_dict, result_to_dict


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
        assert set(s) == {"operator", "description", "severity", "fix"}


def test_badge_dict_is_shields_endpoint():
    b = badge_dict(_run())
    assert b["schemaVersion"] == 1
    assert b["label"] == "eval coverage"
    assert b["message"].endswith("%")
    assert b["color"] in ("brightgreen", "yellow", "red")
