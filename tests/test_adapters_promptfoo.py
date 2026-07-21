"""promptfoo adapter: translate a promptfooconfig into a muteval target."""

from muteval.adapters.promptfoo import (
    _assertion_check,
    _render,
    _suite_eval,
    config_from_promptfoo_dict,
)

SAMPLE = {
    "prompts": ["Answer about {{topic}}. Always cite the source."],
    "defaultTest": {"assert": [{"type": "contains", "value": "source"}]},
    "tests": [
        {"vars": {"topic": "ports"}, "assert": [{"type": "contains", "value": "8080"}]},
        {"vars": {"topic": "keys"}, "assert": [{"type": "not-contains", "value": "password"}]},
    ],
}


def test_render_substitutes_vars():
    assert _render("hi {{ name }}", {"name": "bob"}) == "hi bob"


def test_assertion_translation():
    assert _assertion_check({"type": "contains", "value": "a"})("cat", {}) is True
    assert _assertion_check({"type": "not-contains", "value": "z"})("cat", {}) is True
    assert _assertion_check({"type": "regex", "value": "c.t"})("cat", {}) is True
    assert _assertion_check({"type": "javascript", "value": "..."}) is None  # unsupported


def test_config_built_with_merged_default_asserts():
    cfg = config_from_promptfoo_dict(SAMPLE, run=lambda p, c: "x")
    assert "cite the source" in cfg.prompt
    assert len(cfg.cases) == 2
    # defaultTest assert is merged into every case
    assert any(a["value"] == "source" for a in cfg.cases[0]["_asserts"])
    assert any(a["value"] == "8080" for a in cfg.cases[0]["_asserts"])


def test_suite_eval_honors_all_asserts():
    case = {"_asserts": [
        {"type": "contains", "value": "8080"},
        {"type": "contains", "value": "source"},
    ]}
    assert _suite_eval("port 8080, source: server.md", case) is True
    assert _suite_eval("no port, source: server.md", case) is False
