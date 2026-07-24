"""promptfoo adapter: translate a promptfooconfig into a muteval target."""

import pytest

from muteval.adapters.promptfoo import (
    _assertion_check,
    _render,
    _type_eval,
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
    # ONE eval per translatable assertion TYPE (so the survivor report is per-check)
    assert cfg.eval_names == ["promptfoo:contains", "promptfoo:not-contains"]


def test_type_eval_honors_all_asserts_of_that_type():
    case = {"_asserts": [
        {"type": "contains", "value": "8080"},
        {"type": "contains", "value": "source"},
    ]}
    ev = _type_eval("contains")
    assert ev("port 8080, source: server.md", case) is True
    assert ev("no port, source: server.md", case) is False


def test_skipped_types_warn_and_build(capsys):
    data = {
        "prompts": ["p {{x}}"],
        "tests": [{"vars": {"x": "1"}, "assert": [
            {"type": "contains", "value": "a"},
            {"type": "is-json"},  # unsupported -> skipped, not graded
        ]}],
    }
    cfg = config_from_promptfoo_dict(data, run=lambda p, c: "a")
    err = capsys.readouterr().err
    assert "skipped" in err and "is-json" in err
    assert cfg.eval_names == ["promptfoo:contains"]


def test_all_unsupported_case_raises():
    # A case whose assertions are ALL unsupported must FAIL, not pass vacuously.
    bad = {"prompts": ["p"], "tests": [{"assert": [{"type": "is-json"}]}]}
    with pytest.raises(ValueError, match="unsupported"):
        config_from_promptfoo_dict(bad, run=lambda p, c: "x")
