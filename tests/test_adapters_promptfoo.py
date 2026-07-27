"""promptfoo adapter: translate a promptfooconfig into a muteval target."""

import pytest

from muteval.adapters.promptfoo import (
    _assertion_check,
    _render,
    _resolve_model,
    _type_eval,
    config_from_promptfoo_dict,
)

SAMPLE = {
    "prompts": ["Answer about {{topic}}. Always cite the source."],
    "defaultTest": {"assert": [{"type": "contains", "value": "source"}]},
    "tests": [
        {"vars": {"topic": "ports"}, "assert": [{"type": "contains", "value": "8080"}]},
        {
            "vars": {"topic": "keys"},
            "assert": [{"type": "not-contains", "value": "password"}],
        },
    ],
}


def test_render_substitutes_vars():
    assert _render("hi {{ name }}", {"name": "bob"}) == "hi bob"


def test_assertion_translation():
    assert _assertion_check({"type": "contains", "value": "a"})("cat", {}) is True
    assert _assertion_check({"type": "not-contains", "value": "z"})("cat", {}) is True
    assert _assertion_check({"type": "regex", "value": "c.t"})("cat", {}) is True
    assert bool(_assertion_check({"type": "is-json"})('{"ok": true}', {})) is True
    assert bool(_assertion_check({"type": "is-json"})("not json", {})) is False
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
    case = {
        "_asserts": [
            {"type": "contains", "value": "8080"},
            {"type": "contains", "value": "source"},
        ]
    }
    ev = _type_eval("contains")
    assert ev("port 8080, source: server.md", case) is True
    assert ev("no port, source: server.md", case) is False


def test_skipped_types_warn_and_build(capsys):
    data = {
        "prompts": ["p {{x}}"],
        "tests": [
            {
                "vars": {"x": "1"},
                "assert": [
                    {"type": "contains", "value": "a"},
                    {"type": "javascript"},  # unsupported -> skipped, not graded
                ],
            }
        ],
    }
    cfg = config_from_promptfoo_dict(data, run=lambda p, c: "a")
    err = capsys.readouterr().err
    assert "skipped" in err and "javascript" in err
    assert cfg.eval_names == ["promptfoo:contains"]


def test_is_json_assertion_is_graded():
    data = {
        "prompts": ["Return JSON about {{x}}"],
        "tests": [{"vars": {"x": "ports"}, "assert": [{"type": "is-json"}]}],
    }
    cfg = config_from_promptfoo_dict(data, run=lambda p, c: '{"ok": true}')

    assert cfg.eval_names == ["promptfoo:is-json"]
    assert cfg.evals[0]('{"ok": true}', cfg.cases[0]) is True
    assert cfg.evals[0]("not json", cfg.cases[0]) is False


def test_mixed_unsupported_case_is_dropped_not_fatal(capsys):
    # A suite with one gradeable case + one all-unsupported case should BUILD,
    # dropping (never passing) the ungradeable case and warning about it.
    data = {
        "prompts": ["p {{x}}"],
        "tests": [
            {"vars": {"x": "1"}, "assert": [{"type": "contains", "value": "a"}]},
            {"vars": {"x": "2"}, "assert": [{"type": "javascript"}]},  # all-unsupported
        ],
    }
    cfg = config_from_promptfoo_dict(data, run=lambda p, c: "a")
    err = capsys.readouterr().err
    assert "dropped 1 case" in err
    assert len(cfg.cases) == 1  # the ungradeable case is gone, not vacuously passed
    assert cfg.eval_names == ["promptfoo:contains"]


def test_suite_with_nothing_translatable_fails_closed():
    # If NOTHING in the whole suite is gradeable, fail closed (no vacuous score).
    bad = {"prompts": ["p"], "tests": [{"assert": [{"type": "javascript"}]}]}
    with pytest.raises(ValueError, match="translatable"):
        config_from_promptfoo_dict(bad, run=lambda p, c: "x")


def test_resolve_model_reads_provider(capsys):
    # The model under test is read from the promptfoo `providers:` block.
    assert _resolve_model({"providers": ["openai:gpt-4o"]}, None, None) == "gpt-4o"
    assert _resolve_model({"providers": ["openai:chat:gpt-4o"]}, None, None) == "gpt-4o"
    assert _resolve_model({"providers": [{"id": "openai:gpt-4.1"}]}, None, None) == "gpt-4.1"
    capsys.readouterr()  # drain the "model under test ..." notes


def test_resolve_model_explicit_override_wins():
    data = {"providers": ["openai:gpt-4o"]}
    assert _resolve_model(data, "gpt-4o-mini", None) == "gpt-4o-mini"


def test_resolve_model_no_providers_keeps_default():
    assert _resolve_model({"prompts": ["p"]}, None, None) == "gpt-4o-mini"


def test_new_deterministic_assert_types():
    assert _assertion_check({"type": "contains-any", "value": ["a", "z"]})("cat", {}) is True
    assert _assertion_check({"type": "contains-any", "value": "x, z"})("cat", {}) is False
    assert _assertion_check({"type": "contains-all", "value": ["c", "t"]})("cat", {}) is True
    assert _assertion_check({"type": "contains-all", "value": ["c", "z"]})("cat", {}) is False
    assert _assertion_check({"type": "icontains-any", "value": ["A"]})("cat", {}) is True
    assert _assertion_check({"type": "not-equals", "value": "dog"})("cat", {}) is True
    assert _assertion_check({"type": "starts-with", "value": "ca"})("cat", {}) is True
    assert _assertion_check({"type": "starts-with", "value": "at"})("cat", {}) is False


def test_external_csv_tests_are_loaded(tmp_path):
    (tmp_path / "cases.csv").write_text(
        "topic,__expected\nports,contains: 8080\nkeys,not-contains: password\n",
        encoding="utf-8",
    )
    data = {
        "prompts": ["Answer about {{topic}}"],
        "tests": "file://cases.csv",  # a bare string ref (used to crash with AttributeError)
    }
    cfg = config_from_promptfoo_dict(data, run=lambda p, c: "x", base_dir=tmp_path)
    assert len(cfg.cases) == 2
    assert cfg.cases[0]["topic"] == "ports"
    # __expected translated into an assertion
    assert any(a["value"] == "8080" for a in cfg.cases[0]["_asserts"])
    assert "promptfoo:contains" in cfg.eval_names


def test_external_jsonl_tests_are_loaded(tmp_path):
    (tmp_path / "t.jsonl").write_text(
        '{"vars": {"x": "1"}, "assert": [{"type": "contains", "value": "a"}]}\n'
        '{"vars": {"x": "2"}, "assert": [{"type": "contains", "value": "b"}]}\n',
        encoding="utf-8",
    )
    data = {"prompts": ["p {{x}}"], "tests": ["file://t.jsonl"]}
    cfg = config_from_promptfoo_dict(data, run=lambda p, c: "ab", base_dir=tmp_path)
    assert len(cfg.cases) == 2


def test_external_tests_from_code_or_url_error_clearly(tmp_path):
    # A code-function test generator can't be executed — clear message, not a traceback.
    with pytest.raises(ValueError, match="can't execute"):
        config_from_promptfoo_dict(
            {"prompts": ["p"], "tests": "file://gen.py:make_tests"},
            run=lambda p, c: "x",
            base_dir=tmp_path,
        )
    # A remote URL can't be fetched — clear message.
    with pytest.raises(ValueError, match="fetch/load"):
        config_from_promptfoo_dict(
            {"prompts": ["p"], "tests": "https://example.com/cases.csv"},
            run=lambda p, c: "x",
            base_dir=tmp_path,
        )


def test_external_defaulttest_is_loaded(tmp_path):
    (tmp_path / "dt.yaml").write_text(
        "assert:\n  - type: contains\n    value: source\n", encoding="utf-8"
    )
    data = {
        "prompts": ["Answer about {{topic}}"],
        "defaultTest": "file://dt.yaml",  # external defaultTest (used to crash)
        "tests": [{"vars": {"topic": "ports"}}],
    }
    cfg = config_from_promptfoo_dict(data, run=lambda p, c: "source", base_dir=tmp_path)
    assert any(a["value"] == "source" for a in cfg.cases[0]["_asserts"])
    assert cfg.eval_names == ["promptfoo:contains"]


def test_glob_and_code_prompt_error_clearly(tmp_path):
    with pytest.raises(ValueError, match="glob"):
        config_from_promptfoo_dict(
            {"prompts": ["file://prompt*.txt"], "tests": [{"assert": [{"type": "contains", "value": "a"}]}]},
            run=lambda p, c: "a",
            base_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="code function"):
        config_from_promptfoo_dict(
            {"prompts": ["file://prompt.py:build"], "tests": [{"assert": [{"type": "contains", "value": "a"}]}]},
            run=lambda p, c: "a",
            base_dir=tmp_path,
        )


def test_resolve_model_non_openai_provider_warns_and_falls_back(capsys):
    data = {"providers": ["anthropic:messages:claude-3-5-sonnet"]}
    assert _resolve_model(data, None, None) == "gpt-4o-mini"
    err = capsys.readouterr().err
    assert "isn't an OpenAI-compatible endpoint" in err
    # …but with a compatible base_url, trust the provider's model name.
    assert (
        _resolve_model(data, None, "https://api.example.com/v1")
        == "claude-3-5-sonnet"
    )
