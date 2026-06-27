"""Tests for the zero-config CLI mode (no API calls)."""

import json

import pytest

from muteval.cli import _check_from_spec, _load_cases, main
from muteval.evals import EvalOutcome


def test_check_from_spec_builds_known_checks():
    contains = _check_from_spec("contains:8080", 0.7, "gpt-4o-mini")
    assert contains("port 8080", {}).passed is True
    assert contains("nope", {}).passed is False

    notc = _check_from_spec("not_contains:refund", 0.7, "gpt-4o-mini")
    assert notc("no money back", {}).passed is True
    assert notc("here is your refund", {}).passed is False

    cc = _check_from_spec("contains_case:order_id", 0.7, "gpt-4o-mini")
    assert cc("order A1 shipped", {"order_id": "A1"}).passed is True

    rx = _check_from_spec("regex:ORD-\\d+", 0.7, "gpt-4o-mini")
    assert rx("ref ORD-9", {}).passed is True

    isj = _check_from_spec("is_json", 0.7, "gpt-4o-mini")
    assert isj('{"a":1}', {}).passed is True


def test_check_from_spec_judge_is_lazy_no_api():
    # Building a judge check must not call the API (only invoking it would).
    ev = _check_from_spec("judge:is it polite", 0.7, "gpt-4o-mini")
    assert callable(ev)


def test_check_from_spec_unknown_raises():
    with pytest.raises(ValueError):
        _check_from_spec("bogus:x", 0.7, "gpt-4o-mini")


def test_load_cases_jsonl(tmp_path):
    p = tmp_path / "cases.jsonl"
    p.write_text('{"question":"q1"}\n\n{"question":"q2"}\n', encoding="utf-8")
    cases = _load_cases(str(p))
    assert [c["question"] for c in cases] == ["q1", "q2"]


def test_load_cases_json_list(tmp_path):
    p = tmp_path / "cases.json"
    p.write_text(json.dumps([{"question": "q1"}, {"question": "q2"}]), encoding="utf-8")
    cases = _load_cases(str(p))
    assert len(cases) == 2


def test_dry_run_builds_config_without_api(tmp_path, capsys):
    prompt = tmp_path / "system.txt"
    prompt.write_text("Answer using only the context. Cite the source.", encoding="utf-8")
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"question":"what port?","context":["port 8080"]}\n', encoding="utf-8")

    code = main(
        [
            "run",
            "--prompt-file", str(prompt),
            "--cases", str(cases),
            "--check", "contains:8080",
            "--dry-run",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "dry-run OK" in out
    assert "cases:  1" in out


def test_run_requires_something(capsys):
    code = main(["run", "--dry-run"])
    assert code == 2


def test_load_cases_tolerates_utf8_bom(tmp_path):
    # PowerShell's `Out-File -Encoding utf8` writes a BOM; the loader must cope.
    p = tmp_path / "cases.jsonl"
    p.write_bytes(b'\xef\xbb\xbf{"question":"q1"}\n')
    cases = _load_cases(str(p))
    assert cases == [{"question": "q1"}]
