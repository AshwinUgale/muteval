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
    assert "cases:" in out and "1" in out


def test_run_requires_something(capsys):
    code = main(["run", "--dry-run"])
    assert code == 2


def test_load_cases_tolerates_utf8_bom(tmp_path):
    # PowerShell's `Out-File -Encoding utf8` writes a BOM; the loader must cope.
    p = tmp_path / "cases.jsonl"
    p.write_bytes(b'\xef\xbb\xbf{"question":"q1"}\n')
    cases = _load_cases(str(p))
    assert cases == [{"question": "q1"}]


def test_cli_context_builds_system_mode(tmp_path):
    import argparse
    from muteval.cli import _config_from_flags

    cases = tmp_path / "c.jsonl"
    cases.write_text('{"question":"q"}\n', encoding="utf-8")
    args = argparse.Namespace(
        prompt="Answer from context.", prompt_file=None, cases=str(cases),
        context=["doc A", "doc B"], context_file=None, mutate_model=False, model="gpt-4o-mini",
        check=["contains:x"], judge=None, threshold=0.7, runs_per_mutant=1, scope_include=None, scope_exclude=None,
    )
    cfg = _config_from_flags(args)
    assert cfg.system.context == ("doc A", "doc B")
    assert cfg._system_mode is True


def test_cli_no_context_is_prompt_mode(tmp_path):
    import argparse
    from muteval.cli import _config_from_flags

    cases = tmp_path / "c.jsonl"
    cases.write_text('{"question":"q"}\n', encoding="utf-8")
    args = argparse.Namespace(
        prompt="Answer.", prompt_file=None, cases=str(cases),
        context=None, context_file=None, mutate_model=False, model="gpt-4o-mini",
        check=["contains:x"], judge=None, threshold=0.7, runs_per_mutant=1, scope_include=None, scope_exclude=None,
    )
    cfg = _config_from_flags(args)
    assert cfg._system_mode is False
    assert cfg.system.context is None


def test_cli_mutate_model_builds_system_with_model(tmp_path):
    import argparse
    from muteval.cli import _config_from_flags

    cases = tmp_path / "c.jsonl"
    cases.write_text('{"question":"q"}\n', encoding="utf-8")
    args = argparse.Namespace(
        prompt="Answer.", prompt_file=None, cases=str(cases),
        context=None, context_file=None, mutate_model=True, model="gpt-4o",
        check=["contains:x"], judge=None, threshold=0.7, runs_per_mutant=1, scope_include=None, scope_exclude=None,
    )
    cfg = _config_from_flags(args)
    assert cfg._system_mode is True
    assert cfg.system.model == "gpt-4o"


def test_fail_on_severity_high_gates_when_high_survivors_exist():
    from muteval.cli import main

    # The offline support_bot example has unguarded high-severity survivors.
    code = main([
        "run", "--config", "examples/support_bot/muteval_config.py",
        "--no-color", "--fail-on-severity", "high",
    ])
    assert code == 1


def test_no_severity_gate_passes():
    from muteval.cli import main

    code = main([
        "run", "--config", "examples/support_bot/muteval_config.py", "--no-color",
    ])
    assert code == 0


# --- validity gate: invalid/empty runs must fail closed (exit 2) -------------

_BASELINE_FAIL_CONFIG = '''
from muteval import MutEvalConfig

config = MutEvalConfig(
    prompt="You are a bot.\\n- You must cite the order ID.\\n- Do not lie.",
    cases=[{"order_id": "X1"}],
    run=lambda p, c: "order X1",
    evals=[lambda o, c: False],  # never passes -> baseline FAILS
)
'''

_NO_MUTANTS_CONFIG = '''
from muteval import MutEvalConfig

config = MutEvalConfig(
    prompt="short",  # too short to mutate -> no mutants
    cases=[{"x": 1}],
    run=lambda p, c: "ok",
    evals=[lambda o, c: True],
)
'''


def _write_config(tmp_path, body, name="cfg.py"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_invalid_baseline_exits_2_and_writes_no_badge(tmp_path, capsys):
    cfg = _write_config(tmp_path, _BASELINE_FAIL_CONFIG)
    badge = tmp_path / "badge.json"
    code = main(["run", "--config", cfg, "--no-color", "--badge", str(badge),
                 "--fail-under", "0"])
    assert code == 2  # fail closed BEFORE the (satisfiable) --fail-under 0 gate
    assert not badge.exists()  # never emit a green badge from an invalid run
    err = capsys.readouterr().err
    assert "INVALID" in err


def test_no_mutants_exits_2_by_default(tmp_path):
    cfg = _write_config(tmp_path, _NO_MUTANTS_CONFIG)
    code = main(["run", "--config", cfg, "--no-color"])
    assert code == 2


def test_no_mutants_allow_empty_exits_0(tmp_path):
    cfg = _write_config(tmp_path, _NO_MUTANTS_CONFIG)
    code = main(["run", "--config", cfg, "--no-color", "--allow-empty"])
    assert code == 0


def test_invalid_run_still_writes_json_with_status(tmp_path):
    cfg = _write_config(tmp_path, _BASELINE_FAIL_CONFIG)
    out = tmp_path / "out.json"
    code = main(["run", "--config", cfg, "--no-color", "--json", str(out)])
    assert code == 2
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "baseline_failed"
    assert data["score"] is None


_PARTIAL_ERROR_CONFIG = '''
from muteval import MutEvalConfig

def run(prompt, case):
    if "KEEPLINE stable." not in prompt:
        raise RuntimeError("boom")
    return "ok"

config = MutEvalConfig(
    prompt="- Cite the order ID.\\n- KEEPLINE stable.\\n- Be polite.",
    cases=[{"x": 1}],
    run=run,
    evals=[lambda o, c: True],
)
'''


def test_partial_errors_fail_closed_and_no_badge(tmp_path):
    cfg = _write_config(tmp_path, _PARTIAL_ERROR_CONFIG)
    badge = tmp_path / "b.json"
    out = tmp_path / "o.json"
    code = main(["run", "--config", cfg, "--no-color", "--fail-under", "0",
                 "--badge", str(badge), "--json", str(out)])
    assert code == 2               # fails BEFORE the satisfiable --fail-under 0
    assert not badge.exists()      # no green badge from a partly-errored run
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "partial_errors"
    assert data["error_rate"] > 0.0


def test_allow_mutant_errors_makes_partial_run_pass(tmp_path):
    cfg = _write_config(tmp_path, _PARTIAL_ERROR_CONFIG)
    code = main(["run", "--config", cfg, "--no-color", "--allow-mutant-errors"])
    assert code == 0


def test_dry_run_applies_scope_like_the_real_run(tmp_path):
    # scope_exclude drops every prompt line -> a real run makes 0 mutants, and
    # --dry-run must report the SAME 0 (not ignore scope).
    body = (
        "from muteval import MutEvalConfig\n"
        "config = MutEvalConfig(\n"
        "    prompt='- one.\\n- two.\\n- three.',\n"
        "    cases=[{'x': 1}], run=lambda p, c: 'ok',\n"
        "    evals=[lambda o, c: True], scope_exclude='.*',\n"
        ")\n"
    )
    cfg = _write_config(tmp_path, body, name="scoped.py")
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["run", "--config", cfg, "--no-color", "--dry-run"])
    assert code == 0
    assert "mutants that would run: 0" in buf.getvalue()


def test_init_rag_template_writes_and_runs(tmp_path):
    from muteval.config import load_config
    from muteval.runner import run_mutation_testing

    dest = tmp_path / "rag.py"
    assert main(["init", "--template", "rag", "--path", str(dest)]) == 0
    body = dest.read_text(encoding="utf-8")
    assert "System(" in body and "context=CONTEXT" in body   # System-mode scaffold

    # the scaffold runs keyless and produces a VALID mutation run
    cfg = load_config(str(dest))
    result = run_mutation_testing(cfg)
    assert result.status == "valid"
    assert result.baseline_passed is True


def test_cli_output_survives_a_cp1252_stdout(monkeypatch):
    # Reproduces the Windows cp1252 console cross-platform: the report has box/
    # arrow glyphs that cp1252 can't encode. main() must reconfigure stdout to
    # UTF-8 and not raise UnicodeEncodeError (regression: Windows CI self-check).
    import io
    import sys

    buf = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")
    monkeypatch.setattr(sys, "stdout", buf)
    code = main([
        "run", "--config", "examples/support_bot/muteval_config.py", "--no-color",
    ])
    assert code == 0
