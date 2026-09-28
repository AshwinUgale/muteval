"""CLI & report correctness — from a full audit.

Invalid input must exit 2 with a message (a traceback's exit 1 reads as a failed
gate), every output must tell the same story (JUnit, `results`, `show`, HTML),
and the manifest must actually identify the setup it came from.
"""

import json
import xml.etree.ElementTree as ET

import pytest

from muteval import MutEvalConfig, checks, run_mutation_testing
from muteval.cli import _build_parser, _config_from_flags, _load_run_config, main
from muteval.report import (
    format_report,
    format_report_html,
    format_report_junit,
    result_to_dict,
    run_manifest,
)

PROMPT = "- You must cite the order ID.\n- Never promise refunds.\n- Always be polite."


def _cfg(**kw):
    return MutEvalConfig(
        prompt=PROMPT,
        cases=[{"x": 1}],
        run=lambda p, c: p,  # output tracks the prompt: every mutant changes it
        evals=[lambda o, c: "cite" in o],
        **kw,
    )


def _write(tmp_path, name, text, encoding="utf-8"):
    p = tmp_path / name
    p.write_text(text, encoding=encoding)
    return str(p)


_CONFIG = f"""
from muteval import MutEvalConfig
config = MutEvalConfig(prompt={PROMPT!r}, cases=[{{"x": 1}}],
    run=lambda p, c: p, evals=[lambda o, c: "cite" in o])
"""


# --- invalid input exits 2, with a message ---------------------------------------------


@pytest.mark.parametrize(
    "content, encoding, ok",
    [
        ('["abc123def456"]', "utf-8-sig", True),  # PowerShell Out-File writes a BOM
        ('[{"signature": "abc123def456", "reason": "by design"}]', "utf-8", True),
        ('"abc123def456"', "utf-8", False),  # a bare string (became a set of chars)
        ("not json", "utf-8", False),
    ],
)
def test_accept_file_inputs(tmp_path, capsys, content, encoding, ok):
    cfg = _write(tmp_path, "cfg.py", _CONFIG)
    acc = _write(tmp_path, "acc.json", content, encoding=encoding)
    code = main(["run", "--config", cfg, "--no-color", "--accept-file", acc])
    if ok:
        assert code in (0, 1)
    else:
        assert code == 2 and "--accept-file" in capsys.readouterr().err


def test_missing_accept_file_exits_2(tmp_path, capsys):
    cfg = _write(tmp_path, "cfg.py", _CONFIG)
    code = main(["run", "--config", cfg, "--accept-file", str(tmp_path / "nope.json")])
    assert code == 2 and "can't read" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--json", "--badge"])
def test_unwritable_output_path_exits_2(tmp_path, capsys, flag):
    cfg = _write(tmp_path, "cfg.py", _CONFIG)
    bad = str(tmp_path / "no" / "such" / "dir" / "out.json")
    assert main(["run", "--config", cfg, "--no-color", flag, bad]) == 2
    assert "could not write" in capsys.readouterr().err


def test_config_plus_zero_config_flags_is_an_error(tmp_path, capsys):
    cfg = _write(tmp_path, "cfg.py", _CONFIG)
    assert main(["run", "--config", cfg, "--check", "contains:x"]) == 2
    assert "can't be combined" in capsys.readouterr().err


def test_explicit_model_reaches_promptfoo_programmatically(tmp_path):
    y = _write(
        tmp_path,
        "pf.yaml",
        "prompts: ['Answer {{q}}']\nproviders: [openai:gpt-4o]\n"
        "tests:\n  - vars: {q: x}\n    assert: [{type: contains, value: x}]\n",
    )
    args = _build_parser().parse_args(["run", "--promptfoo", y, "--model", "my-model"])
    assert _load_run_config(args).model_under_test == "my-model"
    args = _build_parser().parse_args(["run", "--promptfoo", y])
    assert _load_run_config(args).model_under_test == "gpt-4o"  # the suite's own


# --- manifest ----------------------------------------------------------------------------


def test_manifest_records_model_sample_and_setup(tmp_path):
    cases = _write(tmp_path, "c.jsonl", '{"input": "hi"}\n')
    args = _build_parser().parse_args(
        [
            "run",
            "--prompt",
            "Be polite.",
            "--cases",
            cases,
            "--check",
            "contains:x",
            "--model",
            "gpt-4o",
        ]
    )
    cfg = _config_from_flags(args)
    r = run_mutation_testing(_cfg(), operators=["flip_negation"])
    m = run_manifest(r, cfg, seed=7, sample=3, max_mutants=10)
    assert m["run"]["model"] == "gpt-4o"  # was null in zero-config mode
    assert m["run"]["sample"] == 3 and m["run"]["max_mutants"] == 10
    a = run_manifest(r, _cfg())["run"]
    b = run_manifest(r, _cfg(runs_per_mutant=3))["run"]
    c = run_manifest(
        r,
        MutEvalConfig(
            prompt=PROMPT,
            cases=[{"x": 1}],
            run=lambda p, c: p,
            evals=[lambda o, c: "refund" in o],
        ),
    )["run"]
    assert a["system_fingerprint"] == c["system_fingerprint"]  # same system...
    assert (
        len({a["config_fingerprint"], b["config_fingerprint"], c["config_fingerprint"]})
        == 3
    )


# --- every output tells the same story ------------------------------------------------------


def _accepted_run():
    r = run_mutation_testing(_cfg())
    first = sorted(r.real_survivors, key=lambda o: o.mutant.signature)[0]
    r.accepted = frozenset({first.mutant.signature})
    return r, first


def test_junit_failures_are_only_new_survivors():
    r, _ = _accepted_run()
    root = ET.fromstring(format_report_junit(r))
    assert int(root.attrib["failures"]) == len(r.new_survivors)
    skipped = [
        t.find("skipped").attrib["message"] for t in root if t.find("skipped") is not None
    ]
    assert any("accepted" in m for m in skipped)


def test_junit_survives_control_characters():
    cfg = MutEvalConfig(
        prompt="- You must cite the order ID.\x1b\n- Never guess.\x0b",
        cases=[{"x": 1}],
        run=lambda p, c: p,
        evals=[lambda o, c: True],
    )
    ET.fromstring(format_report_junit(run_mutation_testing(cfg)))  # parses


def test_terminal_ids_match_json_and_show(tmp_path):
    r, accepted = _accepted_run()
    d = result_to_dict(r)
    ids = {s["signature"]: s["id"] for s in d["survivors"]}
    out = format_report(r, use_color=False)
    for o in r.new_survivors:
        assert f"#{ids[o.mutant.signature]} " in out  # same id as `muteval show`
    assert f"#{ids[accepted.mutant.signature]} " not in out  # hidden, not renumbered


def test_results_after_an_invalid_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    bad = _write(
        tmp_path,
        "bad.py",
        "from muteval import MutEvalConfig\n"
        f"config = MutEvalConfig(prompt={PROMPT!r}, cases=[1], run=lambda p, c: 'x', "
        "evals=[lambda o, c: False])\n",
    )
    assert main(["run", "--config", bad, "--no-color"]) == 2
    capsys.readouterr()
    assert main(["results", "--no-color"]) == 2
    out = capsys.readouterr().out
    assert "INVALID" in out and "caught everything" not in out


def test_html_escapes_untrusted_fields_and_marks_accepted():
    data = {
        "status": "valid",
        "survivors": [
            {
                "id": "<b>1</b>",
                "severity": 'high"><script>x</script>',
                "accepted": True,
                "operator": "op",
                "description": "d",
            }
        ],
    }
    page = format_report_html(data)
    assert "<script>x</script>" not in page and "<b>1</b>" not in page
    assert "accepted" in page


# --- checks --------------------------------------------------------------------------------


def test_on_final_grades_plain_text_and_null():
    ev = checks.on_final(checks.contains("ok"))
    assert ev("ok, plain text", {}).passed is True  # was a JSON error -> "errored"
    assert ev('{"final": null, "trace": []}', {}).passed is False  # was AttributeError


def test_tracelint_fails_closed_on_non_json():
    pytest.importorskip("tracelint")
    ev = checks.tracelint(registry={})
    out = ev("not a trace", {})
    assert out.passed is False and "not a trace" in out.detail


def test_is_json_requires_structure_by_default():
    ev = checks.is_json()
    assert ev('{"a": 1}', {}).passed and ev("[1]", {}).passed
    assert not ev("null", {}).passed and not ev("42", {}).passed
    assert checks.is_json(allow_scalar=True)("42", {}).passed


def test_cites_source_ignores_product_names():
    ev = checks.cites_source()
    assert ev("see [doc-1]", {}).passed and ev("per ORD-9", {}).passed
    assert ev("as in [3]", {}).passed
    assert not ev("Python3 and iPhone15", {}).passed


def test_missing_key_message_does_not_blame_the_evals(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="needed to call 'm'"):
        checks._openai_chat_stdlib("hi", "m")


def test_list_probes_has_plain_descriptions(capsys):
    main(["list", "probes", "--no-color"])
    out = capsys.readouterr().out
    assert "Registry entry point" not in out
    assert "flip rate" in out and "kappa" in out
    json.dumps(out)  # (plain text)
