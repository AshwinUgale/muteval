"""v0.5 P0-8: shareable HTML report (`muteval report --html`)."""

from __future__ import annotations

from muteval.report import format_report_html

SAMPLE = {
    "status": "valid",
    "score": 0.5,
    "effective_score": 0.45,
    "score_ci": [0.2, 0.8],
    "effective_score_ci": [0.18, 0.75],
    "killed": 5,
    "evaluated": 10,
    "total": 12,
    "inert": 2,
    "high_severity_survivors": 1,
    "survivors": [
        {
            "id": 0, "operator": "flip_negation", "severity": "high",
            "description": "inverted 'Do not' -> 'do'",
            "fix": "add an eval for the no-refunds rule",
            "baseline_output": "I cannot promise a refund.",
            "mutant_output": "Sure, refund coming right up!",
        },
    ],
}


def test_html_contains_score_and_survivor():
    doc = format_report_html(SAMPLE)
    assert doc.startswith("<!doctype html>")
    assert "45%" in doc  # effective coverage
    assert "flip_negation" in doc and "HIGH" in doc
    # baseline/mutant diff rows are present.
    assert "dl del" in doc and "dl add" in doc
    assert "refund" in doc


def test_html_escapes_untrusted_output():
    evil = dict(SAMPLE)
    evil["survivors"] = [dict(SAMPLE["survivors"][0],
                              mutant_output="<script>alert(1)</script>")]
    doc = format_report_html(evil)
    assert "<script>alert(1)</script>" not in doc
    assert "&lt;script&gt;" in doc


def test_html_flags_invalid_run():
    doc = format_report_html(dict(SAMPLE, status="budget_exceeded"))
    assert "INVALID" in doc or "INCOMPLETE" in doc


def test_cli_report_roundtrip(tmp_path, monkeypatch, capsys):
    from muteval.cli import main

    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.py"
    cfg.write_text(
        "from muteval import MutEvalConfig, System, EvalOutcome\n"
        "SYSTEM = System(prompt='Be helpful.\\n- Do not promise refunds.\\n"
        "- Cite the source.\\n- Greet politely.', model='gpt-4o-mini')\n"
        "def run(system, case):\n"
        "    return 'no refund' if 'do not promise refunds' in system.prompt.lower() else 'sure refund'\n"
        "def weak(o, c):\n    return EvalOutcome(passed='refund' in o.lower(), name='w')\n"
        "config = MutEvalConfig(system=SYSTEM, cases=[{'q':'r?'}], run=run, evals=[weak], eval_names=['w'])\n"
    )
    main(["run", "--config", str(cfg), "--no-color"])
    capsys.readouterr()
    out = tmp_path / "r.html"
    code = main(["report", "--html", str(out)])
    assert code == 0
    assert out.exists() and "<!doctype html>" in out.read_text()


def test_cli_report_without_run_errors(tmp_path, monkeypatch, capsys):
    from muteval.cli import main

    monkeypatch.chdir(tmp_path)
    code = main(["report", "--html", str(tmp_path / "r.html")])
    assert code == 2
    assert "no run to report" in capsys.readouterr().err
