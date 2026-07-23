"""v0.6: probe report-card HTML (no composite score)."""

from __future__ import annotations

from muteval.probes.base import ProbeResult
from muteval.report import format_probe_card, format_probe_card_html

RESULTS = [
    ProbeResult(name="statistical_adequacy", ok=True, summary="60 cases, adequate", detail="fine"),
    ProbeResult(name="judge_reliability", ok=False, summary="18% flaky", detail="use temperature 0"),
]


def test_card_labels_and_orders_by_tier():
    # Honesty: the card must tell the reader a hygiene WARN != a core WARN, and
    # show the load-bearing lenses first. A hygiene probe passed BEFORE a core
    # probe in the input must render AFTER it.
    txt = format_probe_card(RESULTS, use_color=False)
    assert "(core)" in txt and "(hygiene)" in txt          # tiers are labelled
    assert "core = catches a real eval defect" in txt      # legend present
    assert txt.index("judge_reliability") < txt.index("statistical_adequacy")

    html_doc = format_probe_card_html(RESULTS)
    assert 'class="tier"' in html_doc
    assert html_doc.index("judge_reliability") < html_doc.index("statistical_adequacy")


def test_card_html_renders_pass_and_warn():
    doc = format_probe_card_html(RESULTS)
    assert doc.startswith("<!doctype html>")
    assert "statistical_adequacy" in doc and "judge_reliability" in doc
    assert "PASS" in doc and "WARN" in doc
    assert "no composite score" in doc  # the thesis is stated on the page


def test_card_html_escapes_summary():
    evil = [ProbeResult(name="x", ok=True, summary="<script>alert(1)</script>", detail=None)]
    doc = format_probe_card_html(evil)
    assert "<script>alert(1)</script>" not in doc
    assert "&lt;script&gt;" in doc


def test_card_html_empty():
    doc = format_probe_card_html([])
    assert "No probes ran" in doc


def test_cli_probe_html(tmp_path, monkeypatch, capsys):
    from muteval.cli import main

    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.py"
    cfg.write_text(
        "from muteval import MutEvalConfig, System, EvalOutcome\n"
        "config = MutEvalConfig(prompt='answer the question.', cases=[{'q':'a'},{'q':'b'}],\n"
        "    run=lambda p,c: 'x', evals=[lambda o,c: True], eval_names=['e'])\n"
    )
    out = tmp_path / "card.html"
    code = main(["probe", "--config", str(cfg), "--html", str(out), "--no-color"])
    assert code in (0, 1)  # exit depends on probe verdicts; both are non-error
    assert out.exists() and "<!doctype html>" in out.read_text()
