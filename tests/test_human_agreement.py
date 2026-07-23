"""v0.6: human-agreement probe + `muteval label` worksheet (Cohen's kappa)."""

from __future__ import annotations

import math

from muteval.probes.human_agreement import (
    human_agreement,
    human_agreement_from_rows,
    load_label_rows,
)
from muteval.stats import cohens_kappa, kappa_ci


def test_kappa_perfect_and_chance():
    assert math.isclose(cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0, abs_tol=1e-9)
    # total disagreement on a balanced set -> kappa == -1
    assert math.isclose(cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]), -1.0, abs_tol=1e-9)


def test_kappa_none_on_bad_input():
    assert cohens_kappa([], []) is None
    assert cohens_kappa([1, 0], [1]) is None


def test_kappa_ci_brackets_point_estimate():
    a = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    b = [1, 1, 0, 1, 1, 0, 1, 0, 0, 0]  # mostly agree
    k = cohens_kappa(a, b)
    lo, hi = kappa_ci(a, b, resamples=1000, seed=1)
    assert lo is not None and lo - 1e-9 <= k <= hi + 1e-9


def test_probe_flags_weak_agreement():
    # machine and human agree only at chance level -> low kappa -> not ok.
    pairs = [(True, False), (False, True), (True, True), (False, False),
             (True, False), (False, True)]
    r = human_agreement_from_rows(pairs)
    assert r.metrics["assessed"] is True
    assert r.ok is False  # weak agreement


def test_probe_passes_on_strong_agreement():
    pairs = [(True, True)] * 8 + [(False, False)] * 8 + [(True, False)]
    r = human_agreement_from_rows(pairs)
    assert r.ok is True and r.metrics["kappa"] >= 0.6


def test_probe_not_assessed_without_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = human_agreement(config=None)  # no .muteval/labels.csv present
    assert r.metrics.get("assessed") is False


def test_label_worksheet_roundtrip(tmp_path, monkeypatch):
    from muteval.cli import main

    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.py"
    cfg.write_text(
        "from muteval import MutEvalConfig, System, EvalOutcome\n"
        "SYSTEM = System(prompt='Be helpful.\\n- Cite the source.', model='gpt-4o-mini')\n"
        "def run(system, case):\n    return 'answer ' + case['gt']\n"
        "def has_gt(o, c):\n    return EvalOutcome(passed=c['gt'] in o, name='has_gt')\n"
        "config = MutEvalConfig(system=SYSTEM, cases=[{'gt':'8080'},{'gt':'1986'}], "
        "run=run, evals=[has_gt], eval_names=['has_gt'])\n"
    )
    out = tmp_path / "labels.csv"
    code = main(["label", "--config", str(cfg), "--out", str(out)])
    assert code == 0 and out.exists()

    # A human fills the sheet: agree on the first case, disagree on the second.
    text = out.read_text().splitlines()
    header = text[0]
    assert "human_label" in header
    filled = [header,
              text[1].rsplit(",", 1)[0] + ",pass",
              text[2].rsplit(",", 1)[0] + ",fail"]
    out.write_text("\n".join(filled) + "\n")

    rows = load_label_rows(out)
    assert len(rows) == 2
    assert rows[0] == (True, True) and rows[1] == (True, False)
