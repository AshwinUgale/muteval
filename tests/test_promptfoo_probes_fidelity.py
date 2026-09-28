"""promptfoo fidelity and probe correctness — from a full audit.

promptfoo: `{{var}}` in assertion values was never rendered (a not-contains
guarded nothing), file:// values and vars were literal text, defaultTest.vars
and array vars were ignored, CSV prefixes fell through to `equals`, llm-rubric
ignored its threshold/provider and lost its score, and chat prompts were mangled.

Probes: the bias panel passed an always-SHORTER judge and failed a fair coin;
lower-is-better metrics read as broken; redundancy mis-aligned errored cases;
kappa claimed "substantial agreement" where it's undefined; unassessed probes
showed PASS and a hygiene WARN failed `muteval probe`.
"""

import pytest

from muteval import MutEvalConfig, checks
from muteval.adapters.base import scorer_to_eval
from muteval.adapters.promptfoo import _expected_to_assert, config_from_promptfoo_dict
from muteval.evals import EvalOutcome
from muteval.probes.discrimination import discrimination
from muteval.probes.human_agreement import human_agreement_from_rows
from muteval.probes.judge_bias import run_judge_bias_panel
from muteval.probes.judge_reliability import judge_reliability
from muteval.probes.redundancy import redundancy
from muteval.probes.threshold_calibration import threshold_calibration
from muteval.report import format_probe_card, probe_blocks


def _cfg(tests, **extra):
    data = {"prompts": ["Answer: {{question}}"], "tests": tests, **extra}
    return config_from_promptfoo_dict(data, run=lambda p, c: "stub")


def _grade(cfg, output, case_index=0):
    return {
        n: ev(output, cfg.cases[case_index]).passed
        for n, ev in zip(cfg.eval_names, cfg.evals)
    }


# --- promptfoo ------------------------------------------------------------------------


def test_assertion_values_render_vars():
    cfg = _cfg(
        [
            {
                "vars": {
                    "question": "capital?",
                    "answer": "Paris",
                    "forbidden": "London",
                },
                "assert": [
                    {"type": "contains", "value": "{{answer}}"},
                    {"type": "not-contains", "value": "{{forbidden}}"},
                ],
            }
        ]
    )
    assert _grade(cfg, "It is Paris.") == {
        "promptfoo:contains": True,
        "promptfoo:not-contains": True,
    }
    # The forbidden answer is CAUGHT (it used to pass: "{{forbidden}}" literal).
    assert _grade(cfg, "It is London.")["promptfoo:not-contains"] is False


def test_file_refs_default_vars_and_array_expansion(tmp_path):
    (tmp_path / "q.txt").write_text("What is 2+2?\n", encoding="utf-8")
    (tmp_path / "want.txt").write_text("4", encoding="utf-8")
    data = {
        "prompts": ["{{question}} ({{style}}, {{lang}})"],
        "defaultTest": {"vars": {"style": "brief"}},
        "tests": [
            {
                "vars": {"question": "file://q.txt", "lang": ["en", "fr"]},
                "assert": [{"type": "contains", "value": "file://want.txt"}],
            }
        ],
    }
    cfg = config_from_promptfoo_dict(data, run=lambda p, c: "4", base_dir=tmp_path)
    assert [c["lang"] for c in cfg.cases] == ["en", "fr"]  # one case per array item
    assert all(
        c["question"] == "What is 2+2?" and c["style"] == "brief" for c in cfg.cases
    )
    assert _grade(cfg, "the answer is 4")["promptfoo:contains"] is True


def test_equals_object_and_is_json_schema():
    schema = {"type": "object", "required": ["answer", "source"]}
    cfg = _cfg(
        [
            {
                "vars": {"question": "q"},
                "assert": [
                    {"type": "equals", "value": {"answer": "Paris"}},
                    {"type": "is-json", "value": schema},
                ],
            }
        ]
    )
    g = _grade(cfg, '{"answer": "Paris"}')
    assert g["promptfoo:equals"] is True  # JSON equality, not a Python repr
    assert g["promptfoo:is-json"] is False  # missing the required "source"
    assert _grade(cfg, '{"answer": "Paris", "source": "x"}')["promptfoo:is-json"] is True


def test_negated_regex_and_starts_with():
    cfg = _cfg(
        [
            {
                "vars": {"question": "q"},
                "assert": [
                    {"type": "not-regex", "value": "London"},
                    {"type": "not-starts-with", "value": "Sorry"},
                ],
            }
        ]
    )
    assert all(_grade(cfg, "Paris.").values())
    assert not any(_grade(cfg, "Sorry, London.").values())


def test_llm_rubric_uses_its_threshold_provider_and_keeps_the_score(monkeypatch):
    calls = []

    def fake(prompt, model, base_url=None):
        calls.append((model, base_url))
        return "6"

    monkeypatch.setattr(checks, "_openai_chat_stdlib", fake)
    data = {
        "prompts": ["{{question}}"],
        "providers": ["openai:gpt-4o"],
        "tests": [
            {
                "vars": {"question": "hi"},
                "assert": [
                    {
                        "type": "llm-rubric",
                        "value": "is polite",
                        "threshold": 0.7,
                        "provider": "openai:gpt-4.1",
                    }
                ],
            }
        ],
    }
    cfg = config_from_promptfoo_dict(
        data, run=lambda p, c: "x", base_url="https://x.test/v1"
    )
    outcome = cfg.evals[0]("hello", cfg.cases[0])
    assert calls == [("gpt-4.1", "https://x.test/v1")]  # the assertion's provider
    assert outcome.score == 0.6 and outcome.threshold == 0.7  # score kept
    assert outcome.passed is False  # 0.6 < its own 0.7 threshold
    assert getattr(cfg.evals[0], "is_llm", False)


def test_csv_expected_prefixes():
    assert _expected_to_assert("grade: is polite") == {
        "type": "llm-rubric",
        "value": "is polite",
    }
    assert _expected_to_assert("is-json") == {"type": "is-json"}
    assert _expected_to_assert("javascript: output.length < 5")["type"] == "javascript"
    assert _expected_to_assert("not-icontains: rude")["type"] == "not-icontains"
    assert _expected_to_assert("Paris") == {"type": "equals", "value": "Paris"}


def test_csv_multiple_expected_columns_and_metadata(tmp_path):
    (tmp_path / "cases.csv").write_text(
        "question,__expected1,__expected2,__description\n"
        "capital?,contains: Paris,not-contains: London,geo\n",
        encoding="utf-8",
    )
    data = {"prompts": ["{{question}}"], "tests": "file://cases.csv"}
    cfg = config_from_promptfoo_dict(data, run=lambda p, c: "x", base_dir=tmp_path)
    case = cfg.cases[0]
    assert "__description" not in case and "__expected1" not in case
    assert sorted(a["type"] for a in case["_asserts"]) == ["contains", "not-contains"]


def test_unreproducible_features_are_reported_not_silent(capsys):
    data = {
        "prompts": ["one {{question}}", "two {{question}}"],
        "tests": [
            {
                "vars": {"question": "q"},
                "threshold": 0.5,
                "assert": [{"type": "contains", "value": "a", "weight": 3}],
            },
            {
                "vars": {"question": "q"},
                "options": {"transform": "output.toUpperCase()"},
                "assert": [{"type": "contains", "value": "A"}],
            },
        ],
    }
    cfg = config_from_promptfoo_dict(data, run=lambda p, c: "a")
    err = capsys.readouterr().err
    assert "2 prompts" in err and "weight" in err and "transform" in err
    assert len(cfg.cases) == 1  # the transformed test is dropped, not mis-graded


def test_all_transformed_tests_explain_themselves():
    data = {
        "prompts": ["{{q}}"],
        "defaultTest": {"options": {"transform": "output.trim()"}},
        "tests": [{"vars": {"q": "x"}, "assert": [{"type": "contains", "value": "x"}]}],
    }
    with pytest.raises(ValueError, match="transform"):
        config_from_promptfoo_dict(data, run=lambda p, c: "x")


def test_chat_prompts_are_refused_not_mangled():
    data = {
        "prompts": [
            '[{"role": "system", "content": "Be kind."}, {"role": "user", "content": "{{q}}"}]'
        ],
        "tests": [{"vars": {"q": "x"}, "assert": [{"type": "contains", "value": "x"}]}],
    }
    with pytest.raises(ValueError, match="chat message list"):
        config_from_promptfoo_dict(data, run=lambda p, c: "x")


# --- probes -----------------------------------------------------------------------------

PAIRS = [("short answer.", "short answer. plus padding words for length.", {})] * 20


def test_bias_panel_is_two_sided():
    shorter = lambda a, b, c: "A" if len(a) < len(b) else "B"  # noqa: E731
    tie = lambda a, b, c: "tie"  # noqa: E731
    assert run_judge_bias_panel(shorter, [], PAIRS).ok() is False  # used to pass
    assert run_judge_bias_panel(tie, [], PAIRS).ok() is True


def test_bias_panel_does_not_flag_a_fair_coin_by_noise():
    import random

    flagged = 0
    for seed in range(100):
        rng = random.Random(seed)
        coin = lambda a, b, c, r=rng: r.choice(["A", "B"])  # noqa: E731
        flagged += not run_judge_bias_panel(coin, [], PAIRS).ok()
    assert flagged <= 15  # ~5% expected at 95% confidence


def _exemplar_cfg(evals, names):
    return MutEvalConfig(
        prompt="p",
        cases=[
            {"good": ["clean reply", "kind reply"], "bad": ["toxic reply", "rude toxic"]}
        ],
        run=lambda p, c: "x",
        evals=evals,
        eval_names=names,
    )


def test_lower_is_better_metrics_are_read_correctly():
    toxicity = scorer_to_eval(
        lambda o, c: 0.9 if "toxic" in o else 0.1,
        threshold=0.5,
        name="toxicity",
        higher_is_better=False,
    )
    assert toxicity("clean reply", {}).margin == pytest.approx(0.4)  # positive
    cfg = _exemplar_cfg([toxicity], ["toxicity"])
    assert threshold_calibration(cfg).ok is True  # was "too_lenient"
    d = discrimination(cfg)
    assert d.ok and d.metrics["min_auc"] == 1.0  # was AUC 0.00


def test_discrimination_names_evals_it_could_not_assess():
    def errs_on_bad(o, c):
        if "toxic" in o:
            raise RuntimeError("judge down")
        return EvalOutcome(passed=True, score=1.0, threshold=0.5)

    ok_eval = scorer_to_eval(
        lambda o, c: 0.0 if "toxic" in o else 1.0, threshold=0.5, name="q"
    )
    d = discrimination(_exemplar_cfg([errs_on_bad, ok_eval], ["errs_on_bad", "q"]))
    assert "errs_on_bad" in d.metrics["not_assessable"]
    assert "not assessable" in d.summary


def test_redundancy_aligns_cases_when_an_eval_errors():
    import itertools

    n = itertools.count()

    def a(o, c):
        return EvalOutcome(passed=True, score=float(c["v"]))

    def b(o, c):  # identical to a, but times out on one case
        if c["v"] == 2 and next(n) == 0:
            raise TimeoutError
        return EvalOutcome(passed=True, score=float(c["v"]))

    cfg = MutEvalConfig(
        prompt="p",
        cases=[{"v": v} for v in (5, 2, 9, 1, 7, 3)],
        run=lambda p, c: "x",
        evals=[a, b],
        eval_names=["A", "B"],
    )
    r = redundancy(cfg)
    assert r.ok is False and ["A", "B"] in r.metrics["families"]


def test_reliability_needs_two_runs():
    cfg = MutEvalConfig(
        prompt="p", cases=[1], run=lambda p, c: "x", evals=[lambda o, c: True]
    )
    r = judge_reliability(cfg, runs=1)
    assert r.metrics["assessed"] is False


def test_kappa_is_not_reported_where_undefined_or_unstable():
    one_class = [(True, True)] * 10
    r = human_agreement_from_rows(one_class)
    assert r.metrics["assessed"] is False and "undefined" in r.summary
    # Both raters use both classes, but ~95% "pass": the kappa paradox zone
    # (29/30 agreement can read kappa ~0).
    skewed = [(True, True)] * 28 + [(False, False)] + [(False, True)]
    r = human_agreement_from_rows(skewed)
    assert r.metrics["assessed"] is False and "unstable" in r.summary
    balanced = [(True, True)] * 8 + [(False, False)] * 8 + [(True, False)]
    assert human_agreement_from_rows(balanced).metrics["assessed"] is True


def test_probe_card_shows_na_and_only_core_warns_block():
    from muteval.probes.base import ProbeResult

    na = ProbeResult(
        name="discrimination", ok=True, summary="s", metrics={"assessed": False}
    )
    hygiene_warn = ProbeResult(name="redundancy", ok=False, summary="s")
    core_warn = ProbeResult(name="judge_reliability", ok=False, summary="s")
    assert "N/A" in format_probe_card([na], use_color=False)
    assert not probe_blocks(na) and not probe_blocks(hygiene_warn)
    assert probe_blocks(core_warn)


def test_label_worksheet_refuses_to_overwrite(tmp_path):
    from muteval.cli import _write_label_worksheet

    cfg = MutEvalConfig(
        prompt="- You must cite the order ID.\n- Never guess.",
        cases=[{"x": 1}],
        run=lambda p, c: p,
        evals=[lambda o, c: "must" in o],
    )
    out = tmp_path / "labels.csv"
    n = _write_label_worksheet(cfg, out, mutants=2)
    text = out.read_text(encoding="utf-8")
    assert n == 3 and "mutant:" in text and ",fail," in text  # both classes appear
    with pytest.raises(FileExistsError):
        _write_label_worksheet(cfg, out)
    assert _write_label_worksheet(cfg, out, force=True) > 0
