"""Tests for `muteval check` (the config doctor)."""

from muteval import MutEvalConfig
from muteval.cli import main
from muteval.doctor import all_ok, run_checks


def _good_config():
    return MutEvalConfig(
        prompt="You are a bot.\n- You must cite the order ID.\n- Do not lie.",
        cases=[{"order_id": "X1"}],
        run=lambda p, c: f"order {c['order_id']}",
        evals=[lambda o, c: "order" in o],
        eval_names=["has_order"],
    )


def test_check_passes_on_a_wired_config():
    results = run_checks(_good_config())
    assert all_ok(results)
    assert any(r.name.startswith("baseline passes") and r.ok for r in results)


def test_check_flags_an_eval_that_raises():
    def boom(o, c):
        raise RuntimeError("field missing")

    cfg = MutEvalConfig(
        prompt="You are a bot.\n- Cite the order ID.\n- Do not lie.",
        cases=[{"order_id": "X1"}], run=lambda p, c: "order X1",
        evals=[boom], eval_names=["boom"],
    )
    results = run_checks(cfg)
    assert not all_ok(results)
    # the offending eval is named, and the baseline is reported RED
    assert any("boom" in r.name and not r.ok for r in results)
    assert any(r.name.startswith("baseline passes") and not r.ok for r in results)


def test_check_flags_a_failing_baseline_eval():
    cfg = MutEvalConfig(
        prompt="You are a bot.\n- Cite the order ID.\n- Do not lie.",
        cases=[{"order_id": "X1"}], run=lambda p, c: "order X1",
        evals=[lambda o, c: False],  # never passes on the original system
    )
    results = run_checks(cfg)
    assert not all_ok(results)
    assert any(r.name.startswith("baseline passes") and not r.ok for r in results)


def test_check_no_model_skips_run_and_evals():
    calls = {"n": 0}

    def run(p, c):
        calls["n"] += 1
        return "x"

    cfg = MutEvalConfig(
        prompt="You are a bot.\n- Cite the order ID.\n- Do not lie.",
        cases=[{"x": 1}], run=run, evals=[lambda o, c: True],
    )
    results = run_checks(cfg, use_model=False)
    assert calls["n"] == 0  # no model calls at all
    assert not any(r.name.startswith("run()") for r in results)
    assert all_ok(results)


def test_check_flags_zero_mutants():
    cfg = MutEvalConfig(
        prompt="short",  # too short to mutate
        cases=[{"x": 1}], run=lambda p, c: "ok", evals=[lambda o, c: True],
    )
    results = run_checks(cfg)
    assert any(r.name == "mutants generate" and not r.ok for r in results)


def test_check_only_touches_one_case_by_default():
    calls = {"n": 0}

    def run(p, c):
        calls["n"] += 1
        return "ok"

    cfg = MutEvalConfig(
        prompt="You are a bot.\n- Cite the order ID.\n- Do not lie.",
        cases=[{"x": 1}, {"x": 2}, {"x": 3}],
        run=run, evals=[lambda o, c: True],
    )
    run_checks(cfg)                 # default: first case only
    assert calls["n"] == 1
    calls["n"] = 0
    run_checks(cfg, full=True)      # full: every case
    assert calls["n"] == 3


def test_cli_check_exit_codes(tmp_path):
    good = tmp_path / "good.py"
    good.write_text(
        "from muteval import MutEvalConfig\n"
        "config = MutEvalConfig(prompt='You are a bot.\\n- Cite the order ID.\\n- Do not lie.',\n"
        "    cases=[{'order_id':'X1'}], run=lambda p,c: 'order X1',\n"
        "    evals=[lambda o,c: 'order' in o])\n",
        encoding="utf-8",
    )
    assert main(["check", "--config", str(good), "--no-color"]) == 0

    bad = tmp_path / "bad.py"
    bad.write_text(
        "from muteval import MutEvalConfig\n"
        "config = MutEvalConfig(prompt='You are a bot.\\n- Cite the order ID.\\n- Do not lie.',\n"
        "    cases=[{'order_id':'X1'}], run=lambda p,c: 'order X1',\n"
        "    evals=[lambda o,c: False])\n",
        encoding="utf-8",
    )
    assert main(["check", "--config", str(bad), "--no-color"]) == 2
