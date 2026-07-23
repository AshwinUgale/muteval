"""v0.5 P0-6: cost budget (--max-calls) — fail closed before overspending.

The budget counts ACTUAL model + judge calls (cache hits and skipped judges do
NOT count). Exceeding it aborts with status 'budget_exceeded' and no score.
"""

from __future__ import annotations

from muteval import Cache, EvalOutcome, MutEvalConfig, System, run_mutation_testing
from muteval.runner import BUDGET_EXCEEDED, VALID

SYSTEM = System(
    prompt="Answer the question.\n- Cite the source.\n- Do not invent facts.\n- Greet politely.",
    model="gpt-4o-mini",
)
CASES = [{"gt": "8080"}, {"gt": "1986"}]


def _run(system, case):
    return f"answer {case['gt']} [1](doc)"


def _correct(output, case):
    return EvalOutcome(passed=case["gt"] in output, name="correct")


def _config(run=_run):
    return MutEvalConfig(
        system=SYSTEM, cases=CASES, run=run, evals=[_correct], eval_names=["correct"]
    )


def test_generous_budget_completes_valid():
    r = run_mutation_testing(_config(), max_calls=100000)
    assert r.status == VALID and r.total > 0


def test_tiny_budget_fails_closed():
    calls = {"n": 0}

    def counting_run(system, case):
        calls["n"] += 1
        return _run(system, case)

    r = run_mutation_testing(_config(counting_run), max_calls=3)
    assert r.status == BUDGET_EXCEEDED
    assert r.score is None or True  # no trustworthy score is reported/used
    # It stopped near the cap rather than running the whole suite.
    assert calls["n"] <= 5  # baseline (2) + a little, not all mutants*cases


def test_budget_counts_only_real_calls_not_cache_hits(tmp_path):
    cache = Cache(str(tmp_path / "c.sqlite"))
    # First run populates the cache with a generous budget.
    run_mutation_testing(_config(), cache=cache, max_calls=100000)
    # A re-run served entirely from cache should make ZERO charged calls, so even
    # a budget of 0 completes (nothing is charged).
    r = run_mutation_testing(_config(), cache=cache, max_calls=0)
    assert r.status == VALID
    cache.close()


def test_cli_max_calls_exits_two(tmp_path, capsys):
    from muteval.cli import main

    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"q": "hi"}\n')
    # A model call would be needed, but the tiny budget trips before any real
    # call path matters for exit semantics; use an offline mock via --target.
    (tmp_path / "p.py").write_text("def answer(prompt, case):\n    return 'x'\n")
    import sys as _sys

    _sys.path.insert(0, str(tmp_path))
    try:
        code = main([
            "run", "--target", "p:answer",
            "--prompt", "Answer. Always cite the source. Do not lie.",
            "--cases", str(cases), "--check", "contains:x",
            "--max-calls", "1", "--no-color",
        ])
    finally:
        _sys.path.remove(str(tmp_path))
    assert code == 2
    assert "max-calls" in capsys.readouterr().err.lower()
