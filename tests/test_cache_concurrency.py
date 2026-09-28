"""Cache & concurrency — an optimization must never change a verdict.

From a full audit: the cache keyed eval outcomes on the eval's LABEL (an edited
eval, or two evals sharing a label, served stale/shared verdicts; editing run()
served stale outputs), --concurrency shared mutable metric objects and case
dicts across threads (scores from 0% to 85% on one suite), skip-unchanged
ignored run()'s writes into the case, and adapter judges weren't counted by
--max-calls (a cap of 20 made 80 paid calls).
"""

import itertools
import subprocess
import sys

import pytest

from muteval import Cache, MutEvalConfig, System, checks, run_mutation_testing
from muteval.adapters.deepeval import metric_to_eval
from muteval.cli import main
from muteval.fingerprint import fingerprint
from muteval.report import format_report, result_to_dict
from muteval.runner import select_mutants

PROMPT = (
    "- You must cite the order ID.\n- Never promise refunds.\n- Always be polite.\n"
    "- Keep it under 50 words."
)


def _cached(tmp_path, first, second, **kw):
    """Warm a cache with `first`, then run `second` against it; also run
    `second` with no cache. The two must agree."""
    cache = Cache(str(tmp_path / "c.sqlite"))
    run_mutation_testing(first(), cache=cache, **kw)
    with_cache = run_mutation_testing(second(), cache=cache, **kw)
    cache.close()
    without = run_mutation_testing(second(), **kw)
    return without, with_cache


def _same(a, b):
    da, db = result_to_dict(a), result_to_dict(b)
    da.pop("cache", None)
    db.pop("cache", None)
    return da == db


# --- fingerprints -------------------------------------------------------------------


def test_fingerprint_tracks_what_an_eval_does():
    assert fingerprint(checks.contains("X1")) == fingerprint(checks.contains("X1"))
    assert fingerprint(checks.contains("X1")) != fingerprint(checks.contains("ZZZ"))
    assert fingerprint(checks.llm_judge("r", threshold=0.5)) != fingerprint(
        checks.llm_judge("r", threshold=0.9)
    )
    assert fingerprint(lambda o, c: "a" in o) != fingerprint(lambda o, c: "b" in o)

    def with_default(o, c, word="a"):
        return word in o

    def other_default(o, c, word="b"):
        return word in o

    assert fingerprint(with_default) != fingerprint(other_default)


def test_fingerprint_of_callable_objects_and_cache_version():
    class Metric:
        def __init__(self, t):
            self.threshold = t

        def __call__(self, o, c):
            return True

    assert fingerprint(Metric(0.5)) != fingerprint(Metric(0.9))
    ev = checks.contains("X")
    before = fingerprint(ev)
    ev.cache_version = "rubric-v2"  # depends on something muteval can't see
    assert fingerprint(ev) != before


def test_fingerprint_is_stable_across_processes():
    code = (
        "from muteval import checks; from muteval.fingerprint import fingerprint; "
        "print(fingerprint(checks.contains('X1')), fingerprint(checks.llm_judge('r')))"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        ).stdout
        for _ in range(2)
    }
    assert len(outs) == 1 and next(iter(outs)).strip()


# --- the cache never serves a stale or shared verdict --------------------------------


def _cfg(evals, run=None, **kw):
    return MutEvalConfig(
        prompt=PROMPT,
        cases=[{"order_id": "X1"}],
        run=run or (lambda p, c: "order X1: " + p[:40]),
        evals=evals,
        **kw,
    )


def test_edited_eval_is_not_served_from_cache(tmp_path):
    without, with_cache = _cached(
        tmp_path,
        lambda: _cfg([checks.contains("X1")]),
        lambda: _cfg([checks.contains("ZZZ")]),  # edited: baseline now FAILS
    )
    assert without.status == with_cache.status == "baseline_failed"


def test_evals_sharing_a_label_do_not_share_a_verdict(tmp_path):
    def make():
        return _cfg(
            [checks.contains("X1"), checks.contains("must")],
            eval_names=["contains", "contains"],
        )

    without, with_cache = _cached(tmp_path, make, make)
    assert _same(without, with_cache)
    assert without.killed > 0  # the second check really does catch mutants


def test_edited_run_is_not_served_from_cache(tmp_path):
    # Prompt mode: the model is chosen INSIDE run(); System.key() can't see it.
    without, with_cache = _cached(
        tmp_path,
        lambda: _cfg([checks.contains("X1")], run=lambda p, c: "order X1"),
        lambda: _cfg([checks.contains("X1")], run=lambda p, c: "model down"),
    )
    assert without.status == with_cache.status == "baseline_failed"


def test_dict_outputs_are_cached(tmp_path):
    def make():
        return _cfg(
            [lambda o, c: "X1" in o["final"]],
            run=lambda p, c: {"final": "order X1 " + p[:20], "trace": [{"ok": True}]},
        )

    without, with_cache = _cached(tmp_path, make, make)
    assert with_cache.status == "valid" and with_cache.cache_hits > 0
    assert _same(without, with_cache)


def test_unserializable_outputs_are_just_not_cached(tmp_path):
    marker = object()

    def make():
        return _cfg([lambda o, c: True], run=lambda p, c: {"final": p, "obj": marker})

    without, with_cache = _cached(tmp_path, make, make)
    assert with_cache.status == "valid" and _same(without, with_cache)


# --- run() writing into the case (the used_context side channel) ---------------------

_CTX = ["Orders ship in 2 days.", "Refunds need a manager."]


def _side_channel(counter=None):
    def run(system, case):
        if counter is not None:
            counter["run"] += 1
        case["used_context"] = list(system.context)  # the side channel
        return "Orders ship in 2 days."

    def grounded(output, case):  # grades against the MUTATED context
        return any(output in doc for doc in case["used_context"])

    return MutEvalConfig(
        system=System(prompt="Answer from the docs.", context=_CTX),
        cases=[{"q": "shipping?"}],
        run=run,
        evals=[grounded],
    )


def test_skip_unchanged_respects_case_state():
    # Same output, different case state (the context was mutated): the eval must
    # still run. Skip-unchanged (on at runs_per_mutant=1) used to reuse the
    # baseline pass, so this deterministic suite scored 0 kills with it and
    # more without it (runs_per_mutant=3 turns it off).
    def with_runs(n):
        cfg = _side_channel()
        cfg.runs_per_mutant = n
        return run_mutation_testing(cfg)

    skip_on, skip_off = with_runs(1), with_runs(3)
    assert skip_on.killed > 0
    assert skip_on.killed == skip_off.killed


def test_system_accepts_list_context_and_tools():
    # The README's own form: System(prompt=..., context=[...], tools=[...]).
    # A list inside System.key() crashed mutant de-duplication.
    from muteval.mutators import generate_mutants

    s = System(prompt="Answer from the docs.", context=["a doc.", "b doc."], tools=[1])
    assert s.context == ("a doc.", "b doc.") and s.tools == (1,)
    assert generate_mutants(s)


def test_cache_replays_run_side_effects(tmp_path):
    # (The counter is part of run()'s closure, so it's reset to the same value
    # before the cached run — a different closure value is a different run().)
    counter = {"run": 0}
    cache = Cache(str(tmp_path / "c.sqlite"))
    fresh = run_mutation_testing(_side_channel(counter), cache=cache)
    counter["run"] = 0
    cached = run_mutation_testing(_side_channel(counter), cache=cache)
    cache.close()
    assert counter["run"] == 0  # fully served from cache...
    assert fresh.killed > 0 and _same(fresh, cached)  # ...with the same verdicts


def test_cases_are_isolated_across_mutants():
    cfg = _side_channel()
    run_mutation_testing(cfg)
    assert "used_context" not in cfg.cases[0]  # run() wrote into a private copy


# --- concurrency ----------------------------------------------------------------------


class _StatefulMetric:
    """deepeval-shaped: measure() stores its result on the instance."""

    threshold = 0.5

    def measure(self, tc):
        self.score = 1.0 if "must" in tc else 0.0
        # widen the race window, as a real judge call would
        for _ in range(2000):
            pass
        self.success = self.score >= self.threshold

    def is_successful(self):
        return self.success


def test_concurrency_does_not_change_verdicts():
    def make():
        return _cfg(
            [metric_to_eval(_StatefulMetric(), test_case_factory=lambda o, c: o)],
            run=lambda p, c: p,
        )

    serial = run_mutation_testing(make())
    for _ in range(3):
        assert _same(serial, run_mutation_testing(make(), concurrency=8))
    side_serial = run_mutation_testing(_side_channel())
    assert _same(side_serial, run_mutation_testing(_side_channel(), concurrency=8))


def test_adapter_judges_are_budgeted():
    calls = itertools.count()

    class Paid:
        threshold = 0.5

        def measure(self, tc):
            next(calls)
            self.score = 1.0

        def is_successful(self):
            return True

    cfg = _cfg(
        [metric_to_eval(Paid(), test_case_factory=lambda o, c: o)], run=lambda p, c: p
    )
    r = run_mutation_testing(cfg, max_calls=6, concurrency=4)
    assert r.status == "budget_exceeded"
    assert next(calls) <= 6  # the judge calls count against the cap


# --- labels, sampling, misc -------------------------------------------------------------


def test_eval_names_are_aligned_and_unique():
    evals = [lambda o, c: True, lambda o, c: True, lambda o, c: True]
    cfg = _cfg(evals, eval_names=["GEval", "GEval"])
    assert cfg.eval_names == ["GEval", "GEval#2", "eval_2"]
    with pytest.raises(ValueError):
        _cfg([lambda o, c: True], eval_names=["a", "b"])


def test_negative_sample_and_cap_are_rejected(tmp_path, capsys):
    cfg = _cfg([lambda o, c: True])
    with pytest.raises(ValueError):
        select_mutants(cfg, max_mutants=-1)
    with pytest.raises(ValueError):
        select_mutants(cfg, sample=-3)
    with pytest.raises(SystemExit) as exc:
        main(["run", "--prompt", "p", "--check", "contains:x", "--max-mutants", "-1"])
    assert exc.value.code == 2


def test_system_key_with_mixed_extra_key_types():
    assert System(prompt="p", extra={1: "a", "b": 2}).key()


def test_ragas_string_context_is_one_doc_and_judge_is_budgeted(monkeypatch):
    import types

    from muteval.adapters import ragas as adapter

    captured = {}

    class Sample:
        def __init__(self, **kw):
            captured.update(kw)

    fake = types.ModuleType("ragas.dataset_schema")
    fake.SingleTurnSample = Sample
    monkeypatch.setitem(sys.modules, "ragas", types.ModuleType("ragas"))
    monkeypatch.setitem(sys.modules, "ragas.dataset_schema", fake)
    factory = adapter._default_sample_factory("input", "ctx", None)
    factory("out", {"input": "q", "ctx": "one whole document"})
    assert captured["retrieved_contexts"] == ["one whole document"]  # not chars
    ev = adapter.metric_to_eval(object(), sample_factory=factory, score_fn=lambda s: 0.9)
    assert getattr(ev, "is_llm", False)


def test_cache_provenance_is_reported(tmp_path):
    cache = Cache(str(tmp_path / "c.sqlite"))
    make = lambda: _cfg([checks.contains("X1")])  # noqa: E731
    run_mutation_testing(make(), cache=cache)
    r = run_mutation_testing(make(), cache=cache)
    cache.close()
    assert r.cache_hits > 0
    assert result_to_dict(r)["cache"]["hits"] == r.cache_hits
    assert "served from --cache" in format_report(r, use_color=False)

    cache = Cache(str(tmp_path / "d.sqlite"))
    r = run_mutation_testing(
        _cfg([checks.contains("X1")], runs_per_mutant=3), cache=cache
    )
    assert r.cache_hits is None and "disabled" in r.cache_note
