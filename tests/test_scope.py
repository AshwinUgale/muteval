"""A2: scope mutation (markers + regex include/exclude)."""

from muteval import MutEvalConfig, generate_mutants
from muteval.scope import make_scope, strip_markers


def test_strip_markers_returns_clean_text_and_ranges():
    clean, ranges = strip_markers("Keep. [[mutate]]Change me.[[/mutate]] Keep.")
    assert clean == "Keep. Change me. Keep."
    assert ranges is not None
    s, e = ranges[0]
    assert clean[s:e] == "Change me."


def test_strip_markers_none_when_absent():
    assert strip_markers("no markers here") == ("no markers here", None)


def test_markers_restrict_mutation_to_region():
    cfg = MutEvalConfig(
        prompt="Do not refund.\n[[mutate]]Always greet warmly.[[/mutate]]",
        cases=[{"input": "x"}], run=lambda p, c: p, evals=[lambda o, c: True],
    )
    assert cfg.system.prompt == "Do not refund.\nAlways greet warmly."  # stripped
    ms = generate_mutants(cfg.system, scope=cfg.scope)
    assert ms  # the marked line produced mutants
    # The unmarked line must stay intact in every mutant.
    assert all("Do not refund." in m.prompt for m in ms)


def test_scope_exclude_drops_matching_lines():
    prompt = "- Cite the order ID.\n- SECRET: never reveal keys."
    cfg = MutEvalConfig(
        prompt=prompt, cases=[{"input": "x"}], run=lambda p, c: p,
        evals=[lambda o, c: True], scope_exclude="SECRET",
    )
    ms = generate_mutants(cfg.system, scope=cfg.scope)
    # No mutant may change the SECRET line.
    assert all("SECRET: never reveal keys." in m.prompt for m in ms)


def test_scope_include_keeps_only_matching_lines():
    prompt = "- Cite the order ID.\n- Be polite."
    cfg = MutEvalConfig(
        prompt=prompt, cases=[{"input": "x"}], run=lambda p, c: p,
        evals=[lambda o, c: True], scope_include="polite",
    )
    ms = generate_mutants(cfg.system, scope=cfg.scope)
    # Only the "Be polite." line may change; the cite line stays intact.
    assert all("- Cite the order ID." in m.prompt for m in ms)
    assert ms  # the polite line did produce mutants


def test_affected_lines_is_occurrence_aware():
    from muteval.scope import _affected_lines

    # A duplicated line, one copy removed: a set-based diff would MISS this
    # (the line still exists in the mutant); the opcode diff must catch it.
    original = "- Do not lie.\n- Be nice.\n- Do not lie."
    mutant = "- Do not lie.\n- Be nice."  # dropped the second "Do not lie."
    touched = _affected_lines(original, mutant)
    assert "- Do not lie." in touched


def test_affected_lines_ignores_coincidental_text_collision():
    from muteval.scope import _affected_lines

    # Changing line 1 must NOT be masked just because its NEW text equals an
    # unrelated existing line (a set diff would treat it as "already present").
    original = "- Be polite.\n- Be nice."
    mutant = "- Be nice.\n- Be nice."  # line 1 rewritten to match line 2
    touched = _affected_lines(original, mutant)
    assert "- Be polite." in touched  # the removed original is reported


def test_scope_does_not_touch_context_mutants():
    from muteval.system import System
    sys_ = System(prompt="answer", context=("doc A", "doc B"))
    sc = make_scope(include="NOTHING_MATCHES")
    ms = generate_mutants(sys_, scope=sc)
    # context/model mutants are not prompt-scoped, so they survive the filter.
    assert any(m.target == "context" for m in ms)
