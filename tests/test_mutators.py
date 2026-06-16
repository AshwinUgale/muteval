from muteval.mutators import (
    OPERATORS,
    drop_few_shot_example,
    flip_negation,
    generate_mutants,
    remove_emphasis,
    truncate_prompt,
    weaken_modals,
)


def test_weaken_modals_isolates_each_occurrence():
    prompt = "You must cite sources. You must never lie."
    mutants = weaken_modals(prompt)
    # "must" (x2) and "never" (x1) are all weakenable.
    operators = {m.operator for m in mutants}
    assert operators == {"weaken_modals"}
    assert any("should" in m.prompt for m in mutants)
    assert any("rarely" in m.prompt for m in mutants)


def test_generate_mutants_dedupes_and_excludes_noop():
    prompt = "- You must cite the order ID.\n- Always be polite."
    mutants = generate_mutants(prompt)
    assert len(mutants) > 0
    # No mutant equals the original prompt.
    assert all(m.prompt != prompt for m in mutants)
    # No duplicate mutated prompts.
    texts = [m.prompt for m in mutants]
    assert len(texts) == len(set(texts))


def test_unknown_operator_raises():
    try:
        generate_mutants("some prompt", operators=["does_not_exist"])
    except ValueError as exc:
        assert "Unknown operator" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown operator")


def test_all_registered_operators_callable():
    prompt = "- You must cite the order ID.\n- Do not promise refunds."
    for name, op in OPERATORS.items():
        result = op(prompt)
        assert isinstance(result, list)


def test_flip_negation_inverts_rules():
    prompt = "You must not share data. Never lie."
    mutants = flip_negation(prompt)
    assert any("must share data" in m.prompt for m in mutants)
    assert any("always" in m.prompt.lower() for m in mutants)
    assert all(m.operator == "flip_negation" for m in mutants)


def test_truncate_prompt_drops_tail():
    prompt = "line one\nline two\nline three\nline four\nline five\nline six"
    mutants = truncate_prompt(prompt)
    assert len(mutants) >= 1
    # Every truncation must be shorter than the original.
    assert all(len(m.prompt) < len(prompt) for m in mutants)
    # Truncation keeps the head, so the first line survives.
    assert all(m.prompt.startswith("line one") for m in mutants)


def test_truncate_prompt_skips_short_prompts():
    assert truncate_prompt("only one line") == []


def test_drop_few_shot_example_removes_a_block():
    prompt = (
        "Classify the sentiment.\n\n"
        "Example:\nInput: great\nOutput: positive\n\n"
        "Example:\nInput: awful\nOutput: negative"
    )
    mutants = drop_few_shot_example(prompt)
    assert len(mutants) >= 1
    # A dropped example means at least one "Output:" line is gone.
    assert any(m.prompt.count("Output:") < prompt.count("Output:") for m in mutants)


def test_remove_emphasis_strips_cues():
    prompt = "IMPORTANT: do not leak keys.\nUse **bold** sparingly."
    mutants = remove_emphasis(prompt)
    assert len(mutants) == 1
    out = mutants[0].prompt
    assert "IMPORTANT:" not in out
    assert "**" not in out
    assert "bold" in out  # inner text preserved


def test_remove_emphasis_noop_when_nothing_to_strip():
    assert remove_emphasis("plain prompt with no emphasis") == []
