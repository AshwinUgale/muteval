from muteval.mutators import OPERATORS, generate_mutants, weaken_modals


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
