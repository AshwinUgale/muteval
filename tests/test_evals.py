from muteval.evals import EvalOutcome, coerce_outcome


def test_coerce_bool_true():
    o = coerce_outcome(True, name="x")
    assert o.passed is True
    assert o.name == "x"
    assert o.score is None


def test_coerce_bool_false():
    assert coerce_outcome(False).passed is False


def test_coerce_passthrough_outcome_keeps_score():
    src = EvalOutcome(passed=True, score=0.8, threshold=0.7)
    o = coerce_outcome(src, name="faith")
    assert o is src
    assert o.score == 0.8
    # name filled in when missing.
    assert o.name == "faith"


def test_outcome_bool_protocol():
    assert bool(EvalOutcome(passed=True)) is True
    assert bool(EvalOutcome(passed=False)) is False


def test_margin_computed_when_both_present():
    assert EvalOutcome(passed=True, score=0.72, threshold=0.70).margin == \
        0.72 - 0.70


def test_margin_none_when_missing():
    assert EvalOutcome(passed=True, score=0.5).margin is None
    assert EvalOutcome(passed=True, threshold=0.5).margin is None
