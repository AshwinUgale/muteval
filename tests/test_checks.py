from muteval import checks


def test_contains_case_insensitive_by_default():
    ev = checks.contains("Refund")
    assert ev("we issued a refund", {}).passed is True
    assert ev("no money back", {}).passed is False


def test_not_contains_is_a_guardrail():
    ev = checks.not_contains("refund")
    assert ev("no money back", {}).passed is True
    assert ev("here is your refund", {}).passed is False


def test_contains_case_pulls_value_from_case():
    ev = checks.contains_case("order_id")
    assert ev("your order A123 shipped", {"order_id": "A123"}).passed is True
    assert ev("your order shipped", {"order_id": "A123"}).passed is False


def test_contains_case_missing_key_fails():
    ev = checks.contains_case("order_id")
    out = ev("anything", {})
    assert out.passed is False
    assert out.detail == "missing key"


def test_regex_matches():
    ev = checks.regex_matches(r"\bORD-\d+\b")
    assert ev("ref ORD-99", {}).passed is True
    assert ev("ref ORD-", {}).passed is False


def test_is_json():
    ev = checks.is_json()
    assert ev('{"a": 1}', {}).passed is True
    assert ev("not json", {}).passed is False


def test_equals_strips_by_default():
    ev = checks.equals("expected")
    assert ev("  hello \n", {"expected": "hello"}).passed is True
    assert ev("hello world", {"expected": "hello"}).passed is False


def test_llm_judge_with_injected_judge():
    # Inject a deterministic judge so the test needs no API/openai.
    ev = checks.llm_judge("is it polite?", judge=lambda prompt: 0.8, threshold=0.7)
    out = ev("thank you!", {"input": "hi"})
    assert out.passed is True
    assert out.score == 0.8
    assert out.threshold == 0.7
    # margin enables near-miss reporting.
    assert round(out.margin, 4) == round(0.8 - 0.7, 4)
