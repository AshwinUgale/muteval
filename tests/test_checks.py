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


def test_judge_endpoint_resolution(monkeypatch):
    from muteval.checks import _judge_endpoint

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    # default -> OpenAI
    assert _judge_endpoint() == "https://api.openai.com/v1/chat/completions"
    # an OpenAI-style base ("/v1") gets "/chat/completions" appended
    assert _judge_endpoint("https://api.groq.com/openai/v1") == \
        "https://api.groq.com/openai/v1/chat/completions"
    # trailing slash tolerated
    assert _judge_endpoint("https://api.groq.com/openai/v1/") == \
        "https://api.groq.com/openai/v1/chat/completions"
    # a full endpoint is left as-is
    assert _judge_endpoint("https://x/v1/chat/completions") == \
        "https://x/v1/chat/completions"
    # OPENAI_BASE_URL env is honored when no explicit base_url
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert _judge_endpoint() == "http://localhost:11434/v1/chat/completions"
    # explicit base_url overrides the env
    assert _judge_endpoint("https://api.openai.com/v1") == \
        "https://api.openai.com/v1/chat/completions"


def test_llm_judge_uses_custom_judge_without_network():
    ev = checks.llm_judge("is it polite", judge=lambda prompt: 0.9, threshold=0.5)
    out = ev("Hello, happy to help!", {"input": "hi"})
    assert out.passed is True and out.score == 0.9


def test_cites_source_is_bracket_agnostic():
    ev = checks.cites_source(r"doc-\d+")
    assert ev("see [doc-1] and (doc-2)", {}).passed is True         # ascii brackets
    assert ev("per 【doc-1】", {}).passed is True                    # full-width brackets
    assert ev("supported by doc-3", {}).passed is True              # bare
    assert ev("no citation here", {}).passed is False
    # min_count
    two = checks.cites_source(r"doc-\d+", min_count=2)
    assert two("[doc-1] [doc-2]", {}).passed is True
    assert two("[doc-1] only", {}).passed is False


def test_grounded_preset_uses_context_and_judge_without_network():
    ev = checks.grounded("context", judge=lambda prompt: 0.9, threshold=0.5)
    out = ev("The warranty is 24 months.",
             {"context": ["The warranty is 24 months."]})
    assert out.passed is True and out.score == 0.9 and out.name == "grounded"
    # a strict judge fails it
    ev2 = checks.grounded("context", judge=lambda prompt: 0.1, threshold=0.5)
    assert ev2("made up claim", {"context": "unrelated"}).passed is False
