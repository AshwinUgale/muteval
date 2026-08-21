"""#36: promptfoo llm-rubric judge must see the case's real var, not `input: None`.

A promptfoo suite whose vars are not named `input` (question / query / …) used to
feed the LLM judge a missing input. These tests build a config through the
adapter, stub the judge to capture what it received, and assert the real input
reaches it — and that an existing `input` var is left untouched.
"""

from muteval import checks
from muteval.adapters.promptfoo import config_from_promptfoo_dict
from muteval.evals import EvalOutcome


def _config_with_var(var_name, value, monkeypatch):
    """Build a promptfoo config with one llm-rubric assertion over a single var,
    with the judge stubbed to record the `input` it was given."""
    seen = {}

    def fake_llm_judge(rubric, **kw):
        def _e(output, case):
            seen["input"] = checks._case_get(case, "input")
            return EvalOutcome(passed=True, name="llm_judge")

        return _e

    monkeypatch.setattr(checks, "llm_judge", fake_llm_judge)
    data = {
        "prompts": [f"Answer: {{{{{var_name}}}}}"],
        "providers": ["openai:gpt-4o-mini"],
        "tests": [
            {
                "vars": {var_name: value},
                "assert": [{"type": "llm-rubric", "value": "the answer is correct"}],
            }
        ],
    }
    config = config_from_promptfoo_dict(data)
    idx = config.eval_names.index("promptfoo:llm-rubric")
    config.evals[idx]("some answer", config.cases[0])  # drive the judge path
    return seen


def test_llm_rubric_sees_nonstandard_var(monkeypatch):
    seen = _config_with_var("question", "What is the capital of France?", monkeypatch)
    assert seen["input"] is not None
    assert "What is the capital of France?" in seen["input"]


def test_llm_rubric_preserves_existing_input_var(monkeypatch):
    seen = _config_with_var("input", "the raw input text", monkeypatch)
    # An existing `input` var is passed through unchanged, not re-synthesized.
    assert seen["input"] == "the raw input text"
