"""muteval validation — real OpenAI model, ZERO heavy dependencies.

Calls the OpenAI REST API directly with Python's standard library (urllib) — no
openai SDK, no deepeval/ragas, no jiter/pydantic-core/Rust build.

Story this config tells:
  * faithfulness + relevancy alone catch ~none of the prompt regressions, because
    the answerable cases never force the model off the rails.
  * Adding an UNANSWERABLE case + an `abstains_when_unanswerable` check exercises
    the "if it's not in the context, say I don't know — do not invent facts"
    guardrail, so mutations that remove/invert it get KILLED. The mutation score
    rises — exactly the "write an eval to kill the survivor" loop.

The LLM judges use a 0-10 INTEGER scale (better calibrated than a 0-1 float),
normalized to [0, 1].

Setup:
    pip install certifi
    $env:OPENAI_API_KEY="sk-..."

Run:
    muteval run --config validation/openai_judge_rag/muteval_config.py
"""

import json
import os
import re
import ssl
import urllib.request

from muteval import EvalOutcome, MutEvalConfig

MODEL = os.environ.get("MUTEVAL_EXAMPLE_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("MUTEVAL_JUDGE_MODEL", "gpt-4o-mini")
THRESHOLD = float(os.environ.get("MUTEVAL_THRESHOLD", "0.7"))
_ENDPOINT = "https://api.openai.com/v1/chat/completions"

try:
    import certifi

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL_CTX = ssl.create_default_context()


def _chat(messages, model=MODEL, temperature=0.0):
    """Minimal OpenAI chat call using only the standard library."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    body = json.dumps(
        {"model": model, "temperature": temperature, "messages": messages}
    ).encode("utf-8")
    req = urllib.request.Request(
        _ENDPOINT,
        data=body,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"] or ""


PROMPT = """You are answering a user's question using the provided documentation.
Provide a clear, concise answer grounded in the documentation.
Remember to:
If the question is general (e.g. "hi"), greet the user and do not use the docs.
If the question is specific, locate the pertinent information in the context.
Cite the relevant source from the context to support your answer.
Use a friendly, professional tone.
If the answer is not in the context, say "I don't know" — do not invent facts.
"""


def run(prompt, case):
    context_block = "\n\n---\n\n".join(case["context"])
    user = "Context:\n" + context_block + "\n\nQuestion: " + case["question"] + "\nAnswer:"
    return _chat(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ]
    )


# --- Deterministic guardrail check (free — no API call) ----------------------

_ABSTAIN_PHRASES = (
    "don't know", "do not know", "not in the", "isn't in", "is not in",
    "no information", "cannot find", "couldn't find", "not available",
    "not provided", "not mentioned", "unable to",
)


def abstains_when_unanswerable(output, case):
    """For unanswerable cases the assistant MUST abstain (the guardrail). For
    answerable cases this is N/A and passes."""
    if not case.get("unanswerable"):
        return EvalOutcome(passed=True, name="abstains_when_unanswerable")
    low = output.lower()
    abstained = any(p in low for p in _ABSTAIN_PHRASES)
    return EvalOutcome(passed=abstained, name="abstains_when_unanswerable")


# --- LLM-as-judge checks (0-10 integer scale) --------------------------------


def _score_0_10(rubric, question, context, answer):
    judge_prompt = (
        "You are a strict evaluator grading an AI assistant's answer.\n"
        "Rubric: " + rubric + "\n\n"
        "Documentation:\n" + "\n".join(context) + "\n\n"
        "Question: " + question + "\n"
        "Answer: " + answer + "\n\n"
        "Respond with ONLY an integer from 0 to 10 (10 = perfect). No words."
    )
    text = _chat([{"role": "user", "content": judge_prompt}], model=JUDGE_MODEL).strip()
    nums = re.findall(r"\d+", text)
    n = int(nums[-1]) if nums else 0
    return max(0, min(10, n)) / 10.0


def _judge_eval(name, rubric):
    def _eval(output, case):
        score = _score_0_10(rubric, case["question"], case["context"], output)
        return EvalOutcome(
            passed=score >= THRESHOLD, score=score, threshold=THRESHOLD, name=name
        )

    _eval.__name__ = name
    return _eval


config = MutEvalConfig(
    prompt=PROMPT,
    cases=[
        {
            "question": "What port does the server listen on by default?",
            "context": [
                "The server listens on port 8080 by default. source: config/server.md",
                "Set the PORT environment variable to override it. source: config/server.md",
            ],
            "expected": "Port 8080 by default; override with the PORT env var.",
        },
        {
            "question": "How do I rotate API keys?",
            "context": [
                "API keys are rotated from the dashboard under Settings > Keys. source: security/keys.md",
                "Rotating a key immediately invalidates the previous key. source: security/keys.md",
            ],
            "expected": "Rotate from Settings > Keys; the old key is invalidated immediately.",
        },
        {
            # UNANSWERABLE: the docs say nothing about a request timeout.
            "question": "What is the default request timeout in seconds?",
            "context": [
                "The server listens on port 8080 by default. source: config/server.md",
                "API keys are rotated from the dashboard under Settings > Keys. source: security/keys.md",
            ],
            "expected": "I don't know — it's not in the documentation.",
            "unanswerable": True,
        },
    ],
    run=run,
    # `abstains_when_unanswerable` runs first and is FREE, so mutants that break
    # the guardrail short-circuit before the paid judges.
    evals=[
        abstains_when_unanswerable,
        _judge_eval(
            "faithfulness_judge",
            "Every factual claim in the answer is directly supported by the "
            "documentation, and nothing is invented. Full marks if all claims "
            "are grounded; deduct for any unsupported or invented claim. If the "
            "documentation lacks the answer, correctly saying so is full marks.",
        ),
        _judge_eval(
            "relevancy_judge",
            "The answer appropriately responds to the question given the "
            "documentation. If the documentation does not contain the answer, "
            "correctly stating that (e.g. 'I don't know') is fully appropriate "
            "and earns full marks. Deduct for off-topic or evasive replies.",
        ),
    ],
    eval_names=["abstains_when_unanswerable", "faithfulness_judge", "relevancy_judge"],
    runs_per_mutant=1,
)
