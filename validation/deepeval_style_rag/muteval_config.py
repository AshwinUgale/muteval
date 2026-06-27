"""muteval — deepeval's RAG suite, replicated with ZERO install.

This mirrors deepeval's own RAG example (AnswerRelevancy + Faithfulness — the two
metrics deepeval's getting-started RAG suite ships with), but grades via the
OpenAI REST API directly (stdlib urllib), so it runs without installing deepeval
or compiling pydantic-core/jiter.

When you can install deepeval, the LITERAL-adapter version (using deepeval's real
metric objects) is at validation/deepeval_rag_qdrant/.

Setup:  pip install certifi ; $env:OPENAI_API_KEY="sk-..."
Run:    muteval run --config validation/deepeval_style_rag/muteval_config.py
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
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    body = json.dumps({"model": model, "temperature": temperature, "messages": messages}).encode("utf-8")
    req = urllib.request.Request(
        _ENDPOINT, data=body,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
        return json.load(resp)["choices"][0]["message"]["content"] or ""


# deepeval's example RAG system prompt.
PROMPT = """You're assisting a user who has a question based on the documentation.
Your goal is to provide a clear and concise response that addresses their query
while referencing relevant information from the documentation.
Remember to:
Understand the user's question thoroughly.
If the user's query is general (e.g., "hi," "good morning"), greet them normally
and avoid using the context from the documentation.
If the user's query is specific and related to the documentation, locate and
extract the pertinent information.
Craft a response that directly addresses the user's query and provides accurate
information referring the relevant source from the 'source' field of the fetched
context to support your answer.
Use a friendly and professional tone in your response.
If you cannot find the answer in the provided context, do not pretend to know it.
Instead, respond with "I don't know".
"""


def run(prompt, case):
    ctx = "\n\n---\n\n".join(case["context"])
    user = "Context:\n" + ctx + "\n\nQuestion: " + case["question"] + "\nAnswer:"
    return _chat([{"role": "system", "content": prompt}, {"role": "user", "content": user}])


def _score(rubric, question, context, answer):
    p = (
        "You are a strict evaluator grading an AI assistant's answer.\n"
        "Rubric: " + rubric + "\n\nDocumentation:\n" + "\n".join(context) +
        "\n\nQuestion: " + question + "\nAnswer: " + answer +
        "\n\nRespond with ONLY an integer from 0 to 10 (10 = perfect). No words."
    )
    text = _chat([{"role": "user", "content": p}], model=JUDGE_MODEL).strip()
    nums = re.findall(r"\d+", text)
    return max(0, min(10, int(nums[-1]) if nums else 0)) / 10.0


def _judge(name, rubric):
    def _eval(output, case):
        s = _score(rubric, case["question"], case["context"], output)
        return EvalOutcome(passed=s >= THRESHOLD, score=s, threshold=THRESHOLD, name=name)
    _eval.__name__ = name
    return _eval


config = MutEvalConfig(
    prompt=PROMPT,
    cases=[
        {
            "question": "What port does the server listen on by default?",
            "context": [
                "The server listens on port 8080 by default. source: config/server.md",
                "Set the PORT environment variable to override the default port. source: config/server.md",
            ],
        },
        {
            "question": "How do I rotate API keys?",
            "context": [
                "API keys are rotated from the dashboard under Settings > Keys. source: security/keys.md",
                "Rotating a key immediately invalidates the previous key. source: security/keys.md",
            ],
        },
    ],
    run=run,
    evals=[
        _judge("AnswerRelevancyMetric",
               "How relevant and complete is the answer to the user's question? "
               "Full marks for a direct, complete answer; deduct for incomplete "
               "or redundant content."),
        _judge("FaithfulnessMetric",
               "Is every claim in the answer factually supported by the "
               "documentation, with nothing invented? Full marks if fully "
               "grounded; deduct for any unsupported claim."),
    ],
    eval_names=["AnswerRelevancyMetric", "FaithfulnessMetric"],
    runs_per_mutant=1,
)
