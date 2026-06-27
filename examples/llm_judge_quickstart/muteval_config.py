"""60-second quickstart — real model, ZERO extra installs.

Everything here needs only:  pip install muteval   (+ OPENAI_API_KEY)

muteval is pure Python with no dependencies. The only boilerplate is `run()`,
because muteval is model-agnostic: you tell it how to call YOUR system. Grading
uses the built-in `checks` — no deepeval, no ragas, no openai SDK, no Rust.

The default eval here is a deterministic check (rock-solid baseline). An
LLM-as-judge example is included, commented out — `checks.llm_judge` calls the
model through the standard library, so it needs nothing extra either.

Run:
    pip install muteval certifi          # certifi only if your SSL store is bare
    export OPENAI_API_KEY=sk-...         # PowerShell: $env:OPENAI_API_KEY="sk-..."
    muteval run --config examples/llm_judge_quickstart/muteval_config.py
"""

import json
import os
import ssl
import urllib.request

from muteval import MutEvalConfig
from muteval import checks

MODEL = os.environ.get("MUTEVAL_EXAMPLE_MODEL", "gpt-4o-mini")

try:
    import certifi

    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL = ssl.create_default_context()


def run(prompt, case):
    """Call YOUR system. Here: a tiny stdlib OpenAI call (swap for your app)."""
    body = json.dumps(
        {
            "model": MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "Context:\n"
                    + "\n".join(case["context"])
                    + "\n\nQuestion: "
                    + case["question"],
                },
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as resp:
        return json.load(resp)["choices"][0]["message"]["content"] or ""


config = MutEvalConfig(
    prompt=(
        "Answer the question using only the provided context. Be concise, and "
        "if the answer is not in the context, say you don't know."
    ),
    cases=[
        {
            "question": "What port does the server use by default?",
            "context": ["The server listens on port 8080 by default."],
            "must_contain": "8080",
        },
        {
            "question": "Where are API keys rotated?",
            "context": ["API keys are rotated from the dashboard under Settings > Keys."],
            "must_contain": "Settings",
        },
    ],
    run=run,
    evals=[
        # Deterministic, always green on a correct answer -> reliable baseline.
        checks.contains_case("must_contain"),
        # Want LLM-as-judge grading too? Uncomment — uses the built-in stdlib
        # judge (no extra installs), threshold kept lenient for a clean baseline:
        # checks.llm_judge(
        #     "the answer is grounded in the provided context",
        #     input_key="question", threshold=0.5,
        # ),
    ],
    eval_names=["contains_expected"],
)
