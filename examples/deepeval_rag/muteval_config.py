"""Grade a deepeval RAG suite with muteval.

This shows the headline Tier-2 feature: reuse the deepeval metrics you've
already written, instead of rewriting them as bool functions.

Setup:
    pip install "muteval[deepeval,examples]"
    export OPENAI_API_KEY=sk-...      # deepeval's judge + the system below

Run:
    muteval run --config examples/deepeval_rag/muteval_config.py --max-mutants 6

What happens: muteval mutates the system prompt, re-runs `run()` to get a fresh
answer for each mutant, and scores it with your deepeval metrics. A mutant
"survives" if the metrics still pass despite the degraded prompt — a gap in
your suite.
"""

import os

from muteval import MutEvalConfig

MODEL = os.environ.get("MUTEVAL_EXAMPLE_MODEL", "gpt-4o-mini")

PROMPT = """You are a documentation assistant. Answer using ONLY the provided
context. You must cite the context you used. If the context does not contain
the answer, say "I don't know" — do not guess.
"""


def run(prompt: str, case: dict) -> str:
    """Answer the question using the retrieved context (the system under test)."""
    from openai import OpenAI

    client = OpenAI()
    context_block = "\n\n".join(case["context"])
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Context:\n{context_block}\n\nQuestion: {case['question']}",
            },
        ],
    )
    return resp.choices[0].message.content or ""


def _build_evals():
    """Wrap existing deepeval metrics as muteval evals via the adapter."""
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

    from muteval.adapters.deepeval import metrics_to_evals

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
    ]
    evals = metrics_to_evals(
        metrics,
        input_key="question",
        retrieval_context_key="context",
    )
    names = [type(m).__name__ for m in metrics]
    return evals, names


_evals, _names = _build_evals()

config = MutEvalConfig(
    prompt=PROMPT,
    cases=[
        {
            "question": "What port does the service listen on by default?",
            "context": [
                "The service starts an HTTP server on port 8080 by default.",
                "Set the PORT env var to override the default port.",
            ],
        },
    ],
    run=run,
    evals=_evals,
    eval_names=_names,
    runs_per_mutant=1,
)
