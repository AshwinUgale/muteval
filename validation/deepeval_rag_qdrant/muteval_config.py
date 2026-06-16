"""muteval validation run #1 — against the deepeval RAG example.

Target: confident-ai/deepeval -> examples/rag_evaluation/rag_evaluation_with_qdrant.py
We reuse that example's real system prompt and real RAG metrics. muteval mutates
the prompt and reruns the suite to see which regressions the suite would miss.

Two design notes, both learned the hard way on the first run:

1. We use only the two ANSWER-dependent metrics (AnswerRelevancy, Faithfulness).
   The three Contextual* metrics grade *retrieval*, which is fixed here — so by
   our own thesis they can't catch a prompt mutation, and including them only
   adds baseline noise and cost.
2. async_mode=False forces deepeval's SYNC path. Its async path hangs on some
   Windows setups (httpx-async TLS read never returns); the sync path works.

Setup:
    pip install "muteval[deepeval,examples]"
    export OPENAI_API_KEY=sk-...

Run:
    muteval run --config validation/deepeval_rag_qdrant/muteval_config.py --max-mutants 8
"""

import os

from muteval import MutEvalConfig

MODEL = os.environ.get("MUTEVAL_EXAMPLE_MODEL", "gpt-4o-mini")
# deepeval's judge defaults to gpt-4o (~16x pricier); pin it to mini.
JUDGE_MODEL = os.environ.get("MUTEVAL_JUDGE_MODEL", "gpt-4o-mini")

# --- The exact system prompt from the deepeval example (the mutation target) -
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


def run(prompt: str, case: dict) -> str:
    """Generate an answer from the (mutated) prompt + retrieved context."""
    from openai import OpenAI

    client = OpenAI()
    context_block = "\n\n---\n\n".join(case["context"])
    user = f"Context:\n{context_block}\n\nQuestion: {case['question']}\nAnswer:"
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def _build_evals():
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

    from muteval.adapters.deepeval import metrics_to_evals

    metrics = [
        AnswerRelevancyMetric(model=JUDGE_MODEL, async_mode=False),
        FaithfulnessMetric(model=JUDGE_MODEL, async_mode=False),
    ]
    evals = metrics_to_evals(
        metrics,
        input_key="question",
        retrieval_context_key="context",
    )
    names = [type(m).__name__ for m in metrics]
    return evals, names


_evals, _names = _build_evals()

# --- Airtight, answerable cases so the BASELINE passes ----------------------
config = MutEvalConfig(
    prompt=PROMPT,
    cases=[
        {
            "question": "What port does the server listen on by default?",
            "context": [
                "The server listens on port 8080 by default. source: config/server.md",
                "Set the PORT environment variable to override the default port. "
                "source: config/server.md",
            ],
            "expected": "It listens on port 8080 by default; override it with the PORT env var.",
        },
        {
            "question": "How do I rotate API keys?",
            "context": [
                "API keys are rotated from the dashboard under Settings > Keys. "
                "source: security/keys.md",
                "Rotating a key immediately invalidates the previous key. "
                "source: security/keys.md",
            ],
            "expected": "Rotate API keys from the dashboard under Settings > Keys; "
            "the previous key is invalidated immediately.",
        },
    ],
    run=run,
    evals=_evals,
    eval_names=_names,
    runs_per_mutant=1,
)
