"""Validation run #2 — CONTEXT + MODEL mutation against deepeval's RAG metrics.

The stronger story than prompt-only mutation: muteval corrupts the *retrieved
context* and *downgrades the model*, then asks whether deepeval's RAG suite
notices. It mostly won't — and that's the point.

Why Faithfulness can't catch a poisoned retrieval: Faithfulness checks whether
the answer is grounded in the context the system was GIVEN. If the retriever
returns a subtly wrong doc, the answer is faithful *to that wrong doc* — so the
metric passes a confidently incorrect answer. We grade Faithfulness against the
exact (mutated) context the system used (`used_context`), which is what a real
production eval does.

Setup:
    pip install "muteval[deepeval,examples]"
    export OPENAI_API_KEY=sk-...

Run (focus on the high-value operators):
    muteval run --config validation/deepeval_rag_system/muteval_config.py \
      --operators corrupt_context_doc swap_context_doc drop_context_doc downgrade_model \
      --fail-on-severity high
"""

import os

from muteval import MutEvalConfig, System

JUDGE_MODEL = os.environ.get("MUTEVAL_JUDGE_MODEL", "gpt-4o-mini")

PROMPT = (
    "You are a documentation assistant. Answer using ONLY the provided context, "
    "and cite the source. If the context does not contain the answer, say "
    '"I don\'t know" — do not guess.'
)

SYSTEM = System(
    prompt=PROMPT,
    context=(
        "The API server listens on port 8080 by default. source: config/server.md",
        "Set the PORT environment variable to override the default port. "
        "source: config/server.md",
    ),
    model=os.environ.get("MUTEVAL_EXAMPLE_MODEL", "gpt-4o-mini"),
)


def run(system, case):
    """Generate an answer from the (mutated) context + model, and record the
    exact context used so the eval grades against what the system actually saw."""
    from openai import OpenAI

    client = OpenAI()
    docs = system.context or ()
    case["used_context"] = list(docs)  # what Faithfulness should grade against
    context_block = "\n\n---\n\n".join(docs)
    user = f"Context:\n{context_block}\n\nQuestion: {case['question']}\nAnswer:"
    resp = client.chat.completions.create(
        model=system.model or "gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system.prompt},
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
        retrieval_context_key="used_context",  # the MUTATED context the system saw
    )
    return evals, [type(m).__name__ for m in metrics]


_evals, _names = _build_evals()

config = MutEvalConfig(
    system=SYSTEM,
    cases=[
        {"question": "What port does the server listen on by default?"},
        {"question": "How do I change the port?"},
    ],
    run=run,
    evals=_evals,
    eval_names=_names,
    runs_per_mutant=1,  # deepeval is too flaky here for 3x; get a clean run first
)
