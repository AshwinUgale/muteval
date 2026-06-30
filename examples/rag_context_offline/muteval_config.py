"""Offline demo of CONTEXT mutation (System mode) — no API key.

Shows muteval mutating the *retrieved context* (not just the prompt) and a weak
eval suite failing to notice. The "model" is a deterministic mock that answers
from whatever context it's given, so corrupting/dropping a doc changes the
answer — and the suite, which only checks that an answer was produced, stays
green. That's the RAG blind spot: your evals don't verify the retrieval was
correct.

Run:
    muteval run --config examples/rag_context_offline/muteval_config.py \
      --operators corrupt_context_doc swap_context_doc drop_context_doc clear_context
"""

from muteval import MutEvalConfig, System, checks

SYSTEM = System(
    prompt="Answer the question using ONLY the retrieved context, and cite the source.",
    context=(
        "The API server listens on port 8080 by default. source: config/server.md",
        "Set the PORT environment variable to override the default. source: config/server.md",
    ),
    model="gpt-4o-mini",
)


def run(system, case):
    """Mock RAG: answer straight from the (possibly mutated) retrieved context."""
    docs = system.context or ("",)
    # Record what context was actually used, so an eval *could* grade against it.
    case["used_context"] = list(docs)
    return f"Per the documentation: {docs[0]}"


# A DELIBERATELY WEAK suite: it checks that an answer came back and mentions a
# source — but never verifies the retrieved fact was correct. So a corrupted or
# dropped doc sails through.
config = MutEvalConfig(
    system=SYSTEM,
    cases=[{"question": "What port does the server use by default?"}],
    run=run,
    evals=[
        checks.not_contains("ERROR"),
        checks.contains("source"),
    ],
    eval_names=["answered", "mentions_source"],
)
