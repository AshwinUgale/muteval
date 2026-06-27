"""muteval validation — against RAGAS metrics on a small RAG suite.

Mirror of validation/deepeval_rag_qdrant/, but reusing **RAGAS** metrics via
``muteval.adapters.ragas``. muteval mutates the prompt (and, in system mode, the
retrieved context) and reruns the RAGAS-graded suite to see which regressions
the suite would miss.

RAGAS metrics return a raw [0, 1] score with no built-in pass/fail threshold, so
the adapter applies one (``threshold=``). Because the adapter forwards the score
and threshold to muteval, survivors are reported with their near-miss margin.

Setup:
    pip install "muteval[ragas]" langchain-openai
    export OPENAI_API_KEY=sk-...

Run:
    muteval run --config validation/ragas_rag/muteval_config.py --max-mutants 8

NOTE: RAGAS's public API has moved across versions. This targets the
``SingleTurnSample`` / ``single_turn_score`` interface (ragas >= 0.2) and an
LLM-backed metric. If your version differs, the adapter accepts ``sample_factory``
and ``score_fn`` overrides so you can adapt the wiring without changing muteval.
"""

import os

from muteval import MutEvalConfig

MODEL = os.environ.get("MUTEVAL_EXAMPLE_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("MUTEVAL_JUDGE_MODEL", "gpt-4o-mini")

# --- The system prompt (the mutation target) --------------------------------
PROMPT = """You are answering a user's question using the provided documentation.
Provide a clear, concise answer grounded in the documentation.
Remember to:
If the question is general (e.g. "hi"), greet the user and do not use the docs.
If the question is specific, locate the pertinent information in the context.
Cite the relevant source from the context to support your answer.
Use a friendly, professional tone.
If the answer is not in the context, say "I don't know" — do not invent facts.
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
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from langchain_openai import ChatOpenAI

    from muteval.adapters.ragas import metrics_to_evals

    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    metrics = [Faithfulness(llm=llm), ResponseRelevancy(llm=llm)]
    evals = metrics_to_evals(
        metrics,
        threshold=0.7,
        input_key="question",
        retrieval_context_key="context",
        reference_key="expected",
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
            "expected": "It listens on port 8080 by default; override it with PORT.",
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
