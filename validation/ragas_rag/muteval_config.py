"""muteval validation — REAL RAGAS metrics (Faithfulness, ResponseRelevancy).

Works around ragas issue #2741: current ragas hard-imports
`langchain_community.chat_models.vertexai`, a module recent langchain-community
no longer ships, so `import ragas` crashes. We don't use VertexAI (we use
OpenAI), so we register a tiny shim for that module path BEFORE importing ragas.
This makes ragas import cleanly without pinning to ancient langchain versions.

Setup (Colab or any Linux box with wheels — NOT a wheel-hostile local Python):
    pip install ragas langchain-openai certifi
    export OPENAI_API_KEY=sk-...

Run:
    muteval run --config validation/ragas_rag/muteval_config.py --max-mutants 6
"""

import json
import os
import ssl
import sys
import types
import urllib.request

from muteval import MutEvalConfig

MODEL = os.environ.get("MUTEVAL_EXAMPLE_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("MUTEVAL_JUDGE_MODEL", "gpt-4o-mini")
THRESHOLD = float(os.environ.get("MUTEVAL_THRESHOLD", "0.7"))


def _install_vertexai_shim() -> None:
    """Satisfy ragas's hard `langchain_community.chat_models.vertexai` import
    (ragas #2741) so `import ragas` doesn't crash. We never use VertexAI."""
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        __import__(name)
        return  # already importable; nothing to do
    except Exception:
        pass
    shim = types.ModuleType(name)
    try:
        from langchain_google_vertexai import ChatVertexAI  # type: ignore
    except Exception:  # noqa: BLE001
        class ChatVertexAI:  # minimal stub; unused, just needs to be importable
            pass
    shim.ChatVertexAI = ChatVertexAI
    sys.modules[name] = shim


_install_vertexai_shim()

try:
    import certifi

    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL = ssl.create_default_context()


PROMPT = """You are answering a user's question using the provided documentation.
Provide a clear, concise answer grounded in the documentation.
Cite the relevant source from the context to support your answer.
If the answer is not in the context, say "I don't know" — do not invent facts.
"""


def run(prompt, case):
    """Generate the answer (stdlib OpenAI call)."""
    body = json.dumps(
        {
            "model": MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "Context:\n"
                    + "\n\n---\n\n".join(case["context"])
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


def _build_evals():
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy

    from muteval.adapters.ragas import metrics_to_evals

    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    metrics = [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=emb)]
    evals = metrics_to_evals(
        metrics,
        threshold=THRESHOLD,
        input_key="question",
        retrieval_context_key="context",
        reference_key="expected",
    )
    names = [type(m).__name__ for m in metrics]
    return evals, names


_evals, _names = _build_evals()

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
    ],
    run=run,
    evals=_evals,
    eval_names=_names,
    runs_per_mutant=1,
)
