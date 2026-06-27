"""muteval — a real RAGAS project's suite, replicated with ZERO install.

Modeled on benitomartin/rag-langchain-ragas (a LangChain + FAISS + RAGAS Q&A
pipeline over HuggingFace's LLM-Hallucinations Leaderboard). That repo grades
with faithfulness, answer_relevancy, context_precision, context_recall, and
answer_correctness.

We reuse the three ANSWER-dependent metrics — Faithfulness, AnswerRelevancy,
AnswerCorrectness — because a *prompt* mutation can only move those; the two
context_* metrics grade retrieval, which is fixed here (switch to System-mode +
context operators to exercise those). Graded via the OpenAI REST API directly
(stdlib urllib), so it runs without installing ragas.

When you can install ragas, the LITERAL-adapter version is at
validation/ragas_rag/.

Setup:  pip install certifi ; $env:OPENAI_API_KEY="sk-..."
Run:    muteval run --config validation/ragas_style_rag/muteval_config.py
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


PROMPT = """You are a question-answering assistant for the LLM Hallucinations
Leaderboard documentation. Answer the user's question using ONLY the provided
context. Be accurate and concise, ground every statement in the context, cite
the source, and if the context does not contain the answer, say you don't know
rather than guessing.
"""


def run(prompt, case):
    ctx = "\n\n---\n\n".join(case["context"])
    user = "Context:\n" + ctx + "\n\nQuestion: " + case["question"] + "\nAnswer:"
    return _chat([{"role": "system", "content": prompt}, {"role": "user", "content": user}])


def _score(rubric, question, context, answer, reference=None):
    ref = ("\n\nReference (ground truth): " + reference) if reference else ""
    p = (
        "You are a strict evaluator grading an AI assistant's answer.\n"
        "Rubric: " + rubric + "\n\nContext:\n" + "\n".join(context) +
        "\n\nQuestion: " + question + "\nAnswer: " + answer + ref +
        "\n\nRespond with ONLY an integer from 0 to 10 (10 = perfect). No words."
    )
    text = _chat([{"role": "user", "content": p}], model=JUDGE_MODEL).strip()
    nums = re.findall(r"\d+", text)
    return max(0, min(10, int(nums[-1]) if nums else 0)) / 10.0


def _judge(name, rubric, use_reference=False):
    def _eval(output, case):
        ref = case.get("expected") if use_reference else None
        s = _score(rubric, case["question"], case["context"], output, ref)
        return EvalOutcome(passed=s >= THRESHOLD, score=s, threshold=THRESHOLD, name=name)
    _eval.__name__ = name
    return _eval


config = MutEvalConfig(
    prompt=PROMPT,
    cases=[
        {
            "question": "What are hallucinations in the context of LLM models?",
            "context": [
                "Hallucinations refer to instances where a language model generates "
                "content that does not align with real-world facts or the user's input. "
                "source: hallucinations-leaderboard",
                "The Hallucinations Leaderboard evaluates and compares LLMs by their "
                "tendency to generate hallucinated content. source: hallucinations-leaderboard",
            ],
            "expected": "Hallucinations are when an LLM generates content not aligned "
            "with real-world facts or the user's input; the leaderboard compares models "
            "by how often they hallucinate.",
        },
        {
            "question": "What is the purpose of the Hallucinations Leaderboard?",
            "context": [
                "The Hallucinations Leaderboard is an open effort to evaluate and compare "
                "LLMs based on their tendency to hallucinate. source: hallucinations-leaderboard",
                "It provides insights into the generalization properties and limitations "
                "of models to support more reliable language generators. "
                "source: hallucinations-leaderboard",
            ],
            "expected": "To evaluate and compare LLMs by their tendency to hallucinate, "
            "giving insight into their limitations and supporting more reliable models.",
        },
    ],
    run=run,
    evals=[
        _judge("Faithfulness",
               "Is every claim in the answer factually consistent with the provided "
               "context, with nothing invented? Full marks if fully grounded."),
        _judge("AnswerRelevancy",
               "How pertinent and complete is the answer to the question? Deduct for "
               "incomplete or redundant answers."),
        _judge("AnswerCorrectness",
               "How well does the answer align with the reference ground truth in "
               "accuracy and completeness? Full marks for a close match.",
               use_reference=True),
    ],
    eval_names=["Faithfulness", "AnswerRelevancy", "AnswerCorrectness"],
    runs_per_mutant=1,
)
