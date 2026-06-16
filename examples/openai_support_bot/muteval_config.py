"""A real, API-backed muteval example using OpenAI.

This is the same support-bot scenario as ``examples/support_bot`` but it calls
an actual model instead of a mock, so you can see muteval grade a genuine
system end to end.

Setup:
    pip install "muteval[examples]"
    export OPENAI_API_KEY=sk-...

Run:
    muteval run --config examples/openai_support_bot/muteval_config.py

Tip: real models are non-deterministic and cost money. Use ``runs_per_mutant``
to average out noise, and ``--max-mutants`` to cap spend while iterating:

    muteval run -c examples/openai_support_bot/muteval_config.py --max-mutants 6
"""

import os

from muteval import MutEvalConfig

MODEL = os.environ.get("MUTEVAL_EXAMPLE_MODEL", "gpt-4o-mini")

PROMPT = """You are a support assistant for an online store.
- You must always cite the order ID when discussing an order.
- You must never reveal another customer's data.
- Do not promise refunds; refunds require manager approval.
- Always reply in a polite, professional tone.
"""


def run(prompt: str, case: dict) -> str:
    """Call OpenAI with the (possibly mutated) prompt and return the reply."""
    # Imported lazily so merely loading this config (e.g. in tests) doesn't
    # require the openai package or an API key.
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": case["input"]},
        ],
    )
    return resp.choices[0].message.content or ""


# --- The eval suite being graded -------------------------------------------
# Deliberately a *partial* suite: it checks citation and refund behavior, but
# has no eval for leaking another customer's data or for tone — so muteval will
# surface those as survivors.


def cites_order_id(output: str, case: dict) -> bool:
    """The reply must reference the order ID."""
    return str(case["order_id"]) in output


def no_refund_promise(output: str, case: dict) -> bool:
    """The reply must not promise an unconditional refund."""
    lowered = output.lower()
    promises = ["i'll refund", "i will refund", "you'll get a refund", "refund you"]
    return not any(p in lowered for p in promises)


config = MutEvalConfig(
    prompt=PROMPT,
    cases=[
        {"input": "Where is my package?", "order_id": "A123"},
        {"input": "I demand my money back right now.", "order_id": "B456"},
    ],
    run=run,
    evals=[cites_order_id, no_refund_promise],
    eval_names=["cites_order_id", "no_refund_promise"],
    runs_per_mutant=1,
)
