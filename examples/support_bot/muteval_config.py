"""A fully runnable muteval example — no API keys required.

This uses a deterministic *mock* model so you can see muteval work end to end:

    muteval run --config examples/support_bot/muteval_config.py

The mock "model" reads the prompt and decides how to behave based on which
instructions are present. That lets us demonstrate the core insight: a prompt
mutation only gets *killed* if there's an eval that actually checks the
behavior the mutation broke.

Swap `mock_model` for a real call (OpenAI/Anthropic/your app) and the exact
same config grades your real eval suite.
"""

from muteval import MutEvalConfig

# ---------------------------------------------------------------------------
# The prompt under test
# ---------------------------------------------------------------------------
PROMPT = """You are a support assistant for an online store.
- You must always cite the order ID when discussing an order.
- You must never reveal another customer's data.
- Do not promise refunds; refunds require manager approval.
- Always reply in a polite, professional tone.
"""


# ---------------------------------------------------------------------------
# A deterministic mock "model". In a real config this calls your LLM/app.
# Its behavior depends on which instructions survive in the (mutated) prompt.
# ---------------------------------------------------------------------------
def mock_model(prompt: str, case: dict) -> str:
    p = prompt.lower()
    user = case["input"]
    reply_parts = []

    # Cites the order id only if the prompt still strongly requires it.
    if "must always cite the order id" in p or "must cite the order id" in p:
        reply_parts.append(f"Regarding order {case['order_id']}:")

    # Promises a refund unless strongly told not to.
    if "do not promise refunds" in p:
        reply_parts.append("I can't promise a refund; a manager must approve it.")
    else:
        reply_parts.append("Sure, I'll refund you right away!")

    reply_parts.append(f"(You said: {user})")
    return " ".join(reply_parts)


# ---------------------------------------------------------------------------
# The eval suite being graded. NOTE: deliberately INCOMPLETE so you see a
# survivor — there is no eval for the refund-promise behavior.
# ---------------------------------------------------------------------------
def cites_order_id(output: str, case: dict) -> bool:
    """Eval: the reply must reference the order ID."""
    return str(case["order_id"]) in output


def is_nonempty(output: str, case: dict) -> bool:
    """Eval: the reply must not be empty."""
    return bool(output.strip())


# (Intentionally missing: an eval that checks we never promise refunds.
#  muteval will surface that gap as a SURVIVOR.)


config = MutEvalConfig(
    prompt=PROMPT,
    cases=[
        {"input": "Where is my package?", "order_id": "A123"},
        {"input": "I want my money back", "order_id": "B456"},
    ],
    run=mock_model,
    evals=[cites_order_id, is_nonempty],
    eval_names=["cites_order_id", "is_nonempty"],
    runs_per_mutant=1,
)
