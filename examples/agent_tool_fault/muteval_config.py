"""Offline, keyless demo: muteval + tracelint catch a SILENT tool fault.

A naive payments agent charges a card and always reports success. muteval's
`deny_tool_output` operator makes the charge come back DECLINED — a 200 whose
body says it failed. The agent proceeds anyway, so the final answer still reads
"successful", and the user's semantic eval is blind to it. Only a structural
trace check carrying the tool's declared failure contract (tracelint's
`failure_when`) catches the declined charge.

Run it (shows the [HIGH] survivor with the semantic eval ALONE):

    muteval run --config examples/agent_tool_fault/muteval_config.py --no-color

Then see the fix kill it (0% -> 100%):

    python examples/agent_tool_fault/run_demo.py     # needs: pip install muteval[tracelint]
"""

import json

from muteval import MutEvalConfig, System, checks

# The tool's declared failure contract (a tools.json-shaped registry). A charge
# whose body says {"status": "declined"} is a failure even though transport
# returned 200 — the case structured-error detection is blind to.
TOOL_REGISTRY = {
    "tools": {
        "charge_card": {
            "metadata": {
                "side_effecting": True,
                "failure_when": {"pointer": "/status", "in": ["declined", "failed"]},
            }
        }
    }
}

# The system under test: a payments agent whose one tool output is charge_card's
# result. muteval mutates that output via `deny_tool_output`.
SYSTEM = System(
    prompt="You are a payments assistant. Charge the customer, then confirm the charge.",
    tools=('{"status": "succeeded", "receipt": "R1"}',),
)

CASES = [{"input": "Charge $50 for order A100.", "order_id": "A100", "amount": 50}]


def run(system, case):
    """A deliberately NAIVE agent: it calls charge_card and ALWAYS reports success
    without checking the result. Returns the {"final","trace"} bridge — `.final`
    is what the user's evals grade, `.trace` is what tracelint lints. The
    naive-proceed is the self-contained source of the silent failure (no tracelint
    agent loop, no API key)."""
    raw = system.tools[0] if system.tools else '{"status": "succeeded"}'
    result = json.loads(raw) if isinstance(raw, str) else raw
    final = "Your payment was successful."
    trace = {
        "run_id": "r",
        "steps": [
            {"type": "message", "role": "user", "content": case["input"]},
            {
                "type": "tool_call",
                "call_id": "c1",
                "name": "charge_card",
                "args": {"order_id": case["order_id"], "amount": case["amount"]},
            },
            {"type": "tool_result", "call_id": "c1", "content": result, "status": "ok"},
            {"type": "message", "role": "assistant", "content": final},
        ],
        "final": final,
    }
    return json.dumps({"final": final, "trace": trace})


# The user's realistic semantic eval: does the reply confirm the charge? Reads
# `.final` via on_final and PASSES on the mutant (the answer still says success).
user_eval = checks.on_final(checks.contains("successful"))
user_eval.__name__ = "user_eval:confirms_success"

config = MutEvalConfig(
    system=SYSTEM,
    cases=CASES,
    run=run,
    evals=[user_eval],  # the semantic eval ALONE -> the declined charge SURVIVES
    eval_names=["user_eval:confirms_success"],
    operators=["deny_tool_output"],
)
