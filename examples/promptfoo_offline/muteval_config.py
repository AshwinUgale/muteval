"""Keyless muteval target = the promptfoo suite in this folder.

Point muteval at an existing `promptfooconfig.yaml` and it asks: *would these
promptfoo assertions actually catch a prompt regression?* This demo runs with NO
API key — it injects a tiny deterministic "model" that obeys the prompt's rules
via plain string checks, so you can watch muteval find a coverage gap in ~1s.

    muteval run   --config examples/promptfoo_offline/muteval_config.py --no-color
    muteval probe --config examples/promptfoo_offline/muteval_config.py --no-color

The prompt has three rules (cite a source, no refund promises, reply in English).
The promptfoo suite asserts the first two — but nothing checks the language. So
when muteval deletes the "reply in English" line, the output changes and *every
assertion still passes*: a survivor. That's absence detection — "you have no eval
for this behavior at all" — on a real promptfoo config.

To run it for real against gpt-4o-mini instead of the mock, drop the `run=`
argument (and set OPENAI_API_KEY), or just:  muteval run --promptfoo promptfooconfig.yaml
"""

import os

from muteval.adapters.promptfoo import from_promptfoo

_YAML = os.path.join(os.path.dirname(__file__), "promptfooconfig.yaml")


def mock_model(prompt: str, case: dict) -> str:
    """A deterministic stand-in for an obedient LLM. Each of the prompt's three
    rules maps to one observable behavior, so deleting a rule visibly changes the
    output — exactly what a real model would do."""
    p = prompt.lower()

    # Rule 1 — cite the source document.
    citation = "[kb-123]" if "cite the source" in p else ""

    # Rule 2 — never promise a refund (a manager approves).
    if "never promise a refund" in p:
        body = "I understand your concern. I cannot promise a refund; a manager must approve one."
    else:
        body = "Sure, I'll refund you right away!"

    # Rule 3 — always reply in English (NOTHING in the suite asserts this).
    greeting = "Hello!" if "reply in english" in p else "¡Hola!"

    return f"{greeting} {body} {citation}".strip()


config = from_promptfoo(_YAML, run=mock_model)
