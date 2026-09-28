# Mutation-test your promptfoo suite — in 60 seconds, no API key

You already wrote a [promptfoo](https://github.com/promptfoo/promptfoo) suite. It
passes. But **would it actually fail if your prompt silently got worse?** That's
the question muteval answers: it degrades the prompt ~20 ways and checks whether
*your promptfoo assertions* catch each regression. The ones they miss are
**survivors** — behaviors you have no eval for.

This demo runs entirely offline (a deterministic mock model stands in for the
LLM), so you can see it work with zero setup.

## Run it

```bash
pip install "muteval[promptfoo]"
muteval run --config examples/promptfoo_offline/muteval_config.py --no-color
```

## What you'll see

```
Mutation score: 38%  (5/13 mutants killed, 95% CI 18-64%)
Effective score: 71%  (5/7 — excludes 6 inert mutant(s) whose output didn't change; 95% CI 36-92%)

2 SURVIVED  (output changed but evals didn't notice — real coverage gaps):

  #1 [MED]  SURVIVED  [drop_instruction_lines]
            dropped line: "- Always reply in English."
            fix: add checks.llm_judge("the reply still follows: - Always reply in English.")
  #2 [MED]  SURVIVED  [truncate_prompt]
            truncated prompt — dropped the last 2 of 5 instruction lines (input lines kept), from "- Always reply in English."
```

(The language rule ranks MEDIUM, not HIGH: it's a real gap, but not a safety or
correctness rule. Dropping the refund rule would rank HIGH, and the suite
catches that one.)

## Why this is the point

The prompt has three rules: *cite a source*, *never promise a refund*, *always
reply in English*. The promptfoo suite asserts the first two — so when muteval
deletes those rules, an assertion fails and the mutant is **killed**. But
**nothing asserts the language**, so when muteval deletes "reply in English," the
output changes and *every assertion still passes*. muteval surfaces that as a
survivor: **"you have no eval for this behavior at all."** That's absence
detection — the thing a green test suite can't tell you.

muteval doesn't know in advance which rule is uncovered. It found the gap by
degrading the system and watching which regressions slipped through.

## Run it on YOUR real config

Point muteval straight at your own `promptfooconfig.yaml` (uses `gpt-4o-mini`):

```bash
export OPENAI_API_KEY=sk-...
muteval run --promptfoo promptfooconfig.yaml            # add --dry-run to preview
```

muteval translates promptfoo assertions (`contains`, `icontains`, `not-contains`,
`equals`, `regex`, `is-json`, `llm-rubric` / `model-graded-*`) into graded evals,
one per assertion type, and **skips** what it can't grade (javascript/python/
custom) rather than passing it vacuously. See
[`../promptfoo_demo/`](../promptfoo_demo/) for a real language-tutor config
(with an `llm-rubric` assertion) you can run the same way.
