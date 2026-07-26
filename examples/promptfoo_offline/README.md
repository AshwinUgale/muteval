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
Mutation score: 29%  (6/21 mutants killed, 95% CI 14-50%)
Effective score: 67%  (6/9 — excludes 12 inert mutants whose output didn't change)

3 SURVIVED  (output changed but evals didn't notice — real coverage gaps; 2 HIGH-severity):

  [HIGH] SURVIVED  [drop_instruction_lines]
            dropped line: "- Always reply in English."
            fix: add checks.llm_judge("the reply still follows: - Always reply in English.")
  [HIGH] SURVIVED  [delete_sentences]
            deleted sentence: "- Always reply in English."
  [MED]  SURVIVED  [truncate_prompt]
            truncated prompt — dropped the last 3 of 6 lines
```

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
