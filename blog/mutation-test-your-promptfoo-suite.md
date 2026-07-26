# Your promptfoo suite passes. Would it catch a regression?

*A 60-second, no-API-key way to find the rule your evals forgot to check.*

TL;DR — [promptfoo](https://github.com/promptfoo/promptfoo) tells you whether your
LLM app passes your assertions. It can't tell you whether those *assertions* are
any good — whether they'd go red if your prompt silently got worse.
[muteval](https://github.com/AshwinUgale/muteval) answers that: it degrades your
prompt ~20 ways, reruns *your promptfoo assertions*, and reports which injected
regressions slipped through. The ones that slip through are **survivors** —
behaviors you have no eval for. Here's one it found on a three-rule support
prompt, running entirely offline.

## The layer promptfoo doesn't cover

promptfoo is great at the job it's for: write assertions, run them against your
prompt, gate CI, ship. The whole point is catching regressions before users do.

But a green suite tells you nothing if it would *stay* green while your system got
worse. In software we don't trust a test suite because it passes — we trust it
because we've watched it fail when the code breaks. There's no equivalent
discipline for evals. You write the assertions, they go green, and you hope.

The classic answer to "is my test suite any good?" is **mutation testing**: inject
a bug, rerun the tests, and any bug the tests *don't* catch is a gap. `mutmut` and
Stryker have done this for code for years. muteval does it for evals — and it
reads your promptfoo config directly.

## A concrete example (reproducible, no key)

Here's a small support-assistant prompt with three rules:

```yaml
prompts:
  - |
    You are a customer-support assistant.
    - Always cite the source document in square brackets, like [kb-123].
    - Never promise a refund; only a manager can approve one.
    - Always reply in English.

    Question: {{question}}

tests:
  - vars: { question: "Can I get a refund for my order?" }
    assert:
      - type: contains
        value: "[kb-"
      - type: not-icontains
        value: "i'll refund"
```

The suite asserts two of the three rules — citation and the refund policy. Nothing
asserts the language. That's a realistic omission: you assert what you remember to
assert.

Point muteval at it:

```bash
muteval run --config examples/promptfoo_offline/muteval_config.py --no-color
```

(The demo swaps in a deterministic mock model so it runs with no API key; against
your real config it's `muteval run --promptfoo promptfooconfig.yaml`.)

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

When muteval deletes the *citation* or *refund* rule, an assertion fails and the
mutant is **killed** — your suite works. When it deletes **"reply in English,"**
the output changes and *every assertion still passes*. That's the survivor:
muteval degraded the system, and your evals didn't blink.

The important part: **muteval didn't know in advance which rule was uncovered.**
It found the gap by construction — by breaking each rule and watching which
breakage your suite ignored. That's absence detection, and it's exactly the thing
a passing test run can't show you.

## What it's honest about

muteval only claims what it can back up:

- It **excludes inert mutants** — 12 of the 21 changes here produced output
  identical to the baseline, so they're not counted as blind spots (they're not
  evidence of anything). The "effective score" is over the mutants that actually
  changed behavior.
- Survivors are **candidates to triage, not verdicts** — each one is an output
  change your suite didn't catch, with a suggested starter check. You decide
  whether it matters.
- It reports **confidence intervals** and won't emit a score at all if your
  baseline suite doesn't pass first.
- It **skips promptfoo assertions it can't grade** (javascript/python/custom)
  rather than passing them vacuously and inflating the number.

## Try it on yours

```bash
pip install "muteval[promptfoo]"
muteval run --promptfoo promptfooconfig.yaml     # your real config, gpt-4o-mini
```

If it comes back 100%, your assertions are tighter than most. If it finds a
survivor, you just learned about a regression your CI would have shipped. Either
way you now know something a green checkmark couldn't tell you.

muteval is MIT-adjacent (Apache-2.0), dependency-free at the core, and the
promptfoo adapter is one `pip` extra. Repo:
[github.com/AshwinUgale/muteval](https://github.com/AshwinUgale/muteval).
