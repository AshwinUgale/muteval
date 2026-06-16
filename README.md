# muteval

[![CI](https://github.com/AshwinUgale/muteval/actions/workflows/ci.yml/badge.svg)](https://github.com/AshwinUgale/muteval/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/muteval.svg)](https://pypi.org/project/muteval/)
[![Python versions](https://img.shields.io/pypi/pyversions/muteval.svg)](https://pypi.org/project/muteval/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Mutation testing for your LLM eval suite.**

Your evals are passing. That doesn't mean they work.

`muteval` answers the question every eval suite quietly dodges: *would my
evals actually fail if my system silently got worse?* It deliberately degrades
the thing under test, reruns **your existing eval suite** against each degraded
version (a "mutant"), and reports a **mutation score** — the percentage of
injected regressions your evals caught. The ones they miss are **survivors**:
concrete blind spots in your eval coverage.

It's `mutmut`/Stryker, but for evals.

```
Mutation score: 23%  [█████░░░░░░░░░░░░░░░░░░░]  (5/22 mutants killed)

17 SURVIVED  (these regressions slipped past your evals — coverage gaps):

  SURVIVED  [flip_negation]
            inverted "Do not" -> "Do" (near: ...Do not promise refunds...)
  SURVIVED  [drop_instruction_lines]
            dropped line: "You must never reveal another customer's data."
```

---

## Why this exists

Regression-testing tools (promptfoo, deepeval, OpenAI Evals, LangSmith) catch
regressions in your *system*. None of them tell you whether your *evals* are
good enough to catch those regressions in the first place. That meta-layer is
the gap `muteval` fills.

The technique — mutation testing — is the established answer to "is my test
suite any good?" in software engineering, and has been studied for LLM
in-context-learning systems in research (e.g. the MILE framework, arXiv
2409.04831). `muteval` brings it to working eval suites as a tool-agnostic,
developer-facing package.

How muteval differs from neighbouring tools (the two axes that matter):

| Tool | Mutates the… | Measures… |
| --- | --- | --- |
| promptfoo red team | input (jailbreaks) | your system's safety |
| Giskard | input (typos, swaps) | your model's robustness |
| deepeval synth data | output / ground truth | a metric's calibration |
| **muteval** | **the system (prompt → context → tools)** | **your eval suite's coverage** |

## Install

```bash
pip install muteval
```

## Quick start (runs offline, no API key)

```bash
git clone https://github.com/AshwinUgale/muteval
cd muteval
pip install -e .
muteval run --config examples/support_bot/muteval_config.py
```

You'll see a mutation score and a list of survivors — because the example's
eval suite is deliberately missing checks.

Want it against a real model? See `examples/openai_support_bot/` (needs
`pip install "muteval[examples]"` and an `OPENAI_API_KEY`).

## How it works

You describe your system and evals in a small Python config:

```python
from muteval import MutEvalConfig

config = MutEvalConfig(
    prompt=MY_SYSTEM_PROMPT,        # the thing under test
    cases=[...],                    # inputs to your system
    run=lambda prompt, case: ...,   # call your LLM/app, return output text
    evals=[...],                    # each: (output, case) -> bool  (True = pass)
)
```

Then:

1. **Baseline.** muteval confirms your eval suite passes on the original
   prompt. (If it doesn't, the score is meaningless — fix that first.)
2. **Mutate.** It generates mutants by degrading the prompt — weakening
   instructions, inverting rules, dropping lines, truncating, removing
   emphasis, deleting few-shot examples.
3. **Grade.** It reruns your eval suite against each mutant. A mutant is
   **killed** if your evals fail (good — they caught it) and **survives** if
   they still pass (bad — a gap).
4. **Score.** `killed / total`. Write evals to kill the survivors, and watch
   the number climb.

## Gate CI

```bash
muteval run --config muteval_config.py --fail-under 75
```

Exits non-zero if your eval coverage drops below 75%, so a PR that weakens your
evals fails the build.

## Mutation operators

| Operator | What it injects |
| --- | --- |
| `weaken_modals` | softens strong instructions (`must` → `should`) |
| `flip_negation` | inverts a rule (`do not` → `do`, `never` → `always`) |
| `drop_instruction_lines` | deletes a single instruction line |
| `delete_sentences` | deletes a single sentence (prose prompts) |
| `truncate_prompt` | clips the tail of the prompt |
| `drop_few_shot_example` | removes one few-shot example block |
| `remove_emphasis` | strips `**bold**` / `IMPORTANT:` cues |

## Roadmap

`muteval` v0 mutates **prompts**. The thesis scales well beyond that:

- [ ] Mutate **retrieved context** (RAG) — corrupt/swap/drop retrieved docs
- [ ] Mutate **tool outputs** for agent eval suites
- [ ] Model-swap mutants (downgrade the model, see if evals notice)
- [ ] LLM-driven semantic mutations (beyond rule-based string edits)
- [ ] Adapters for promptfoo / deepeval test definitions
- [ ] Statistical handling for non-deterministic suites (confidence intervals)
- [ ] HTML / Markdown reports and a shareable score badge

The endgame is the standard way teams *certify* their evals before trusting an
AI system in production.

## Contributing

This is an early, open project and contributions are very welcome — especially
new mutation operators and tool adapters. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE).
