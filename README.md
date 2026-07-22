# muteval

[![CI](https://github.com/AshwinUgale/muteval/actions/workflows/ci.yml/badge.svg)](https://github.com/AshwinUgale/muteval/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/muteval.svg)](https://pypi.org/project/muteval/)
[![Python versions](https://img.shields.io/pypi/pyversions/muteval.svg)](https://pypi.org/project/muteval/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Mutation testing for your LLM evals — find out if they'd actually catch a regression.**

Your evals are passing. That doesn't mean they work.

muteval answers the question every eval suite quietly dodges: *would my evals
actually fail if my system silently got worse?* It deliberately degrades the
thing under test, reruns **your existing eval suite** against each degraded
version (a "mutant"), and reports a **mutation score** — the percentage of
injected regressions your evals caught. The ones they miss are **survivors**:
concrete blind spots in your eval coverage.

It's `mutmut` / Stryker, but for evals.

```text
Mutation score: 33%  [████████░░░░░░░░░░░░░░░░]  (2/6 mutants killed, 95% CI 10-70%)

2 SURVIVED  (output changed but evals didn't notice — real coverage gaps; 1 HIGH-severity):

  [HIGH] SURVIVED  [delete_sentences]
            deleted sentence: "If the answer is not in the context, say you don't know."
            fix: add checks.grounded("context")   ← muteval suggests the eval that would catch it
  [MED ] SURVIVED  [weaken_modals]
            weakened "ONLY" -> "preferably" (near: answer using ONLY the provided context)
```

---

## Install

```bash
pip install muteval        # pure Python, zero required dependencies
```

That's the whole install. The core drags in **no** heavy LLM SDKs; optional
adapters (`muteval[deepeval]`, `muteval[ragas]`, `muteval[promptfoo]`) only
matter if you already use those tools.

## 60-second quickstart (no API key)

```bash
muteval init --template rag        # scaffold a config (or --template basic)
muteval check --config muteval_config.py   # validate wiring + baseline first
muteval run   --config muteval_config.py
```

`muteval init` writes a runnable config with the four things you supply clearly
marked. `muteval check` is the doctor — it validates your pipeline, evals, and
baseline *before* a full run and tells you exactly which layer is broken. Then
`run` gives you a mutation score and a ranked list of survivors, each with a
suggested eval to close the gap.

Prefer zero-config? Point muteval at a prompt + cases and let it call the model:

```bash
export OPENAI_API_KEY=sk-...
muteval run --prompt-file system.txt --cases cases.jsonl --model gpt-4o-mini \
  --judge "the answer is grounded in the provided context" --fail-under 75
```

## Why this exists

Regression tools (promptfoo, deepeval, OpenAI Evals, LangSmith) catch
regressions in your *system*. None tell you whether your *evals* are good enough
to catch those regressions in the first place. That meta-layer is the gap
muteval fills — mutation testing is the established answer to "is my test suite
any good?" in software engineering, brought to LLM eval suites.

| Tool | Mutates the… | Measures… |
| --- | --- | --- |
| promptfoo red team | input (jailbreaks) | your system's safety |
| Giskard | input (typos, swaps) | your model's robustness |
| deepeval synth data | output / ground truth | a metric's calibration |
| **muteval** | **the system (prompt → context → tools → model)** | **your eval suite's coverage** |

## How it works

Describe your system + evals in a small config, then muteval:

1. **Baseline** — confirms your suite passes on the *original* system. If it
   doesn't, muteval **refuses to score** (a red baseline makes every number
   meaningless) rather than hand you a misleading 100%.
2. **Mutate** — generates mutants by degrading the prompt / retrieved context /
   tool outputs / model (18 operators).
3. **Grade** — reruns your suite against each mutant. **Killed** = your evals
   caught it (good); **survived** = they missed it (a gap).
4. **Score** — `killed / evaluated`, with a 95% confidence interval, severity
   ranking, near-miss margins, and a suggested fix per survivor.

```python
from muteval import MutEvalConfig, checks

config = MutEvalConfig(
    prompt=SYSTEM_PROMPT,                 # the thing under test
    cases=[{"input": "...", "order_id": "A123"}],
    run=my_run_fn,                        # call your LLM/app -> output text
    evals=[                               # your existing checks, graded by muteval
        checks.contains_case("order_id"),
        checks.grounded("context"),       # LLM-judge preset (any OpenAI-compatible endpoint)
    ],
)
```

## Trustworthy by design

A coverage number you can't trust is worse than none. muteval **fails closed**:

- **Red or errored baseline → no score** (status `baseline_failed`/`errored`),
  CLI exits non-zero, no badge.
- **Partial mutant errors** above a budget → `partial_errors`, not a score over a
  shrunken denominator. `--max-error-rate` / `--allow-mutant-errors` to accept.
- **Non-determinism** → strict-majority verdicts over `runs_per_mutant`, Wilson
  confidence intervals on the score, flaky-mutant flagging.
- **Cosmetic changes** → output-diffing separates real coverage gaps from
  "observationally unchanged" mutants.

In a controlled, CI-enforced experiment the mutation score rises monotonically
with eval-suite coverage (**0 → 33 → 67 → 100%** across two domains). See
[FINDINGS.md](FINDINGS.md), and [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for
when to distrust the number.

## What it can mutate (18 operators)

**Prompt:** `weaken_modals`, `flip_negation`, `drop_instruction_lines`,
`delete_sentences`, `truncate_prompt`, `drop_few_shot_example`, `remove_emphasis`.
**Retrieved context (RAG):** `drop_context_doc`, `clear_context`,
`corrupt_context_doc`, `swap_context_doc`, `shuffle_context`,
`duplicate_context_doc`, `truncate_context_doc`.
**Model:** `downgrade_model`. **Tools (agents):** `drop_tool_output`,
`corrupt_tool_output`, `swap_tool_output`.

Pass a `System(prompt=..., context=[...], tools=[...], model=...)` to make
context / tools / model mutable for RAG and agent suites. Bring your own operator
with `register_operator`, and scope which parts of the prompt mutate with
`[[mutate]]…[[/mutate]]` markers or `--scope-include/-exclude`.

## Gate CI + coverage badge

```bash
muteval run --config muteval_config.py --fail-under 75 --badge badge.json
```

Exits non-zero if coverage drops below 75%, so a PR that weakens your evals fails
the build. `--fail-on-severity high` gates on any unguarded high-severity gap.
Copy `examples/ci/github-actions.yml` to run it on every PR and publish a
[shields.io](https://shields.io) eval-coverage badge.

## Bring your existing metrics

Already have a suite? Reuse its metrics instead of rewriting them:

```python
from deepeval.metrics import FaithfulnessMetric
from muteval.adapters.deepeval import metrics_to_evals

evals = metrics_to_evals([FaithfulnessMetric()], input_key="question",
                         retrieval_context_key="context")
```

Adapters for **deepeval**, **RAGAS**, and **promptfoo** (`pip install
"muteval[deepeval|ragas|promptfoo]"`). Or use the built-in framework-free
`checks` — including `llm_judge` / `grounded` that hit **any OpenAI-compatible
endpoint** (OpenAI, Groq, Gemini, GitHub Models, Ollama…) via `base_url=`, using
only the standard library.

## Adopting it on your own suite

Pointing muteval at a real system is a ~1-hour integration, not plug-and-play —
[docs/ADOPTION.md](docs/ADOPTION.md) has the honest checklist, a
"where-it-breaks → what-to-change" table, judge-selection guidance, and the four
pieces you supply. Start with `muteval check` and fix a green baseline first.

## Roadmap

Shipped: prompt/context/tool/model mutation · deepeval/RAGAS/promptfoo adapters ·
scored evals + near-miss reporting · severity ranking + `--fail-on-severity` ·
confidence intervals + majority-vote stability · output-diffing · fail-closed
validity gate · `muteval check` doctor · RAG scaffold + adoption guide.

Next: LLM-driven semantic mutations · HTML/Markdown reports · the eval-quality
probe layer (`muteval probe`) maturing into a standalone eval-evaluator.

## Contributing

Early, open project — contributions welcome, especially new operators and
adapters. See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed [Apache-2.0](LICENSE).
