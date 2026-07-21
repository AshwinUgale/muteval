# muteval

[![CI](https://github.com/AshwinUgale/muteval/actions/workflows/ci.yml/badge.svg)](https://github.com/AshwinUgale/muteval/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/muteval.svg)](https://pypi.org/project/muteval/)
[![Python versions](https://img.shields.io/pypi/pyversions/muteval.svg)](https://pypi.org/project/muteval/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Mutation testing for your LLM evals — find out if they'd actually catch a regression.**

Your evals are passing. That doesn't mean they work.

`muteval` answers the question every eval suite quietly dodges: *would my
evals actually fail if my system silently got worse?* It deliberately degrades
the thing under test, reruns **your existing eval suite** against each degraded
version (a "mutant"), and reports a **mutation score** — the percentage of
injected regressions your evals caught. The ones they miss are **survivors**:
concrete blind spots in your eval coverage.

It's `mutmut`/Stryker, but for evals.

**Zero dependencies, no eval framework required.** muteval is pure Python — `pip
install muteval` and you're running; it never drags in heavy LLM SDKs. Use the
built-in checks (including a standard-library LLM judge) with nothing else
installed, or reuse your existing deepeval / RAGAS metrics if you already have
them.

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
pip install muteval        # pure Python, zero required dependencies
```

That's the whole install. The core has no dependencies; optional adapters
(`muteval[deepeval]`, `muteval[ragas]`) only matter if you already use those
tools.

## Quick start — no Python, one command

You don't need a config file or a `run()` function. Give muteval a prompt, a
cases file, and the checks you care about — it calls the model for you:

```bash
export OPENAI_API_KEY=sk-...

muteval run \
  --prompt-file system.txt \
  --cases cases.jsonl \
  --model gpt-4o-mini \
  --check not_contains:refund \
  --judge "the answer is grounded in the provided context" \
  --fail-under 75
```

`cases.jsonl` is one JSON object per line, e.g.
`{"question": "what port?", "context": ["The server listens on port 8080."]}`.

Built-in checks: `contains:TEXT`, `not_contains:TEXT`, `contains_case:KEY`,
`regex:PATTERN`, `is_json`, `equals`, and `judge:<rubric>` (LLM-as-judge, stdlib —
no extra installs). Add `--dry-run` to validate your setup without spending a
token. For custom pipelines/metrics, use `--config muteval_config.py` instead.

Testing a RAG app? Add `--context-file knowledge.txt` (or `--context "doc"`,
repeatable). muteval then also drops/clears retrieved docs (`drop_context_doc`,
`clear_context`) and checks whether your evals notice the degraded retrieval.

Using a specific model? Add `--mutate-model` to also swap it for a weaker one
(`downgrade_model`) and see whether your evals catch the cheaper model.

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

### Start from scratch

No eval framework? You need nothing but muteval. Scaffold a config and grade
with built-in checks in two lines (`checks.llm_judge` calls the model via the
standard library — no `openai` package needed). See
`examples/llm_judge_quickstart/` for a runnable end-to-end example:

```bash
muteval init                       # writes muteval_config.py you can edit
muteval run --config muteval_config.py
```

```python
from muteval import MutEvalConfig, checks

config = MutEvalConfig(
    prompt=SYSTEM_PROMPT,
    cases=[{"input": "where is my order?", "order_id": "A123"}],
    run=my_run_fn,
    evals=[
        checks.contains_case("order_id"),   # answer must cite the order id
        checks.not_contains("refund"),      # never promise a refund
        checks.llm_judge("is it polite?"),  # generic LLM-as-judge (stdlib, no SDK)
    ],
)
```

## How it works

You describe your system and evals in a small Python config:

```python
from muteval import MutEvalConfig

config = MutEvalConfig(
    prompt=MY_SYSTEM_PROMPT,        # the thing under test
    cases=[...],                    # inputs to your system
    run=lambda prompt, case: ...,   # call your LLM/app, return output text
    evals=[...],                    # each: (output, case) -> bool | EvalOutcome
)
```

Evals may return a plain `bool` or a scored `EvalOutcome(passed, score,
threshold)`. When a score is present, survivors that only *barely* passed are
flagged as **near misses** (`↳ near miss: passed Faithfulness by only +0.020`) —
your eval almost caught the regression.

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

Copy `examples/ci/github-actions.yml` to `.github/workflows/muteval.yml` and it
runs on every PR automatically — set once, then it guards your eval suite forever.

### Eval-coverage badge

`muteval run --json out.json --badge badge.json` writes machine-readable results
and a [shields.io](https://shields.io) endpoint payload. The CI template publishes
the badge on `main`; then add to your README (replace `OWNER/REPO`):

```markdown
[![eval coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/OWNER/REPO/main/.github/badges/eval-coverage.json)](https://github.com/OWNER/REPO)
```

## Advanced

- **Choose what mutates** — `--operators weaken_modals flip_negation ...` runs a
  subset. `--sample N --seed S` runs a reproducible random subset for cheap CI.
- **Scope the prompt** — wrap mutable regions in `[[mutate]] ... [[/mutate]]`, or
  use `--scope-include REGEX` / `--scope-exclude REGEX` (line-level) so muteval
  only mutates the parts you care about.
- **Bring your own operator** — `from muteval import register_operator` (or pass
  a callable in `operators=[...]`). Operator factories let you parametrize the
  built-ins: `make_weaken_modals(pairs)`, `make_downgrade_model(ladder)`.
- **Mutate beyond the prompt** — `System(prompt, context=[...], tools=[...],
  model=...)` makes retrieved context, tool outputs, and the model itself
  mutable (RAG/agent suites).

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
| `drop_context_doc` | drops one retrieved document (RAG) |
| `clear_context` | removes all retrieved context (retrieval failure) |
| `corrupt_context_doc` | injects a plausible-but-wrong fact into a doc |
| `swap_context_doc` | replaces a doc with an irrelevant one (bad retrieval) |
| `shuffle_context` | reverses doc order (position sensitivity) |
| `duplicate_context_doc` | duplicates a doc (redundant noise) |
| `truncate_context_doc` | clips a doc's tail (cut-off chunk) |
| `downgrade_model` | swaps the model for a weaker one (model-swap) |
| `drop_tool_output` | drops one tool output (agents) |
| `corrupt_tool_output` | corrupts one tool output (wrong tool result) |
| `swap_tool_output` | swaps a tool output for an irrelevant one |

## Mutating retrieved context (RAG)

The mutation target isn't limited to a prompt string. Pass a `System` and your
`run` receives the mutated system, so muteval can degrade the **retrieved
context** and see whether your evals actually depend on retrieval quality:

```python
from muteval import MutEvalConfig, System

config = MutEvalConfig(
    system=System(prompt=SYSTEM_PROMPT, context=["doc1", "doc2"]),
    cases=[{"question": "..."}],
    run=lambda system, case: my_rag_answer(system.prompt, system.context, case),
    evals=[...],
)
```

The `drop_context_doc` and `clear_context` operators now produce mutants; if your
suite still passes when a relevant doc is dropped, that's a survivor.

## Adapters (optional)

**You don't need any framework to use muteval.** But if you already have a suite
in another tool, reuse its metrics instead of rewriting them. The **deepeval**
adapter wraps your existing metrics as muteval evals:

```python
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from muteval import MutEvalConfig
from muteval.adapters.deepeval import metrics_to_evals

metrics = [AnswerRelevancyMetric(threshold=0.7), FaithfulnessMetric()]
config = MutEvalConfig(
    prompt=SYSTEM_PROMPT,
    cases=[{"question": "...", "context": ["doc1", "doc2"]}],
    run=my_run_fn,                       # how to regenerate output from a prompt
    evals=metrics_to_evals(metrics, input_key="question",
                           retrieval_context_key="context"),
)
```

Install with `pip install "muteval[deepeval]"`. See `examples/deepeval_rag/`.

**RAGAS** works the same way (`pip install "muteval[ragas]"`). RAGAS metrics
return a raw score, so you supply a threshold — and survivors get near-miss
margins for free:

```python
from ragas.metrics import Faithfulness, ResponseRelevancy
from muteval.adapters.ragas import metrics_to_evals

evals = metrics_to_evals(
    [Faithfulness(), ResponseRelevancy()],
    threshold=0.7,
    input_key="question",
    retrieval_context_key="context",
)
```

(A promptfoo adapter is next.) Writing a new adapter is small — see
`src/muteval/adapters/base.py` for the contract.

## Roadmap

`muteval` started by mutating **prompts**. The thesis scales well beyond that:

- [x] Mutate **retrieved context** (RAG) — `drop_context_doc`, `clear_context`
      (corrupt/swap operators next)
- [x] **Scored evals + near-miss reporting** (`EvalOutcome`)
- [x] **deepeval** and **RAGAS** adapters — promptfoo adapter next
- [ ] Mutate **tool outputs** for agent eval suites
- [ ] Model-swap mutants (downgrade the model, see if evals notice)
- [ ] LLM-driven semantic mutations (beyond rule-based string edits)
- [ ] Statistical handling for non-deterministic suites (confidence intervals)
- [ ] HTML / Markdown reports and a shareable score badge

The endgame is the standard way teams *certify* their evals before trusting an
AI system in production.

## Does it work?

See [FINDINGS.md](FINDINGS.md). In a controlled experiment the mutation score
rises monotonically with eval-suite coverage (0% → 28% → 56% → 72%), and
`validation/` holds reproducible runs — including against real deepeval metrics.

## Limitations

muteval is honest about what it can't do and when to distrust the number —
see [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Contributing

This is an early, open project and contributions are very welcome — especially
new mutation operators and tool adapters. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE).
