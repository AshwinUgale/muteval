# CLAUDE.md — muteval

Context for any Claude/agent working in this repo. Read this first.

## What muteval is

**Mutation testing for LLM eval suites.** It answers a question no other tool
packages: *would my evals actually fail if my system silently got worse?*

It deliberately degrades the thing under test (the prompt today; retrieved
context, tool outputs, and model choice on the roadmap), reruns the user's
**existing eval suite** against each degraded version (a "mutant"), and reports
a **mutation score** = % of injected regressions the evals caught. Mutants the
evals fail to catch are **survivors** — concrete blind spots in eval coverage.

One-liner: *"`mutmut`/Stryker, but for evals."*

## The core thesis (don't lose this)

Two axes define the product and keep it distinct from look-alikes:

1. **What gets mutated:** the *system under test* (prompt/context/tools) —
   NOT the input (that's red-teaming / robustness testing) and NOT the
   output/ground-truth (that's synthetic-data metric calibration).
2. **What gets measured:** the *quality of the eval suite* — NOT the robustness
   or safety of the system.

The killer property is **absence detection**: by mutating the system and
running the user's real pipeline through their real suite, muteval surfaces the
survivor where *nothing fails* — i.e. "you have no eval for this behavior at
all." Calibration-style tools can't show that.

## Positioning vs. related work

- **promptfoo** (red teaming): mutates *inputs* (jailbreaks, encodings) to test
  the *system's* safety. Different axis.
- **Giskard**: mutates *inputs* (typos, entity swaps) for *model robustness*
  scoring. Different axis.
- **DeepEval** synthetic data: mutates *outputs* to calibrate a *metric*. The
  closest competitor; Confident AI could productize this. Our edge: tool-
  agnostic + the packaged mutation-score/survivor abstraction + absence
  detection.
- **heerkubadia/MutEval** (GitHub, ~1 star, no package): a research project
  doing prompt-mutation *robustness* testing for HumanEval code-gen. Same-ish
  name, opposite axis. Not a competitor; does not hold the PyPI/npm names.
- Academic prior art: **MILE** (arXiv 2409.04831) — mutation testing of test
  suites for in-context-learning systems. Validates the concept; not a product.

**Strategy:** muteval should *consume* promptfoo/Giskard/deepeval as input
generators, not fight them. Market message: "they mutate your inputs to test
your system; muteval mutates your system to test your evals."

## Names claimed (state as of June 2026)

- PyPI: `muteval` — claimed.
- npm: `muteval` — claimed (placeholder package pointing to the Python one).
- GitHub: `AshwinUgale/muteval`.
- Consider later: a GitHub org, domain (muteval.dev), social handles.

## Repo layout

- `pyproject.toml`, `src/muteval/` — the real Python package.
  - `mutators.py` — 7 mutation operators (weaken_modals, flip_negation,
    drop_instruction_lines, delete_sentences, truncate_prompt,
    drop_few_shot_example, remove_emphasis). Registered in `OPERATORS`.
  - `adapters/deepeval.py` — wrap deepeval metrics as muteval evals
    (metric_to_eval / metrics_to_evals). Behind the `[deepeval]` extra.
  - `runner.py` — engine: baseline check -> generate mutants -> grade -> score.
  - `report.py` — terminal report (score bar + survivors).
  - `config.py` — `MutEvalConfig` (user-facing API) + `load_config`.
  - `cli.py` — `muteval run --config ... [--fail-under N] [--operators ...]`.
- `examples/support_bot/` — runs offline (mock model, no API key); scores ~25%
  on purpose to demonstrate survivors.
- `examples/openai_support_bot/` — real OpenAI-backed example (`[examples]`).
- `examples/deepeval_rag/` — uses the deepeval adapter (`[deepeval]`).
- `tests/` — pytest; all green (19 tests).
- `js/` — npm placeholder package (`package.json`, `index.js`, README, LICENSE).
  Publish npm from this folder: `cd js && npm publish --access public`.

## Conventions

- Apache-2.0 license.
- Core stays dependency-free; integrations (deepeval/promptfoo/model SDKs) go
  behind optional extras.
- Every mutation operator and public function gets a test.
- A `Mutant.description` must be human-actionable — it's what users see in the
  survivor report.
- Run `pytest` before any commit. CLI sanity check:
  `muteval run --config examples/support_bot/muteval_config.py`.

## Roadmap (priority order)

1. Mutate **retrieved context** (RAG) — corrupt/swap/drop docs. This is the
   defensible moat; prioritize over polishing prompt operators.
2. Mutate **tool outputs** for agent eval suites.
3. Model-swap mutants (downgrade model, see if evals notice).
4. LLM-driven semantic mutations (beyond rule-based edits).
5. Adapters that consume existing promptfoo / deepeval suites. (deepeval adapter SHIPPED — src/muteval/adapters/deepeval.py; promptfoo next.)
6. Statistical handling for non-deterministic suites (confidence intervals).
7. Markdown/HTML report + shareable score badge.

## What matters most right now

Validate demand, not competitor speed. Get muteval running on 2-3 real eval
suites, publish a "your evals scored X%" writeup, ship tool-agnostic adapters.
Mindshare > moat for an OSS land-grab.
