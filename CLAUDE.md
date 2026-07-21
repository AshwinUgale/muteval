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
  - `system.py` — `System` (the mutation target: prompt + context + tools +
    model + extra) and `as_system` (promotes a bare string for back-compat).
  - `evals.py` — `EvalOutcome` (passed + score + threshold + margin) and
    `coerce_outcome`. Evals may return `bool` OR `EvalOutcome`.
  - `checks.py` — framework-free eval factories (contains, not_contains,
    contains_case, regex_matches, is_json, equals, llm_judge).
  - `mutators.py` — 18 operators: 7 prompt (weaken_modals, flip_negation,
    drop_instruction_lines, delete_sentences, truncate_prompt,
    drop_few_shot_example, remove_emphasis) + 7 context (drop_context_doc,
    clear_context, corrupt_context_doc, swap_context_doc, shuffle_context,
    duplicate_context_doc, truncate_context_doc) + 1 model (downgrade_model)
    + 3 tool (drop/corrupt/swap_tool_output). Custom ops via register_operator;
    operator factories make_weaken_modals/make_downgrade_model. All accept `str | System`. `Mutant` carries a `System`
    (with `.prompt` back-compat property). Registered in `OPERATORS`.
  - `adapters/base.py` — the adapter contract + helpers (case_get,
    scorer_to_eval). Read this before writing a new adapter.
  - `adapters/deepeval.py` — wrap deepeval metrics; returns `EvalOutcome` with
    score/threshold. `[deepeval]` extra.
  - `adapters/ragas.py` — wrap RAGAS metrics (score + threshold). `[ragas]` extra.
  - `runner.py` — engine: baseline check -> generate mutants -> grade -> score;
    records near-miss margins for survivors. Works on `System` via `config.invoke`.
  - `report.py` — terminal report (score bar + survivors + near-miss lines);
    survivors are sorted by severity (HIGH first) with [HIGH]/[MED]/[LOW] tags.
  - `stats.py` — Wilson confidence interval + min-sample-size (dependency-free).
    Score is a proportion; reported as `X% [95% CI lo-hi]`. `runs_per_mutant`
    now uses a MAJORITY vote (`config.kill_threshold`, default 0.5) so judge
    noise doesn't flip verdicts; `MutationResult.flaky` lists mutants that did.
  - `severity.py` — ranks each mutant: `OPERATOR_SEVERITY` base (invert/corrupt
    = high, drop/weaken = medium, cosmetic = low) escalated one level when the
    change touches safety/correctness text (`CRITICAL_PATTERNS`). `severity_of`,
    `severity_rank`. Outcomes carry `.severity`; result has `high_severity_survivors`.
  - `config.py` — `MutEvalConfig` (accepts `prompt=` legacy OR `system=`) +
    `load_config`.
  - `cli.py` — `muteval run` (zero-config flags OR `--config`) with
    `--fail-under N`, `--fail-on-severity {high,medium,low}` (gate on any
    real survivor at/above that severity), `--operators`, `--sample/--seed`,
    `--scope-include/--scope-exclude`, `--context/--context-file`,
    `--mutate-model`, `--dry-run`; plus `muteval init`.
- `examples/support_bot/` — runs offline (mock model, no API key); scores ~23%
  on purpose to demonstrate survivors.
- `examples/openai_support_bot/` — real OpenAI-backed example (`[examples]`).
- `examples/deepeval_rag/` — uses the deepeval adapter (`[deepeval]`).
- `examples/rag_context_offline/` — OFFLINE System-mode demo (mock model, no
  key): corrupt_context_doc flips an answer, a weak eval misses it ->
  [HIGH] survivor. Proves context mutation + diffing + severity end-to-end.
- `validation/deepeval_rag_qdrant/`, `validation/ragas_rag/` — real-metric
  validation configs, prompt mutation (need an API key).
- `validation/deepeval_rag_system/` — System-mode: mutates CONTEXT + MODEL.
  Grades Faithfulness against the *mutated* `used_context`, so a poisoned
  retrieval survives (the stronger, less-gimmicky result). See its NOTES.md.
- `validation/eval_quality_experiment/` — controlled, API-free experiment proving
  the mutation score tracks eval-suite quality (0→28→56→72%). See `FINDINGS.md`.
- `tests/` — pytest; all green (109 tests).
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

SHIPPED: context mutation (drop/clear/corrupt/swap/shuffle/duplicate/truncate),
tool-output mutation (drop/corrupt/swap), model-swap (downgrade_model) — all via
the `System` target. Adapters: deepeval + ragas (promptfoo next). Severity
ranking + `--fail-on-severity` gate. Output-diffing (inert/equivalent mutants).

Open:
1. LLM-driven semantic mutations (beyond rule-based edits; behind an extra).
2. promptfoo adapter (cross-tool generality for the writeup).
3. Statistical handling for non-deterministic suites (confidence intervals;
   `runs_per_mutant` exists, CIs do not).
4. Markdown/HTML report + shareable score badge.
5. User-supplied `CRITICAL_PATTERNS` / per-domain severity overrides via config.

Also shipped beyond the original roadmap: scored evals + near-miss reporting
(EvalOutcome), framework-free `checks`, `muteval init`, severity ranking, the
`--fail-on-severity` CI gate, an offline context-mutation demo
(examples/rag_context_offline/), and a controlled eval-quality experiment
(validation/eval_quality_experiment/, see FINDINGS.md).

## Active plan

The A (scope/custom/sampling) and B (context/tool/model mutation) plan in
`docs/PLAN-A-scope-B-system-mutation.md` is COMPLETE. 18 operators; CLI has
--operators/--sample/--seed/--scope-include/--scope-exclude/--context/--mutate-model.
Next candidates: LLM-driven semantic mutations (behind an extra), confidence
intervals for noisy suites, HTML report + score badge, promptfoo adapter.

## Next-level roadmap

Phased plan — (1) best muteval [TRUST only: stability, generalize, one real
judge, limits], (1.5) capability expansion [agents/Trace, LLM mutations —
AFTER trust gate], (2) eval-evaluator [probe report card], (3) adoption — in
`docs/ROADMAP-next-level.md`. Hard TRUST GATE between phase 1 and 2.

## Outreach / go-to-market

Target list + plan for running muteval against real OSS repos to find
eval gaps, contribute, and publish: `docs/OUTREACH-targets-and-plan.md`.
Lead with a gift not a pitch; filter outreach by HIGH severity.

## What matters most right now

Validate demand, not competitor speed. Get muteval running on 2-3 real eval
suites, publish a "your evals scored X%" writeup, ship tool-agnostic adapters.
Mindshare > moat for an OSS land-grab.
