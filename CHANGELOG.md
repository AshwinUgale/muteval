# Changelog

All notable changes to muteval are documented here. This project adheres to
[Semantic Versioning](https://semver.org) (pre-1.0: minor versions may introduce
additive features; the public API is not yet frozen — that lands at 1.0).

## [Unreleased]

- Survivor IDs in `muteval results` and `muteval show` now start at 1 for more
  natural human-facing CLI output.
- `muteval run` now numbers survivors (`#1`, `#2`, …), matching the IDs used by
  `muteval results` / `muteval show`, so you can inspect one without re-running.
- **tracelint integration (agent suites).** A new `deny_tool_output` operator
  mutates a tool output into a domain failure returned as transport success
  (HTTP 200 carrying `{"status": "declined"}`) — the fault structured-error
  detection is blind to. A new deterministic, no-judge eval `checks.tracelint()`
  (behind the `muteval[tracelint]` extra) lints the agent's execution trace and
  kills such a mutant even when the final answer still reads clean, and
  `checks.on_final()` lets ordinary output checks grade the `{"final","trace"}`
  bridge. When a tool-fault mutant survives, the report now names the exact
  deterministic check that would catch it. See `examples/agent_tool_fault/`.

## [0.8.0] — 2026-07-28

- **promptfoo: run the model your suite actually uses.** The adapter now reads the
  model under test from the promptfoo `providers:` block instead of always defaulting
  to `gpt-4o-mini`; an explicit `--model` still wins, and a provider muteval can't call
  directly falls back with a warning. (`from_promptfoo` now defaults `model=None` = auto.)
- **promptfoo: graceful degrade on unsupported asserts.** A case whose assertions are all
  untranslatable types (javascript/python/…) is now *dropped with a warning* instead of
  aborting the whole run; muteval fails closed only if nothing in the suite is gradeable.
- **promptfoo: external test files + more assert types.** `tests: file://cases.csv` (also
  `.jsonl`/`.json`/`.yaml`) and an external `defaultTest: file://…` are now loaded instead
  of crashing; code-function / remote sources (`.py:fn`, `https://`, `huggingface://`) get a
  clear error, not a traceback. Added `contains-any`/`-all`, `icontains-any`/`-all`,
  `not-equals`, `starts-with` assertion translations. Verified against promptfoo's own 194
  example configs: clean build rate **88 → 100**, cryptic errors **19 → 0**.
- **GitHub Action** (`action.yml`) — mutation-test your promptfoo suite in CI in a few
  lines; see `docs/ci.md` and `examples/github_action/mutation-test.yml`.

- **Keyless promptfoo demo** (`examples/promptfoo_offline/`) — `muteval run
  --config examples/promptfoo_offline/muteval_config.py` degrades a support-bot
  prompt and finds the rule its promptfoo suite forgot to assert, with **no API
  key** (a deterministic mock model). Plus a recipe README and a walkthrough
  (`blog/mutation-test-your-promptfoo-suite.md`) for adopting muteval on an
  existing promptfoo config.

## [0.7.0] — 2026-07-24

- Add optional JUnit XML output via `muteval run --junit PATH`.

Adoption pass, driven by a three-way audit of onboarding, integration, and UX.
All additive — no behavior a 0.6 user relied on was removed.

### Reach the easy on-ramps
- **`check`, `probe`, and `label` now accept the same inputs as `run`** —
  `--promptfoo` and the zero-config flags, not just a Python `--config`. The
  doctor and the probe report card finally work on every entry point.
- **`muteval list [operators|checks|probes]`** — discover the operators, built-in
  checks, and probes from the CLI.
- **Clean config errors** — a hand-edited config that raises (SyntaxError,
  NameError, …) now prints `your config <path> raised <Error>`, not a traceback.
- **`muteval run` auto-picks `./muteval_config.py`** when no source is given.
- **`eval_names` auto-derived** from your eval function names — no parallel list to
  hand-duplicate.
- deepeval/ragas adapters raise a `pip install "muteval[…]"` hint when missing.

### Any provider for the system under test
- **`--base-url` / `OPENAI_BASE_URL`** for the model under test (not just the
  judge) — Groq, Gemini-compat, GitHub Models, Ollama, a local server. Threaded
  through zero-config and the promptfoo adapter.

### promptfoo adapter, honest
- **One eval per assertion type** (`promptfoo:contains`, `promptfoo:llm-rubric`) so
  survivors and severity stay per-check.
- **Warns** on skipped unsupported assertions (is-json/javascript/…) and
  **refuses** a case whose assertions are all unsupported — instead of passing it
  vacuously and inflating the score.

### Custom targets
- `--endpoint` POSTs `context`/`model`/`tools` too (retrieval/model mutations reach
  a deployed pipeline), plus `--header` for auth. muteval warns when
  `--target`/`--endpoint` is combined with context/model mutation.

### CLI polish
- `run --help` flags grouped (input / mutation / cost & speed / CI gates / output).

[0.7.0]: https://github.com/AshwinUgale/muteval/releases/tag/v0.7.0

## [0.6.0] — 2026-07-23

The first release since 0.3.1, packaging three internal milestones: "provably
honest" (verification hardening), "adopt in an hour" (ingestion + performance),
and "the eval-evaluator, validated" (the probe layer). Everything below is
additive — no behavior a 0.3.x user relied on was removed. The fail-closed
validity gate, Wilson CIs, and majority-vote stability from 0.3.x are unchanged
and now backed by reference cross-checks and Monte-Carlo coverage tests.

### Trust & verification
- **Reference cross-checks** against `statsmodels`, `scipy`, `scikit-learn`,
  `krippendorff`, and `pingouin` (behind the test-only `[verify]` extra):
  Wilson/Jeffreys intervals to 1e-6, AUC/Spearman to 1e-9, Krippendorff's alpha,
  Cohen's d, and ICC(2,1) all validated against the established libraries.
- **Property-based tests** (Hypothesis) over the statistics and the runner
  (intervals stay in `[0,1]`, `killed ≤ evaluated ≤ total`, `effective ≥ point`,
  CI brackets the point estimate).
- **Monte-Carlo coverage** — Wilson and Jeffreys intervals empirically cover in
  `[0.93, 0.97]` across a `p × n` grid.
- **Determinism** — a single `seed` threads through the whole run; same config +
  seed produces byte-identical JSON on every OS × Python version.
- **Secret redaction** — API keys never appear in emitted JSON or logs;
  `schema_version` added to the result payload.
- **CI matrix** — Python 3.9–3.13 × ubuntu/macos/windows, 90% coverage gate,
  `mypy` type-check gate, and muteval dogfooded with `mutmut`.
- **Jeffreys (Beta-Binomial) interval** added alongside Wilson for very small n.

### Adoption & performance
- **Zero-config ingestion** — run straight from a `promptfoo` config
  (`--promptfoo`), a deepeval test file, or a pytest path; no `.py` config needed.
- **Bring-your-own target** — point at a callable (`--target pkg.mod:fn`) or a
  deployed HTTP endpoint (`--endpoint URL`); no `run()` wrapper required.
- **Caching** — `--cache runs.sqlite` memoizes outputs + eval outcomes; an
  identical re-run makes zero model/judge calls.
- **Concurrency** — `--concurrency N` evaluates mutants in parallel with
  order-preserving, serial-identical results.
- **Cost control** — `--max-calls` / `--budget-usd` fail closed before overspend;
  cheap rule-based evals run before judges and short-circuit kills.
- **Triage UX** — last run persisted to `.muteval/last_run.json`; `muteval
  results` (ranked survivors), `muteval show <id>` (baseline→mutant diff), and
  `muteval report --html` (shareable standalone report).
- **Typing & plugins** — `py.typed` ships; `docs/PLUGINS.md` documents the
  operator/probe/adapter/reporter extension points with a contract test.

### The eval-evaluator (`muteval probe`)
- **Report card** across six lenses, no composite score: judge reliability
  (flip-rate + Krippendorff's alpha + ICC(2,1)), discrimination (AUC + Cohen's d),
  statistical adequacy (Wilson/Jeffreys), redundancy (Spearman + connected
  families), threshold calibration, and **human agreement** (Cohen's κ via
  `muteval label`). A separate **judge-bias panel** (position/verbosity/
  self-preference) ships as a library function for pairwise A/B judges — it needs
  a pairwise-judge harness, so it is not part of the default card.
- Every probe has a CI test asserting its signal is monotonic in injected
  severity and hits its endpoints.
- **Autofix verify loop** — `autofix.suggest_and_verify` proposes an eval for a
  survivor and confirms it actually kills the mutant while the baseline stays
  green; only verified suggestions are returned.
- **Eval-quality proof** extended to four CI-enforced domains (support bot, code
  review, RAG, HR policy): score rises monotonically 0% → 100% with coverage.
- `muteval probe --html` renders the report card.

### Fixes
- Force UTF-8 stdout so the CLI report renders on Windows consoles (cp1252)
  instead of raising `UnicodeEncodeError`.
- Satisfy the `mypy` verify gate (`stream.reconfigure` probe; typed `operators`).

## [0.3.1] and earlier

See the git history. 0.3.x delivered the fail-closed validity gate, partial-error
handling, Wilson confidence intervals, the `muteval check` doctor, the RAG
scaffold (`init --template rag`), OpenAI-compatible `base_url` judges, and the
first four probes upgraded to their prior-art methods.

[0.6.0]: https://github.com/AshwinUgale/muteval/releases/tag/v0.6.0
