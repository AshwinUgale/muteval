# Changelog

All notable changes to muteval are documented here. This project adheres to
[Semantic Versioning](https://semver.org) (pre-1.0: minor versions may introduce
additive features; the public API is not yet frozen — that lands at 1.0).

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
