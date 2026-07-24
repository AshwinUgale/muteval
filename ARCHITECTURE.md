# muteval architecture

A one-page map of the codebase for contributors. muteval mutation-tests an eval
suite: it degrades the *system under test*, reruns your *existing evals* against
each degraded version, and reports which regressions the evals missed.

## The flow of a run

```
MutEvalConfig  (system + cases + run + evals)
      │
      ▼
select_mutants ──▶ mutators.py apply operators to the System
      │                (prompt / context / tools / model)
      ▼
runner.py: for each mutant → config.invoke(system, case) → output
      │                    → grade output with every eval
      ▼
verdict per mutant (killed / survived / inert / errored), with a validity gate
      │
      ▼
report.py: score (+ Wilson CI) · ranked survivors · suggested fix per survivor
```

## Core modules (`src/muteval/`)

| File | Responsibility |
| --- | --- |
| `system.py` | `System` — the mutation target (prompt + context + tools + model). `as_system` promotes a bare prompt string for back-compat. |
| `config.py` | `MutEvalConfig` (what the user supplies) + `load_config` (executes a Python config file — a trust boundary). |
| `evals.py` | `EvalOutcome` (passed + score + threshold + margin) and `coerce_outcome`. Evals may return `bool` or `EvalOutcome`. |
| `checks.py` | Framework-free eval factories (`contains`, `regex_matches`, `is_json`, `llm_judge`, `grounded`, …). The judge speaks any OpenAI-compatible endpoint via `base_url`. |
| `mutators.py` | The mutation operators (prompt / context / tool / model) + the `OPERATORS` registry + `register_operator`. |
| `runner.py` | The engine: baseline gate → generate mutants → grade → score. Owns the fail-closed validity states. |
| `runners.py` | Built-in `run` helpers so users don't write one: `openai_run`, `callable_run` (`--target`), `http_run` (`--endpoint`). |
| `report.py` | Terminal report, HTML report, probe report card, JSON, badge, secret redaction. |
| `stats.py` | Dependency-free Wilson / Jeffreys intervals, Spearman, AUC, Krippendorff α, ICC. |
| `severity.py` | Ranks each mutant (invert/corrupt = high, drop/weaken = medium, cosmetic = low). |
| `suggest.py` | The `fix:` line under each survivor (operator-aware starter eval). |
| `doctor.py` | `muteval check` — layered, cheapest-first validation of a config + baseline. |
| `cache.py` | Optional sqlite memoization of outputs + eval outcomes (`--cache`). |
| `cli.py` | The command line: `run / init / check / probe / results / show / report / label / list`. |

## Adapters (`src/muteval/adapters/`)

Point muteval at another framework's suite. `base.py` is the contract;
`promptfoo.py`, `deepeval.py`, `ragas.py` implement it. Each lives behind an
optional extra so the core stays dependency-free.

## Probes (`src/muteval/probes/`)

The eval-quality "report card" (`muteval probe`) — separate lenses on suite
quality: `judge_reliability`, `discrimination`, `statistical_adequacy`,
`redundancy`, `threshold_calibration`, `human_agreement` (+ a `judge_bias` panel
for pairwise judges). Registered via `register_probe`. Dependency direction is
one-way: **probes → core, never core → probes.**

## Extension points

- **Operator:** `register_operator(name, fn)` — `fn(target) -> list[Mutant]`.
- **Probe:** `register_probe(name, fn)` — `fn(config) -> ProbeResult`.
- **Adapter:** implement `adapters/base.py`.
- **Custom pipeline:** just pass your own `run(system, case)` and `evals` in a
  Python config — muteval never constrains what they call.

See [docs/PLUGINS.md](docs/PLUGINS.md) for the full API and
`tests/test_plugin_contract.py` for a worked third-party example.

## Tests & CI gates (`tests/`)

- `pytest` — unit + property (Hypothesis) + edge + determinism tests.
- `mypy` — type gate (config in `pyproject.toml`).
- `test_ci_coverage.py` (slow) — Monte-Carlo interval coverage.
- `test_stats_reference.py` — cross-checks the hand-rolled stats against
  statsmodels/scipy/sklearn/pingouin (`[verify]` extra).
- `test_eval_quality.py` — enforces the central claim (score rises with suite
  coverage across four domains).
- CI matrix: Python 3.9–3.13 × ubuntu/macos/windows, 90% coverage gate.

## Conventions

Apache-2.0. Core dependency-free; integrations behind extras. Every operator and
public function gets a test. A `Mutant.description` must be human-actionable — it
is what users read in the survivor report.
