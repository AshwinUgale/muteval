# Contributing to muteval

Thanks for your interest! muteval is a focused, honest tool — mutation testing
for LLM eval suites — and contributions that keep it *trustworthy and simple* are
very welcome. This guide gets you productive fast.

New to the codebase? Read [ARCHITECTURE.md](ARCHITECTURE.md) first — it's a
one-page map of how a run flows through the code.

## Development setup

```bash
git clone https://github.com/AshwinUgale/muteval
cd muteval
pip install -e ".[dev]"          # editable install + test/type tools
pytest -q                        # the suite (all green)
muteval run --config examples/support_bot/muteval_config.py   # no API key needed
```

Before opening a PR, run what CI runs:

```bash
pytest -q                        # unit + property + edge tests
mypy                             # type gate (config in pyproject.toml)
pytest -q tests/test_ci_coverage.py -m slow   # (optional) Monte-Carlo coverage
```

Reference cross-checks against statsmodels/scipy/etc. need the `[verify]` extra
(`pip install -e ".[dev,verify]"`); they're skipped if those libs aren't present.

## Good first contributions

- **A new mutation operator** — the highest-value, smallest-surface area. ~15
  lines + a test (see below). Ideas: paraphrase an instruction, reorder few-shot
  examples, corrupt a tool schema.
- **A new probe** — an eval-quality lens (`src/muteval/probes/`). Must catch a
  *real* eval defect and degrade honestly ("not assessed" when it can't run).
- **An adapter** — let people point muteval at another eval framework's suite
  (see `adapters/base.py` + `adapters/promptfoo.py`).
- **A reporter / output format** — Markdown, JUnit XML, a nicer HTML.
- **Docs & examples** — especially an agent/tools example, or a real API-backed
  walkthrough.

Browse issues labeled **`good first issue`** and **`help wanted`**. For anything
larger than a single operator/probe, open an issue first so we agree on the shape.

## How to add a mutation operator

1. Write `def my_operator(target: str | System) -> list[Mutant]` in
   `src/muteval/mutators.py` (use `as_system(target)` to accept both). Each
   `Mutant` carries a `System` and a **human-actionable `description`** — that
   string is what users see in the survivor report, so make it specific.
2. Register it: `register_operator("my_operator", my_operator)` (or add it to the
   `OPERATORS` dict). Custom operators can also be passed directly via
   `config.operators=[my_operator]` without global registration.
3. Add a test in `tests/test_mutators.py`. **Every operator gets a test.**
4. `muteval list operators` should now show it.

The same pattern applies to probes (`register_probe`) — the full extension API is
in [docs/PLUGINS.md](docs/PLUGINS.md), with a contract test in
`tests/test_plugin_contract.py`.

## Guidelines

- **Keep the core dependency-free.** Optional integrations (deepeval, ragas,
  promptfoo, model SDKs) go behind extras in `pyproject.toml`.
- **Fail closed and stay honest.** muteval never emits a misleading score; new
  code should preserve the validity gates and the "candidate, not verdict"
  framing. If you're unsure, that's a great thing to raise in the issue.
- **Every public function and operator gets a test.**
- **Update `CHANGELOG.md`** under the top section for user-visible changes.

## Pull request checklist

- [ ] `pytest -q` passes and new behavior has a test
- [ ] `mypy` passes
- [ ] `CHANGELOG.md` updated (for user-visible changes)
- [ ] docs/README touched if the change is user-facing
- [ ] the PR description says *what* and *why* in a sentence or two

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you agree to uphold it — be kind, be constructive.
