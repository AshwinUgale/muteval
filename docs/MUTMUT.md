# Dogfooding: mutation-testing muteval with `mutmut`

muteval is a mutation tester; the most credible quality signal we can give is to
**mutation-test our own code** with an established tool (`mutmut`) and report the
score. A surviving mutant in `stats.py`/`runner.py`/`probes/` means our own test
suite has a blind spot — exactly the thing muteval exists to find in others.

## How to run

```bash
pip install -e ".[dev]" mutmut
mutmut run          # config in pyproject.toml [tool.mutmut]
mutmut results      # list survivors
mutmut show <id>    # inspect a specific surviving mutant
```

Config (in `pyproject.toml`):

```toml
[tool.mutmut]
paths_to_mutate = ["src/muteval/stats.py"]   # start with the load-bearing math
tests_dir = ["tests/"]
```

We start by mutating `stats.py` (the confidence-interval / rank / alpha math the
whole "honest number" claim rests on) and will widen `paths_to_mutate` to
`runner.py` and `probes/` as survivors are driven to zero.

## Known setup note

`mutmut` (v3) copies the project into a `mutants/` sandbox and runs the suite
there. Our adapter tests (`tests/test_adapters_*.py`) import optional extras and
fail to collect inside that sandbox, which aborts the run early. Scope the run to
the tests that exercise the mutated module so the sandbox stays import-clean, e.g.
run against `tests/test_properties.py tests/test_edge_cases.py
tests/test_stats_reference.py` (these fully cover `stats.py`). The `mutants/`
sandbox dir is gitignored.

## Baseline (recorded 2026-07-22)

First real run — `paths_to_mutate = ["src/muteval/stats.py"]`, tested against a
**reduced suite** (`test_edge_cases.py` + `test_stats_reference.py` only; the
property + MC-coverage tests were excluded for speed):

| metric | value |
| --- | --- |
| mutants generated | 335 |
| killed | 231 |
| survived | 44 |
| not covered by this suite | 60 |
| **mutation score** (killed / (killed+survived)) | **84.0%** |

This is a **lower bound**: it excludes `test_properties.py` (400 Hypothesis
examples over the interval math) and the Monte-Carlo coverage test, both of which
exercise `stats.py` hard. The full stats-covering suite scores higher; re-run with
all four stats tests to record the real ceiling.

### Where the survivors are (and why)

Survivors concentrate in the `_betacf` / `_betai` continued-fraction internals
(Numerical Recipes incomplete-beta). That's the classic mutation-testing signal:
the tests assert the *end-to-end* interval value to 1e-6, so a mutation deep in
the iteration that the fixed-point still converges through (or that only shifts
the 13th digit) survives. Options, in order of honesty:
1. add targeted asserts on `_betai(a,b,x)` against `scipy.special.betainc` at
   several (a,b,x) — kills most `_betacf`/`_betai` survivors directly; or
2. document specific survivors as **equivalent mutants** (the change cannot alter
   the observable output within tolerance) with the reasoning, here.

Do NOT paper over them by loosening tolerances. Next action: add the
`_betai`-vs-`scipy.special.betainc` cross-check, re-run, and update this table.

### Reproduce this number

```toml
[tool.mutmut]
paths_to_mutate = ["src/muteval/stats.py"]
tests_dir = ["tests/"]
also_copy = ["src/muteval"]   # copy the whole package into the sandbox
```

```bash
mutmut run && mutmut results
```

Note: `mutmut` v3 runs in a copied `mutants/` sandbox; `also_copy = ["src/muteval"]`
is required so the full package (not just the mutated file) is importable there.
Run with only the `[dev]` extras so the optional-framework adapter tests are
skipped cleanly.

## Why this is in the v0.4 "Provably honest" gate

The reference cross-checks prove the math matches gold-standard libraries; the
property/coverage tests prove the invariants hold; `mutmut` proves the **tests
themselves** would catch a regression in that math. Together they are the
difference between "looks rigorous" and "is rigorous".
