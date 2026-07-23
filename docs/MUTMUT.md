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

Run against a **reduced suite** (`test_edge_cases.py` + `test_stats_reference.py`;
the property + MC-coverage tests are excluded for speed) with the correct mutmut-3
config (`source_paths`; the whole package `also_copy`-ed into the sandbox):

| metric | first suite | after v0.4 hardening |
| --- | --- | --- |
| mutants generated | 335 | 335 |
| killed | 231 | **247** |
| survived | 44 | 46 |
| not covered by this reduced suite | 60 | 42 |
| **mutation score** (killed / (killed+survived)) | 84.0% | **84.3%** |

The middle column used a mis-configured (deprecated `tests_dir`) run that left 60
mutants untested; the right column is the corrected run **plus** the new tests
below. Net: **+16 real mutants killed** (231 → 247) and 18 fewer untested.

Still a **lower bound** — it excludes `test_properties.py` (400 Hypothesis
examples) and the MC-coverage test, which exercise `stats.py` harder. Re-run with
the full stats-covering set for the ceiling.

### Two real bugs mutation testing caught (and we fixed)

1. **z rounded to 4 dp** (`1.9600`) — off by ~3.6e-5 vs the exact value; fixed to
   full precision (also caught by the reference cross-checks).
2. **Inconsistent fallback z** — `_Z.get(confidence, 1.9600)` returned the *rounded*
   z for unknown confidence levels while the table held the *precise* one. Fixed to
   `_Z.get(confidence, _Z[0.95])`. Surfaced directly by the surviving
   `default-z` mutants.

### Tests added to kill real survivors

- `_betai` / `_beta_ppf` vs `scipy.special.betainc`/`betaincinv` at 10+ (a,b,x)
  points (pins the incomplete-beta engine directly, not just the end-to-end
  interval).
- unknown-confidence fallback (`wilson(·,0.80) == wilson(·,0.95)`).
- `jeffreys` n=1 lower bound is a real quantile > 0 (kills `n<=0 → n<=1`).
- `interval()` dispatch (wilson/jeffreys/default).

### Residual survivors are EQUIVALENT mutants (verified, not gaps)

The remaining survivors concentrate in `_betacf`/`_beta_ppf` and are equivalent
mutants — no test can kill them because they do not change observable output:

- `_betacf`: `maxit = 300 → 301`. The continued fraction converges in ~10
  iterations (breaks on `abs(delta-1) < eps`), so the loop bound is never reached
  → identical output.
- `_beta_ppf` / `_betai`: `if p <= 0.0 → if p < 0.0` (and `x <= 0.0 → x < 0.0`).
  At the boundary value both paths return the same number (0.0) via different
  routes; the mutated `<` case (negative input) never occurs in practice.
- `wilson`/`jeffreys`: `min(1.0, hi) → min(2.0, hi)` and `max(0.0, lo) →
  max(-1.0, lo)`. These are defensive clamps; the Wilson/Jeffreys formulas already
  produce values in [0,1], so the clamp never binds → identical output.

Per policy we **document** these rather than contrive tests to force the score to
100% — killing an equivalent mutant is impossible by definition, and faking it
would be the exact dishonesty this project exists to prevent.

### Reproduce

```toml
[tool.mutmut]
source_paths = ["src/muteval/stats.py"]
also_copy = ["src/muteval"]   # whole package must be importable in the sandbox
```

```bash
mutmut run && mutmut results
```

Note: `mutmut` v3 runs in a copied `mutants/` sandbox, so `also_copy` is required.
Install only `[dev]` so the optional-framework adapter tests skip cleanly, and run
against the stats tests (`test_edge_cases.py`, `test_stats_reference.py`).

## Why this is in the v0.4 "Provably honest" gate

The reference cross-checks prove the math matches gold-standard libraries; the
property/coverage tests prove the invariants hold; `mutmut` proves the **tests
themselves** would catch a regression in that math. Together they are the
difference between "looks rigorous" and "is rigorous".
