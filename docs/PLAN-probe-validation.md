# Plan: per-probe validation (the trust gate for the eval-evaluator)

**Goal.** Before the probes graduate from "bonus layer" into a standalone
eval-evaluator, each one must earn the same trust the mutation score did: a
controlled experiment proving it **fires when it should** (a genuinely bad eval)
and **stays quiet when it shouldn't** (a good eval), ideally tracking severity
monotonically — and **enforced in CI** so it can't silently regress.

**Why this is cleanly doable.** Every probe can be validated **deterministically
with synthetic inputs** — no API keys, no LLM, fully reproducible. We *construct*
the ground truth (a metric we know is redundant, a judge we know is noisy) and
check the probe measures it. This is the same trick that made the eval-quality
experiment CI-safe.

**Policy.** A probe stays labeled *experimental* in the report card until its
validation lands and passes. The evaluator ships only probes that cleared the
gate.

## Shared harness

Mirror the eval-quality experiment structure:

```
validation/probe_validation/
  validate_statistical_adequacy.py
  validate_judge_reliability.py
  validate_discrimination.py
  validate_redundancy.py
tests/test_probe_validation.py     # parametrized, enforces each in CI
```

Each `validate_*.py` exposes the controlled inputs + expected outcomes; the test
loads them and asserts the probe behaves. Seed all randomness for determinism.

---

## Probe 1 — statistical_adequacy
**Claim:** flags when a suite has too few cases to trust its pass rate (Wilson CI
too wide / below the n needed for a target precision).
**Controlled setup:** suites at a fixed pass rate (say 0.9) with N ∈ {3, 10, 30,
100} cases.
**Expected:** reported CI **width shrinks monotonically** as N grows; the probe
**WARNs** at tiny N and **PASSes** once N clears the required-n for the target.
**Assertions:**
- CI width strictly decreases with N (monotonic).
- WARN at N=3; PASS at N=100 (endpoints).
- required-n matches `stats.min_samples_for_precision` for the target.
**Effort:** S (pure math, already have `stats.py`).

## Probe 2 — judge_reliability
**Claim:** flags a noisy judge via verdict-flip rate over N re-runs.
**Controlled setup:** a fixed output graded by **synthetic judges with a known
flip probability** p ∈ {0.0, 0.2, 0.4}, seeded.
**Expected:** measured flip rate **rises monotonically** with p and ≈ p within
tolerance; deterministic judge (p=0) → 0 flips.
**Assertions:**
- flip_rate(p=0.0) == 0 → PASS.
- flip_rate monotonic in p; |measured − p| < tolerance.
- WARN triggers above the reliability threshold (e.g. p=0.4).
**Effort:** S–M (inject a seeded stochastic eval; enough runs for a stable rate).

## Probe 3 — discrimination
**Claim:** flags a metric that doesn't separate good answers from bad ones.
**Controlled setup:** labeled good/bad exemplars per case, graded by:
- a **discriminating** metric (scores good high, bad low), and
- a **non-discriminating** one (e.g. always 1.0, or length-only).
Optionally a gradient of metrics with increasing separation.
**Expected:** large good−bad **gap** for the discriminating metric → PASS; ~0 gap
for the non-discriminating one → WARN.
**Assertions:**
- gap(discriminating) is large and > gap(non-discriminating).
- non-discriminating metric → WARN; discriminating → PASS.
- (gradient) gap increases monotonically with designed separation.
**Effort:** M.

## Probe 4 — redundancy
**Claim:** flags pairs of metrics that measure the same thing (Pearson r > 0.9).
**Controlled setup:** a metric set over fixed cases where **two are near-duplicates**
(identical scores) and the rest are **independent by construction**; optionally a
gradient of pairs at r ∈ {0.0, 0.5, 0.95, 1.0}.
**Expected:** the duplicate pair correlates ≈ 1.0 and is flagged; independent
pairs stay below threshold and aren't.
**Assertions:**
- duplicate pair r ≈ 1.0 and appears in the flagged set.
- independent pairs r < 0.9 and are NOT flagged.
- (gradient) only pairs with r ≥ 0.9 are flagged.
**Effort:** S–M.

---

## Acceptance bar (per probe)
1. **Endpoints:** clean/good input → no warning; clearly-bad input → warning.
2. **Monotonic** where a severity gradient exists (the measured signal tracks the
   injected severity).
3. **Deterministic** (seeded) so it's CI-enforceable, no keys.

## Suggested order
statistical_adequacy (S, pure math) → redundancy (S–M) → judge_reliability (S–M)
→ discrimination (M). Land each with its `validate_*` + the CI assertion before
moving on; flip its report-card label from *experimental* to *validated* only
once green.

## After the gate
When all four are validated **and** the launch has produced real demand for a
"rate my eval suite" product, that's the signal to branch the eval-evaluator into
its own repo/org (composing muteval). Until both hold, the probes stay the bonus
layer — no premature fork.
