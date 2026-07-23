# Extending muteval (plugin API)

muteval has three stable extension points. Each is a small, documented contract
with a test in `tests/test_plugin_contract.py` — a third-party plugin that
satisfies the contract keeps working across releases.

The guiding rule (muteval's core thesis): extensions stay **orthogonal**. A
custom *operator* only produces mutated systems; a custom *eval* only grades
output; a custom *probe* only rates the suite. None of them reach across, so the
mutation-score abstraction stays clean.

## 1. Custom mutation operators

An operator degrades the system under test. It is
`fn(target) -> list[Mutant]`, where `target` is a `System` or a bare prompt
string (call `as_system(target)` to normalize).

```python
from muteval import as_system
from muteval.mutators import Mutant, register_operator

def shout(target):
    sys = as_system(target)
    return [Mutant(
        operator="shout",
        description="UPPERCASED the whole prompt",
        system=sys.with_prompt(sys.prompt.upper()),
    )]

register_operator("shout", shout)   # now runs by default and via --operators shout
```

`Mutant(operator, description, system, target="prompt")` — `description` is what
users see in the survivor report, so make it actionable. Set `target` to
`"context"`, `"tools"`, or `"model"` if you mutate those (affects severity).

You don't have to register globally: pass callables straight through —
`operators=[shout]` in `MutEvalConfig` or `generate_mutants(system, operators=[shout])`.

## 2. Custom evals (checks / framework adapters)

An eval is any `fn(output, case) -> bool | EvalOutcome`. Return an `EvalOutcome`
to carry a score/threshold (enables near-miss reporting). Tag `fn.is_llm = True`
if it makes a paid model call, so the runner orders it AFTER cheap checks and
counts it against `--max-calls`.

```python
from muteval import EvalOutcome

def mentions_price(output, case):
    return EvalOutcome(passed="$" in output, name="mentions_price")
```

To wrap a third-party metric, use the adapter helpers in `muteval.adapters.base`:

```python
from muteval.adapters.base import scorer_to_eval, case_get

# any (output, case) -> float scorer, compared to a threshold
faithfulness = scorer_to_eval(my_metric.score, threshold=0.7, name="faithfulness")
```

`case_get(case, key)` reads a field from a dict-or-object case. See
`adapters/deepeval.py` and `adapters/ragas.py` for full examples.

## 3. Custom probes (eval-quality lenses)

A probe rates the eval SUITE along one dimension. It is
`fn(config) -> ProbeResult`, registered like an operator.

```python
from muteval.probes.base import ProbeResult, register_probe

def has_negative_case(config):
    ok = any(getattr(c, "get", lambda k: None)("unanswerable") for c in config.cases)
    return ProbeResult(
        name="has_negative_case", ok=ok,
        summary="suite includes an unanswerable case" if ok else "no negative case",
        detail="Add a case with no supported answer to test the abstain path.",
    )

register_probe("has_negative_case", has_negative_case)
```

`ProbeResult(name, ok, summary, detail=None, metrics={})`. There is deliberately
**no composite score** — probes are a report card of separate signals.

## Stability

The signatures above (`register_operator`, `Mutant`, `register_probe`,
`ProbeResult`, `scorer_to_eval`, `case_get`, the eval `(output, case)` shape) are
covered by `tests/test_plugin_contract.py`. Changes to them are breaking changes
and will be called out in the changelog.
