# Autoevals with a structured JSON profile

This is a **controlled, offline wiring example**, using invented facts and a
deterministic stand-in for an LLM. It demonstrates how to reuse an Autoevals
scorer through `muteval.adapters.base.scorer_to_eval`, without adding a framework
adapter or changing muteval's dependency-free core.

The original prompt requires a `name` and `city`, with `null` for an unknown
city. The config scopes mutation to these two bullet rules (`scope_include`),
leaving the introductory sentence unchanged. muteval deletes each rule in turn and calls the renderer with the
**mutated prompt**. The renderer then omits the city or invents `"Atlantis"`.
The input facts and expected answers remain unchanged.

## Run without an API key

From the repository root, with Python 3.10 or later:

```bash
python -m pip install -e '.[dev]' autoevals
muteval run --config examples/autoevals_profile_json/muteval_config.py --no-color
MUTEVAL_PROFILE_SUITE=strong muteval run \
  --config examples/autoevals_profile_json/muteval_config.py --no-color
python -m pytest -q examples/autoevals_profile_json/test_example.py
```

The environment-variable syntax above is for a POSIX shell. Set
`MUTEVAL_PROFILE_SUITE` to `strong` in your shell to select the stronger suite;
the default is `weak`. Neither mode calls a model or needs a Braintrust account.
Autoevals is needed for the strong suite; these optional integration tests skip
when it is not installed.

Expected results for these two deliberately constructed mutations:

| Suite | Checks | Baseline | Killed | Changed survivors | Mutation score |
| --- | --- | --- | --- | --- | --- |
| Weak | Valid JSON only | Pass | 0/2 | 2 | 0% |
| Strong | Valid JSON + Autoevals `ExactMatch` | Pass | 2/2 | 0 | 100% |

Both defective outputs are valid JSON. Parsing alone says nothing about missing
fields or unsupported facts. The stronger suite compares the whole profile
against its reference, after normalizing whitespace and object-key order. The
wrapper retains score and threshold in `EvalOutcome`. A skipped (`None`),
non-finite, or errored Autoevals score raises an error instead of becoming a
pass or a kill. muteval's default validity gates withhold a trusted score if the
baseline fails or scoring is unavailable.

When consuming the Python result directly, require `result.status == "valid"`
before interpreting `result.score`: an invalid `partial_errors` result may
still expose a numeric diagnostic over the remaining mutants. The CLI rejects
that run by default.

## Connect your own pipeline

The config factory accepts a replacement `run(prompt, case) -> str`. That
function must send **its `prompt` argument** and `case["facts"]` to the real
system, and return its JSON output text. Do not use the global original prompt,
replay a saved answer, feed the expected answer to the model, or impose the
mutated rules again in an unmodified wrapper.

Keep the same cases, mutation operators, and generator for the weak/strong
comparison. First run `muteval check` with your config and establish a passing
baseline. Retain errors and invalid run statuses; do not report a mutation score
over a silently filtered subset.

This example proves the integration and the constructed blind spot. Its 100%
result is **not** evidence about a real model, LLM judge quality, or coverage of
all profile requirements. Exact matching is suitable for these small extraction
fixtures, not open-ended personality descriptions with many valid answers.
For those, supply validated checks for your actual requirements, and inspect
changed survivors before treating them as regressions.
