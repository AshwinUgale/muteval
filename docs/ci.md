# Run muteval in CI

Your promptfoo suite tells you whether your assertions **pass**. muteval tells you
whether they'd **catch a regression** — it degrades your prompt and reruns *your*
promptfoo assertions to find the ones that protect nothing. Wiring that into CI turns
"the suite is green" into "the suite would actually notice if the prompt got worse."

## The GitHub Action (recommended)

muteval ships a composite Action. Add this to your repo at
`.github/workflows/mutation-test.yml`:

```yaml
name: mutation-test evals
on:
  pull_request:
    paths: ["promptfooconfig.yaml", "prompts/**"]
  schedule:
    - cron: "0 6 * * 1"      # weekly, Mondays 06:00 UTC
  workflow_dispatch: {}

jobs:
  muteval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: AshwinUgale/muteval@v0.7.0     # pin a release tag
        with:
          config: promptfooconfig.yaml
          fail-on-severity: high             # fail on any HIGH-severity blind spot
          sample: 30                         # fast/cheap; drop for a full run
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

A ready-to-copy version with artifact upload lives in
[`examples/github_action/mutation-test.yml`](https://github.com/AshwinUgale/muteval/blob/main/examples/github_action/mutation-test.yml).

### Inputs

| Input | Description |
|-------|-------------|
| `config` (required) | Path to your `promptfooconfig.yaml`. |
| `fail-on-severity` | Fail if any surviving mutant is at/above `high` \| `medium` \| `low`. |
| `fail-under` | Fail if the mutation score is below this percent. |
| `model` | Override the model under test. Default: read from your `providers:` block. |
| `base-url` | OpenAI-compatible base URL (Groq / Gemini-compat / Ollama / local). |
| `sample` / `seed` | Sample N mutants for a fast, reproducible run. |
| `cache` | sqlite cache path so an identical re-run makes zero model calls. |
| `max-calls` | Cap model+judge calls; fails closed before overspending. |
| `junit` / `badge` | Write JUnit XML / a shields.io coverage-badge endpoint. |
| `extra-args` | Any extra flags passed through to `muteval run`. |

## Without the Action (any CI)

It's just two commands, so any CI works:

```yaml
- run: pip install "muteval[promptfoo]"
- run: muteval run --promptfoo promptfooconfig.yaml --fail-on-severity high --sample 30
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Notes worth knowing

- **It needs an API key.** muteval regenerates the system-under-test's output to test
  your assertions, so even a suite of pure `contains`/`regex` asserts needs a model call.
- **It runs your real model.** muteval reads the first entry of your `providers:` block, so
  it mutates your prompt against the model your suite uses — not a default. A provider it
  can't call directly (e.g. a native Anthropic endpoint) falls back to `gpt-4o-mini` with a
  warning; pass `base-url` + `model` to point it at your real one.
- **Schedule it, don't gate every PR on a full run.** Mutation testing runs your suite once
  per mutant, so it's heavier than a single eval. Use `sample`/`cache` on PRs, or run the
  full thing on a weekly schedule.
- **The gate is honest.** muteval exits non-zero only when it produced a *trustworthy*
  result below your threshold; a failed baseline or an empty run is reported as invalid, not
  a false pass.
