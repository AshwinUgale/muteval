# Contributing to muteval

Thanks for your interest! muteval is early-stage, which means there's lots of
high-impact work available and your contribution can shape the project.

## Good first contributions

- **New mutation operators.** The highest-value area. An operator takes a
  prompt and yields `Mutant`s. See `src/muteval/mutators.py` — adding one is
  ~15 lines plus a test. Ideas: negation flips, reorder instructions, truncate
  the prompt, swap examples in few-shot prompts.
- **Tool adapters.** Let users point muteval at an existing promptfoo or
  deepeval suite instead of writing a config by hand.
- **Reports.** Markdown/HTML output, a score badge, a JSON export.
- **Docs and examples.** A real (API-backed) example for OpenAI/Anthropic.

## Development setup

```bash
git clone https://github.com/REPLACE_ME/muteval
cd muteval
pip install -e ".[dev]"
pytest
muteval run --config examples/support_bot/muteval_config.py
```

## How to add a mutation operator

1. Write a function `my_operator(prompt: str) -> list[Mutant]` in
   `mutators.py`.
2. Register it in the `OPERATORS` dict.
3. Add a test in `tests/test_mutators.py`.
4. Each `Mutant` needs a clear `description` — it's what users see in the
   survivor report, so make it actionable.

## Guidelines

- Keep the core dependency-free where possible. Optional integrations
  (deepeval, promptfoo, model SDKs) go behind extras.
- Every operator and public function gets a test.
- Run `pytest` before opening a PR.
- Open an issue to discuss anything larger than a single operator before you
  build it, so we can agree on the approach.

## Code of Conduct

Be kind and constructive. We follow the
[Contributor Covenant](https://www.contributor-covenant.org/).
