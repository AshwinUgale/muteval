# Security policy

## Reporting a vulnerability

Please open a private security advisory on the GitHub repository, or email the
maintainer, rather than filing a public issue. We aim to acknowledge within a
few days.

## Threat model & trust boundaries

muteval is a developer tool you run locally / in your own CI. Two things are
worth understanding before you run it.

### 1. Config files are executed as code

`muteval run --config path/to/muteval_config.py` (and `muteval.load_config`)
**executes that Python file** to obtain the `config` object. A muteval config is
program code by design — it wires up your model call and your evals.

Consequently:

- **Only run configs you wrote or have reviewed.** Treat a muteval config like
  any other script in your repo.
- **Never run a config from an untrusted source** — one pasted into an issue,
  downloaded from the internet, or fetched at runtime. It would run with your
  privileges and can read your environment (including API keys).
- A declarative (YAML/TOML) config path for untrusted sources is on the roadmap
  (ROADMAP-master §2.1); until then, the `.py` config is trusted input.

### 2. Secrets

muteval never stores your API keys; it reads them from the environment at call
time (e.g. `OPENAI_API_KEY`, `GEMINI_API_KEY`) exactly like the SDKs do.

- **Machine-readable output is redacted.** `result_to_dict` (the `--json` output)
  scrubs anything matching common key formats (`sk-…`, `gsk_…`, `AIza…`,
  `Authorization: …`) before returning, so a mutated prompt or an error string
  that echoed a key cannot leak into CI logs or a committed badge. This is
  enforced by `tests/test_output.py`.
- Still, **treat muteval's stdout/verbose logs as potentially sensitive** if your
  prompts or eval code print secrets themselves — redaction covers the structured
  JSON contract, not arbitrary text your own `run`/evals emit.

## Supported versions

muteval is pre-1.0; security fixes land on the latest released minor. Pin a
version in CI and upgrade deliberately.
