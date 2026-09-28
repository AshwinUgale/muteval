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

- **Everything muteval emits is redacted, through one function**
  (`muteval.redact.redact`): the terminal report, `--json`, JUnit, HTML, the
  manifest, `muteval check`, the probe card, and CLI error lines. (Before
  0.12, only the JSON and manifest were; a provider error that echoed a key was
  printed verbatim to the terminal and JUnit.) Two layers:
  - **Known shapes:** provider key prefixes (`sk-…` incl. `sk-proj-`/`sk-ant-`,
    `gsk_…`, `AIza…`, `ghp_…`/`github_pat_…`, `hf_…`, `xai-…`, `xox?-…`,
    `AKIA…`), `Bearer <token>`, credential assignments (`OPENAI_API_KEY=…`,
    `"api_key": "…"`, `token: …`, `Authorization: Bearer …`), and URL query
    credentials (`?key=…`, `&access_token=…`).
  - **Your actual secrets:** the exact value of any environment variable whose
    name contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL` or `AUTH`
    (8+ chars), so a key in a format no pattern knows (Azure, a proxy token)
    is still caught when it came from your environment.

  Enforced across every output format by `tests/test_secret_redaction.py`.
- **The judge only calls the endpoint you configured.** A zero-config
  `judge:<rubric>` check uses `--base-url` (override with `--judge-base-url` /
  `--judge-model`), so `OPENAI_API_KEY` is sent only where you pointed it. (Before
  0.12 the judge ignored `--base-url` and sent the key to api.openai.com.)
- Still, **treat logs as potentially sensitive** if your own `run`/evals print
  secrets directly (muteval doesn't intercept their stdout), or a secret sits in
  your prompt in a shape no pattern knows and it isn't in your environment.
  Redaction is defense in depth, not a guarantee.

## Supported versions

muteval is pre-1.0; security fixes land on the latest released minor. Pin a
version in CI and upgrade deliberately.
