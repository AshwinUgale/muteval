# Changelog

All notable changes to muteval are documented here. This project adheres to
[Semantic Versioning](https://semver.org) (pre-1.0: minor versions may introduce
additive features; the public API is not yet frozen — that lands at 1.0).

## [Unreleased]

Operators, severity & scope: every mutant an edit a real person could make,
severity from WHAT changed, scoping fast and exact. Mutation scores on the
bundled examples are unchanged or move by one mutant; survivor severities move
(both ways) and some survivor signatures change (see below).

- **Severity comes from what the edit touched, not its description.** Escalation
  matched the description, which always contains the modal word itself
  (never/always/must/do not) plus neighbouring text, so almost every weakened or
  dropped rule ranked HIGH ("Never use emojis" next to corrupted retrieval) and
  substrings escalated ("mustard", "nevertheless", "illegally", "police"). Mutants
  now carry a `focus` (the original line/sentence/doc); escalation needs a
  safety/correctness CONTENT word there, word-bounded (refunds, customer data,
  passwords, citations, "don't know", guessing/inventing, ...). `truncate_prompt`
  and `remove_emphasis` now escalate on what they cut. When two operators produce
  the same mutant, the more severe framing is kept.
- **Scoping is fast and exact.** Any active scope ran a character-level diff over
  the whole prompt per mutant: 287 s on a 5k-char prompt with
  `--scope-include` (now 0.03 s). A marker hugging a line or sentence rejected
  that line's own deletion (the line break sits outside the marker).
  `--scope-include never` kept an "Always -> never" flip (it matched the MUTATED
  text); include/exclude now match your original lines. Stray or nested
  `[[/mutate]]` markers, which leaked into the prompt, are rejected.
- **Operators no longer produce edits no one would make:**
  - `weaken_modals`: "must not" -> "should not" (was "should avoid share");
    no second overlapping "must" mutant; "the only exception" / "if and only if"
    / "You are required to" left alone or weakened grammatically; "You do not
    have access" isn't treated as a command; "If unsure, do not guess" and
    "**Do not**" are.
  - `flip_negation`: contractions (can't, won't, doesn't, shouldn't, curly ’);
    never "not always" -> "not never"; never inside a quoted literal.
  - `weaken_numeric_threshold`: direction from the phrase attached to the number
    ("no fewer than 3" was TIGHTENED to 6); "0.5" and "1,000" are one number
    ("0.10", "1,1" before); list markers and versions are skipped.
  - `remove_emphasis`: only UPPERCASE markers with ":" (it deleted "Note that",
    "Important details" and blank lines, and turned `__init__` into `init`); now
    also de-emphasizes ALL-CAPS NEVER/MUST/ONLY (YES/NO untouched).
  - `drop_few_shot_example`: drops blocks SHAPED like demonstrations (repeated
    "Label: content" lines, or Input/Output, Q/A, User/Assistant), not any block
    mentioning "example" or "output:"; the rest of the prompt is byte-identical.
  - `delete_sentences` doesn't split at "e.g."/"i.e." or delete headings and
    lead-ins; `drop_instruction_lines`/`swap`/`paraphrase` skip headings and short
    lead-ins ("## Rules", "Follow these steps:").
  - `truncate_prompt` cuts the tail of the instruction lines wherever the input
    placeholders sit (it never fired with a placeholder on line 1).
  - `corrupt_context_doc` changes a FACT number, not an id ("doc-1" -> "doc-2"
    made the RAG quickstart's mutants inert), never produces "do not not" /
    "can not't", and names the edited token in its description.
  - Structured (dict) docs and tool outputs: no crash (a dict doc aborted
    generation), and swap/deny/corrupt keep the output's type.
  - Robustness operators (#56 follow-ups): paraphrase never edits quoted
    literals and tidies "JSON,." / sentence capitals; swap never reorders
    numbered steps.
  - More placeholder forms protected: `{0}`, `{}`, `{case.q}`, `{q!r}`,
    `{q:>10}`, `$question`, `%(q)s`, `%s`.
- **The RAG quickstart shows what it promises.** `muteval init --template rag`
  reported 0 survivors (its mock ignored the prompt). The mock now obeys the
  prompt's abstention rule and the cases include an unanswerable question, so
  the run surfaces the real gap: nothing checks the "say you don't know" rule.
- **`muteval check` noise check.** It now calls `run()` twice and grades each
  LLM judge twice on one case, and WARNs (without blocking) when either varies
  at settings that can't absorb it. It also reports a robustness-only operator
  set as not ready, and accepts dict outputs (the `{"final", "trace"}` agent
  bridge was a fatal "expected str").
- `--dry-run` mirrors the real run's validity: exit 2 when no scored mutant
  would run (it exited 0 for a run that would be invalid), and it splits scored
  vs robustness mutants.
- `suggest`: a fix for a line with quotes in it isn't cut at the first quote and
  is valid Python. `autofix`: samples the case whose output changed, and verifies
  a candidate by the suite's `runs_per_mutant` majority.
- Signatures change for mutants whose descriptions now name their content
  (`swap_context_doc`, `clear_context`, `shuffle_context`, `truncate_context_doc`,
  `truncate_prompt`, `remove_emphasis`, `corrupt_*`, tool operators) — so an
  accepted survivor no longer survives a completely different doc. Re-accept.

Cache & concurrency: an optimization must never change a verdict. Default runs
without `--cache` / `--concurrency` score exactly as before.

- **The cache keys on what an eval does, not its name.** Outcomes were keyed on
  the eval's LABEL: editing `contains("X1")` to `contains("ZZZ")` kept serving
  the old verdicts (a baseline that should fail came back valid), a changed
  threshold kept the old score, and two evals sharing a label shared one result
  (`--check contains:8080 --check contains:BANANA`). Outputs ignored the `run`
  function, so editing `run` (e.g. the model it calls in prompt mode) served
  stale outputs. v2 keys: an output on the system + case + a fingerprint of
  `run`; an outcome on the output + case + a fingerprint of the eval (its code,
  closure values, defaults, thresholds, the simple globals it reads; new
  `muteval.fingerprint`). Old cache entries are never read. Set `cache_version`
  on an eval that depends on a file or remote rubric. The result reports how
  many lookups the cache served (JSON `cache`).
- **The cache replays `run()`'s writes into the case.** A cache hit skipped
  `run()`, so an eval reading `case["used_context"]` graded stale state. The
  post-run case is now stored with the output and restored on a hit. Dict
  outputs (the `{"final", "trace"}` agent bridge) are cached instead of
  crashing sqlite; anything that can't round-trip through JSON just isn't cached.
- **`--concurrency` no longer changes verdicts.** Cases and deepeval metric
  objects were shared across threads: a stateful metric scored 0%–85% on one
  suite, and a `run()` writing into the case leaked into other mutants. Each
  (mutant, case, run) now gets a private copy of the case, and the deepeval
  adapter measures a shallow copy of its metric per call (a lock if it can't be
  copied). Queued mutants are cancelled once `--max-calls` is hit.
- **Skip-unchanged no longer changes verdicts.** It reused the baseline's pass
  whenever the output was identical, even when `run()` had written different
  data into the case (the mutated context an eval grades against): 0/11 kills
  vs 4/11 with it off. It now also requires the post-run case state to match.
- **Adapter judges are budgeted.** The deepeval and ragas adapters never set
  `is_llm`, so `--max-calls 20` made 80 paid calls, the judges ran before cheap
  checks, and the "free" canary called them. They're now tagged.
- **`System(context=[...], tools=[...])` works.** The README's own form crashed
  mutant generation (`unhashable type: 'list'`); lists are normalized to tuples.
- Eval labels are unique and aligned: duplicates get `#2`, `#3` (two deepeval
  `GEval` metrics), a shorter `eval_names` list is filled in, and a longer one is
  an error. Zero-config checks are labelled by their full spec (`contains:8080`).
- Smaller: `--max-mutants -1` silently dropped the last mutant (now rejected,
  as is a negative `--sample`); `System.key()` crashed on `extra` with mixed key
  types; one unkeyable baseline sample marked its case undetermined; the ragas
  adapter split a string context into characters.
- `Cache.get_outcome`/`set_outcome` now take `(output, case, eval_fingerprint)`
  instead of `(system, case, label)`; `Cache.lookup_output`/`store_output` carry
  the post-run case. (Passing a `Cache` to `run_mutation_testing` is unchanged.)

Security: secrets and the judge endpoint.

- **One redaction point for every output.** Only the JSON and manifest were
  redacted; a provider error that echoed a key (e.g.
  `…generateContent?key=AIza…`, `401: Bearer …`) was printed verbatim in the
  terminal report, JUnit, `muteval check`, and CLI error lines. All outputs now
  go through `muteval.redact` (terminal, JSON, JUnit, HTML — including an old
  unredacted JSON fed to `muteval report` — manifest, doctor, probe card, CLI
  errors).
- **The pattern covers what it missed.** `Bearer <token>` (the token survived
  `Authorization: Bearer …`), `OPENAI_API_KEY=…` / `GITHUB_TOKEN: …` style
  assignments, `"api_key": "…"`, GitHub (`ghp_`, `github_pat_`), Hugging Face,
  xAI, Slack and AWS keys, and URL query credentials. Plus the exact values of
  secret-named environment variables, for key formats no pattern knows. False
  positives like `max_tokens: 256` are left alone.
- **The zero-config judge no longer sends your key to OpenAI.** A `judge:<rubric>`
  check ignored `--base-url`, so with a Groq/Gemini/GitHub Models setup the judge
  called api.openai.com with that provider's key. It now uses `--base-url`; new
  `--judge-base-url` / `--judge-model` pick a different judge endpoint explicitly.

Gates & validity: from a full audit of the trust core, every way a run could be
scored or gated on evidence it doesn't have. Default single-run suites score
exactly as before; runs with a noisy judge, ties, or odd eval return values now
fail closed instead of passing.

- **Unresolved ties are gated.** An all-tied run used to be `valid` with no
  score: `--fail-on-severity` exited 0 and `--fail-under` crashed with a
  `TypeError`. One resolved mutant of 17 read "100%" and passed. Now all-tied is
  `no_confident_score`, and any tie above `max_unresolved_rate` (new; default
  0.0 = fail closed, like `max_error_rate`; CLI `--max-unresolved-rate`) is
  `partial_unresolved`. An odd `runs_per_mutant` can't tie.
- **The built-in judge reads the score it asked for.** The parser took the LAST
  number and only rescaled values > 1, so "0/10" and "3/10" parsed as 1.0 (a
  perfect pass) and a bare "1" meant 1.0. It now reads "N/10" / "N out of 10",
  else the first number, on the 0-10 scale; an empty reply, no number, or a
  value outside 0-10 raises, so the mutant is *errored* (it used to be a 0.0
  score — a kill).
- **The baseline is judged by the mutants' rule.** It was graded once while
  mutants needed a majority of `runs_per_mutant` runs; a flaky judge that would
  "kill" the unmodified original half the time still yielded a valid run. Now
  the baseline is graded `runs_per_mutant` times: if the original would itself
  be killed (or ties), the run is `baseline_failed`, and the report shows the
  original's pass rate as a noise floor. JSON: `baseline_pass_rate`.
- **Noise kills leave the effective score.** A kill whose outputs match what the
  original itself produced (any baseline sample, under `output_key`) didn't
  change behavior, so it isn't detection. It is now excluded from the effective
  score like an inert survivor; a prompt-independent system no longer scores an
  effective 100%. `--fail-under` gates on the lower of the raw and effective
  scores (identical to the raw score when there are no noise kills). JSON:
  `noise_kills`.
- **Non-verdicts fail closed.** A NaN metric score counted as a kill (`nan >=
  threshold` is False): the ragas adapter and `scorer_to_eval` scored 1.0 on a
  metric that returned NaN everywhere. An eval returning a number (`0.2`) or a
  string (`"fail"`) was truthy, so it passed. Both now raise, so the mutant is
  *errored*. A promptfoo-style `{"pass": bool, "score": ...}` dict is accepted.
- `--fail-under` must be a percent in [0, 100]; a fraction such as `0.8` (the
  style `--max-error-rate` takes) is rejected instead of passing almost any run.
- A run where only robustness operators ran has its own status,
  `no_scored_mutants`; the CLI no longer says "every mutant errored" for it.

Mutant realism: fixes found by running muteval on a real external suite (a
promptfoo PR-scope classifier), where malformed, meaning-preserving,
input-deleting and wording-only mutants distorted the score. **Scores can move**
on existing suites: they get more honest, not uniformly higher or lower. Some
survivor signatures change (`delete_sentences`, `paraphrase_instruction`,
`truncate_prompt`); re-accept those if you use `--accept-file`.

- **Robustness operators are no longer scored.** `paraphrase_instruction` and
  `swap_adjacent_instructions` make meaning-preserving edits, so a survivor is
  healthy, not a coverage gap. They still run; the report lists any that flipped
  a verdict ("an eval may key on wording/order"), and JUnit marks them skipped.
  Custom operators can opt in via `register_operator(name, fn,
  intent="robustness")`. Both also get a LOW base severity; they previously had
  no entry and fell back to MEDIUM, then escalated to HIGH.
- **Mutants never delete the input.** Any mutant that drops an input
  placeholder (`{{var}}`, `{var}`, `${VAR}`) is discarded: a guaranteed kill
  that inflated the score. `truncate_prompt` now only truncates the instructions
  above the input block.
- **`delete_sentences` makes one change.** It used to collapse every newline in
  the prompt too, and glue unpunctuated lines (headings, the input template)
  into one "sentence". It now removes only the sentence's span. When a line held
  a single sentence, the mutant now dedupes against `drop_instruction_lines`
  (it was the same deletion, previously counted twice).
- **No ungrammatical mutants.** `weaken_modals` only weakens an imperative
  `do not` ("Do not X" → "Try not to X"), never a descriptive one ("changes do
  not make …" → "changes try not to make …"). `paraphrase_instruction` rewrites
  `do not` → `never` (was "avoid follow …"), `make sure` → `be sure`,
  `ensure` → `make sure`, `in order to` → `to`, and tidies deletions (no
  stranded "not", no leading space, the sentence's capital restored).
  `weaken_modals` and `flip_negation` preserve case ("Do not" → "Do", "MUST" →
  "SHOULD"). `flip_negation` still inverts descriptive rules, since that is a
  real inversion.
- **Behavior-level equivalence (opt-in).** New `output_key=` (the part of the
  output that *is* the behavior, e.g. a classifier's label) and
  `baseline_runs=N` (sample the baseline's variance; outputs only, no extra
  judge calls). With free-text output, every survivor used to look like a real
  gap because the wording always differs. Survivors whose output couldn't be
  told apart from baseline noise are flagged *undetermined*. Defaults are
  unchanged.
- The effective-score line now uses the resolved denominator (it used
  `evaluated`, which disagreed with the number when there were unresolved ties).
- JSON: `robustness`, `brittle`, `noisy_cases`, `undetermined`
  (`schema_version` → 6).

## [0.11.0] — 2026-09-16

- **Accept a survivor as "untested by design".** Each survivor now shows a stable
  `accept: <signature>` (operator + the exact edit); list those in a JSON file and
  pass `muteval run --accept-file PATH` (or `config.accepted_survivors=[...]`) and
  they split out of the actionable list and stop tripping `--fail-on-severity` — so
  a decided gap stops resurfacing as noise. Signatures are tied to the change text,
  so editing that part of the prompt re-surfaces the accepted mutation (it's a new
  decision). The mutation score is unchanged — the eval still doesn't cover it. JSON
  gains a per-survivor `signature`/`accepted` and a top-level `accepted` count
  (`schema_version` → 5).
- Add a keyless Autoevals JSON-profile example using `scorer_to_eval`: compare
  JSON-only checks with exact-profile checks on two controlled prompt mutations.

## [0.10.0] — 2026-09-11

- **Flaky verdicts are now attributed per eval.** The report and JSON
  (`flaky_by_eval`) show which eval *dimension* the flips came from — a rubric a
  judge can't answer consistently is a bug in the eval question, so rewrite that
  dimension before adding runs.
- **Judge/model provenance in the result.** Records the model under test and any
  judge model muteval can introspect (its own `llm_judge` / `grounded`) as
  `model_under_test` / `judge_models`, so scores are comparable across time — a
  silent model bump replaces the coin a majority vote stabilizes. (`schema_version`
  → 4.)
- **Unresolved (tied) verdicts are now first-class.** Under strict majority, a
  dead-even split over `runs_per_mutant` (the judge straddled 50%) used to
  silently default to "survived". It's now marked `unresolved` and excluded from
  the score's numerator AND denominator — the score, its Wilson CI, and the
  survivor list are computed over the *resolved* set, with the unresolved count
  reported separately (and an all-tied run reports no confident score rather than
  a misleading one). `resolved` + `unresolved` added to the JSON
  (`schema_version` → 3). Only affects even `runs_per_mutant`.
- New `weaken_numeric_threshold` operator: loosens a numeric constraint in the
  prompt (an upper bound goes up, a lower bound down — "at most 3" → "at most 6"),
  firing only on a number near a bound word. Aimed at a behavior class that
  faithfulness/relevancy suites say nothing about, so it discriminates suites
  whose mutation score is otherwise inflated by prompt-tail operators.
- New opt-in positive control: `muteval run --canary` (or `run_mutation_testing(..., canary=True)`)
  feeds the rule-based checks a blank and a nonsense output and warns if the suite
  passes both — i.e. it may not be discriminating (or it's a guardrail-only suite).
  Separates a genuine 0% mutation score from a harness that isn't scoring. Off by
  default (it calls the checks an extra time; skips LLM judges). `canary_caught`
  is added to the JSON (`schema_version` → 2).

## [0.9.0] — 2026-08-21

- Survivor IDs in `muteval results` and `muteval show` now start at 1 for more
  natural human-facing CLI output.
- `muteval run` now numbers survivors (`#1`, `#2`, …), matching the IDs used by
  `muteval results` / `muteval show`, so you can inspect one without re-running.
- **Fix (promptfoo adapter):** an `llm-rubric` / `model-graded` assertion on a
  suite whose vars aren't named `input` now shows the judge the case's real vars
  instead of `User input: None`. An existing `input` var is left untouched. (#36)
- **tracelint integration (agent suites).** A new `deny_tool_output` operator
  mutates a tool output into a domain failure returned as transport success
  (HTTP 200 carrying `{"status": "declined"}`) — the fault structured-error
  detection is blind to. A new deterministic, no-judge eval `checks.tracelint()`
  (behind the `muteval[tracelint]` extra) lints the agent's execution trace and
  kills such a mutant even when the final answer still reads clean, and
  `checks.on_final()` lets ordinary output checks grade the `{"final","trace"}`
  bridge. When a tool-fault mutant survives, the report now names the exact
  deterministic check that would catch it. See `examples/agent_tool_fault/`.

## [0.8.0] — 2026-07-28

- **promptfoo: run the model your suite actually uses.** The adapter now reads the
  model under test from the promptfoo `providers:` block instead of always defaulting
  to `gpt-4o-mini`; an explicit `--model` still wins, and a provider muteval can't call
  directly falls back with a warning. (`from_promptfoo` now defaults `model=None` = auto.)
- **promptfoo: graceful degrade on unsupported asserts.** A case whose assertions are all
  untranslatable types (javascript/python/…) is now *dropped with a warning* instead of
  aborting the whole run; muteval fails closed only if nothing in the suite is gradeable.
- **promptfoo: external test files + more assert types.** `tests: file://cases.csv` (also
  `.jsonl`/`.json`/`.yaml`) and an external `defaultTest: file://…` are now loaded instead
  of crashing; code-function / remote sources (`.py:fn`, `https://`, `huggingface://`) get a
  clear error, not a traceback. Added `contains-any`/`-all`, `icontains-any`/`-all`,
  `not-equals`, `starts-with` assertion translations. Verified against promptfoo's own 194
  example configs: clean build rate **88 → 100**, cryptic errors **19 → 0**.
- **GitHub Action** (`action.yml`) — mutation-test your promptfoo suite in CI in a few
  lines; see `docs/ci.md` and `examples/github_action/mutation-test.yml`.

- **Keyless promptfoo demo** (`examples/promptfoo_offline/`) — `muteval run
  --config examples/promptfoo_offline/muteval_config.py` degrades a support-bot
  prompt and finds the rule its promptfoo suite forgot to assert, with **no API
  key** (a deterministic mock model). Plus a recipe README and a walkthrough
  (`blog/mutation-test-your-promptfoo-suite.md`) for adopting muteval on an
  existing promptfoo config.

## [0.7.0] — 2026-07-24

- Add optional JUnit XML output via `muteval run --junit PATH`.

Adoption pass, driven by a three-way audit of onboarding, integration, and UX.
All additive — no behavior a 0.6 user relied on was removed.

### Reach the easy on-ramps
- **`check`, `probe`, and `label` now accept the same inputs as `run`** —
  `--promptfoo` and the zero-config flags, not just a Python `--config`. The
  doctor and the probe report card finally work on every entry point.
- **`muteval list [operators|checks|probes]`** — discover the operators, built-in
  checks, and probes from the CLI.
- **Clean config errors** — a hand-edited config that raises (SyntaxError,
  NameError, …) now prints `your config <path> raised <Error>`, not a traceback.
- **`muteval run` auto-picks `./muteval_config.py`** when no source is given.
- **`eval_names` auto-derived** from your eval function names — no parallel list to
  hand-duplicate.
- deepeval/ragas adapters raise a `pip install "muteval[…]"` hint when missing.

### Any provider for the system under test
- **`--base-url` / `OPENAI_BASE_URL`** for the model under test (not just the
  judge) — Groq, Gemini-compat, GitHub Models, Ollama, a local server. Threaded
  through zero-config and the promptfoo adapter.

### promptfoo adapter, honest
- **One eval per assertion type** (`promptfoo:contains`, `promptfoo:llm-rubric`) so
  survivors and severity stay per-check.
- **Warns** on skipped unsupported assertions (is-json/javascript/…) and
  **refuses** a case whose assertions are all unsupported — instead of passing it
  vacuously and inflating the score.

### Custom targets
- `--endpoint` POSTs `context`/`model`/`tools` too (retrieval/model mutations reach
  a deployed pipeline), plus `--header` for auth. muteval warns when
  `--target`/`--endpoint` is combined with context/model mutation.

### CLI polish
- `run --help` flags grouped (input / mutation / cost & speed / CI gates / output).

[0.7.0]: https://github.com/AshwinUgale/muteval/releases/tag/v0.7.0

## [0.6.0] — 2026-07-23

The first release since 0.3.1, packaging three internal milestones: "provably
honest" (verification hardening), "adopt in an hour" (ingestion + performance),
and "the eval-evaluator, validated" (the probe layer). Everything below is
additive — no behavior a 0.3.x user relied on was removed. The fail-closed
validity gate, Wilson CIs, and majority-vote stability from 0.3.x are unchanged
and now backed by reference cross-checks and Monte-Carlo coverage tests.

### Trust & verification
- **Reference cross-checks** against `statsmodels`, `scipy`, `scikit-learn`,
  `krippendorff`, and `pingouin` (behind the test-only `[verify]` extra):
  Wilson/Jeffreys intervals to 1e-6, AUC/Spearman to 1e-9, Krippendorff's alpha,
  Cohen's d, and ICC(2,1) all validated against the established libraries.
- **Property-based tests** (Hypothesis) over the statistics and the runner
  (intervals stay in `[0,1]`, `killed ≤ evaluated ≤ total`, `effective ≥ point`,
  CI brackets the point estimate).
- **Monte-Carlo coverage** — Wilson and Jeffreys intervals empirically cover in
  `[0.93, 0.97]` across a `p × n` grid.
- **Determinism** — a single `seed` threads through the whole run; same config +
  seed produces byte-identical JSON on every OS × Python version.
- **Secret redaction** — API keys never appear in emitted JSON or logs;
  `schema_version` added to the result payload.
- **CI matrix** — Python 3.9–3.13 × ubuntu/macos/windows, 90% coverage gate,
  `mypy` type-check gate, and muteval dogfooded with `mutmut`.
- **Jeffreys (Beta-Binomial) interval** added alongside Wilson for very small n.

### Adoption & performance
- **Zero-config ingestion** — run straight from a `promptfoo` config
  (`--promptfoo`), a deepeval test file, or a pytest path; no `.py` config needed.
- **Bring-your-own target** — point at a callable (`--target pkg.mod:fn`) or a
  deployed HTTP endpoint (`--endpoint URL`); no `run()` wrapper required.
- **Caching** — `--cache runs.sqlite` memoizes outputs + eval outcomes; an
  identical re-run makes zero model/judge calls.
- **Concurrency** — `--concurrency N` evaluates mutants in parallel with
  order-preserving, serial-identical results.
- **Cost control** — `--max-calls` / `--budget-usd` fail closed before overspend;
  cheap rule-based evals run before judges and short-circuit kills.
- **Triage UX** — last run persisted to `.muteval/last_run.json`; `muteval
  results` (ranked survivors), `muteval show <id>` (baseline→mutant diff), and
  `muteval report --html` (shareable standalone report).
- **Typing & plugins** — `py.typed` ships; `docs/PLUGINS.md` documents the
  operator/probe/adapter/reporter extension points with a contract test.

### The eval-evaluator (`muteval probe`)
- **Report card** across six lenses, no composite score: judge reliability
  (flip-rate + Krippendorff's alpha + ICC(2,1)), discrimination (AUC + Cohen's d),
  statistical adequacy (Wilson/Jeffreys), redundancy (Spearman + connected
  families), threshold calibration, and **human agreement** (Cohen's κ via
  `muteval label`). A separate **judge-bias panel** (position/verbosity/
  self-preference) ships as a library function for pairwise A/B judges — it needs
  a pairwise-judge harness, so it is not part of the default card.
- Every probe has a CI test asserting its signal is monotonic in injected
  severity and hits its endpoints.
- **Autofix verify loop** — `autofix.suggest_and_verify` proposes an eval for a
  survivor and confirms it actually kills the mutant while the baseline stays
  green; only verified suggestions are returned.
- **Eval-quality proof** extended to four CI-enforced domains (support bot, code
  review, RAG, HR policy): score rises monotonically 0% → 100% with coverage.
- `muteval probe --html` renders the report card.

### Fixes
- Force UTF-8 stdout so the CLI report renders on Windows consoles (cp1252)
  instead of raising `UnicodeEncodeError`.
- Satisfy the `mypy` verify gate (`stream.reconfigure` probe; typed `operators`).

## [0.3.1] and earlier

See the git history. 0.3.x delivered the fail-closed validity gate, partial-error
handling, Wilson confidence intervals, the `muteval check` doctor, the RAG
scaffold (`init --template rag`), OpenAI-compatible `base_url` judges, and the
first four probes upgraded to their prior-art methods.

[0.6.0]: https://github.com/AshwinUgale/muteval/releases/tag/v0.6.0
