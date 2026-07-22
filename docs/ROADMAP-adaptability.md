# Adaptability roadmap: make muteval a no-brainer to adopt

The goal of this track is a single thing: **shrink the distance between `pip
install muteval` and a first useful result on the user's *own* suite** — from the
~1-hour integration it is today toward the near-zero-config experience of mutmut
/ Stryker.

## The guiding insight (from mutmut)

mutmut is frictionless because it stands on **standardized infrastructure it can
auto-discover**: your source tree, `pytest`, and coverage data. It never asks you
to describe your system — it plugs into interfaces the ecosystem already agreed
on. LLM-eval land has none of those standards (no standard run function, eval
format, or judge), which is the root cause of every adoption seam.

So the strategy is two-pronged:

1. **Consume the standards that *are* emerging** (promptfoo/deepeval/ragas/pytest
   evals, callable/HTTP pipelines) so most users write no config at all.
2. **Make the wiring that remains thin, and repeated runs cheap** (caching,
   cheap-checks-first) so running it isn't a tax.

Everything below serves one of those two.

## Already shipped toward this (0.3.x)

- `muteval check` doctor — validates wiring + baseline, names the broken layer.
- `muteval init --template basic|rag` — runnable scaffolds, four blocks marked.
- Framework-free `checks` incl. `llm_judge`/`grounded` with `base_url=` for any
  OpenAI-compatible endpoint (no `json_schema` needed).
- deepeval / ragas / promptfoo adapters.
- Fail-closed validity gate + `docs/ADOPTION.md` ("where it breaks" guide).

## Build queue (priority order)

### P0-1 — Auto-consume existing eval setups (zero-config ingestion)
**Problem:** writing a `MutEvalConfig` is the biggest blank-page cost; most users
already have evals somewhere.
**Build:** one-command ingestion for the common formats, no config authored:
`muteval run --promptfoo promptfooconfig.yaml`, `--deepeval path/to/test_*.py`,
`--pytest path/`. Extend `from_promptfoo` into a general `from_*` family that
returns a ready `MutEvalConfig`.
**Effort:** M per format (promptfoo mostly exists).
**Done when:** a user with a promptfoo/deepeval suite runs muteval against it
without writing a config file.

### P0-2 — Reuse the user's pipeline (callable / endpoint targets)
**Problem:** asking users to wrap their app in a `run()` is friction mutmut never
imposes; they usually already have a callable or an HTTP endpoint.
**Build:** `--target package.module:function` (importable callable) and
`--endpoint URL` (OpenAI-compatible or a small adapter contract), mirroring the
`module:attribute` pattern. No `run()` written.
**Effort:** M.
**Done when:** muteval can drive an existing function or HTTP service as the
system under test with a flag, not a wrapper.

### P0-3 — Caching + incremental runs (results DB)
**Problem:** the cost multiplier (re-run the suite per mutant) is muteval's
biggest adoption tax, and CI re-runs pay it every time.
**Build:** a local results store keyed by (system hash, mutant, case, eval).
Re-runs skip unchanged mutants; `--since` / incremental mode only tests what
changed. This is mutmut's actual killer trick.
**Effort:** L.
**Done when:** an unchanged mutant is never re-graded; a CI re-run after a small
change costs a fraction of a full run.

### P1-4 — Cheap-checks-first execution
**Problem:** every mutant currently pays for LLM-judge calls even when a cheap
deterministic check would have killed it.
**Build:** order evals cheapest-first (rule-based before judge), short-circuit a
mutant as killed on the first failing check, and skip judge calls on
provably-output-unchanged mutants *before* spending them.
**Effort:** S–M.
**Done when:** most mutants are resolved with zero judge calls; judge spend
scales with genuinely-ambiguous cases only.

### P1-5 — Frictionless triage UX (results / show / report)
**Problem:** triaging survivors currently means reading one big terminal dump or
re-running.
**Build:** persist the last run; add `muteval results` (list survivors from last
run, ranked), `muteval show <id>` (baseline-vs-mutant output diff for one
survivor), and an HTML/Markdown report. Reuse existing severity + suggested-fix.
**Effort:** M.
**Done when:** a user can inspect a specific survivor's diff without re-running,
and share a report.

### P2-6 — Auto-detecting / interactive `init`
**Problem:** `init` still starts from a template, not the user's repo.
**Build:** `muteval init` detects nearby `promptfooconfig.yaml` / deepeval tests /
a likely pipeline module and offers to wire them ("found promptfooconfig.yaml —
use it? [Y/n]"), generating a config pointed at real files.
**Effort:** M.
**Done when:** in a repo that already has evals, `init` produces a config that
runs against them with no manual editing.

### P2-7 — Documented, stable plugin API
**Problem:** "adaptable" scales fastest when *others* adapt it (Stryker's model).
**Build:** promote `register_operator`, the adapter contract
(`adapters/base.py`), and the probe registry into a documented, versioned public
plugin API — custom operators, adapters, and reporters — with a short
"write-your-own" guide.
**Effort:** M (mostly docs + API stabilization).
**Done when:** a third party can add an operator/adapter/reporter against a
documented contract without reading core internals.

## The two that matter most

**P0-1 (auto-consume existing evals)** and **P0-3 (caching/incremental)** are what
actually separate muteval from mutmut-level frictionlessness — they're the two
things software got for free (a standard test interface, cheap deterministic
tests) that LLM-land doesn't have. Ship those and adoption stops requiring a
debugging marathon; the rest is polish that compounds on top.
