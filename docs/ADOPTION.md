# Adopting muteval on a real eval suite

Honest expectation first: running muteval against a **real** target (someone's
RAG app + their eval suite) is a **~1-hour integration, not plug-and-play.** The
mutation engine is turnkey; wiring it to *your* system is the work. This guide
tells you exactly what you supply, the order to do it in, and — most usefully —
**where it tends to break and what to change**, distilled from real integrations.

If you just want to see muteval work, start with the offline examples
(`examples/rag_context_offline/`, `examples/support_bot/`) — those need no keys
and no wiring. This guide is for pointing it at *your own* or a third-party suite.

---

## What you supply (the four pieces)

muteval is deliberately tool-agnostic: it works with anything because you provide
the glue. Every integration is these four things.

1. **A re-runnable pipeline** — `run(system, case) -> output_text`. Given the
   (possibly mutated) `System` and a case, produce a **fresh** output. This is
   muteval's hard requirement: it must be able to mutate the prompt/context/model
   and get a *new* output. Scoring a static file of pre-generated answers does
   NOT work.
2. **An eval suite** — `evals`, a list of `(output, case) -> bool | EvalOutcome`.
   These are the checks you're *grading*. They can be hand-written `checks`,
   deepeval/ragas/promptfoo metrics via the adapters, or your own functions that
   call a target repo's metric classes.
3. **A judge** (only if your evals use an LLM) — a model + endpoint + key.
4. **Test cases** — `cases`, the inputs fed to `run` and every eval.

`muteval init` scaffolds a config with these four blocks clearly marked —
`--template basic` (prompt-only) or `--template rag` (System mode; mutates the
retrieved context, runs keyless with a mock retriever you replace with yours).

---

## Prerequisites: does your target even fit?

Before investing, confirm BOTH (muteval can't help without them):

- [ ] **A real, programmatic eval suite** — actual metric calls (ragas/deepeval/
      custom `(output, expected) -> score`), not just a README mention.
- [ ] **A callable pipeline** you can invoke for a fresh output — a
      `query()`/`answer()`/chain, not a fixed answers.csv.

If the target only *scores a static file*, or has a pipeline but no eval suite (or
vice-versa), it does not fit. See `docs/OUTREACH-targets-and-plan.md` for a worked
vetting checklist.

---

## Step-by-step integration (do them in this order)

1. **Wire the pipeline into `run()`.** Return a serialized output that carries
   everything your evals need (answer text, retrieved ids, citations…). JSON is
   fine — the evals parse it back.
2. **Wire the metrics into `evals`.** Each returns `EvalOutcome(passed, score,
   threshold)`. If a metric is skipped/NA on a case, return `passed=True` (a skip
   is not a failure).
3. **Pick a judge** (if needed) — see the judge notes in the table below.
4. **Get a GREEN baseline.** The eval suite must PASS on the *original* (unmutated)
   system. muteval 0.1.4 **fails closed** if it doesn't — that's a feature, but it
   means the baseline is the first thing to get right.
5. **Preflight** (see below) — validate the wiring with ~4 calls before running.
6. **1-mutant trial** — `run_mutation_testing(config, sample=1)`; confirm
   `status == "valid"`.
7. **Full run** — remove `sample`, or cap it to control cost.

Steps 4–5 are where most time goes. Do NOT skip to the full run.

---

## Where it breaks, and what to change

This table is the real punch-list — the failure modes real integrations hit and
the fix for each. None of these are muteval bugs; they're the friction of wiring a
tool to a live system, and knowing them up front turns eight detours into two.

| Symptom | Cause | What to change |
| --- | --- | --- |
| `ImportError` / `ModuleNotFoundError` on the target's package | Target needs Python 3.11+ (uses `StrEnum`, `datetime.UTC`, `asyncio.timeout`) or a dependency SDK changed its API (e.g. `together.error` removed) | Run on the right Python, or add small compat shims for the missing stdlib symbols; stub a stale SDK's missing attribute. |
| Bizarre import crash (`'__file__' has no attribute 'endswith'`, `partially initialized module 'torch'`) | A heavy ML dep (torch/transformers/sentence-transformers) is version-mismatched or half-installed (common on Colab) | Don't install what you don't need. For small-corpus retrieval, use TF-IDF (scikit-learn) instead of sentence-transformers — no torch at all. |
| Importing a metric drags in torch you don't use | The target's `metrics/__init__.py` eagerly imports *all* metrics, incl. torch-based ones | Stub the unused torch modules in `sys.modules` before importing the metric you want (a `ModuleType` with `__getattr__` returning dummy classes). |
| `BadRequestError: model does not support response_format json_schema` | Your judge model doesn't support strict structured outputs (many free/open models don't) | Use a model that does (real OpenAI models, Groq `openai/gpt-oss-20b`/`120b`), OR monkeypatch the judge's `parse` to use `json_object` + manual pydantic validation (works on any model). |
| A metric scores 0 on obviously-correct answers | The judge is too weak/noisy for that metric (esp. claim-by-claim NLI) | Drop that metric, or switch to a stronger judge. Watch the judge-reliability probe. A noisy judge silently poisons the whole score. |
| **Baseline fails** on a correct system | A format mismatch (e.g. the model cites with `【doc-1】` full-width brackets but your extractor expects `[doc-1]`), or a noisy judge | Turn on verbose baseline debug: print each eval's score AND the actual output per case. Fix the extraction/format; only adjust thresholds if the baseline *legitimately* clears them. |
| `status: partial_errors` (CLI exits non-zero, badge n/a) | Too many mutants errored (usually free-tier rate/token limits) | Shrink the run (fewer cases × mutants × metrics), raise `--max-error-rate` if a few flaky calls are acceptable, or spread across days. muteval fails closed on purpose — the number over a shrunken denominator isn't trustworthy. |
| Everything shows up as `inert` / no mutants worth anything | The pipeline ignores the thing you're mutating (e.g. a promptless retriever) | Mutate the surface that actually drives behavior. For a promptless RAG pipeline, mutate **context** (`--operators drop_context_doc corrupt_context_doc …`), not the prompt. |
| Survivors that a human thinks are fine | Output text changed but behavior didn't (near-equivalent mutation); severity-by-pattern can over-flag | Inspect baseline vs mutant outputs (the human-agreement step). A raw mutation score should never be read without it — a survivor is only a real gap if a human agrees the mutant is meaningfully worse. |

---

## Choosing a judge

If your evals use an LLM judge, two things bite most often:

- **Structured-output support.** Many judge integrations (and some target repos)
  request a strict `json_schema` response, which **free/open models often don't
  support** (Groq's Llama, etc.) — you'll get a 400. Use a model that does
  (real OpenAI models; Groq `openai/gpt-oss-20b`/`120b`), or a judge that asks for
  plain text. muteval's own `checks.llm_judge` asks for a plain 0-10 score (no
  `json_schema`), so it works on any model.
- **Point the built-in judge anywhere.** `checks.llm_judge(rubric,
  base_url="…/v1", model="…")` (or the `OPENAI_BASE_URL` env var) hits any
  OpenAI-compatible endpoint — OpenAI, Groq, Gemini's compat API, GitHub Models,
  Ollama, a local server — using just `OPENAI_API_KEY`. So you can run a full
  suite on a free judge.
- **Judge reliability.** A weak judge gives noisy verdicts (e.g. 0.0 on a correct
  answer) and silently poisons the score. If a metric misbehaves, use a stronger
  judge or drop that metric — and watch the `judge_reliability` probe.

## Validate before you run: `muteval check`

Run the built-in doctor before a full run. It validates the wiring layer by layer
(cheapest first) and surfaces **per-eval baseline diagnostics**, so a wiring or
format bug costs ~1 call instead of a whole run:

```
muteval check --config your_config.py          # structural checks + 1-case run/evals
muteval check --config your_config.py --no-model   # 0-call structural checks only
muteval check --config your_config.py --full       # baseline over EVERY case
```

It reports, in order: config loads · cases/evals present · mutants generate (0
calls) · `run()` returns text (1 call) · each eval returns a valid outcome with its
score · baseline passes on the original system. A red baseline shows *which* eval
failed and its score — so a format mismatch (e.g. the citation-bracket bug above)
or a noisy judge is obvious instead of an opaque "baseline failed". Exit code is
0 when ready, non-zero otherwise (so you can gate CI on it). Fix the `FAIL` rows,
re-check, then `muteval run`.

---

## Cost & rate limits

muteval **re-runs the whole pipeline + eval suite for every mutant**, so cost ≈
`baseline × (1 + number of mutants)`. Judge-heavy suites multiply fast. Control it:

- Start tiny — a few cases, a curated ~8–10 mutants, the minimum metric set.
- Trim the metric suite to your hypothesis (testing grounding? citation +
  faithfulness is enough; you don't need every judge metric).
- On free tiers, the binding limit is usually **tokens- or requests-per-day**, not
  wall-clock. Pick a provider with headroom, keep the run small, and let
  fail-closed handling resume you if a cap trips.

See `docs/LIMITATIONS.md` for when to distrust the number.

---

## Minimal template

```python
from muteval import MutEvalConfig
from muteval.system import System
from muteval.evals import EvalOutcome

# 1. YOUR PIPELINE — return a fresh output for the (possibly mutated) system.
def run(system, case):
    ...
    return output_text

# 2. YOUR EVALS — grade the output; return EvalOutcome(passed, score, threshold).
def my_check(output, case):
    ...
    return EvalOutcome(passed=..., score=..., threshold=...)

# 3. YOUR JUDGE — only if an eval uses an LLM (endpoint/model/key). See the table.

# 4. YOUR CASES.
cases = [...]

config = MutEvalConfig(
    system=System(prompt="...", context=(...)),  # the mutation target
    cases=cases,
    run=run,
    evals=[my_check],
    max_error_rate=0.0,   # fail closed on any mutant error (raise to tolerate flaky judges)
)
```

Then: `muteval run --config your_config.py` (add `--dry-run` first to see the
mutant count without spending calls).
