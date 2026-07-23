# Findings: does muteval's mutation score measure eval-suite quality?

muteval's central claim is a *meta* one: the mutation score isn't a property of
your system, it's a property of your **eval suite** — a higher score means your
evals would catch more real regressions. This document reports what we've
actually measured against that claim, and is deliberately honest about what the
evidence does and does not show.

## TL;DR

- In a controlled experiment where a deterministic system genuinely reacts to
  prompt and context mutations, the effective mutation score **rose
  monotonically with eval-suite coverage, from 0% (empty suite) to 100%
  (complete suite)** — and it holds across **four independent domains**
  (support bot, code review, RAG grounding, HR policy), each climbing 0% -> 100%
  with coverage (e.g. support bot `0 -> 33 -> 67 -> 100%`; code review
  `0 -> 35 -> 71 -> 100%`). The relationship is **enforced in CI**
  (`tests/test_eval_quality.py`) across all four, so it can't silently regress.
  The metric behaves as advertised.
- On a **live GPT-4o-mini run**, a standard faithfulness + relevancy RAG suite
  caught **0 of 24** prompt regressions; adding one unanswerable case + an
  abstention check raised it to **25% (6/24)**, killing exactly the mutants that
  break the abstain/don't-invent guardrail.
- Survivors are actionable: adding the specific evals that cover a survivor
  *kills* it and lifts the score, which is the intended workflow.
- Some survivors are **equivalent mutants** (a weakening that doesn't change
  behavior, or dropping a context doc irrelevant to the case). These are a known
  limitation of mutation testing and are discussed below, not hidden.

## The claim, stated precisely

> If eval suite B catches strictly more behavioral regressions than suite A,
> then `mutation_score(B) >= mutation_score(A)` on the same system and mutants.

This is the falsifiable core. If a stronger suite did *not* score higher, the
metric would be measuring noise. So we built suites of strictly increasing
coverage over one system and measured.

## Experiment 1 — controlled, deterministic, no API

`validation/eval_quality_experiment/run_experiment.py`

We use a support-assistant prompt with four rules (cite the order ID; don't
promise refunds; never reveal another customer's data; be polite) plus two
retrieved context documents. The "model" is a deterministic function whose
output **faithfully reflects which rules survive in the prompt and which
documents survive in the context** — so a mutation that drops or inverts a rule
genuinely produces a violating answer, exactly as a degraded real system would.
The eval checks are independent of the prompt.

We then grade the *same* system + mutants with four suites of increasing
coverage (effective mutation score, i.e. excluding output-unchanged mutants):

| Suite | Checks added | Support bot | Code review |
|-------|--------------|-------------|-------------|
| S0 smoke | output non-empty | **0%** | **0%** |
| S1 basic | + a core rule (cite / injection) | **33%** | **35%** |
| S2 good | + more rules (refund+leak / approve+cite) | **67%** | **71%** |
| S3 strong | + polite/concise + grounded in context | **100%** | **100%** |

The score climbs strictly with coverage in both domains, and the uncaught set
shrinks at each step until the complete suite catches every degrading mutant.
This is the result the claim predicts: **the mutation score is a faithful proxy
for how much real degradation the suite would catch** — and, because the two
domains agree and it's enforced in CI, it isn't a single-example fluke.

### The intended workflow, demonstrated

Each step from S(n) to S(n+1) means writing the specific evals a survivor named;
those evals kill exactly the mutants that break that behavior and lift the score.
That is the loop muteval is meant to drive: *a survivor names a missing eval; you
write it; the score goes up.*

## Honest caveats

- **Equivalent mutants.** Some of S3's 7 survivors are not real regressions:
  weakening `must -> should` on a rule the model still honors, or dropping the
  billing context doc that the port question never needed. Classic mutation
  testing has the same problem; a survivor is a *candidate* gap, not a proof of
  one. Quantifying and filtering equivalents is future work (and the right place
  to engage the related-work literature, e.g. MILE, arXiv 2409.04831).
- **Deterministic model.** Experiment 1 removes LLM non-determinism on purpose,
  to isolate the measurement. It demonstrates the metric behaves correctly; it
  does **not** by itself prove the result holds under a noisy real model — that's
  what Experiment 2 and the real-metric configs are for.
- **One system, hand-built suites.** This is a controlled demonstration, not a
  broad empirical study across many real repos. Scaling to several public suites
  is the next step toward a publishable claim.

## Experiment 2 — a live LLM run on GPT-4o-mini

`validation/openai_judge_rag/` (real model, real LLM-as-judge evals, run on
GPT-4o-mini). A small RAG assistant graded by a **faithfulness** judge and a
**relevancy** judge — the two metrics almost every RAG eval suite starts with.

| Eval suite | Mutation score | Mutants killed |
|------------|----------------|----------------|
| faithfulness + relevancy (2 answerable cases) | **0%** | 0 / 24 |
| + 1 unanswerable case + an `abstains_when_unanswerable` check | **25%** | 6 / 24 |

Two things make this more than a toy — read them with the *Scope* caveat in
"What this supports" below, which keeps the claim honest:

1. **The standard suite was completely blind.** Faithfulness + relevancy caught
   **none** of 24 prompt regressions — not inverting "do not invent facts", not
   dropping "cite the source", not deleting half the prompt. The faithfulness
   judge returned a flat 1.0 on *every* mutant (every survivor is the same
   `+0.300` near-miss). When the test cases have answers sitting in the context,
   these metrics simply can't see prompt degradation. That is exactly the kind of
   blind spot muteval exists to surface on a given suite (read with the *Scope*
   caveat below — this is a known property of reference-free metrics, not proof a
   suite is "broken").

2. **The kills were surgical.** Adding one unanswerable case + an abstention
   check killed exactly the 6 mutants that break the "if it's not in the
   context, say I don't know — do not invent facts" guardrail (the weaken/flip of
   that line, the two operators that delete it, and both truncations that chop
   off the prompt tail where it lives). The eval we added caught precisely the
   regressions it should, and nothing it shouldn't. The remaining 18 survivors
   are honest, named gaps — citation, tone, greeting handling — each a concrete
   "write this eval next."

A bonus finding from building this suite: the LLM judge itself was the weakest
link. A first cut silently scored a perfect, fully-grounded answer **0.0** (a
parsing artifact — the judge's reasoning mentioned a "0"), and a second cut gave
a mushy **0.5** to the same answer (the 0-1 float scale invites hedging). Both
would have sat invisibly inside a "passing" suite. Switching to a 0-10 integer
scale with last-number parsing fixed it. Brittle judges are part of *why* green
suites lie — and mutation testing is what dragged the problem into the light.

## Experiment 3 — replicated deepeval and RAGAS suites (GPT-4o-mini)

To check the blindness isn't an artifact of one rubric set, two more live configs
replicate real frameworks' metric suites, graded by GPT-4o-mini (the *literal*
framework objects are exercised by the adapter configs in the same folders, which
need the package installed):

| Suite (`validation/...`) | Metrics | Mutation score |
|--------------------------|---------|----------------|
| `deepeval_style_rag` | AnswerRelevancy + Faithfulness (deepeval's RAG example prompt) | **0%** (0/12) |
| `ragas_style_rag` | Faithfulness + AnswerRelevancy + AnswerCorrectness (modeled on `benitomartin/rag-langchain-ragas`) | **0%** (0/9) |

Same story across three independent metric sets (these two plus the pair in
Experiment 2): the faithfulness / relevancy / correctness judges saturate at the
top of their scale on answerable cases — every survivor is the identical
`+0.300` / `+0.200` near-miss — so prompt regressions pass straight through. The
remedy is the one demonstrated in Experiment 2: add a case + check that actually
exercises the degraded behavior, and survivors convert to kills.

## Experiment 4 — literal deepeval AND ragas metrics (real adapters, run in Colab)

With deepeval actually installed (`pip install -e ".[deepeval]"`), muteval grades
through deepeval's **real** `AnswerRelevancyMetric` + `FaithfulnessMetric` objects
— confirmed by deepeval's own "✨ running ... Metric" banners during the run. This
upgrades Experiment 3's deepeval result from a stdlib replica to the actual
framework code.

| Suite | Metrics | Mutation score |
|-------|---------|----------------|
| deepeval RAG example (literal adapter, GPT-4o-mini, `--max-mutants 8`) | AnswerRelevancy + Faithfulness | **0%** (0/7, 1 errored) |
| ragas RAG suite (literal adapter, GPT-4o-mini, `--max-mutants 6`) | Faithfulness + ResponseRelevancy | **0%** (0/6) |

Baseline passed; every survivor was a `+0.500` near-miss (Answer Relevancy pinned
at the top of its scale). The same blindness seen with the replicas holds with
deepeval's genuine metrics: standard relevancy + faithfulness did not catch the
injected prompt regressions on these answerable cases.

Both runs use the frameworks' genuine metric objects. With **real ragas**,
`ResponseRelevancy` stayed near the top of its scale on every mutant (survivors
are ~+0.28 near-misses) — the same blindness deepeval showed. Getting real ragas
to import at all required a *patched* ragas: the released version hard-imports a
removed langchain module (issue #2741; fix in PR #2746). That the leading RAG-eval
library doesn't currently import on a fresh install is itself a small data point
on eval-tooling fragility.

## Real-metric validation (reproducible with an API key)

To show the same machinery works on other frameworks' metrics, two more configs
reuse real frameworks' metrics via muteval's adapters:

- **deepeval** — `validation/deepeval_rag_qdrant/` reuses the *actual* metrics
  and system prompt from deepeval's own RAG example
  (`confident-ai/deepeval -> examples/rag_evaluation/rag_evaluation_with_qdrant.py`),
  using `AnswerRelevancyMetric` + `FaithfulnessMetric`.
- **RAGAS** — `validation/ragas_rag/` reuses `Faithfulness` + `ResponseRelevancy`
  via the ragas adapter, with a score threshold and near-miss reporting.

Run either with:

```bash
pip install "muteval[deepeval]"   # or muteval[ragas] langchain-openai
export OPENAI_API_KEY=sk-...
muteval run --config validation/deepeval_rag_qdrant/muteval_config.py --max-mutants 8
```

These are non-deterministic, so report them with `runs_per_mutant > 1` and treat
the score as an estimate. They demonstrate adapter coverage end-to-end; the
controlled experiment is what isolates the core measurement claim.

## What this supports — and what it doesn't

Supports: muteval surfaces concrete eval-coverage gaps a green suite is hiding,
and its score moves in the right direction as a suite improves. That is a strong,
defensible basis for "your evals scored X% — here are the regressions they'd
miss."

Does not (yet) support: a quantitative claim that the score predicts real-world
regression-catch rate across many production suites, or a handle on equivalent
mutants. Those are the experiments that would turn this from a compelling tool
into a citable result.

**Scope of the RAG-blindness result (Experiments 2–4).** These illustrate a
*known* limitation, not a new discovery: reference-free metrics
(faithfulness/relevancy) measure grounding, not correctness — so on answerable
cases they pass prompt regressions that don't change the grounded answer.
Competent teams already mitigate this with labeled/correctness evals and
retrieval-quality checks, and it does **not** apply when the retrieved documents
*are* the source of truth (there, faithful = correct). The point is not
"faithfulness is broken" — it's that muteval **surfaces this gap on your specific
suite automatically and names the eval that closes it.** Read every muteval
result as a per-suite diagnostic, never a universal verdict.

## Reproduce

```bash
pip install -e ".[dev]"
pytest -q
python validation/eval_quality_experiment/run_experiment.py            # offline, no key
muteval run --config validation/openai_judge_rag/muteval_config.py     # live, needs OPENAI_API_KEY
```
