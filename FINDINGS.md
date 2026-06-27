# Findings: does muteval's mutation score measure eval-suite quality?

muteval's central claim is a *meta* one: the mutation score isn't a property of
your system, it's a property of your **eval suite** — a higher score means your
evals would catch more real regressions. This document reports what we've
actually measured against that claim, and is deliberately honest about what the
evidence does and does not show.

## TL;DR

- In a controlled experiment where a deterministic system genuinely reacts to
  prompt and context mutations, the mutation score **rose monotonically with
  eval-suite coverage — 0% -> 28% -> 56% -> 72%** — and the survivors narrowed to
  a concrete, nameable set at each step. The metric behaves as advertised.
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

We then grade the *same* system + mutants with four suites:

| Suite | Checks | Baseline | Mutation score | Mutants killed |
|-------|--------|----------|----------------|----------------|
| S0 smoke | output non-empty | PASS | **0%** | 0 / 25 |
| S1 basic | + cites order ID | PASS | **28%** | 7 / 25 |
| S2 good | + no refund promise, no data leak | PASS | **56%** | 14 / 25 |
| S3 strong | + polite, + grounded in context | PASS | **72%** | 18 / 25 |

The score climbs strictly with coverage, and the uncaught-operator set shrinks:

- **S0** misses everything (only an empty output would fail it).
- **S1** still misses refund, data-leak, politeness and all context regressions.
- **S2** closes refund and data-leak gaps; still misses politeness and context.
- **S3** additionally catches politeness and a dropped *relevant* context doc;
  the remaining 7 survivors are residual (see caveats).

This is the result the claim predicts: **the mutation score is a faithful proxy
for how much real degradation the suite would catch.**

### The intended workflow, demonstrated

Going from S2 -> S3 means writing two more evals (`is_polite`,
`grounded_in_context`). Those evals kill 4 additional mutants and lift the score
from 56% to 72%. That is exactly the loop muteval is meant to drive: *a survivor
names a missing eval; you write it; the score goes up.*

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

Two things make this a real finding, not a demo:

1. **The standard suite was completely blind.** Faithfulness + relevancy caught
   **none** of 24 prompt regressions — not inverting "do not invent facts", not
   dropping "cite the source", not deleting half the prompt. The faithfulness
   judge returned a flat 1.0 on *every* mutant (every survivor is the same
   `+0.300` near-miss). When the test cases have answers sitting in the context,
   these metrics simply can't see prompt degradation. That is exactly the "your
   green suite is lying" failure muteval exists to surface.

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

## Reproduce

```bash
pip install -e ".[dev]"
pytest -q
python validation/eval_quality_experiment/run_experiment.py            # offline, no key
muteval run --config validation/openai_judge_rag/muteval_config.py     # live, needs OPENAI_API_KEY
```
