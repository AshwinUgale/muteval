# I ran mutation testing on deepeval's own RAG example. It caught 1 of 8 regressions.

*Your evals are passing. That doesn't mean they work.*

TL;DR — I built [muteval](https://github.com/AshwinUgale/muteval), a tool that
does mutation testing for LLM eval suites: it deliberately degrades the system
under test, reruns *your existing evals*, and reports how many of the injected
regressions your evals actually caught. I pointed it at deepeval's own RAG
example. It scored **12%** — and it didn't notice when I inverted the prompt's
anti-hallucination rule from *"do not pretend to know"* into *"do pretend to
know."*

## The thing nobody tests

We've gotten good at testing LLM systems. promptfoo, deepeval, Giskard,
LangSmith — pick your flavor, write some metrics, gate CI, ship. The whole
point is to catch regressions before they reach production.

But there's a layer underneath that nobody checks: **are your evals any good?**
A green eval suite tells you nothing if the suite would stay green while your
system quietly got worse. In software, we don't trust a test suite because it
passes — we trust it because we've seen it *fail* when the code breaks. There's
no equivalent discipline for evals. You write them, they go green, and you hope.

The classic answer to "is my test suite any good?" is **mutation testing**:
inject a bug, rerun the tests, and any bug the tests *don't* catch is a gap.
`mutmut` and Stryker have done this for code for years. muteval does it for
evals.

## What "mutating the system" means

This is the part that makes muteval different from the tools it sits next to:

| Tool | Mutates the… | Measures… |
| --- | --- | --- |
| promptfoo red team | **input** (jailbreaks) | your system's safety |
| Giskard | **input** (typos, swaps) | your model's robustness |
| deepeval synth data | **output / ground truth** | a metric's calibration |
| **muteval** | **the system** (prompt → context → tools) | **your eval suite's coverage** |

Red-teaming and robustness tools mutate your *inputs* to test your *system*.
muteval mutates your *system* to test your *evals*. The payoff is **absence
detection**: by degrading the prompt and running your real pipeline through your
real suite, it finds the regression where *nothing fails* — i.e. "you have no
eval for this behavior at all."

## The experiment

I wanted a real, recognizable target, not a toy. So I used deepeval's own
published RAG example
([`rag_evaluation_with_qdrant.py`](https://github.com/confident-ai/deepeval/blob/main/examples/rag_evaluation/rag_evaluation_with_qdrant.py)):
its actual system prompt and its actual metrics.

The prompt is a normal, sensible RAG instruction set: cite the source, use a
professional tone, and — critically — *"if you cannot find the answer in the
provided context, do not pretend to know it. Instead, respond with 'I don't
know'."*

muteval mutated that prompt 8 ways (weaken instructions, invert rules, drop
lines, truncate) and reran the suite's two answer-dependent metrics —
**AnswerRelevancy** and **Faithfulness** — against each mutant.

## The result

**Mutation score: 12% — 1 of 8 mutants killed. 7 survived.** Here's what slipped
through:

```
SURVIVED  flip_negation   "cannot" -> "can"   (inverts the retrieval guard)
SURVIVED  flip_negation   "do not" -> "do"    (inverts the anti-hallucination rule)
SURVIVED  flip_negation   "don't" -> "do"     (corrupts the "I don't know" refusal)
SURVIVED  drop_line       "Understand the user's question thoroughly."
SURVIVED  drop_line       "avoid using the context from the documentation."
SURVIVED  drop_line       "extract the pertinent information."
SURVIVED  drop_line       "Remember to:"
```

Read the top three again. I told the system to **hallucinate** — flipped "do
not pretend to know" into "do pretend to know," turned "if you cannot find the
answer" into "if you can," corrupted the refusal phrase — and the eval suite
stayed **green**. The metrics never noticed the guardrails were switched off.

## Why this happens

It's not a deepeval bug, and it's not unique to deepeval — it's structural, and
deepeval is just a popular, well-built suite that happened to be in front of me.
AnswerRelevancy asks *"does the answer address the question?"* Faithfulness asks
*"is the answer grounded in the retrieved context?"* Neither asks *"did the
system refuse when it should have?"* or *"did it follow its own safety rules?"*
So a prompt change that disables refusal behavior is invisible to them — on an
answerable question, the answer is still relevant and still grounded. The eval
has no opinion about the behavior you actually care about.

That's the gap muteval is built to surface: not "this metric is miscalibrated,"
but "**you have no eval for this at all.**"

## Honest caveats

I'd rather you trust the finding than the decimal:

- Small sample — 8 mutants, 2 cases. The *finding* (guardrail mutations survive)
  is robust; the *12%* is directional.
- The judge here was gpt-4o-mini; a published number should be re-run with a
  stronger judge.
- 4 additional mutants errored on API timeouts and were excluded.

The number will move. The blind spot won't.

## Try it on your own suite

```bash
pip install muteval
```

muteval is tool-agnostic and wraps your existing evals — there's a deepeval
adapter so you can point it at metrics you've already written. It's early and
open source (Apache-2.0). If your suite scores higher than 12%, great — now you
have a number. If it scores lower, you just found out before your users did.

Repo, docs, and the exact config from this post:
**https://github.com/AshwinUgale/muteval**

*They mutate your inputs to test your system. muteval mutates your system to
test your evals.*
