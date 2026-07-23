# Finding: khoj-ai/khoj — the answer-quality eval scores correctness only

**Target:** [khoj-ai/khoj](https://github.com/khoj-ai/khoj) — self-hostable
personal-AI / RAG assistant (~30k+ stars, actively maintained).

**Date:** 2026-07-01. **Tool:** muteval.

## The claim (precise, not overstated)

khoj's answer-quality eval (`tests/evals/eval.py`) grades responses with a
**single binary LLM judge**, `evaluate_response_with_gemini`:

> "Determine if the agent response contains the key information from the ground
> truth. Focus on factual correctness rather than exact wording. … decision:
> TRUE if response contains key information, FALSE otherwise."

That is the *only* answer-quality signal in the suite (the other graders are an
MCQ exact-match for GPQA and an IR precision/recall grader for retrieval).

Meanwhile khoj's own system prompt (`personality` in
`src/khoj/processor/conversation/prompts.py`) explicitly **requires** behaviors
the judge never checks:

> "Provide inline citations to documents and websites referenced. Add them
> inline in markdown format to directly support your claim."
> "Ask crisp follow-up questions … when a helpful response cannot be provided
> from the provided notes."

So muteval's prediction is structural: **mutations that degrade citation or
grounding behavior — while leaving the ground-truth fact in the answer — will
pass khoj's correctness judge undetected.** khoj has no eval that would catch a
regression in the very behaviors its prompt is engineered to produce.

## Reproduce

Offline (no key — demonstrates the mechanism against khoj's real prompt):

```
muteval run --config validation/khoj_rag/muteval_config_offline.py
```

Result (output-diffing ON, so inert mutants are excluded from the score):

```
Mutation score: 0%   (0/22 mutants killed)
Effective score: 0%  (0/4 — excludes 18 inert mutants whose output didn't change)

4 SURVIVED (real coverage gaps):
  [MED] drop_instruction_lines  dropped "- Provide inline citations to documents…"
  [MED] delete_sentences        deleted "- Provide inline citations to documents…"
  [MED] truncate_prompt         dropped the last 6 of 11 lines
  [MED] drop_few_shot_example   dropped the "# Style" block
```

Every survivor removes khoj's citation instruction; the mock answer keeps the
correct fact (so the correctness judge passes) but stops citing sources. That is
the blind spot, made concrete.

Real (khoj's actual Gemini judge + a real model on khoj's real prompt — one key):

```
export GEMINI_API_KEY=...
muteval run --config validation/khoj_rag/muteval_config.py \
  --operators drop_instruction_lines delete_sentences truncate_prompt
```

## Fidelity — what is khoj's vs. what is our harness

VERBATIM FROM KHOJ (this is what makes the finding legitimate):
- The **system prompt** under mutation is khoj's `personality` +
  `notes_conversation`, copied unchanged.
- The **judge** is khoj's `evaluate_response_with_gemini` — same Gemini model,
  same prompt text, same TRUE/FALSE correctness rubric.

APPROXIMATED (a minimal harness, not khoj's full server):
- We call the model directly with khoj's system prompt + the reference article,
  rather than booting the khoj server and its live retriever. The retrieved
  context here is the gold FRAMES-style reference; khoj in production retrieves
  it. This does not affect the finding, which is about what the *judge* measures.
- Cases are a small FRAMES-style sample embedded in the config; khoj's harness
  loads the full `google/frames-benchmark` via `datasets`. Swap in the real
  loader for a full run.

## Honest caveats (before any outreach)

1. This is a **candidate** gap surfaced by muteval, confirmed by reading khoj's
   code — not a bug in khoj's system. khoj's eval does exactly what it claims
   (measure correctness); the point is the *absence* of citation/grounding evals.
2. Severity is MED, not HIGH: citation loss is a quality/trust regression, not a
   safety/correctness failure. Frame it that way — do not inflate it.
3. Whether khoj *wants* a citation eval is their call. The gift is the
   observation + a ready-to-merge eval that would close the gap, not a verdict.

## The gift (what to actually offer the maintainer)

"Your `tests/evals/eval.py` correctness judge wouldn't catch a regression that
drops inline citations, even though your system prompt requires them — here's a
reproducible case, plus a small `has_inline_citation` eval you could add." That
is the "kill the survivor" loop: add the eval, the mutant dies, coverage rises.
