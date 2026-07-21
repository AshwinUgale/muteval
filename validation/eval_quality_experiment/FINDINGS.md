# Does muteval's score actually mean anything? (self-validation)

muteval's central claim: **a better eval suite kills more mutants.** If that's
true, the mutation score is a real measure of eval quality. If it's not, muteval
is theatre. So we test it directly, in a controlled, API-free setting.

## Setup

One fixed system (a support bot with a prompt + retrieved context) and a
deterministic "model" that faithfully obeys the rules still present in the
prompt and grounds answers only in the context still retrieved. We grade it with
**four eval suites of increasing coverage**, from "nonempty only" to a strong
suite that checks citation, refund policy, data-leak, tone, and grounding.

We report the **effective** score (which excludes inert/equivalent mutants whose
output never changed — no output-based eval could catch those, so counting them
would be dishonest).

## Result

| Suite | Effective score | Raw score |
| --- | --- | --- |
| S0 — nonempty only | **0%** | 0% |
| S1 — basic (cite) | **33%** | 21% |
| S2 — good (cite + refund + leak) | **67%** | 41% |
| S3 — strong (+ polite + grounded) | **100%** | 62% |

**The effective score rises monotonically from 0% (no evals) to 100% (complete
coverage).** Empty suite scores zero; complete suite scores one hundred; every
step in between is an improvement. The metric tracks eval-suite quality end to
end.

Two honest notes this surfaced:
- The **raw** score understates good suites because it counts equivalent mutants
  (e.g. shuffling identical context) as survivors. The **effective** score is the
  trustworthy one — always read that.
- This is a controlled experiment (deterministic model, single system) — it
  proves the *mechanism* is sound. For results on real LLM-judge metrics see
  `validation/deepeval_rag_qdrant/` and `validation/deepeval_rag_system/`.

## This claim is now a test

The relationship above is enforced in `tests/test_eval_quality.py`
(monotonic increase + endpoints at 0% and 100%). If a change ever breaks the
link between eval quality and muteval's score, CI fails. The core thesis is
verified continuously, not just demonstrated once.
