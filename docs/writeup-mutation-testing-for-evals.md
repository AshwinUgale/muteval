# Your LLM evals are passing. That doesn't mean they work.

Every team shipping an LLM feature eventually writes evals — automated checks
that score the model's output so a regression fails CI instead of a customer.
It's the right instinct. But it hides a question nobody asks: **are the evals
themselves any good?** A suite that passes tells you nothing about whether it
*would have failed* had the system quietly gotten worse.

I built [muteval](https://github.com/AshwinUgale/muteval) to answer that, and
this is what I learned pointing it at real eval suites.

## The idea: mutation testing, for evals

In software testing there's an old, established answer to "is my test suite any
good?": **mutation testing**. You deliberately inject bugs into the code, rerun
the tests, and measure how many bugs the tests catch. Tools like `mutmut` and
Stryker do this. A test suite that catches 40% of injected bugs has a coverage
problem, no matter how green it looks.

muteval applies the same move to LLM evals. It deliberately **degrades the
system under test** — weakens a prompt instruction, drops a retrieved document,
swaps in a cheaper model — reruns **your existing eval suite** against each
degraded version (a "mutant"), and reports a **mutation score**: the percentage
of injected regressions your evals caught. The ones they miss are
**survivors** — concrete blind spots in your coverage.

Two axes keep it distinct from things it looks like. Red-teaming tools mutate
the *input* to test the *system's* safety. Calibration tools mutate the *output*
to tune a *metric*. muteval mutates the *system* to measure the *eval suite's
coverage*. Its killer property is **absence detection**: it surfaces the case
where *nothing fails* — i.e. "you have no eval for this behavior at all."

## Does the number mean anything?

A coverage score is only useful if it tracks reality. So before trusting it on
anyone's suite, I ran a controlled experiment: take a fixed system, write eval
suites of deliberately increasing quality, and check that the mutation score
rises with them. It does — monotonically, **0 → 33 → 67 → 100%**, and the same
shape holds across two different domains. That relationship is enforced in CI, so
it can't silently regress. It's not proof the number is perfect; it's evidence
the number *behaves like a coverage metric should*.

## A real run, honestly

Then I pointed muteval at a real, recognizable eval suite —
[Vectara's open-rag-eval](https://github.com/vectara/open-rag-eval) — driving its
actual citation metric, on a free model, and I mutated the grounding prompt.

The first run scored **33%**. Citation-checking caught mutations that made the
model stop citing, but four grounding/abstention degradations survived —
including deleting *"if the answer isn't in the context, say you don't know."*
The model kept citing sources while becoming freer to hallucinate, and a
citation metric can't see that.

So I did what muteval suggests: added a grounding eval and an out-of-context test
case. **50%** — the blatant removal now got caught. I added a few adversarial
probes (questions the model could answer from world knowledge but that weren't in
the corpus). **60%** — more caught.

Two survivors persisted: subtle *weakenings* ("don't" → "try not to", "ONLY" →
"preferably"). Here's the honest part. I inspected the actual outputs, and on
every probe the mutated system **still refused correctly** — it never
hallucinated. The prompt text changed but the behavior didn't. Those weren't real
gaps; they were near-equivalent mutations. muteval flagged them because the output
changed and no eval fired, but a human looking at the outputs agrees the mutant is
fine.

That's the real lesson, and it's a better one than a clean 100%: **catching
grounding regressions needs the right metric *and* adversarial test cases — and a
survivor isn't a gap until a human agrees the mutant is meaningfully worse.** A
mutation score should never be read without that last step.

## A coverage tool has to refuse to lie

The most dangerous thing a tool like this can do is hand you a confident,
meaningless number. Two external code reviews hammered muteval on exactly this,
and the fixes shaped its design. muteval **fails closed**:

- If your suite doesn't pass on the *original* system, there's no score — a red
  baseline makes every mutant "fail" too, faking 100%. muteval reports the run
  invalid and the CLI exits non-zero.
- If some mutants error (a flaky judge, a rate limit), it won't quietly score
  over a shrunken denominator; it flags `partial_errors` unless you opt in.
- For non-deterministic judges it uses majority-vote verdicts and reports a
  confidence interval, so a single flaky verdict can't swing the number.
- It diffs outputs to separate real coverage gaps from cosmetic changes.

A number you can't trust is worse than no number. That principle is the whole
project.

## Try it

```bash
pip install muteval
muteval init --template rag                 # a runnable starting config
muteval check --config muteval_config.py    # validate wiring + baseline first
muteval run   --config muteval_config.py
```

It's pure Python with zero required dependencies, works with your existing
deepeval / RAGAS / promptfoo metrics or its own framework-free checks, and its
built-in LLM judge points at any OpenAI-compatible endpoint (including free ones).
Adopting it on a real suite is a real ~1-hour integration, not plug-and-play —
there's an honest [adoption guide](https://github.com/AshwinUgale/muteval/blob/main/docs/ADOPTION.md)
with a "where it breaks → what to change" table, because pointing any eval tool
at a live system hits dependency and judge friction, and pretending otherwise
helps no one.

## What it is, and isn't

muteval is a way to **audit your eval suite** — to find the regressions it would
miss before a customer does. It is *not* a measure of your system's correctness
or safety, and it's an early, honest project, not a finished product. But the
question it answers is real, and most teams have never asked it: your evals are
passing — would they actually catch it if your system got worse?

*muteval is open source (Apache-2.0): <https://github.com/AshwinUgale/muteval>*
