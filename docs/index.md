# muteval

**Mutation testing for your LLM evals — find out if they'd actually catch a regression.**

Your evals are passing. That doesn't mean they work. muteval deliberately degrades the
system under test (the prompt, retrieved context, tool outputs, or model), reruns your
**existing** eval suite against each degraded version (a "mutant"), and reports a
**mutation score** — the percentage of injected regressions your evals caught. The ones
they miss are **survivors**: candidate coverage gaps for you to triage.

It's `mutmut` / Stryker, but for evals.

[See it run on a promptfoo suite → Example](real-report.html){ .md-button .md-button--primary }
[Read the limits](LIMITATIONS.md){ .md-button }

---

## Install

```bash
pip install muteval        # pure Python, zero required dependencies
```

## 60-second quickstart (no API key)

```bash
muteval init --template rag                  # scaffold a config
muteval check --config muteval_config.py     # validate wiring + baseline first
muteval run   --config muteval_config.py     # score + ranked survivors + suggested fixes
```

## What it is — and isn't

muteval is a **per-suite diagnostic**. What that means:

- A survivor is a **candidate** gap — flagged for review, not an automatic verdict on your system.
- It measures eval **coverage** (would the suite notice a regression?), not whether an
  eval is **correct** — that needs labels.
- It **fails closed**: a red or errored baseline yields *no score*, never a misleading 100%.

For the full picture, read the [Limitations](LIMITATIONS.md) and the
[Findings](findings.md) for what's been measured.

## Where to go next

- **[Example — a promptfoo suite](real-report.html)** — install, point it at a config, read the two reports.
- **[Adopting muteval](ADOPTION.md)** — a ~1-hour integration guide for your own suite.
- **[Findings](findings.md)** — does the mutation score actually track eval quality? (Yes, across four domains, CI-enforced.)
- **[Plugins](PLUGINS.md)** — extend with your own operators, probes, adapters, and reporters.
- **[Prior art](PRIOR-ART.md)** — how this relates to mutation testing and meta-evaluation research.
