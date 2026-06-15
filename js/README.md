# muteval (npm)

**Mutation testing for LLM eval suites** — measure whether your evals would
actually catch a regression.

> This npm package currently **reserves the name** while the JavaScript/TypeScript
> port is in progress. The working implementation is available today in Python.

## Use it now (Python)

```bash
pip install muteval
```

- PyPI: https://pypi.org/project/muteval
- Source & roadmap: https://github.com/AshwinUgale/muteval

## What it does

`muteval` deliberately degrades the thing under test (your prompt — and soon
retrieved context and tool outputs), reruns your existing eval suite against
each mutant, and reports a **mutation score**: the percentage of injected
regressions your evals caught. The ones they miss are **survivors** — concrete
blind spots in your eval coverage.

It's `mutmut`/Stryker, but for evals.

## Status

JS/TS API coming. Star or watch the
[GitHub repo](https://github.com/AshwinUgale/muteval) for updates, and see
[CONTRIBUTING](https://github.com/AshwinUgale/muteval/blob/main/CONTRIBUTING.md)
if you'd like to help build the Node port.

## License

[Apache-2.0](LICENSE)
