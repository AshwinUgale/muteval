"""Tests for output-diffing: separating real coverage gaps from inert mutants."""

from muteval import MutEvalConfig, run_mutation_testing


def test_inert_mutants_when_output_ignores_prompt():
    # The system ignores the prompt entirely -> every mutant produces the same
    # output -> no eval could catch them -> all survivors are INERT, not gaps.
    cfg = MutEvalConfig(
        prompt="- You must cite the order ID.\n- Do not promise refunds.\n"
        "- Always be polite.",
        cases=[{"order_id": "X1"}],
        run=lambda p, c: "a fixed answer that ignores the prompt",
        evals=[lambda o, c: bool(o.strip())],
    )
    result = run_mutation_testing(cfg)
    assert len(result.survivors) > 0
    assert len(result.real_survivors) == 0
    assert len(result.inert_survivors) == len(result.survivors)
    # Nothing actually degraded the output -> there is NO observed-degradation
    # evidence, so the effective score is undefined (None), not a vacuous 1.0.
    assert result.effective_score is None
    assert all(o.output_changed is False for o in result.survivors)
    # format_report must NOT crash when effective_score is None (all inert).
    from muteval.report import format_report

    text = format_report(result, use_color=False)
    assert "Effective score: n/a" in text


def test_real_survivors_when_output_depends_on_prompt():
    # Output reflects the prompt, so mutations change it; a weak eval misses them.
    def run(p, c):
        out = []
        if "cite the order id" in p.lower():
            out.append(f"order {c['order_id']}")
        if "do not promise refunds" in p.lower():
            out.append("no refund")
        else:
            out.append("refund promised")
        return " ".join(out)

    cfg = MutEvalConfig(
        prompt="- You must cite the order ID.\n- Do not promise refunds.",
        cases=[{"order_id": "X1"}],
        run=run,
        evals=[lambda o, c: bool(o.strip())],  # weak: only non-empty
    )
    result = run_mutation_testing(cfg)
    # At least one mutant changed the output but slipped past the weak eval.
    assert len(result.real_survivors) >= 1
    assert any(o.output_changed is True for o in result.survivors)
