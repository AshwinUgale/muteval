"""Human-readable terminal reporting for a MutationResult."""

from __future__ import annotations

from muteval.runner import MutationResult


def _bar(score: float, width: int = 24) -> str:
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


def format_report(result: MutationResult, use_color: bool = True) -> str:
    def c(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    lines = []
    lines.append("")
    lines.append(c("muteval — mutation testing for your eval suite", "1"))
    lines.append("")

    if result.baseline_error:
        lines.append(c(f"⚠  Baseline ERRORED: {result.baseline_error}", "33"))
        lines.append(
            "   The eval suite raised on the original prompt; results below may "
            "be unreliable."
        )
        lines.append("")
    elif not result.baseline_passed:
        lines.append(
            c(
                "⚠  Baseline FAILED: your eval suite does not pass on the "
                "original prompt.",
                "33",
            )
        )
        lines.append(
            "   Fix your evals/system so the baseline is green before trusting "
            "the score below."
        )
        lines.append("")

    if result.total == 0:
        lines.append("No mutants were generated. Is your prompt long enough?")
        return "\n".join(lines)

    pct = result.score * 100
    score_color = "32" if pct >= 80 else "33" if pct >= 50 else "31"
    lines.append(
        f"Mutation score: {c(f'{pct:.0f}%', score_color)}  "
        f"[{_bar(result.score)}]  "
        f"({result.killed}/{result.evaluated} mutants killed)"
    )
    if result.errored:
        lines.append(
            c(
                f"   {result.errored} mutant(s) errored and were excluded "
                "(e.g. API timeouts). Re-run to retry them.",
                "33",
            )
        )

    # Honest score: drop inert (output-unchanged) survivors from the denominator.
    inert = result.inert_survivors
    if inert:
        eff = result.effective_score * 100
        eff_color = "32" if eff >= 80 else "33" if eff >= 50 else "31"
        lines.append(
            f"Effective score: {c(f'{eff:.0f}%', eff_color)}  "
            f"({result.killed}/{result.evaluated - len(inert)} — excludes "
            f"{len(inert)} inert mutant(s) whose output didn't change)"
        )
    lines.append("")

    survivors = result.survivors
    if not survivors:
        lines.append(
            c("✓ No survivors — your evals caught every injected regression.", "32")
        )
        return "\n".join(lines)

    real = result.real_survivors
    if real:
        from muteval.severity import HIGH, LOW, MEDIUM, severity_rank

        real = sorted(real, key=lambda o: severity_rank(o.severity or MEDIUM))
        n_high = sum(1 for o in real if o.severity == HIGH)
        header = c(f"{len(real)} SURVIVED", "31") + (
            "  (output changed but evals didn't notice — real coverage gaps"
        )
        if n_high:
            header += "; " + c(f"{n_high} HIGH-severity", "1;31")
        lines.append(header + "):")
        lines.append(
            c("  ranked by severity: ", "2")
            + c("HIGH", "31") + c(" › ", "2") + c("MED", "33") + c(" › ", "2")
            + c("LOW", "2")
        )
        lines.append("")
        _sev_color = {HIGH: "31", MEDIUM: "33", LOW: "2"}
        _sev_label = {HIGH: "HIGH", MEDIUM: "MED ", LOW: "LOW "}
        for o in real:
            sev = o.severity or MEDIUM
            tag = c(f"[{_sev_label[sev]}]", _sev_color[sev])
            lines.append(f"  {tag} {c('SURVIVED', '31')}  [{o.mutant.operator}]")
            lines.append(f"            {o.mutant.description}")
            if o.min_margin is not None and o.closest_eval:
                lines.append(
                    c(
                        f"            ↳ near miss: passed {o.closest_eval} by only "
                        f"+{o.min_margin:.3f}",
                        "33",
                    )
                )

    if inert:
        lines.append("")
        lines.append(
            c(f"{len(inert)} inert", "2")
            + "  (output identical to baseline — equivalent mutants, NOT eval "
            "blind spots; excluded from the effective score):"
        )
        for o in inert:
            lines.append(
                f"  {c('inert', '2')}     [{o.mutant.operator}] {o.mutant.description}"
            )

    lines.append("")
    lines.append(
        "Each real survivor is an output change your evals would NOT notice. "
        "Write an eval that fails on it, then re-run."
    )
    return "\n".join(lines)
