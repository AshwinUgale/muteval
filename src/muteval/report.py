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
    lines.append("")

    survivors = result.survivors
    if not survivors:
        lines.append(
            c("✓ No survivors — your evals caught every injected regression.", "32")
        )
        return "\n".join(lines)

    lines.append(
        c(f"{len(survivors)} SURVIVED", "31")
        + "  (these regressions slipped past your evals — coverage gaps):"
    )
    lines.append("")
    for o in survivors:
        lines.append(f"  {c('SURVIVED', '31')}  [{o.mutant.operator}]")
        lines.append(f"            {o.mutant.description}")
    lines.append("")
    lines.append(
        "Each survivor is a change to your system your evals would NOT notice. "
        "Write an eval that fails on it, then re-run."
    )
    return "\n".join(lines)
