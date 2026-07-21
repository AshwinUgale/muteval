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
    lo, hi = result.score_ci
    lines.append(
        f"Mutation score: {c(f'{pct:.0f}%', score_color)}  "
        f"[{_bar(result.score)}]  "
        f"({result.killed}/{result.evaluated} mutants killed, "
        f"95% CI {lo * 100:.0f}-{hi * 100:.0f}%)"
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
        elo, ehi = result.effective_score_ci
        lines.append(
            f"Effective score: {c(f'{eff:.0f}%', eff_color)}  "
            f"({result.killed}/{result.evaluated - len(inert)} — excludes "
            f"{len(inert)} inert mutant(s) whose output didn't change; "
            f"95% CI {elo * 100:.0f}-{ehi * 100:.0f}%)"
        )

    flaky = result.flaky
    if flaky:
        lines.append(
            c(
                f"   {len(flaky)} mutant(s) flipped verdict between runs (judge "
                "noise) — raise runs_per_mutant to stabilize.",
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
        from muteval.suggest import suggest_eval

        for o in real:
            sev = o.severity or MEDIUM
            tag = c(f"[{_sev_label[sev]}]", _sev_color[sev])
            lines.append(f"  {tag} {c('SURVIVED', '31')}  [{o.mutant.operator}]")
            lines.append(f"            {o.mutant.description}")
            lines.append(c(f"            fix: {suggest_eval(o)}", "36"))
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


def format_probe_card(results, use_color: bool = True) -> str:
    """Render probe results as an eval-quality report card (no composite score)."""

    def c(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    lines = ["", c("muteval — eval quality report card", "1"), ""]
    if not results:
        lines.append("No probes ran.")
        return "\n".join(lines)
    for r in results:
        tag = c("PASS", "32") if r.ok else c("WARN", "33")
        lines.append(f"  [{tag}] {c(r.name, '1')}")
        lines.append(f"         {r.summary}")
        if r.detail:
            lines.append(c(f"         {r.detail}", "2"))
        lines.append("")
    return "\n".join(lines)


def result_to_dict(result) -> dict:
    """Machine-readable summary of a MutationResult (for --json / CI / reports)."""
    from muteval.suggest import suggest_eval

    return {
        "baseline_passed": result.baseline_passed,
        "baseline_error": result.baseline_error,
        "score": round(result.score, 4),
        "effective_score": round(result.effective_score, 4),
        "score_ci": [round(x, 4) for x in result.score_ci],
        "effective_score_ci": [round(x, 4) for x in result.effective_score_ci],
        "killed": result.killed,
        "evaluated": result.evaluated,
        "total": result.total,
        "errored": result.errored,
        "inert": len(result.inert_survivors),
        "high_severity_survivors": len(result.high_severity_survivors),
        "survivors": [
            {
                "operator": o.mutant.operator,
                "description": o.mutant.description,
                "severity": o.severity,
                "fix": suggest_eval(o),
            }
            for o in result.real_survivors
        ],
    }


def badge_dict(result, label: str = "eval coverage") -> dict:
    """A shields.io endpoint payload for the effective mutation score."""
    pct = round(result.effective_score * 100)
    color = "brightgreen" if pct >= 80 else "yellow" if pct >= 50 else "red"
    return {"schemaVersion": 1, "label": label, "message": f"{pct}%", "color": color}
