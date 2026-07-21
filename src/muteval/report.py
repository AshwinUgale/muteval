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

    # Invalid / empty runs are terminal — there is NO trustworthy score to show.
    if result.baseline_error:
        lines.append(c("⚠  INVALID RUN — baseline ERRORED", "1;31"))
        lines.append(f"   {result.baseline_error}")
        lines.append(
            "   The eval suite raised on the ORIGINAL system, so there is no "
            "trustworthy score. Fix the error and re-run."
        )
        return "\n".join(lines)
    if not result.baseline_passed:
        lines.append(c("⚠  INVALID RUN — baseline FAILED", "1;31"))
        lines.append(
            "   Your eval suite does not pass on the ORIGINAL system, so a "
            "mutation score would be meaningless (every mutant 'fails' too). "
            "Fix the baseline, then re-run."
        )
        return "\n".join(lines)
    if result.total == 0:
        lines.append(c("⚠  NO MUTANTS — nothing to test", "33"))
        lines.append(
            "   No mutants were generated (prompt too short, or operators/scope "
            "filtered them all out). No score."
        )
        return "\n".join(lines)
    if result.evaluated == 0:
        lines.append(c("⚠  INVALID RUN — no mutant produced a clean verdict", "1;31"))
        lines.append(
            f"   All {result.total} mutant(s) errored (e.g. API failures). No "
            "score — investigate the failures and re-run."
        )
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

    # Effective score: drop observationally-unchanged survivors from the
    # denominator (their output didn't change on the samples we ran).
    inert = result.inert_survivors
    if inert and result.effective_score is not None:
        eff = result.effective_score * 100
        eff_color = "32" if eff >= 80 else "33" if eff >= 50 else "31"
        elo, ehi = result.effective_score_ci
        lines.append(
            f"Effective score: {c(f'{eff:.0f}%', eff_color)}  "
            f"({result.killed}/{result.evaluated - len(inert)} — excludes "
            f"{len(inert)} inert mutant(s) whose output didn't change; "
            f"95% CI {elo * 100:.0f}-{ehi * 100:.0f}%)"
        )
    elif inert:
        # Every evaluated mutant was observationally unchanged -> no observed
        # degradation to score. Say so rather than crash on a None effective score.
        lines.append(
            c(
                f"Effective score: n/a  (all {len(inert)} evaluated mutant(s) "
                "left the output unchanged on this run — nothing to score)",
                "33",
            )
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
            c(f"{len(inert)} observationally unchanged", "2")
            + "  (output identical to baseline on this run — NOT eval blind "
            "spots; excluded from the effective score. For a stochastic system "
            "this is not proof of equivalence — raise runs_per_mutant):"
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

    score = result.score
    eff = result.effective_score
    return {
        "status": result.status,
        "baseline_passed": result.baseline_passed,
        "baseline_error": result.baseline_error,
        "score": round(score, 4) if score is not None else None,
        "effective_score": round(eff, 4) if eff is not None else None,
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
    eff = result.effective_score
    if eff is None or result.status != "valid":
        return {"schemaVersion": 1, "label": label, "message": "n/a",
                "color": "lightgrey"}
    pct = round(eff * 100)
    color = "brightgreen" if pct >= 80 else "yellow" if pct >= 50 else "red"
    return {"schemaVersion": 1, "label": label, "message": f"{pct}%", "color": color}
