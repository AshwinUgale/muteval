"""Human-readable terminal reporting for a MutationResult."""

from __future__ import annotations

import html
from typing import List

from muteval.redact import redact, redact_obj
from muteval.runner import MutationResult


def _bar(score: float, width: int = 24) -> str:
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


def _robustness_lines(result: MutationResult, c) -> List[str]:
    """The meaning-preserving (robustness) section: never scored. Surviving
    these is healthy; one that flipped a verdict means an eval keys on wording
    or order (or the system is sensitive to it) — worth a look, not a gap."""
    robust = result.robustness
    if not robust:
        return []
    brittle = result.brittle
    if not brittle:
        return [
            c(
                f"   robustness: {len(robust)} meaning-preserving edit(s) (paraphrase/"
                "reorder), none flipped a verdict — not scored.",
                "2",
            )
        ]
    out = [
        c(
            f"   robustness: {len(brittle)}/{len(robust)} meaning-preserving edit(s) "
            "flipped a verdict — an eval may key on exact wording/order (or the "
            "system is sensitive to it). Not scored:",
            "33",
        )
    ]
    for o in brittle:
        out.append(
            c(
                f"      [{o.mutant.operator}] {o.mutant.description}  "
                f"(caught by {o.failing_eval})",
                "2",
            )
        )
    return out


def format_report(result: MutationResult, use_color: bool = True) -> str:
    """The terminal report. Redacted: error strings (baseline_error, mutant
    errors) can echo request URLs, auth headers, or a key in a prompt."""
    return redact(_format_report(result, use_color))


def _format_report(result: MutationResult, use_color: bool = True) -> str:
    def c(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    lines = []
    lines.append("")
    lines.append(c("muteval — mutation testing for your eval suite", "1"))
    lines.append("")

    # Invalid / empty runs are terminal — there is NO trustworthy score to show.
    if result.status == "budget_exceeded":
        lines.append(c("⚠  INCOMPLETE — call budget exceeded (--max-calls)", "1;31"))
        lines.append(
            "   Stopped before finishing, so there is no trustworthy score. "
            "Raise --max-calls (or narrow with --sample) and re-run."
        )
        return "\n".join(lines)
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
        rate = result.baseline_pass_rate
        if rate is not None and 0.0 < rate < 1.0:
            lines.append(
                f"   The ORIGINAL system passed only {rate * 100:.0f}% of its graded "
                "runs — by the same majority rule applied to mutants, it would "
                "itself be 'killed', so every kill would be indistinguishable from "
                "noise. Stabilize the judge/system (or its rubric), then re-run."
            )
        else:
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
        if not result.regression_total and result.robustness:
            lines.append(c("⚠  NO SCORE — only meaning-preserving operators ran", "33"))
            lines.append(
                "   Robustness operators (paraphrase, reorder) are reported, never "
                "scored. Add regression operators for a mutation score."
            )
            lines.extend(_robustness_lines(result, c))
            return "\n".join(lines)
        lines.append(c("⚠  INVALID RUN — no mutant produced a clean verdict", "1;31"))
        lines.append(
            f"   All {result.total} mutant(s) errored (e.g. API failures). No "
            "score — investigate the failures and re-run."
        )
        return "\n".join(lines)

    if result.score is None:  # every evaluated mutant tied — no confident score
        lines.append(c("⚠  NO CONFIDENT SCORE — every mutant's verdict tied", "1;31"))
        lines.append(
            f"   All {result.unresolved} evaluated mutant(s) were unresolved "
            "(the judge straddled 50%). Use an ODD runs_per_mutant — ties can't "
            "happen then."
        )
        return "\n".join(lines)

    pct = result.score * 100
    score_color = "32" if pct >= 80 else "33" if pct >= 50 else "31"
    lo, hi = result.score_ci
    lines.append(
        f"Mutation score: {c(f'{pct:.0f}%', score_color)}  "
        f"[{_bar(result.score)}]  "
        f"({result.killed}/{result.resolved} mutants killed, "
        f"95% CI {lo * 100:.0f}-{hi * 100:.0f}%)"
    )
    if result.unresolved:
        from muteval.runner import PARTIAL_UNRESOLVED

        if result.status == PARTIAL_UNRESOLVED:
            lines.append(
                c(
                    f"   ⚠  INVALID for CI — {result.unresolved}/{result.evaluated} "
                    f"mutant(s) unresolved ({result.unresolved_rate * 100:.0f}% > "
                    "allowed budget; verdict tied over runs_per_mutant). The score "
                    "above is over the few RESOLVED mutants and is shown for "
                    "diagnosis only; the CLI exits non-zero and the badge is n/a. "
                    "Use an odd runs_per_mutant, or raise --max-unresolved-rate.",
                    "1;31",
                )
            )
        else:
            lines.append(
                c(
                    f"   {result.unresolved} unresolved (verdict tied over "
                    "runs_per_mutant; excluded from the score — an odd "
                    "runs_per_mutant can't tie).",
                    "33",
                )
            )
    rate = result.baseline_pass_rate
    if rate is not None and rate < 1.0:
        lines.append(
            c(
                f"   noise floor: the ORIGINAL system failed {(1 - rate) * 100:.0f}% of "
                "its graded runs — kills at that rate are noise, not detection.",
                "33",
            )
        )
    if result.errored:
        from muteval.runner import PARTIAL_ERRORS

        if result.status == PARTIAL_ERRORS:
            lines.append(
                c(
                    f"   ⚠  INVALID for CI — {result.errored}/{result.total} "
                    f"mutant(s) errored ({result.error_rate * 100:.0f}% > allowed "
                    "budget). Score above is over a SHRUNKEN denominator and is "
                    "shown for diagnosis only; the CLI exits non-zero and the "
                    "badge is n/a. Re-run, or raise --max-error-rate to accept it.",
                    "1;31",
                )
            )
        else:
            lines.append(
                c(
                    f"   {result.errored} mutant(s) errored and were excluded "
                    "(e.g. API timeouts). Re-run to retry them.",
                    "33",
                )
            )

    # Effective score: drop mutants that didn't change observed behavior —
    # unchanged survivors (inert) AND unchanged kills (noise) — from the score.
    inert = result.inert_survivors
    noise = result.noise_kills
    if inert or noise:
        parts = []
        if inert:
            parts.append(f"{len(inert)} inert mutant(s) whose output didn't change")
        if noise:
            parts.append(
                f"{len(noise)} noise kill(s) — 'caught' on output the original "
                "itself produces"
            )
        excluded = " and ".join(parts)
        if result.effective_score is not None:
            eff = result.effective_score * 100
            eff_color = "32" if eff >= 80 else "33" if eff >= 50 else "31"
            elo, ehi = result.effective_score_ci
            k, n = result.effective_counts
            lines.append(
                f"Effective score: {c(f'{eff:.0f}%', eff_color)}  "
                f"({k}/{n} — excludes {excluded}; "
                f"95% CI {elo * 100:.0f}-{ehi * 100:.0f}%)"
            )
        else:
            # No mutant changed observed behavior -> nothing to score. Say so
            # rather than crash on a None effective score.
            lines.append(
                c(
                    f"Effective score: n/a  (no mutant changed the observed "
                    f"behavior — excludes {excluded}; nothing to score)",
                    "33",
                )
            )

    if result.canary_caught is False:
        lines.append(
            c(
                "   ⚠  suite sanity: your evals passed a blank AND a nonsense "
                "output — they may not be discriminating (expected only for a "
                "guardrail-only suite). Read the score below with that in mind.",
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
        fbe = result.flaky_by_eval
        if fbe:
            top = ", ".join(
                f"{k} ({v})" for k, v in sorted(fbe.items(), key=lambda kv: -kv[1])
            )
            lines.append(
                c(
                    f"      flaky by eval — rewrite these rubric dimensions before "
                    f"adding runs: {top}",
                    "2",
                )
            )
    prov = []
    if result.model_under_test:
        prov.append(f"model under test: {result.model_under_test}")
    if result.judge_models:
        prov.append(f"judge: {', '.join(result.judge_models)}")
    if prov:
        lines.append(
            c("   " + " · ".join(prov) + " (pin these to compare over time)", "2")
        )
    if result.cache_hits is not None:
        lines.append(
            c(
                f"   cache: {result.cache_hits} lookup(s) served from --cache (keyed "
                "on your run/eval code; set cache_version on an eval that reads "
                "files or remote state)",
                "2",
            )
        )
    elif result.cache_note:
        lines.append(c(f"   cache: {result.cache_note}", "2"))
    undetermined = result.undetermined_survivors
    if result.noisy_cases and undetermined:
        lines.append(
            c(
                f"   {len(undetermined)} survivor(s) undetermined: the baseline's own "
                f"output varied on {result.noisy_cases} case(s), so a change there "
                "can't be told apart from sampling noise. Counted as gaps "
                "(conservative) — set output_key= to the part of the output that is "
                "the behavior (e.g. the label).",
                "33",
            )
        )
    lines.extend(_robustness_lines(result, c))
    lines.append("")

    survivors = result.survivors
    if not survivors:
        lines.append(
            c("✓ No survivors — your evals caught every injected regression.", "32")
        )
        return "\n".join(lines)

    accepted_n = len(result.accepted_survivors)
    real = result.new_survivors
    if not real and accepted_n:
        lines.append(
            c(
                f"✓ No new survivors — {accepted_n} accepted (untested by design) and "
                "no new coverage gaps.",
                "32",
            )
        )
        return "\n".join(lines)
    if real:
        from muteval.severity import HIGH, LOW, MEDIUM, severity_rank

        real = sorted(real, key=lambda o: severity_rank(o.severity or MEDIUM))
        n_high = sum(1 for o in real if o.severity == HIGH)
        accepted_note = (
            f" ({accepted_n} accepted — untested by design — hidden)"
            if accepted_n
            else ""
        )
        header = c(f"{len(real)} SURVIVED", "31") + (
            f"  (output changed but evals didn't notice — real coverage gaps"
            f"{accepted_note}"
        )
        if n_high:
            header += "; " + c(f"{n_high} HIGH-severity", "1;31")
        lines.append(header + "):")
        lines.append(
            c("  ranked by severity: ", "2")
            + c("HIGH", "31")
            + c(" › ", "2")
            + c("MED", "33")
            + c(" › ", "2")
            + c("LOW", "2")
        )
        lines.append("")
        _sev_color = {HIGH: "31", MEDIUM: "33", LOW: "2"}
        _sev_label = {HIGH: "HIGH", MEDIUM: "MED", LOW: "LOW"}
        from muteval.suggest import suggest_eval

        for i, o in enumerate(real, start=1):
            sev = o.severity or MEDIUM
            raw_tag = f"[{_sev_label[sev]}]"
            tag = c(raw_tag, _sev_color[sev]) + " " * (len("[HIGH]") - len(raw_tag))
            lines.append(
                f"  #{i} {tag} {c('SURVIVED', '31')}  [{o.mutant.operator}]  "
                + c(f"accept: {o.mutant.signature}", "2")
            )
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
            + "  (output matched the baseline on the samples we ran — NOT eval "
            "blind spots; excluded from the effective score. For a stochastic "
            "system this is not proof of equivalence — see docs/LIMITATIONS.md):"
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


# Not every lens carries the same weight — be honest about it in the output.
# core:     catches a real, common eval defect (trust a WARN here).
# validity: the "is the eval actually correct?" check — needs labels.
# hygiene:  a sanity check; a WARN is a footnote, not a crisis.
# Only lenses that actually run in the `muteval probe` card belong here. The
# judge-bias panel is a separate library function (needs a pairwise A/B judge),
# so it is deliberately NOT listed as a card lens.
_PROBE_TIER = {
    "judge_reliability": "core",
    "discrimination": "core",
    "human_agreement": "validity",
    "threshold_calibration": "hygiene",
    "statistical_adequacy": "hygiene",
    "redundancy": "hygiene",
}
_PROBE_TIER_LEGEND = (
    "core = catches a real eval defect · validity = needs labels · "
    "hygiene = sanity check (a WARN here is a footnote) · N/A = not assessed"
)


def probe_assessed(r) -> bool:
    """False when a probe couldn't assess anything (no exemplars / labels /
    runs): shown as N/A, never as a PASS."""
    return (getattr(r, "metrics", None) or {}).get("assessed", True) is not False


def probe_blocks(r) -> bool:
    """Does this probe result fail `muteval probe`? A WARN from a core or
    validity lens does; a hygiene WARN is a footnote, and N/A never blocks."""
    return probe_assessed(r) and not r.ok and _PROBE_TIER.get(r.name) != "hygiene"


def format_probe_card(results, use_color: bool = True) -> str:
    """Render probe results as an eval-quality report card (no composite score)."""

    def c(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    lines = [
        "",
        c("muteval — eval quality report card", "1"),
        c(f"  {_PROBE_TIER_LEGEND}", "2"),
        "",
    ]
    if not results:
        lines.append("No probes ran.")
        return redact("\n".join(lines))
    # Show core lenses first, then validity, then hygiene, then anything custom.
    order = {"core": 0, "validity": 1, "hygiene": 2}
    ranked = sorted(results, key=lambda r: order.get(_PROBE_TIER.get(r.name, ""), 3))
    for r in ranked:
        if not probe_assessed(r):
            tag = c("N/A ", "2")
        else:
            tag = c("PASS", "32") if r.ok else c("WARN", "33")
        tier = _PROBE_TIER.get(r.name, "")
        tier_str = c(f"  ({tier})", "2") if tier else ""
        lines.append(f"  [{tag}] {c(r.name, '1')}{tier_str}")
        lines.append(f"         {r.summary}")
        if r.detail:
            lines.append(c(f"         {r.detail}", "2"))
        lines.append("")
    return redact("\n".join(lines))


def format_probe_card_html(
    results, title: str = "muteval — eval quality report card"
) -> str:
    """Render the probe panel as a standalone HTML page. Deliberately NO composite
    score — the panel is a set of separately-interpretable signals."""
    order = {"core": 0, "validity": 1, "hygiene": 2}
    ranked = sorted(results, key=lambda r: order.get(_PROBE_TIER.get(r.name, ""), 3))
    cards = []
    for r in ranked:
        state = "na" if not probe_assessed(r) else ("pass" if r.ok else "warn")
        badge = "N/A" if not probe_assessed(r) else ("PASS" if r.ok else "WARN")
        tier = _PROBE_TIER.get(r.name, "")
        tier_html = f'<span class="tier">{tier}</span>' if tier else ""
        detail = (
            f'<div class="pd">{html.escape(redact(r.detail))}</div>' if r.detail else ""
        )
        cards.append(
            f"""<div class="card {state}">
  <div class="chd"><span class="badge {state}">{badge}</span>
    <span class="pn">{html.escape(r.name)}</span>{tier_html}</div>
  <div class="psum">{html.escape(redact(r.summary))}</div>
  {detail}
</div>"""
        )
    body = "\n".join(cards) or "<p>No probes ran.</p>"
    n_warn = sum(1 for r in results if not r.ok)
    subtitle = (
        f"{n_warn} of {len(results)} probes flagged an issue — {_PROBE_TIER_LEGEND}"
        if results
        else "no probes ran"
    )
    return f"""<!doctype html>
<meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1f2328}}
 h1{{font-size:1.4rem;margin-bottom:.2rem}} .muted{{color:#656d76}}
 .card{{border:1px solid #d0d7de;border-left-width:5px;border-radius:8px;padding:.7rem 1rem;margin:.7rem 0}}
 .card.pass{{border-left-color:#2ea043}} .card.warn{{border-left-color:#d29922}}
 .chd{{display:flex;gap:.6rem;align-items:center}} .pn{{font-family:ui-monospace,monospace;font-weight:600}}
 .badge{{font-size:.72rem;font-weight:700;padding:.1rem .45rem;border-radius:4px;color:#fff}}
 .badge.pass{{background:#2ea043}} .badge.warn{{background:#d29922}}
 .badge.na{{background:#8b949e}}
 .psum{{margin:.4rem 0}} .pd{{color:#656d76;font-size:.9rem}}
 .tier{{font-size:.7rem;color:#656d76;border:1px solid #d0d7de;border-radius:4px;padding:.05rem .35rem;text-transform:uppercase;letter-spacing:.03em}}
</style>
<h1>{html.escape(title)}</h1>
<p class="muted">{subtitle} — no composite score (each signal stands on its own).</p>
{body}
<p class="muted" style="margin-top:2rem;font-size:.85rem">Generated by muteval.</p>
"""


# The JSON schema version. Bump on any breaking change to result_to_dict's shape;
# consumers can branch on it. Snapshotted in tests/test_output.py.
RESULT_SCHEMA_VERSION = 6

# Secrets never reach any output: every formatter below goes through the one
# redaction point in muteval.redact (see its module doc). Kept under the old
# name for back-compat.
_redact = redact_obj


def _severity_sorted(survivors):
    """Survivors ordered HIGH severity first (stable), for triage."""
    from muteval.severity import MEDIUM, severity_rank

    return sorted(survivors, key=lambda o: severity_rank(o.severity or MEDIUM))


def result_to_dict(result) -> dict:
    """Machine-readable summary of a MutationResult (for --json / CI / reports).

    Secrets (API keys) are redacted from all string fields before returning.
    """
    from muteval.suggest import suggest_eval

    score = result.score
    eff = result.effective_score
    return _redact(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": result.status,
            "baseline_passed": result.baseline_passed,
            "baseline_error": result.baseline_error,
            "score": round(score, 4) if score is not None else None,
            "effective_score": round(eff, 4) if eff is not None else None,
            "score_ci": [round(x, 4) for x in result.score_ci],
            "effective_score_ci": [round(x, 4) for x in result.effective_score_ci],
            "killed": result.killed,
            "evaluated": result.evaluated,
            "resolved": result.resolved,
            "unresolved": result.unresolved,
            "total": result.total,
            "errored": result.errored,
            "error_rate": round(result.error_rate, 4),
            "inert": len(result.inert_survivors),
            "high_severity_survivors": len(result.high_severity_survivors),
            "canary_caught": result.canary_caught,
            "model_under_test": result.model_under_test,
            "judge_models": list(result.judge_models),
            "flaky_by_eval": result.flaky_by_eval,
            "accepted": len(result.accepted_survivors),
            # Meaning-preserving mutants: evaluated but never scored. "brittle"
            # are the ones that flipped an eval verdict anyway.
            "robustness": len(result.robustness),
            "brittle": [
                {
                    "operator": o.mutant.operator,
                    "description": o.mutant.description,
                    "failing_eval": o.failing_eval,
                }
                for o in result.brittle
            ],
            "noisy_cases": result.noisy_cases,
            "undetermined": len(result.undetermined_survivors),
            "noise_kills": len(result.noise_kills),
            "cache": (
                {"hits": result.cache_hits, "note": result.cache_note}
                if result.cache_hits is not None or result.cache_note
                else None
            ),
            "baseline_pass_rate": (
                round(result.baseline_pass_rate, 4)
                if result.baseline_pass_rate is not None
                else None
            ),
            "survivors": [
                {
                    "id": i,
                    "operator": o.mutant.operator,
                    "description": o.mutant.description,
                    "severity": o.severity,
                    "signature": o.mutant.signature,
                    "accepted": o.mutant.signature in result.accepted,
                    "fix": suggest_eval(o),
                    "baseline_output": o.baseline_output,
                    "mutant_output": o.mutant_output,
                }
                for i, o in enumerate(_severity_sorted(result.real_survivors), start=1)
            ],
        }
    )


def format_report_junit(result: MutationResult) -> str:
    """Serialize mutation outcomes as a JUnit XML document.

    Each generated mutant is one testcase. A killed mutant passes; a survivor
    is a failure because the eval suite missed the injected regression. Runtime
    errors are represented as testcase errors. Invalid baselines still produce
    a valid, empty suite with a suite-level error message.
    """
    from xml.etree.ElementTree import Element, SubElement, tostring

    from muteval.mutators import ROBUSTNESS

    def _robust(o) -> bool:
        return o.mutant.intent == ROBUSTNESS

    suite = Element(
        "testsuite",
        {
            "name": "muteval",
            "tests": str(result.total),
            "failures": str(
                sum(
                    1
                    for o in result.outcomes
                    if not o.errored and not o.killed and not _robust(o)
                )
            ),
            "errors": str(result.errored),
            "skipped": str(
                sum(1 for o in result.outcomes if _robust(o) and not o.errored)
            ),
        },
    )
    if result.total == 0 and (result.baseline_error or not result.baseline_passed):
        suite.set("tests", "1")
        suite.set("errors", "1")
        case = SubElement(suite, "testcase", {"classname": "muteval", "name": "baseline"})
        msg = redact(result.baseline_error or "baseline failed")
        error = SubElement(case, "error", {"message": msg})
        error.text = msg
    else:
        for outcome in result.outcomes:
            name = redact(f"{outcome.mutant.operator}: {outcome.mutant.description}")
            case = SubElement(suite, "testcase", {"classname": "muteval", "name": name})
            if outcome.errored:
                msg = redact(outcome.error or "mutant errored")
                error = SubElement(case, "error", {"message": msg})
                error.text = msg
            elif _robust(outcome):
                # Meaning-preserving edit: never a CI failure either way.
                note = (
                    "robustness (not scored): a meaning-preserving edit flipped "
                    f"{outcome.failing_eval}"
                    if outcome.killed
                    else "robustness (not scored): survived, as expected"
                )
                SubElement(case, "skipped", {"message": note})
            elif not outcome.killed:
                failure = SubElement(case, "failure", {"message": "mutation survived"})
                failure.text = redact(outcome.mutant.description)
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + tostring(suite, encoding="unicode")
        + "\n"
    )


def run_manifest(result, config, operators=None, seed=None) -> dict:
    """A reproducible-run manifest: provenance (version, model, seed, operator
    set, config fingerprint, timestamp) + the machine-readable result. Committing
    this next to a real-LLM-judge run makes the number auditable and repeatable.
    Secrets are redacted."""
    import hashlib
    import platform
    import sys as _sys
    from datetime import datetime, timezone

    from muteval import __version__

    system = getattr(config, "system", None)
    key = (
        repr(system.key()) if system is not None else repr(getattr(config, "prompt", ""))
    )
    fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return _redact(
        {
            "manifest_version": 1,
            "muteval_version": __version__,
            "python": _sys.version.split()[0],
            "platform": platform.platform(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run": {
                "model": getattr(system, "model", None) if system is not None else None,
                "operators": list(operators) if operators else "all",
                "seed": seed,
                "n_cases": len(config.cases) if config.cases else 0,
                "eval_names": list(config.eval_names),
                "runs_per_mutant": config.runs_per_mutant,
                "system_fingerprint": fingerprint,
            },
            "result": result_to_dict(result),
        }
    )


def badge_dict(result, label: str = "eval coverage") -> dict:
    """A shields.io endpoint payload for the effective mutation score."""
    eff = result.effective_score
    if eff is None or result.status != "valid":
        return {
            "schemaVersion": 1,
            "label": label,
            "message": "n/a",
            "color": "lightgrey",
        }
    pct = round(eff * 100)
    color = "brightgreen" if pct >= 80 else "yellow" if pct >= 50 else "red"
    return {"schemaVersion": 1, "label": label, "message": f"{pct}%", "color": color}


def _diff_html(base: str, mutant: str) -> str:
    """A minimal line-diff of two outputs as escaped, colored HTML rows."""
    import difflib

    rows = []
    for ln in difflib.unified_diff(
        base.splitlines(),
        mutant.splitlines(),
        fromfile="baseline",
        tofile="mutant",
        lineterm="",
    ):
        cls = "add" if ln.startswith("+") else "del" if ln.startswith("-") else "ctx"
        rows.append(f'<div class="dl {cls}">{html.escape(ln)}</div>')
    return "".join(rows) or '<div class="dl ctx">(no textual diff)</div>'


def format_report_html(data: dict, title: str = "muteval — eval coverage report") -> str:
    """Render a result_to_dict() payload (or a saved last_run.json) as a
    self-contained HTML report: score, survivors, and baseline→mutant diffs."""
    # The payload may be an older / hand-edited JSON file, not one this version
    # wrote (already redacted) — redact again before rendering.
    data = redact_obj(data)

    def pct(x):
        return "n/a" if x is None else f"{round(x * 100)}%"

    status = data.get("status", "unknown")
    valid = status == "valid"
    eff = data.get("effective_score")
    bar_color = (
        "#2ea043" if (eff or 0) >= 0.8 else "#d29922" if (eff or 0) >= 0.5 else "#f85149"
    )
    ci = data.get("effective_score_ci") or [0, 0]
    survivors = data.get("survivors", [])

    cards = []
    for s in survivors:
        sev = (s.get("severity") or "medium").lower()
        base, mut = s.get("baseline_output"), s.get("mutant_output")
        diff = (
            _diff_html(base, mut)
            if (base is not None and mut is not None)
            else '<div class="dl ctx">(output unchanged / not captured)</div>'
        )
        cards.append(
            f"""<div class="card {sev}">
  <div class="chd"><span class="sev {sev}">{sev.upper()}</span>
    <span class="op">{html.escape(str(s.get("operator", "")))}</span>
    <span class="cid">#{s.get("id", "")}</span></div>
  <div class="desc">{html.escape(str(s.get("description", "")))}</div>
  <div class="fix"><b>fix:</b> {html.escape(str(s.get("fix", "") or "—"))}</div>
  <div class="diff">{diff}</div>
</div>"""
        )
    cards_html = (
        "\n".join(cards)
        or '<p class="ok">No survivors — your evals caught every injected regression.</p>'
    )

    banner = (
        ""
        if valid
        else (
            f'<div class="warn">⚠ INVALID / INCOMPLETE run (status: {html.escape(status)}) '
            "— the score below is not trustworthy.</div>"
        )
    )

    return f"""<!doctype html>
<meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1f2328}}
 h1{{font-size:1.4rem}} .muted{{color:#656d76}}
 .warn{{background:#ffebe9;border:1px solid #ff818266;padding:.6rem .8rem;border-radius:6px;margin:1rem 0;color:#a40e26}}
 .score{{font-size:2.4rem;font-weight:700}}
 .track{{height:12px;background:#eaeef2;border-radius:6px;overflow:hidden;margin:.4rem 0 1rem}}
 .fill{{height:100%;background:{bar_color}}}
 .stats{{display:flex;gap:1.5rem;flex-wrap:wrap;margin:.5rem 0 1.5rem}} .stats div b{{display:block;font-size:1.2rem}}
 .card{{border:1px solid #d0d7de;border-left-width:5px;border-radius:8px;padding:.8rem 1rem;margin:.8rem 0}}
 .card.high{{border-left-color:#f85149}} .card.medium{{border-left-color:#d29922}} .card.low{{border-left-color:#9aa0a6}}
 .chd{{display:flex;gap:.6rem;align-items:center}} .op{{font-family:ui-monospace,monospace;font-weight:600}} .cid{{color:#8b949e;margin-left:auto}}
 .sev{{font-size:.72rem;font-weight:700;padding:.1rem .4rem;border-radius:4px;color:#fff}}
 .sev.high{{background:#f85149}} .sev.medium{{background:#d29922}} .sev.low{{background:#9aa0a6}}
 .desc{{margin:.4rem 0}} .fix{{color:#0969da;font-size:.9rem;margin:.3rem 0}}
 .diff{{background:#f6f8fa;border-radius:6px;padding:.4rem;font-family:ui-monospace,monospace;font-size:.82rem;overflow:auto;margin-top:.5rem}}
 .dl{{white-space:pre-wrap}} .dl.add{{background:#e6ffec;color:#116329}} .dl.del{{background:#ffebe9;color:#a40e26}} .dl.ctx{{color:#656d76}}
 .ok{{color:#116329;font-weight:600}}
</style>
<h1>{html.escape(title)}</h1>
{banner}
<div class="score">{pct(eff)} <span class="muted" style="font-size:1rem">effective coverage</span></div>
<div class="track"><div class="fill" style="width:{round((eff or 0) * 100)}%"></div></div>
<div class="stats">
 <div><b>{pct(data.get("score"))}</b>raw score</div>
 <div><b>{ci[0] * 100:.0f}–{ci[1] * 100:.0f}%</b>95% CI</div>
 <div><b>{data.get("killed", 0)}/{data.get("evaluated", 0)}</b>killed / evaluated</div>
 <div><b>{data.get("inert", 0)}</b>inert (excluded)</div>
 <div><b>{data.get("high_severity_survivors", 0)}</b>high-severity survivors</div>
</div>
<h2>Survivors <span class="muted">({len(survivors)})</span></h2>
{cards_html}
<p class="muted" style="margin-top:2rem;font-size:.85rem">Generated by muteval. Each survivor is an output change your evals did not catch — write an eval that fails on it, then re-run.</p>
"""
