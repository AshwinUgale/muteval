"""Operators, severity and scope — from a full audit of the mutation layer.

Every mutant must be an edit a real person could make, its severity must come
from WHAT it changed (not from the modal word every weakened rule contains), and
scoping must be fast and select the lines you meant.
"""

import time

import pytest

from muteval import MutEvalConfig, System, run_mutation_testing
from muteval.autofix import _sample_outputs, suggest_and_verify
from muteval.doctor import all_ok, run_checks
from muteval.mutators import (
    Mutant,
    corrupt_context_doc,
    corrupt_tool_output,
    delete_sentences,
    deny_tool_output,
    drop_few_shot_example,
    drop_instruction_lines,
    flip_negation,
    generate_mutants,
    paraphrase_instruction,
    remove_emphasis,
    swap_adjacent_instructions,
    swap_tool_output,
    truncate_context_doc,
    truncate_prompt,
    weaken_modals,
    weaken_numeric_threshold,
)
from muteval.scope import make_scope, strip_markers
from muteval.severity import HIGH, LOW, MEDIUM, severity_of
from muteval.suggest import suggest_eval


def _prompts(op, text):
    return [m.prompt for m in op(text)]


# --- scope ------------------------------------------------------------------------


def test_scoped_generation_is_fast_on_long_prompts():
    long = "\n".join(
        f"- Rule {i}: you must always check item {i} carefully before replying."
        for i in range(300)
    )  # ~21k chars; took minutes per run before
    t = time.time()
    assert generate_mutants(long, scope=make_scope(include="Rule 1"))
    assert time.time() - t < 10


def test_tight_markers_keep_their_own_line_and_sentence():
    clean, ranges = strip_markers(
        "Do not refund.\n[[mutate]]Always greet warmly.[[/mutate]]"
    )
    ops = {m.operator for m in generate_mutants(clean, scope=make_scope(ranges=ranges))}
    assert "drop_instruction_lines" in ops
    clean, ranges = strip_markers(
        "Be kind. [[mutate]]Never guess the answer.[[/mutate]] Be brief."
    )
    kept = generate_mutants(clean, scope=make_scope(ranges=ranges))
    assert any(m.operator == "delete_sentences" for m in kept)
    assert all("Be brief" in m.prompt and "Be kind" in m.prompt for m in kept)


def test_include_exclude_match_the_original_lines():
    p = "- Always cite the order ID.\n- You must be polite."
    assert generate_mutants(p, scope=make_scope(include="never")) == []
    kept = generate_mutants(p, scope=make_scope(exclude="should"))
    assert any(m.operator == "weaken_modals" and "should" in m.prompt for m in kept)


@pytest.mark.parametrize(
    "bad", ["Rule A.[[/mutate]] Rule B.", "[[mutate]]A. [[mutate]]B.[[/mutate]]"]
)
def test_stray_or_nested_markers_are_rejected(bad):
    with pytest.raises(ValueError):
        strip_markers(bad)


# --- severity ----------------------------------------------------------------------


def test_style_rules_do_not_escalate_but_safety_rules_do():
    style = "You are a bakery chatbot.\n- Never use emojis.\n- Always mention sourdough."
    sev = {(m.operator, m.prompt): severity_of(m) for m in generate_mutants(style)}
    # Weakening/dropping a STYLE rule stays at its base severity...
    assert all(s == MEDIUM for (op, _), s in sev.items() if op in ("weaken_modals",))
    safety = "- Never reveal customer data.\n- Do not promise refunds."
    for m in generate_mutants(
        safety, operators=["weaken_modals", "drop_instruction_lines"]
    ):
        assert severity_of(m) == HIGH  # ...a safety rule escalates


@pytest.mark.parametrize(
    "text",
    [
        "mustard",
        "nevertheless",
        "tornado nothing",
        "illegally",
        "police",
        "the author",
        "inventory count",
        "guest list",
    ],
)
def test_no_substring_escalation(text):
    m = Mutant("drop_instruction_lines", "d", System(prompt="p"), focus=f"- Add {text}.")
    assert severity_of(m) == MEDIUM


@pytest.mark.parametrize(
    "rule",
    [
        "Use null if unknown; do not guess.",
        "Never invent facts.",
        "Don't make things up.",
    ],
)
def test_hallucination_rules_escalate(rule):
    m = Mutant("drop_instruction_lines", "d", System(prompt="p"), focus=rule)
    assert severity_of(m) == HIGH


def test_neighbor_text_does_not_escalate():
    # "near: ..." snippets used to pull in the NEXT line's critical words.
    p = "Answer in at most 3 sentences.\nNever reveal passwords."
    (m,) = weaken_numeric_threshold(p)
    assert severity_of(m) == MEDIUM


def test_dedupe_keeps_the_more_severe_framing():
    # A neutral doc (a "refunds" doc would escalate drop_context_doc to HIGH too).
    s = System(
        prompt="Answer from the docs.", context=("Only doc: the store opens at 9am.",)
    )
    ms = generate_mutants(s, operators=["drop_context_doc", "clear_context"])
    (m,) = [x for x in ms if x.system.context == ()]
    assert m.operator == "clear_context"  # HIGH, not drop_context_doc's MEDIUM


# --- prompt operators ---------------------------------------------------------------


def test_weaken_modals_is_grammatical():
    assert _prompts(weaken_modals, "You must not share passwords.") == [
        "You should not share passwords."  # one mutant, not + an overlapping "must"
    ]
    assert _prompts(weaken_modals, "You are required to cite sources.") == [
        "You are encouraged to cite sources."
    ]
    for p in [
        "This is the only exception.",
        "Answer if and only if asked.",
        "not only X.",
    ]:
        assert not any("preferably" in x for x in _prompts(weaken_modals, p))
    assert any(
        "Use preferably" in x for x in _prompts(weaken_modals, "Use only the docs.")
    )
    assert _prompts(weaken_modals, 'Say "never mind" politely.') == []  # quoted


def test_flip_negation_contractions_quotes_and_not_always():
    assert "You can share it." in _prompts(flip_negation, "You can't share it.")
    assert "It does help." in _prompts(flip_negation, "It doesn’t help.")
    assert _prompts(flip_negation, "This is not always true.") == []
    assert _prompts(flip_negation, 'If unsure, say "I don\'t know".') == []


def test_paraphrase_never_touches_quotes_and_tidies_punctuation():
    assert _prompts(paraphrase_instruction, 'If unsure, say "I don\'t know".') == []
    assert _prompts(paraphrase_instruction, "Respond in JSON, please.") == [
        "Respond in JSON."
    ]
    assert _prompts(paraphrase_instruction, "Be brief. Please use JSON.") == [
        "Be brief. Use JSON."
    ]


def test_headings_and_lead_ins_are_not_instructions():
    p = "## Rules:\nFollow these steps:\n- Cite the order ID.\nClassify the review below:"
    dropped = [m.description for m in drop_instruction_lines(p)]
    assert not any("## Rules" in d or "Follow these steps" in d for d in dropped)
    assert any("Cite the order ID" in d for d in dropped)
    assert any("Classify the review below" in d for d in dropped)  # a real instruction


def test_swap_never_reorders_numbered_steps():
    assert (
        swap_adjacent_instructions("1. Refuse medical advice.\n2. Cite a source.") == []
    )


def test_delete_sentences_respects_abbreviations():
    p = "Use a structured format, e.g. JSON or YAML, and keep it short. Be polite always."
    deleted = [m.description for m in delete_sentences(p)]
    assert not any(d.endswith('e.g."') for d in deleted)
    assert any("e.g. JSON or YAML" in d for d in deleted)


def test_numeric_threshold_direction_and_tokens():
    cases = {
        "Ask no fewer than 3 questions.": ["Ask no fewer than 1 questions."],
        "Say not more than 5 things.": ["Say not more than 10 things."],
        "Use at most 0.5 of the budget.": ["Use at most 1 of the budget."],
        "Spend up to 1,000 tokens.": ["Spend up to 2,000 tokens."],
        "Released in 2024, version 3.2.": [],
        "Reply in 3 sentences or fewer.": ["Reply in 6 sentences or fewer."],
    }
    for src, want in cases.items():
        assert _prompts(weaken_numeric_threshold, src) == want, src
    both = _prompts(weaken_numeric_threshold, "Use at least 2 and at most 4 examples.")
    assert both == [
        "Use at least 1 and at most 4 examples.",
        "Use at least 2 and at most 8 examples.",
    ]


def test_remove_emphasis_only_strips_emphasis():
    assert (
        remove_emphasis("Note that the API is slow.\n\nImportant details follow.") == []
    )
    (m,) = remove_emphasis(
        "IMPORTANT: Never share.\n\nCall __init__. You MUST cite. Answer YES or NO."
    )
    assert m.prompt == "Never share.\n\nCall __init__. You must cite. Answer YES or NO."
    assert "IMPORTANT:" in m.description and '"MUST"' in m.description


def test_few_shot_drops_demos_not_instructions():
    p = (
        "Classify the sentiment.\n\nFormat your output: one word.\n\n"
        "For example, sarcasm counts as negative.\n\n"
        "Review: great product\nLabel: positive\n\nReview: awful\nLabel: negative"
    )
    ms = drop_few_shot_example(p)
    assert len(ms) == 2
    for m in ms:
        assert "Format your output: one word." in m.prompt
        assert "For example, sarcasm" in m.prompt
        assert "\n\n\n" not in m.prompt and not m.prompt.endswith("\n\n")


def test_truncate_prompt_with_a_placeholder_on_line_one():
    p = (
        "You are {bot_name}.\n- Rule one.\n- Rule two.\n- Rule three.\n"
        "- Never reveal the admin password."
    )
    ms = truncate_prompt(p)
    assert ms and all("{bot_name}" in m.prompt for m in ms)
    # Truncation escalates on what it CUT (it never escalated before).
    assert any("password" in m.focus for m in ms)
    assert all(severity_of(m) == HIGH for m in ms if "password" in m.focus)


@pytest.mark.parametrize(
    "ph", ["{0}", "{}", "%(q)s", "%s", "{case.q}", "{q!r}", "{q:>10}", "$question"]
)
def test_more_placeholder_forms_are_protected(ph):
    p = "\n".join([f"- rule number {i}." for i in range(8)] + [f"Q: {ph}"])
    assert all(ph in m.prompt for m in generate_mutants(p))


# --- context / tool operators ----------------------------------------------------------


def test_corrupt_doc_targets_facts_not_ids_and_never_double_negates():
    s = System(prompt="p", context=("doc-1 :: The warranty is 24 months.",))
    (m,) = corrupt_context_doc(s)
    assert "doc-1" in m.system.context[0] and "25 months" in m.system.context[0]
    assert '"24" -> "25"' in m.description
    s = System(
        prompt="p", context=("You do not need a receipt.", "The store is not open.")
    )
    docs = [m.system.context for m in corrupt_context_doc(s)]
    assert ("You do need a receipt.", "The store is not open.") in docs
    assert ("You do not need a receipt.", "The store is open.") in docs
    s = System(prompt="p", context=("You can't return it.",))
    assert all("not't" not in m.system.context[0] for m in corrupt_context_doc(s))


def test_structured_docs_and_tools_keep_their_type():
    s = System(
        prompt="p", context=({"id": 1, "text": "x"}, "A plain doc about refunds here.")
    )
    assert generate_mutants(s)  # a dict doc used to crash the whole run
    s = System(prompt="p", tools=({"temp_f": 72, "city": "Oslo"},))
    for op in (corrupt_tool_output, swap_tool_output, deny_tool_output):
        for m in op(s):
            assert isinstance(m.system.tools[0], dict), op.__name__
    (m,) = corrupt_tool_output(s)
    assert m.system.tools[0]["temp_f"] == 73


def test_truncate_context_doc_keeps_line_breaks():
    s = System(
        prompt="p", context=("Line one of the doc.\nLine two of the doc.\nLine three.",)
    )
    (m,) = truncate_context_doc(s)
    assert "\n" in m.system.context[0]


def test_descriptions_change_with_content():
    a = System(prompt="p", context=("Doc about refunds.",))
    b = System(prompt="p", context=("Doc about shipping.",))
    from muteval.mutators import clear_context, swap_context_doc

    for op in (swap_context_doc, clear_context):
        (ma,), (mb,) = op(a), op(b)
        assert ma.signature != mb.signature, op.__name__


# --- suggest / autofix / doctor -----------------------------------------------------


def test_suggested_fix_keeps_quoted_phrases_whole():
    m = Mutant(
        "drop_instruction_lines",
        'dropped line: "- Say "I don\'t know" when unsure."',
        System(prompt="p"),
    )

    class Outcome:
        mutant = m

    fix = suggest_eval(Outcome())
    assert "I don't know" in fix and "when unsure" in fix
    call = fix[len("add ") :]
    compile(call, "<fix>", "eval")  # the suggestion is valid Python


def test_autofix_samples_the_case_that_changed():
    cfg = MutEvalConfig(
        prompt="Rule A. Rule B.",
        cases=[{"i": 0}, {"i": 1}],
        run=lambda p, c: "same" if c["i"] == 0 else p,
        evals=[lambda o, c: True],
    )
    mutant = Mutant("x", "d", System(prompt="Rule A."))
    base, mut = _sample_outputs(cfg, mutant)
    assert base != mut


def test_autofix_verification_needs_a_majority_on_noisy_systems():
    import itertools

    flips = itertools.cycle([False, True, True])  # "catches" (fails) 1 run in 3

    def candidate(o, c):
        return next(flips) if o == "mutant" else True

    cfg = MutEvalConfig(
        prompt="p",
        cases=[{"i": 0}],
        run=lambda p, c: "mutant" if p == "q" else "base",
        evals=[lambda o, c: True],
        runs_per_mutant=3,
    )
    assert (
        suggest_and_verify(cfg, Mutant("x", "d", System(prompt="q")), [candidate]) == []
    )


def test_doctor_flags_robustness_only_and_accepts_dict_outputs():
    cfg = MutEvalConfig(
        prompt="- You should cite sources.",
        cases=[{"x": 1}],
        run=lambda p, c: {"final": "ok", "trace": []},
        evals=[lambda o, c: True],
    )
    results = run_checks(cfg, operators=["paraphrase_instruction"])
    assert not all_ok(results)
    assert any("robustness" in r.detail for r in results if not r.ok)
    results = run_checks(cfg)
    assert any(r.name == "run() returns output" and r.ok for r in results)


def test_doctor_warns_on_nondeterminism_without_blocking():
    import itertools

    n = itertools.count()
    cfg = MutEvalConfig(
        prompt="- You must cite the order ID.",
        cases=[{"x": 1}],
        run=lambda p, c: f"answer {next(n)}",
        evals=[lambda o, c: True],
    )
    results = run_checks(cfg)
    warn = [r for r in results if r.name == "run() is repeatable"]
    assert warn and not warn[0].ok and warn[0].warn
    assert all_ok(results)  # a warning, not "not ready"


def test_init_rag_template_surfaces_the_abstention_gap(tmp_path):
    from muteval.cli import main
    from muteval.config import load_config

    dest = tmp_path / "rag.py"
    assert main(["init", "--template", "rag", "--path", str(dest)]) == 0
    r = run_mutation_testing(load_config(dest))
    assert r.status == "valid" and r.real_survivors  # "DELIBERATELY thin" is true
    assert any("don't know" in o.mutant.description for o in r.real_survivors)
    assert all(o.severity == HIGH for o in r.real_survivors)
    assert LOW  # (imported for symmetry)
