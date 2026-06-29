# Plan: (A) user-controlled mutation scope + custom operators, (B) context/tool/model mutation

Persistent design doc so this can be picked up on any machine. Status reflects
the repo as of the System refactor + output-diffing work.

## Current state (don't re-discover this)

- `src/muteval/system.py` — `System(prompt, context, tools, model, extra)` is the
  mutation target; `as_system()` promotes a bare string for back-compat.
- `src/muteval/mutators.py` — operators take `Target` (System|str), return
  `Mutant(system=...)`. Registry: `OPERATORS`. `generate_mutants(target, operators=None)`.
  - Prompt operators: weaken_modals, flip_negation, drop_instruction_lines,
    delete_sentences, truncate_prompt, drop_few_shot_example, remove_emphasis.
  - Context operators ALREADY exist: drop_context_doc, clear_context.
- `src/muteval/runner.py` — engine + output-diffing (real vs inert survivors,
  `effective_score`), error resilience.
- `src/muteval/runners.py` — `openai_run` (stdlib-only) so zero-config CLI works.
- `src/muteval/cli.py` — zero-config (`--prompt-file/--cases/--check/--judge/--model`)
  OR `--config file.py`. Has `--operators` (select subset) and `--max-mutants`.
- `src/muteval/config.py` — MutEvalConfig (prompt/system, cases, run, evals...).

Guiding rule: **mutants come from the system, never from the evals.** Both A and
B must preserve that orthogonality (it's what enables absence detection).

---

## Part A — let users control WHAT gets mutated

Goal: steer which parts of the system get mutated, and allow custom operators —
without ever coupling mutant generation to the user's evals.

- [x] **A1 Operator selection** — DONE (`--operators`, `generate_mutants(operators=)`).

- [ ] **A2 Scoping** (mutate only chosen regions of the prompt)
  - Design: implement as a POST-generation FILTER, so we don't touch every
    operator. For each mutant, locate the changed span vs the original and keep
    it only if that span overlaps the scope.
  - Two ways to specify scope:
    a. Inline markers in the prompt, e.g. `[[mutate]] ... [[/mutate]]`. If any
       markers are present, only those regions are mutable.
    b. Regex flags: `--scope-include REGEX` / `--scope-exclude REGEX` (line-level).
  - Files: new `src/muteval/scope.py` (parse markers, compute mutable spans,
    filter mutants); thread `scope=` through `generate_mutants`; add CLI flags
    in `cli.py`; add `MutEvalConfig.scope`.
  - Tests: scoped prompt yields mutants only inside the region; exclude-regex
    drops mutants whose change lands on matching lines.

- [ ] **A3 Custom operators** (bring your own)
  - Public API: `muteval.register_operator(name, fn)` AND config-level operators
    that accept callables (not just names).
  - `generate_mutants` currently takes operator NAMES; extend to accept callables.
    Add `MutEvalConfig.operators: list[str | Callable] | None`.
  - Files: `mutators.py` (register_operator + accept callables), `config.py`,
    export in `__init__.py`.
  - Tests: register a custom op, confirm it runs and shows in the report.

- [ ] **A5 Sampling** — `--sample N` (+ `--seed`) deterministic random subset for
  cheap runs. Files: `cli.py`, `mutators.py`/`runner.py`.

- [ ] **A4 Per-operator params** (lower priority) — operator factories, e.g.
  custom modal pairs, truncate fractions. Optional.

---

## Part B — mutate context, tools, model (the RAG/agent moat)

Goal: extend mutation past the prompt to the rest of the System. Higher-demand
frontier (RAG/agents).

- [x] **B0 System foundation** — DONE (system.py; operators take Target).

- [ ] **B1 The `run()` contract is the linchpin** (do this FIRST — unblocks all of B)
  - For context/tool/model mutation to change anything, `run()` must CONSUME the
    mutated System, not just the prompt. Today the contract is
    `run(prompt, case) -> output`.
  - Plan: support `run(system, case)` in addition to legacy `run(prompt, case)`.
    Detect via signature inspection, or add an explicit `run_system` field.
    Back-compat: a 2-arg `run(prompt, case)` keeps working (receives
    `system.prompt`).
  - Files: `runner.py` (pass System where supported), `config.py` (document the
    new contract), `runners.py` (`openai_run` should use `system.context` and
    `system.model`).

- [ ] **B2 More context operators** (build on drop_context_doc / clear_context)
  - corrupt_context_doc (inject a plausible-but-wrong fact), swap_context_doc
    (replace with an irrelevant doc), shuffle_context (reorder — position
    sensitivity), duplicate_context_doc, truncate_context_doc.
  - Start rule-based; LLM-driven semantic corruption later (behind an extra).
  - Files: `mutators.py` + `OPERATORS`; a test per operator.

- [ ] **B5 Zero-config CLI for context** — `--cases` JSONL may carry a per-case
  `context`; `openai_run` injects it; enable context operators from the CLI.
  Files: `cli.py`, `runners.py`.

- [ ] **B3 Model-swap operator** — `downgrade_model`: set `System.model` to a
  weaker model from a configurable ladder; `run`/`openai_run` must honor
  `system.model`. Files: `mutators.py`, `runners.py`, maybe `--model-ladder`.

- [ ] **B4 Tool-output operators** (agents, after context) — define the
  tool-output shape in `System.tools`; operators: drop_tool_output,
  corrupt_tool_output, swap_tool_output.

---

## Recommended order

1. **B1** run(System) contract (unblocks all of B).
2. **B2** context operators + **B5** CLI context (the moat + demand).
3. **A2** scoping + **A3** custom operators (high user value, low coupling).
4. **B3** model-swap → **A5** sampling → **B4** tools → **A4** params.

## Open questions

- Scope marker syntax: `[[mutate]]` vs an HTML-comment style?
- run(System) detection: signature inspection vs an explicit `run_system` field?
- corrupt_context_doc: rule-based vs LLM-driven (cost)? Start rule-based.
- Output-diff semantics for context mutants where the output *should* change
  (e.g. cleared context -> "I don't know"): inert-detection still holds (output
  changed), but document how killed/survived reads for context.

## Don't-break list

- `as_system` back-compat: bare-string prompts and 2-arg `run(prompt, case)`
  must keep working.
- Output-diffing (real vs inert) must apply to context/tool mutants too.
- Core stays dependency-free; LLM-driven operators go behind optional extras.
- Every new operator and public function gets a test (current: 64 passing).
