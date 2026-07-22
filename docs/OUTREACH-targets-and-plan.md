# muteval outreach: find real eval gaps, contribute, get noticed

A living plan + target list for running muteval against real OSS projects,
finding genuine eval-coverage gaps, and turning them into contributions + content.

## The play (and why it works)

"Show, don't tell." Find a real eval gap in a real project, hand it to the
maintainer as a gift, contribute the fix, and write it up. This is simultaneously
demand validation, content, a contribution track record, and relationship-building.

### Success factors (do these or it backfires)
1. **Lead with a gift, not a pitch.** "I found your eval suite wouldn't catch X
   (reproducible case attached) — thought you'd want to know" > "use my tool".
2. **PR/issue > proposal.** Open a documented issue or small PR. Maintainers
   respond to artifacts, not sales offers.
3. **Filter with severity.** Only reach out about HIGH-severity (safety/correctness)
   gaps. "Your evals don't check tone" is not worth an email.
4. **Be respectful in public.** Give the maintainer a heads-up / draft before you
   publish. Frame gaps as structural, never as a dig. This decides whether they
   share your post or resent it.
5. **Treat adoption/repost as upside, not the goal.** The blog + findings are the
   win regardless.

### Realistic odds (per the strategy discussion)
- Find ≥1 genuine issue across several repos: HIGH (~70–85%).
- Maintainer engages positively: MODERATE (~30–50%).
- Maintainer adds muteval: LOWER (~10–30%).
- Maintainer reposts blog: LOW (~5–20%, higher after rapport).

## Per-repo vetting checklist (run BEFORE investing time)

A candidate only qualifies if ALL are true:
- [ ] **Real, runnable eval suite** — actual eval files (deepeval/ragas/custom),
      not just a mention. Check: search the repo for `import deepeval`,
      `from ragas`, or a `tests/`/`eval*/` dir with metric calls.
- [ ] **Re-runnable system** — you can invoke the system to produce a fresh output
      (muteval's hard requirement). Pure "score a fixed CSV" setups don't fit.
- [ ] **Active** — commits in the last ~2–3 months.
- [ ] **Responsive maintainer** — recent issues get replies within days (skim the
      issues tab). This is the one I could NOT verify for you — check it yourself.
- [ ] **Mid-size** — big enough to matter, small enough to notice you. Rough band:
      ~300–8k stars; avoid mega-projects (LangChain etc.) where you'll be invisible.

## VETTED TARGETS (re-verified 2026-07-21 after external review)

Hard-won lesson from this pass: our first "Bucket 1" leaned on data from a
rate-limited scrape, and a reviewer correctly flagged that two of them
(`leisurelyleon/ragline`, `OpenAgentHQ/openagent-eval`) can't be independently
located. So **verification status is now explicit and nothing is "vetted" until
it passes the clone-and-run gate below.** Confirmation legend:
- **[API✓]** confirmed live via GitHub REST API this session.
- **[search✓]** confirmed via web search this session.
- **[reviewer]** reviewer-cited, but I could NOT independently confirm this
  session — treat as a lead, clone-verify first.
- **[unverified]** neither — do not invest until cloned.

### The clone-and-run gate (ALL must hold before a repo earns the spreadsheet)
```
git clone succeeds
one documented command runs the pipeline
one documented command runs the evals
at least one system component is mutable (prompt / context / retrieval / model)
baseline passes twice consecutively
repository has a real (permissive) license
maintainer activity visible in the last ~3 months
```

### Phase 1 — establish muteval credibility (smoke tests; a dramatic survivor is NOT the goal)
Objective: prove install is clean, the baseline gate works, mutants generate,
results reproduce, partial errors fail closed, reports diagnose survivors.

1. **sanmaxdev/ragproof**  [reviewer]  — reviewer's top keyless smoke test:
   bundled no-key example, deterministic retrieve/citation/abstention/injection
   metrics, re-runnable `retrieve()`/`answer()`, a tagged PyPI `ragproof` 1.0.0.
   **Caveat (important): I could NOT surface it via the GitHub API or web search
   this session — clone-verify before trusting it.** Also note it is itself an
   eval product, so it proves muteval *works*; it is not a "we found an app's
   blind spot" story.
2. **aswithabukka/Evaluation-First-Testing-Harness-for-RAG-and-Agents**  [reviewer/search]
   — real runner + evaluators + a `demo_rag` adapter; ~65 keyless unit tests,
   but the E2E quickstart wants an OpenRouter/OpenAI key; 0★, portfolio project.
   Integration fixture, low outreach value. Mutate the bundled `demo_rag` system
   / adapter retriever+generator config — **NOT** its statistical release gate
   (that's ordinary program logic, not a muteval system mutation).
3. **Your own deterministic fixture** (`examples/rag_context_offline/`) — fully
   in your control; the cleanest possible reproducibility check.

### Phase 1B — replacements needed (do NOT call these vetted)
- **leisurelyleon/ragline**  [unverified] — our earlier "offline, 2★" entry came
  from the rate-limited run; the reviewer couldn't locate it and neither could I.
  PAUSE until cloned.
- **OpenAgentHQ/openagent-eval**  [unverified] — same; REMOVE until confirmed.
- **Raudaschl/rag-fusion**  [API✓] — LIVE: 946★, MIT, active (pushed 2026-04-26),
  ships an eval harness over NFCorpus/BEIR. Solid replacement candidate, BUT its
  keyless path is **retrieval-only**; generator-prompt mutations need its
  API-backed methods. Vet against the gate before investing.

### Phase 2 — find a publishable external gap
1. **vectara/open-rag-eval**  [API✓]  — **best outreach target.** 370★,
   Apache-2.0, active org (pushed 2026-06-02), right in the attention band.
   **CONNECTOR MODE REQUIRED** — its static "score an existing answers file" mode
   does NOT meet muteval's re-run requirement; you must drive a Vectara /
   LangChain / LlamaIndex / small custom connector. Default branch is **`dev`**
   per the live API (confirm the working branch/commit before integrating).
   Mutation surface: connector prompt template · generator model · retriever
   top-k/query · retrieved contexts. Strongest place for a credible
   "severe grounding-prompt degradation survives the metric config" finding.
2. **Marker-Inc-Korea/AutoRAG**  [prior-pass ✓, ~4.8k★]  — strong technical case
   study, heavier setup; run AFTER Vectara. **Design trap:** AutoRAG optimizes
   across module combinations and may *compensate* for injected damage by picking
   another config. For a legit test: freeze one completed pipeline, use a small
   fixed dataset, **disable optimization/search**, mutate only the selected
   prompt / model / top-k / context, then evaluate the frozen pipeline with the
   existing metrics. Prompt-maker nodes are graded via downstream generation
   (good for prompt mutations) — your adapter must preserve that flow.
3. **vibrantlabsai/ragas** example  [search✓]  — ragas MOVED from
   `explodinggradients` to `vibrantlabsai` (old links redirect). Use the
   "Evaluate and Improve a RAG App" example
   (`docs/howtos/applications/evaluate-and-improve-rag.md` /
   `ragas_examples/improve_rag`): it runs a BM25-backed RAG per dataset row then
   evaluates the fresh response — meets the re-run requirement. (The old
   `docs/getstarted/rag_eval.md` path still exists too.) Name-recognition
   benchmark, **not** primary outreach — huge framework, big issue/PR queue, and
   a survivor reflects one tutorial config, not ragas generally. Good blog
   section ("muteval vs the official ragas example"), framed narrowly.

### Removed / demoted
- **NirDiamant/RAG_Techniques**  [search✓ exists] — **remove from outreach.**
  ~27.6k★ (far outside the 300–8k band), a notebook tutorial *collection* with no
  single eval-ownership boundary, and a **non-commercial custom license** (not
  permissive OSS). Our earlier `evaluation/evalute_rag.py` entry point is likely
  outdated. Use only as an optional content experiment: convert one runnable eval
  notebook into a script and demo muteval compatibility.
- Previous-pass disqualifications stand: TonicAI/tonic_validate (metric library,
  static-CSV examples), SciPhi-AI/R2R (pipeline but no eval suite),
  umbertogriffo/rag-chatbot (unit tests, no metrics), firecrawl/rag-arena (human
  Elo voting + TS, stale), AlaGrine & aaronjimv (no eval suite),
  prasadshreyas/rag-evaluation (renamed to `0xshre`, now inaccessible).

### Updated shortlist
| Rank | Target | Purpose | Outreach | Verified |
| ---: | --- | --- | --- | --- |
| 1 | sanmaxdev/ragproof | first keyless validation | Medium | reviewer — clone-verify |
| 2 | vectara/open-rag-eval | first credible external finding | High | API✓ |
| 3 | AutoRAG (frozen pipeline) | technical case study | Med–High | prior pass |
| 4 | vibrantlabsai/ragas example | name-recognition benchmark | Low–Med | search✓ |
| 5 | aswithabukka demo adapter | integration testing | Low | reviewer/search |
| — | Raudaschl/rag-fusion | Phase-1B replacement candidate | Low–Med | API✓ |
| — | NirDiamant/RAG_Techniques | optional content demo only | Low | search✓ |

### What a valid first result looks like (methodology — don't run every operator at once)
Start with 8–20 deliberate mutants on a small dataset:
- **Prompt:** remove grounding instruction · remove abstention instruction ·
  weaken citation requirement · inject a conflicting output-format instruction.
- **Context:** drop highest-ranked passage · truncate retrieved context ·
  reorder passages · inject an irrelevant passage.
- **Retrieval/config:** reduce top-k · raise similarity cutoff · disable reranking.
- **Model:** only when an explicit valid downgrade mapping exists.

For each survivor, save: baseline output · mutated output · mutation description ·
per-eval scores · why the output is materially worse · why the suite still passed ·
three repeated runs · exact commit SHA · exact config + dataset hash. **Only call
it a genuine gap when a human looks at baseline vs mutant and immediately agrees
the mutant is meaningfully degraded.**

## Candidate pool (~15)

Confidence: **[V]** = I verified some facts via fetch; **[L]** = strong lead from
search, eval-presence/size plausible but UNVERIFIED; vet before investing.

### A. Eval-centric example/app repos (best fit — evals are first-class)
| Repo | Eval type | Conf | Notes / what to verify |
| --- | --- | --- | --- |
| relari-ai/examples | continuous-eval | [L] | End-to-end LLM apps + eval pipelines. Strong fit; check activity. |
| TonicAI/tonic_validate | custom + ragas | [L] | RAG eval framework; target its example apps, not the lib core. |
| Marker-Inc-Korea/AutoRAG | custom (AutoML eval) | [L] | RAG eval/optimization; eval at core. Verify size/activity. |
| RulinShao/RAG-evaluation-harnesses | custom | [L] | Research RAG eval suite. May be academic/less responsive. |
| explodinggradients/ragas (examples/) | ragas | [L] | Framework itself — mutate its example pipelines, not the lib. |

### B. Application repos likely shipping evals
| Repo | Eval type | Conf | Notes / what to verify |
| --- | --- | --- | --- |
| NirDiamant/RAG_Techniques | deepeval + ragas | [L] | Popular; notebooks use deepeval for correctness/faithfulness. May be large. |
| umbertogriffo/rag-chatbot | custom/ragas? | [L] | Mid-size RAG chatbot; confirm eval suite exists. |
| firecrawl/rag-arena | feedback-based eval | [L] | RAG eval via user voting (mendable). Different eval style. |
| pmaske-aihub/rag-application | ragas | [L] | Uses ragas (precision/recall/faithfulness). Likely small. |
| aaronjimv/open-source-web-chatbot-using-rag | ragas | [L] | Small tutorial; easy to run, maybe low-value. |
| prasadshreyas/rag-evaluation | ragas + DSPy | [L] | QA RAG + ragas. Small but clean. |
| AlaGrine/RAG_chatabot_with_Langchain | custom | [L] | Tutorial-grade; easy run, responsive likely. |

### C. Larger / stretch (verify they're not too big)
| Repo | Eval type | Conf | Notes |
| --- | --- | --- | --- |
| SciPhi-AI/R2R | custom? | [V] | 7.9k stars, very active (v3.6.5). Big & corporate; eval presence unconfirmed. Stretch. |
| NirDiamant/GenAI_Agents | mixed | [L] | Popular agent cookbook; some eval content. Large. |

### EXCLUDED (verified poor fits — don't waste time)
- **weaviate/Verba** — RAG Evaluation is "planned", not implemented; 7.7k stars (too big).
- **truefoundry/cognita** — ARCHIVED Mar 2026 (dead).

## Find more yourself (I can't run GitHub code search; you can)

- Topic, sorted by recently updated:
  https://github.com/topics/ragas-evaluation?o=desc&s=updated
  https://github.com/topics/rag-evaluation
- GitHub code search (logged in), to find repos USING the frameworks:
  - `from ragas import path:/(test|eval)/`  
  - `import deepeval path:/(test|eval)/`  
  - `assert_test deepeval`  ·  `ragas evaluate( `
  Sort results by "Recently indexed" and filter to repos in the size band.

## Tracking table (fill as you go)

| Repo | Vetted? | Ran? | Top finding (severity) | Outreach (issue/PR) | Response | Blog? |
| --- | --- | --- | --- | --- | --- | --- |
| (e.g. relari-ai/examples) | | | | | | |

## Suggested first batch (3 to start)
Pick a spread: one eval-centric (relari-ai/examples), one popular app
(NirDiamant/RAG_Techniques), one small/easy (prasadshreyas/rag-evaluation).
Run muteval against each (remember the re-run requirement + system-mode for
context/model mutation), collect HIGH-severity survivors, and only contact the
maintainer where the finding is genuinely surprising and reproducible.
