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
