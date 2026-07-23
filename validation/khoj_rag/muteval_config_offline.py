"""muteval vs. khoj-ai/khoj — OFFLINE mechanism demo (no API key).

This reproduces khoj's REAL eval design against khoj's REAL system prompt, then
shows the blind spot muteval predicts. It runs fully offline with a deterministic
mock model so anyone can reproduce the *mechanism* in one command; the genuine,
model-backed finding is in ``muteval_config.py`` (needs GEMINI_API_KEY).

What is REAL here (copied verbatim from the khoj repo):
  * The system prompt = khoj's ``personality`` + ``notes_conversation`` templates
    (src/khoj/processor/conversation/prompts.py). Note it explicitly instructs:
    "Provide inline citations to documents and websites referenced."
  * khoj's ONLY answer-quality eval (tests/evals/eval.py ::
    evaluate_response_with_gemini) is a single binary judge:
    "does the response contain the key information from the ground truth?
     TRUE/FALSE" — factual correctness vs. ground truth, nothing else.

What is MOCKED (so it runs with no key):
  * The model. The mock answers from the retrieved context and — like a real LLM
    — only adds an inline citation WHEN the system prompt still tells it to.
    khoj's correctness judge is reproduced offline as a ground-truth-substring
    check (the real config uses khoj's actual Gemini judge prompt).

The point: khoj's suite scores ONLY factual correctness. So a mutation that
strips khoj's own citation instruction degrades the system (answers stop citing
sources — a behavior khoj's prompt explicitly requires) while the correctness
judge stays green. That mutant SURVIVES = khoj has no eval for citation behavior.

Run:
    muteval run --config validation/khoj_rag/muteval_config_offline.py \
      --operators drop_instruction_lines delete_sentences truncate_prompt
"""

from muteval import EvalOutcome, MutEvalConfig, System

# --- khoj's REAL system prompt (verbatim from prompts.py) --------------------
# personality + notes_conversation, stitched as khoj does for a notes/RAG turn.
KHOJ_SYSTEM_PROMPT = """
You are Khoj, a smart, curious, empathetic and helpful personal assistant.
Use your general knowledge and past conversation with the user as context to inform your responses.

# Style
- Your responses should be helpful, conversational and tuned to the user's communication style.
- Provide inline citations to documents and websites referenced. Add them inline in markdown format to directly support your claim.
  For example: "The weather today is sunny [1](https://weather.com)."
- Do not respond with raw programs or scripts in your final response unless you know the user is a programmer or has explicitly requested code.

Use my personal notes and our past conversations to inform your response.
Ask crisp follow-up questions to get additional context, when a helpful response cannot be provided from the provided notes or past conversations.
""".strip()


SYSTEM = System(
    prompt=KHOJ_SYSTEM_PROMPT,
    model="gpt-4o-mini",
)


# --- FRAMES-style cases: question + ground truth + a Wikipedia-like snippet ---
CASES = [
    {
        "question": "How many FIFA World Cup titles has Brazil won?",
        "ground_truth": "5",
        "reference": "Brazil national football team has won the FIFA World Cup a record five times. source: en.wikipedia.org/wiki/Brazil_national_football_team",
    },
    {
        "question": "In what year did the Chernobyl disaster occur?",
        "ground_truth": "1986",
        "reference": "The Chernobyl disaster occurred on 26 April 1986. source: en.wikipedia.org/wiki/Chernobyl_disaster",
    },
    {
        "question": "What is the boiling point of water at sea level in Celsius?",
        "ground_truth": "100",
        "reference": "At standard atmospheric pressure water boils at 100 degrees Celsius. source: en.wikipedia.org/wiki/Boiling_point",
    },
]


def run(system, case):
    """Deterministic mock of khoj's answer turn.

    Reads the (possibly mutated) system prompt to decide behavior, exactly as a
    real LLM would follow/ignore instructions that are present/absent:
      * always answers with the correct fact from the reference, and
      * adds an inline markdown citation ONLY IF the prompt still asks for
        "inline citations" (khoj's own requirement).
    """
    prompt = system.prompt.lower()
    wants_citation = "inline citation" in prompt
    fact = case["ground_truth"]
    ref = case["reference"]
    url = ref.split("source:", 1)[-1].strip()
    if wants_citation:
        return f"The answer is {fact} [1]({url})."
    # Citation instruction was mutated away -> model stops citing (still correct).
    return f"The answer is {fact}."


# --- khoj's REAL eval, reproduced: binary factual-correctness judge -----------
def khoj_correctness_judge(output, case):
    """Offline stand-in for khoj's evaluate_response_with_gemini: TRUE if the
    response contains the key info from the ground truth (factual correctness
    only — exactly what khoj grades). The real config calls khoj's Gemini judge.
    """
    passed = str(case["ground_truth"]).lower() in output.lower()
    return EvalOutcome(passed=passed, name="khoj_correctness")


config = MutEvalConfig(
    system=SYSTEM,
    cases=CASES,
    run=run,
    # khoj's suite = this ONE check. Nothing verifies citations / grounding.
    evals=[khoj_correctness_judge],
    eval_names=["khoj_correctness"],
    runs_per_mutant=1,
)
