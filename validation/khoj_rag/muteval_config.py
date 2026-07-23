"""muteval vs. khoj-ai/khoj — REAL model + khoj's REAL Gemini judge.

Reproduces khoj's own evaluation methodology and points muteval at khoj's own
system prompt, to answer: *would khoj's answer-quality eval catch a regression
in the behaviors khoj's prompt explicitly requires (inline citations, grounding)?*

Fidelity — what is copied verbatim from the khoj repo:
  * SYSTEM PROMPT: khoj's ``personality`` + ``notes_conversation`` templates
    (src/khoj/processor/conversation/prompts.py). It explicitly instructs the
    model to "Provide inline citations to documents and websites referenced."
  * THE JUDGE: khoj's ``evaluate_response_with_gemini`` (tests/evals/eval.py) —
    same Gemini model, same prompt, same TRUE/FALSE factual-correctness rubric.
    This is khoj's ONLY answer-quality signal.

Fidelity — what is approximated (documented in NOTES.md):
  * We call the model directly with khoj's system prompt + the reference article,
    instead of booting the full khoj server + live retriever. The prompt and the
    grader are khoj's; the plumbing around them is a faithful minimal harness.

Setup (one key):
    pip install certifi
    export GEMINI_API_KEY=...        # used for BOTH answer-gen and khoj's judge

Run:
    muteval run --config validation/khoj_rag/muteval_config.py
    # See only the real coverage gaps (citation/grounding survivors):
    muteval run --config validation/khoj_rag/muteval_config.py \
      --operators drop_instruction_lines delete_sentences truncate_prompt
"""

import json
import os
import re
import ssl
import urllib.request

from muteval import EvalOutcome, MutEvalConfig, System

GEN_MODEL = os.environ.get("KHOJ_GEN_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = os.environ.get("GEMINI_EVAL_MODEL", "gemini-2.5-flash")

try:
    import certifi

    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:  # noqa: BLE001
    _SSL = ssl.create_default_context()


def _gemini(prompt: str, model: str, json_mode: bool = False) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    gen_cfg = {"response_mime_type": "application/json"} if json_mode else {}
    body = json.dumps(
        {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen_cfg}
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120, context=_SSL) as resp:
        data = json.load(resp)
    return data["candidates"][0]["content"]["parts"][0]["text"] or ""


# --- khoj's REAL system prompt (verbatim: personality + notes_conversation) ---
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

SYSTEM = System(prompt=KHOJ_SYSTEM_PROMPT, model=GEN_MODEL)

# FRAMES-style cases: question + ground truth + a Wikipedia reference article.
CASES = [
    {
        "question": "How many FIFA World Cup titles has Brazil won?",
        "ground_truth": "five (5)",
        "reference": "Brazil national football team has won the FIFA World Cup a record five times (1958, 1962, 1970, 1994, 2002). source: https://en.wikipedia.org/wiki/Brazil_national_football_team",
    },
    {
        "question": "On what date did the Chernobyl disaster occur?",
        "ground_truth": "26 April 1986",
        "reference": "The Chernobyl disaster began on 26 April 1986 at the No. 4 reactor. source: https://en.wikipedia.org/wiki/Chernobyl_disaster",
    },
    {
        "question": "Who wrote the novel 'One Hundred Years of Solitude'?",
        "ground_truth": "Gabriel Garcia Marquez",
        "reference": "One Hundred Years of Solitude is a 1967 novel by Colombian author Gabriel Garcia Marquez. source: https://en.wikipedia.org/wiki/One_Hundred_Years_of_Solitude",
    },
]


def run(system, case):
    """Generate khoj's answer: khoj's (possibly mutated) system prompt + the
    reference article + the question."""
    prompt = (
        system.prompt
        + "\n\nUser's Notes:\n-----\n"
        + case["reference"]
        + "\n\nQuestion: "
        + case["question"]
    )
    return _gemini(prompt, GEN_MODEL)


# --- khoj's REAL judge, verbatim prompt from tests/evals/eval.py -------------
def khoj_correctness_judge(output, case):
    evaluation_prompt = f"""
    Compare the following agent response with the ground truth answer.
    Determine if the agent response contains the key information from the ground truth.
    Focus on factual correctness rather than exact wording.

    Query: {case["question"]}
    Agent Response: {output}
    Ground Truth: {case["ground_truth"]}

    Provide your evaluation in the following json format:
    {{"explanation":"<1 short sentence>", "decision":"<TRUE if response contains key information, FALSE otherwise>"}}
    """
    raw = _gemini(evaluation_prompt, JUDGE_MODEL, json_mode=True)
    try:
        parsed = json.loads(re.sub(r"^```json\s*|\s*```$", "", raw.strip()))
        decision = str(parsed.get("decision", "")).upper() == "TRUE"
    except Exception:  # noqa: BLE001
        decision = "TRUE" in raw.upper()
    return EvalOutcome(passed=decision, name="khoj_correctness")


config = MutEvalConfig(
    system=SYSTEM,
    cases=CASES,
    run=run,
    evals=[khoj_correctness_judge],  # khoj's entire answer-quality suite
    eval_names=["khoj_correctness"],
    runs_per_mutant=1,
)
