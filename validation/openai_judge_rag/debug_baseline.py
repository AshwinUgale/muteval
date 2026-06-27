"""Show baseline answers AND the judges' raw replies, so we can see exactly why
the baseline fails (parse artifact vs genuinely low score).

Run (with OPENAI_API_KEY set):
    python validation/openai_judge_rag/debug_baseline.py
"""

import importlib.util
from pathlib import Path

cfg_path = Path(__file__).with_name("muteval_config.py")
spec = importlib.util.spec_from_file_location("dbg_cfg", cfg_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Wrap the module's _chat so we can see every judge reply verbatim.
_orig_chat = mod._chat


def _logged_chat(messages, model=mod.MODEL, temperature=0.0):
    reply = _orig_chat(messages, model=model, temperature=temperature)
    last = messages[-1]["content"]
    if "Rubric:" in last or "rubric" in last.lower():
        print(f"   [judge raw reply] {reply!r}")
    return reply


mod._chat = _logged_chat
cfg = mod.config

print("=" * 70)
for i, case in enumerate(cfg.cases, 1):
    answer = cfg.invoke(cfg.system, case)
    print(f"CASE {i}: {case['question']}")
    print(f"ANSWER: {answer}")
    for ev, name in zip(cfg.evals, cfg.eval_names):
        oc = ev(answer, case)
        verdict = "PASS" if oc.passed else "FAIL"
        print(f"   {name:<20} parsed={oc.score}  thr={oc.threshold} -> {verdict}")
    print("-" * 70)
