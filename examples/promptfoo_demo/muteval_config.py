"""muteval target = the promptfoo suite in this folder.

The suite is a real promptfoo project (the language-tutor feedback prompt from
adelmuursepp/promptfoo-demo-evals). This 2-line wrapper lets BOTH
`muteval run --config ...` and `muteval probe --config ...` grade the same
promptfoo config, so you get the coverage report and the probe card from one
target. (`muteval run --promptfoo promptfooconfig.yaml` works too, but `probe`
takes `--config`.)

Runs on OPENAI_API_KEY (gpt-4o-mini); no promptfoo install needed beyond PyYAML.
"""

import os

from muteval.adapters.promptfoo import from_promptfoo

config = from_promptfoo(
    os.path.join(os.path.dirname(__file__), "promptfooconfig.yaml")
)
