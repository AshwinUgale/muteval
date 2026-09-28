"""One redaction point for everything muteval emits.

Error strings are the risky path: a provider exception can echo the request
URL (``...generateContent?key=AIza...``), an ``Authorization: Bearer ...``
header, or a config that embedded a key. Every output — terminal report,
JSON, JUnit, HTML, manifest, ``muteval check``, the probe card, CLI error lines
— goes through ``redact`` so what gets printed never depends on which format
you picked.

Two layers:

* **Known shapes** (``_SECRET_RE``): provider key prefixes, ``Bearer`` tokens,
  ``key=`` / ``"api_key": "..."`` assignments, URL query credentials.
* **Your actual secrets**: the exact value of any environment variable whose
  name looks secret (``*KEY*``, ``*TOKEN*``, ``*SECRET*``, ``*PASSWORD*``, ...),
  so a key in a format no pattern knows (Azure, a proxy token) is still caught
  when it came from your environment.

This is defense in depth, not a guarantee: a secret that matches no pattern and
doesn't come from the environment can still be printed.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Tuple

REDACTED = "[REDACTED]"

_SECRET_RE = re.compile(
    "|".join(
        [
            # Provider key prefixes (OpenAI sk-/sk-proj-, Anthropic sk-ant-, Groq,
            # Google, GitHub classic + fine-grained, Hugging Face, xAI, Slack, AWS).
            r"\bsk-[A-Za-z0-9_\-]{8,}",
            r"\bgsk_[A-Za-z0-9_\-]{8,}",
            r"\bAIza[A-Za-z0-9_\-]{20,}",
            r"\bgh[pousr]_[A-Za-z0-9]{16,}",
            r"\bgithub_pat_[A-Za-z0-9_]{20,}",
            r"\bhf_[A-Za-z0-9]{16,}",
            r"\bxai-[A-Za-z0-9_\-]{16,}",
            r"\bxox[abprs]-[A-Za-z0-9\-]{10,}",
            r"\bAKIA[0-9A-Z]{16}\b",
            # "Bearer <token>" anywhere (headers, error echoes).
            r"(?i:\bbearer\s+)[A-Za-z0-9._~+/=\-]{8,}",
            # key = value / "key": "value" / key: value, for credential-ish key
            # NAMES — including prefixed ones (OPENAI_API_KEY=, GITHUB_TOKEN:,
            # client_secret:). The name must end right at the separator, so
            # "max_tokens: 512" or "secretary: Ann" don't match.
            r"(?i:\b[\w\-]*?(?:api[_-]?key|access[_-]?token|auth[_-]?token|token"
            r"|secret|password|passwd|authorization)[\"']?\s*[:=]\s*[\"']?)"
            r"(?i:bearer\s+)?[^\s\"'&,;}]+",
            # Credentials in a URL query string (?key=..., &access_token=...).
            r"(?i:[?&](?:key|api[_-]?key|access[_-]?token|token|sig)=)[^&\s\"'#]+",
        ]
    )
)

# Environment variable NAMES whose values are treated as secrets.
_SECRET_ENV_NAME = re.compile(
    r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH", re.IGNORECASE
)
_MIN_ENV_SECRET_LEN = 8


@lru_cache(maxsize=8)
def _env_pattern(values: Tuple[str, ...]):
    if not values:
        return None
    # Longest first, so a secret that contains another is removed whole.
    alts = sorted(values, key=len, reverse=True)
    return re.compile("|".join(re.escape(v) for v in alts))


def _env_secret_values() -> Tuple[str, ...]:
    return tuple(
        sorted(
            {
                v
                for k, v in os.environ.items()
                if _SECRET_ENV_NAME.search(k) and len(v) >= _MIN_ENV_SECRET_LEN
            }
        )
    )


def redact(text: str) -> str:
    """Remove secret-looking substrings and your secret env-var values."""
    if not text:
        return text
    env = _env_pattern(_env_secret_values())
    if env is not None:
        text = env.sub(REDACTED, text)
    return _SECRET_RE.sub(_keep_prefix, text)


def _keep_prefix(m: "re.Match[str]") -> str:
    """Keep a readable label ('Bearer ', 'api_key=', '?key=') and redact the
    value, so the output still says WHAT was hidden."""
    s = m.group(0)
    for label in (
        re.match(r"(?i)bearer\s+", s),
        re.match(r"(?i)[?&][A-Za-z_\-]+=", s),
        re.match(r"(?i)[A-Za-z_\-]+[\"']?\s*[:=]\s*[\"']?(?:bearer\s+)?", s),
    ):
        if label and label.end() < len(s):
            return s[: label.end()] + REDACTED
    return REDACTED


def redact_obj(obj: Any) -> Any:
    """Recursively redact every string in a JSON-like structure."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj
