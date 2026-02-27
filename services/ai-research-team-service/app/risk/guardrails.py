from __future__ import annotations

import re
from dataclasses import dataclass


INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"reveal\s+system\s+prompt",
    r"you\s+are\s+now\s+developer",
    r"override\s+safety",
    r"bypass\s+guardrails",
]

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


@dataclass
class GuardrailResult:
    allowed: bool
    sanitized_text: str
    blocked_reason: str | None
    pii_types: list[str]


def sanitize_prompt(text: str) -> GuardrailResult:
    if not text or not text.strip():
        return GuardrailResult(allowed=False, sanitized_text="", blocked_reason="empty_query", pii_types=[])

    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailResult(
                allowed=False,
                sanitized_text="",
                blocked_reason="prompt_injection_detected",
                pii_types=[],
            )

    sanitized = text
    pii_types: list[str] = []
    for pii_type, pii_pattern in PII_PATTERNS.items():
        if pii_pattern.search(sanitized):
            pii_types.append(pii_type)
            sanitized = pii_pattern.sub(f"[REDACTED_{pii_type.upper()}]", sanitized)

    return GuardrailResult(
        allowed=True,
        sanitized_text=sanitized,
        blocked_reason=None,
        pii_types=sorted(set(pii_types)),
    )
