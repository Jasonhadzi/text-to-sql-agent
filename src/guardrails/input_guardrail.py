"""Input guardrail — deterministic prompt-injection screening."""

from __future__ import annotations

import re

SUSPICIOUS_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+instructions",
    r"drop\s+table",
    r"delete\s+from",
    r"insert\s+into",
    r"update\s+\w+\s+set",
    r";\s*(drop|delete|insert|update|alter|create)",
    r"exfiltrate",
    r"run\s+tool",
]


def check_input_safety(question: str) -> list[str]:
    """Basic deterministic prompt-injection screening."""
    flags: list[str] = []
    lower = question.lower()
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, lower):
            flags.append(f"Suspicious pattern: {pattern}")
    return flags
