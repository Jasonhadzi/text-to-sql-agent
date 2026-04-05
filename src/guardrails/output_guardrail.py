"""Output guardrail — validates response quality before returning to user.

Checks that every response has a citation and that the answer is grounded
in actual query results. Blocks responses that fail validation.

# TODO: Abhirami/Amna — build from scratch
"""

from __future__ import annotations


def check_output_safety(response: str) -> list[str]:
    """Validate response has citation and is grounded in data.

    Returns a list of issues found. Empty list means the response is safe.
    """
    return []  # TODO: Abhirami/Amna — implement citation and grounding checks
