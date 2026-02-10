"""PII redaction for result previews."""

from __future__ import annotations

from typing import Any

# Default PII column names (case-insensitive matching)
DEFAULT_PII_COLUMNS = {"name", "email", "phone", "address"}

REDACTED = "[REDACTED]"


def redact_preview(
    rows: list[dict[str, Any]],
    pii_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return a copy of *rows* with PII columns replaced by ``[REDACTED]``.

    Parameters
    ----------
    rows:
        List of row dicts as returned by the executor.
    pii_columns:
        Column names to redact.  Falls back to ``DEFAULT_PII_COLUMNS``.
    """
    if not rows:
        return rows

    pii_set = {c.lower() for c in (pii_columns or DEFAULT_PII_COLUMNS)}

    redacted_rows: list[dict[str, Any]] = []
    for row in rows:
        new_row = {}
        for key, value in row.items():
            if key.lower() in pii_set:
                new_row[key] = REDACTED
            else:
                new_row[key] = value
        redacted_rows.append(new_row)
    return redacted_rows
