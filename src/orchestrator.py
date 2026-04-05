"""Pipeline orchestrator — drives the multi-agent text-to-SQL workflow."""

from __future__ import annotations

import json
from uuid import uuid4

from agents import Runner

from src.agents.clarification import clarification_agent
from src.agents.nlq_agent import nlq_agent
from src.agents.query_router import query_router_agent
from src.agents.rag_agent import rag_agent
from src.guardrails.input_guardrail import check_input_safety
from src.models.schemas import (
    ClarificationDecision,
    FinalResponse,
    QueryRoute,
    SQLCandidate,
)
from src.tools.redact import redact_preview
from src.tools.run_logger import RunLogger
from src.tools.schema_introspect import get_schema_summary, load_csv_to_duckdb
from src.tools.sql_execute import execute_sql
from src.tools.sql_validate import inject_limit, validate_sql

MAX_SQL_ATTEMPTS = 3


async def run_pipeline(question: str, csv_path: str) -> FinalResponse:
    """Execute the full text-to-SQL agent pipeline and return a ``FinalResponse``."""

    run_id = uuid4().hex[:12]
    logger = RunLogger(run_id)
    logger.log("setup", "run_started", {"question": question, "csv_path": csv_path})
    print(f"\n[run_id={run_id}] Starting pipeline...")

    # ------------------------------------------------------------------
    # Stage 0 — Load data + introspect schema (deterministic)
    # ------------------------------------------------------------------
    print("[Stage 0] Loading CSV into DuckDB and introspecting schema...")
    conn = load_csv_to_duckdb(csv_path)
    schema = get_schema_summary(conn)
    logger.save_json_artifact("schema.json", schema)
    logger.log("schema", "introspected", {"tables": len(schema.tables), "notes": schema.notes})

    schema_text = schema.format_for_prompt()

    # ------------------------------------------------------------------
    # Stage 1 — Input safety check (deterministic guardrail)
    # ------------------------------------------------------------------
    print("[Stage 1] Checking input safety...")
    safety_issues = check_input_safety(question)
    if safety_issues:
        logger.log("safety", "flagged", {"issues": safety_issues})
        print(f"  [WARN] Safety flags: {safety_issues}")
    else:
        logger.log("safety", "passed", {})

    # ------------------------------------------------------------------
    # Stage 1a — Clarification gate (LLM)
    # ------------------------------------------------------------------
    print("[Stage 1a] Checking whether question is SQL-ready...")
    clarification_input = (
        f"## User Question\n{question}\n\n"
        f"## Database Schema\n{schema_text}"
    )
    clarification_result = await Runner.run(clarification_agent, clarification_input)
    clarification: ClarificationDecision = clarification_result.final_output
    logger.log("clarification", "completed", clarification)

    if not clarification.is_clear:
        logger.log("clarification", "needs_user_input", {
            "question": clarification.clarifying_question,
            "reasons": clarification.ambiguity_reasons,
        })
        print(f"  Clarification required: {clarification.clarifying_question}")
        return FinalResponse(
            question=question,
            answer=clarification.clarifying_question,
        )

    logger.log("clarification", "routed", {"route_to": clarification.route_to})

    # ------------------------------------------------------------------
    # Stage 1b — Query routing (LLM)
    # ------------------------------------------------------------------
    print("[Stage 1b] Routing question to relevant tables...")
    route_input = (
        f"## User Question\n{question}\n\n"
        f"## Database Schema\n{schema_text}"
    )
    route_result = await Runner.run(query_router_agent, route_input)
    route: QueryRoute = route_result.final_output
    logger.log("query_router", "completed", route)
    routed_tables = route.relevant_tables
    routed_tables_text = ", ".join(routed_tables) if routed_tables else "(none selected)"
    print(f"  Routed tables: {routed_tables_text}")

    # ------------------------------------------------------------------
    # Stage 2 — NLQ Agent: question → SQL (with retry loop)
    # ------------------------------------------------------------------
    validated_sql: str | None = None
    errors_feedback: str = ""

    for attempt in range(MAX_SQL_ATTEMPTS):
        print(f"[Stage 2] NLQ Agent generating SQL (attempt {attempt + 1}/{MAX_SQL_ATTEMPTS})...")

        nlq_input = (
            f"## User Question\n{question}\n\n"
            f"## Routed Tables\n{routed_tables_text}\n\n"
            f"## Database Schema\n{schema_text}"
        )
        if errors_feedback:
            nlq_input += (
                f"\n\n## Previous Validation Errors — Fix These\n{errors_feedback}"
            )

        nlq_result = await Runner.run(nlq_agent, nlq_input)
        sql_candidate: SQLCandidate = nlq_result.final_output
        logger.log("nlq_agent", f"attempt_{attempt}", sql_candidate)
        print(f"  SQL: {sql_candidate.sql[:120]}...")

        # ---- Deterministic SQL validation (hard gate) ----
        print(f"[Stage 3] Validating SQL (attempt {attempt + 1})...")
        det_val = validate_sql(sql_candidate.sql, schema)
        logger.log("validation_deterministic", f"attempt_{attempt}", det_val)

        if det_val.has_blockers:
            blocker_msgs = [i.message for i in det_val.issues if i.severity == "blocker"]
            errors_feedback = "\n".join(f"- {m}" for m in blocker_msgs)
            print(f"  [BLOCKED] {blocker_msgs}")
            continue

        # Inject LIMIT if needed
        final_sql = sql_candidate.sql
        if det_val.recommended_fix and "LIMIT" in det_val.recommended_fix:
            final_sql = inject_limit(final_sql)

        validated_sql = final_sql
        logger.log("validation", "passed", {"attempt": attempt})
        logger.save_artifact("query.sql", validated_sql)
        break

    if validated_sql is None:
        logger.log("pipeline", "failed", {"reason": "SQL validation failed after max attempts"})
        print("[FAILED] Could not produce a valid SQL query.")
        return FinalResponse(
            question=question,
            answer="I was unable to generate a valid SQL query for your question after multiple attempts. "
                   "Please try rephrasing your question.",
        )

    # ------------------------------------------------------------------
    # Stage 4 — Execute SQL
    # ------------------------------------------------------------------
    print("[Stage 4] Executing SQL...")
    try:
        exec_result = execute_sql(conn, validated_sql, run_id)
    except Exception as exc:
        logger.log("execution", "error", {"error": str(exc)})
        print(f"  [ERROR] Execution failed: {exc}")
        return FinalResponse(
            question=question,
            sql=validated_sql,
            answer=f"The SQL query was valid but execution failed: {exc}",
        )

    logger.log("execution", "completed", {
        "row_count": exec_result.row_count,
        "execution_ms": exec_result.execution_ms,
    })
    print(f"  Returned {exec_result.row_count} rows in {exec_result.execution_ms}ms")

    # Redact PII from preview
    preview_rows = redact_preview(exec_result.preview_rows, schema.pii_columns)

    # ------------------------------------------------------------------
    # Stage 5 — RAG Agent: results → grounded answer
    # ------------------------------------------------------------------
    print("[Stage 5] RAG Agent generating answer...")
    rag_input = (
        f"## User Question\n{question}\n\n"
        f"## Routed Tables\n{routed_tables_text}\n\n"
        f"## SQL Query\n```sql\n{validated_sql}\n```\n\n"
        f"## Execution Summary\n{exec_result.row_count} rows returned in {exec_result.execution_ms}ms\n\n"
        f"## Results Preview (first {len(preview_rows)} rows)\n"
        f"```json\n{json.dumps(preview_rows, indent=2, default=str)}\n```"
    )
    rag_result = await Runner.run(rag_agent, rag_input)
    final: FinalResponse = rag_result.final_output

    # Attach structured data from execution so the API can surface
    # grounded tables/charts to the frontend.
    final.preview_rows = preview_rows
    final.columns = exec_result.columns

    logger.log("rag_agent", "completed", final)
    logger.save_artifact("final.md", _format_final_md(final))
    logger.save_json_artifact("final.json", final)

    print(f"\n[run_id={run_id}] Pipeline completed successfully.")
    print(f"  Artifacts saved to: outputs/runs/{run_id}/")
    return final


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_final_md(resp: FinalResponse) -> str:
    """Format the final response as a readable Markdown document."""
    parts = [
        f"# Query Report\n",
        f"## Question\n{resp.question}\n",
        f"## Business Context\n{resp.business_context_summary}\n",
        f"## SQL\n```sql\n{resp.sql}\n```\n",
        f"## Execution\n{resp.execution_summary}\n",
        f"## Analysis\n{resp.analysis}\n",
        f"## Answer\n{resp.answer}\n",
    ]
    return "\n".join(parts)
