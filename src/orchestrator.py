"""Pipeline orchestrator — drives the multi-agent text-to-SQL workflow."""

from __future__ import annotations

import json
import re
from uuid import uuid4

from agents import Runner

from src.agents.business_context import business_context_agent
from src.agents.technical_spec import technical_spec_agent
from src.agents.sql_writer import sql_writer_agent
from src.agents.sql_evaluator import sql_evaluator_agent
from src.agents.analysis import analysis_agent
from src.agents.synthesis import synthesis_agent
from src.models.schemas import (
    AnalysisReport,
    BusinessContext,
    FinalResponse,
    SQLCandidate,
    SchemaSummary,
    TechnicalSpec,
    ValidationResult,
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
    # Stage 1 — Input safety check (basic deterministic)
    # ------------------------------------------------------------------
    print("[Stage 1] Checking input safety...")
    safety_issues = _check_input_safety(question)
    if safety_issues:
        logger.log("safety", "flagged", {"issues": safety_issues})
        print(f"  [WARN] Safety flags: {safety_issues}")
    else:
        logger.log("safety", "passed", {})

    # ------------------------------------------------------------------
    # Stage 1b — Business context (LLM)
    # ------------------------------------------------------------------
    print("[Stage 1b] Extracting business context...")
    biz_input = (
        f"## User Question\n{question}\n\n"
        f"## Database Schema\n{schema_text}"
    )
    biz_result = await Runner.run(business_context_agent, biz_input)
    biz_ctx: BusinessContext = biz_result.final_output
    logger.log("business_context", "completed", biz_ctx)
    logger.save_json_artifact("business_context.json", biz_ctx)
    print(f"  Business goal: {biz_ctx.business_goal}")

    # ------------------------------------------------------------------
    # Stage 2 — Technical spec (LLM, sequential)
    # ------------------------------------------------------------------
    print("[Stage 2] Generating technical specification...")
    tech_input = (
        f"## User Question\n{question}\n\n"
        f"## Business Context\n{json.dumps(biz_ctx.model_dump(), indent=2)}\n\n"
        f"## Database Schema\n{schema_text}"
    )
    tech_result = await Runner.run(technical_spec_agent, tech_input)
    tech_spec: TechnicalSpec = tech_result.final_output
    logger.log("technical_spec", "completed", tech_spec)
    logger.save_json_artifact("technical_spec.json", tech_spec)
    print(f"  Task: {tech_spec.task}")

    # ------------------------------------------------------------------
    # Stage 3 + 4 — SQL generation + validation loop
    # ------------------------------------------------------------------
    validated_sql: str | None = None
    errors_feedback: str = ""

    for attempt in range(MAX_SQL_ATTEMPTS):
        print(f"[Stage 3] Writing SQL (attempt {attempt + 1}/{MAX_SQL_ATTEMPTS})...")

        sql_input = (
            f"## Technical Specification\n{json.dumps(tech_spec.model_dump(), indent=2)}\n\n"
            f"## Database Schema\n{schema_text}"
        )
        if errors_feedback:
            sql_input += (
                f"\n\n## Previous Validation Errors — Fix These\n{errors_feedback}"
            )

        sql_result = await Runner.run(sql_writer_agent, sql_input)
        sql_candidate: SQLCandidate = sql_result.final_output
        logger.log("sql_writer", f"attempt_{attempt}", sql_candidate)
        print(f"  SQL: {sql_candidate.sql[:120]}...")

        # ---- Deterministic validation ----
        print(f"[Stage 4] Validating SQL (attempt {attempt + 1})...")
        det_val = validate_sql(sql_candidate.sql, schema)
        logger.log("validation_deterministic", f"attempt_{attempt}", det_val)

        # ---- LLM advisory validation (run even if deterministic passes) ----
        eval_input = (
            f"## User Question\n{question}\n\n"
            f"## Technical Specification\n{json.dumps(tech_spec.model_dump(), indent=2)}\n\n"
            f"## Proposed SQL\n```sql\n{sql_candidate.sql}\n```\n\n"
            f"## Database Schema\n{schema_text}\n\n"
            f"## Engine Constraints\nDuckDB dialect. Only table retail_transactions_typed allowed. SELECT only."
        )
        eval_result = await Runner.run(sql_evaluator_agent, eval_input)
        llm_eval: ValidationResult = eval_result.final_output
        logger.log("validation_llm", f"attempt_{attempt}", llm_eval)

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
    # Stage 5 — Execute
    # ------------------------------------------------------------------
    print("[Stage 5] Executing SQL...")
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
    # Stage 6a — Analysis (LLM)
    # ------------------------------------------------------------------
    print("[Stage 6a] Generating analysis report...")
    analysis_input = (
        f"## User Question\n{question}\n\n"
        f"## Business Context\n{json.dumps(biz_ctx.model_dump(), indent=2)}\n\n"
        f"## SQL Query\n```sql\n{validated_sql}\n```\n\n"
        f"## Query Results ({exec_result.row_count} rows, {exec_result.execution_ms}ms)\n"
        f"### Preview (first {len(preview_rows)} rows)\n"
        f"```json\n{json.dumps(preview_rows, indent=2, default=str)}\n```"
    )
    analysis_result = await Runner.run(analysis_agent, analysis_input)
    report: AnalysisReport = analysis_result.final_output
    logger.log("analysis", "completed", report)
    logger.save_json_artifact("analysis.json", report)
    print(f"  Executive summary: {report.executive_summary[:1] if report.executive_summary else '(empty)'}...")

    # ------------------------------------------------------------------
    # Stage 6b — Synthesis (LLM)
    # ------------------------------------------------------------------
    print("[Stage 6b] Synthesizing final response...")
    synthesis_input = (
        f"## Original Question\n{question}\n\n"
        f"## Business Context Summary\n{biz_ctx.business_goal}\n\n"
        f"## SQL Query\n```sql\n{validated_sql}\n```\n\n"
        f"## Execution Summary\n{exec_result.row_count} rows returned in {exec_result.execution_ms}ms\n\n"
        f"## Results Preview\n```json\n{json.dumps(preview_rows, indent=2, default=str)}\n```\n\n"
        f"## Analysis Report\n"
        f"### Executive Summary\n" + "\n".join(f"- {s}" for s in report.executive_summary) + "\n\n"
        f"### Key Findings\n" + "\n".join(f"- {f}" for f in report.key_findings) + "\n\n"
        f"### Caveats\n" + "\n".join(f"- {c}" for c in report.caveats) + "\n\n"
        f"### Suggested Next Questions\n" + "\n".join(f"- {q}" for q in report.suggested_next_questions)
    )
    synth_result = await Runner.run(synthesis_agent, synthesis_input)
    final: FinalResponse = synth_result.final_output
    logger.log("synthesis", "completed", final)
    logger.save_artifact("final.md", _format_final_md(final))
    logger.save_json_artifact("final.json", final)

    print(f"\n[run_id={run_id}] Pipeline completed successfully.")
    print(f"  Artifacts saved to: outputs/runs/{run_id}/")
    return final


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUSPICIOUS_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+instructions",
    r"drop\s+table",
    r"delete\s+from",
    r"insert\s+into",
    r"update\s+\w+\s+set",
    r";\s*(drop|delete|insert|update|alter|create)",
    r"exfiltrate",
    r"run\s+tool",
]


def _check_input_safety(question: str) -> list[str]:
    """Basic deterministic prompt-injection screening."""
    flags: list[str] = []
    lower = question.lower()
    for pattern in _SUSPICIOUS_PATTERNS:
        if re.search(pattern, lower):
            flags.append(f"Suspicious pattern: {pattern}")
    return flags


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
