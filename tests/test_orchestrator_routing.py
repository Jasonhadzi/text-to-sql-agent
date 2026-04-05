"""Tests for orchestrator stage ordering around clarification and routing."""

import asyncio

import pytest

from src import orchestrator
from src.models.schemas import ClarificationDecision, QueryRoute, SchemaSummary, SQLCandidate


def test_clear_question_runs_clarification_then_router_then_nlq(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure clear questions follow: clarification -> query router -> NLQ agent."""

    call_order: list[str] = []

    class _DummyResult:
        def __init__(self, final_output):
            self.final_output = final_output

    class _FakeRunner:
        @staticmethod
        async def run(agent, _input):
            call_order.append(agent.name)
            if agent.name == "ClarificationAgent":
                return _DummyResult(
                    ClarificationDecision(
                        is_clear=True,
                        route_to="query_router",
                        clarifying_question="",
                        ambiguity_reasons=[],
                    )
                )
            if agent.name == "QueryRouterAgent":
                return _DummyResult(
                    QueryRoute(
                        relevant_tables=["retail_transactions_typed"],
                        reasoning="Revenue question maps to transactions table.",
                        confidence=0.9,
                    )
                )
            if agent.name == "NLQAgent":
                raise RuntimeError("stop-after-router")
            return _DummyResult(SQLCandidate(sql="SELECT 1"))

    monkeypatch.setattr(orchestrator, "Runner", _FakeRunner)
    monkeypatch.setattr(orchestrator, "load_csv_to_duckdb", lambda _csv_path: object())
    monkeypatch.setattr(orchestrator, "get_schema_summary", lambda _conn: SchemaSummary())

    with pytest.raises(RuntimeError, match="stop-after-router"):
        asyncio.run(orchestrator.run_pipeline("Show monthly revenue trend", "dummy.csv"))

    assert call_order[:3] == [
        "ClarificationAgent",
        "QueryRouterAgent",
        "NLQAgent",
    ]
