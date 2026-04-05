"""Tests for orchestrator stage ordering around clarification and routing."""

import asyncio

import pytest

from src import orchestrator
from src.models.schemas import ClarificationResult, RoutingResult, SchemaSummary, SQLCandidate


def test_clear_question_runs_clarification_then_router_then_nlq(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure clear questions follow: clarification -> query router -> NLQ agent."""

    call_order: list[str] = []

    class _DummyResult:
        def __init__(self, final_output):
            self.final_output = final_output

    class _FakeRunner:
        @staticmethod
        async def run(agent, _input, run_config=None):
            call_order.append(agent.name)
            if agent.name == "ClarificationAgent":
                return _DummyResult(
                    ClarificationResult(
                        is_clear=True,
                        clarifying_question="",
                        original_question="Show monthly revenue trend",
                        interpreted_intent="Trend of revenue by month",
                        confidence=0.9,
                    )
                )
            if agent.name == "QueryRouterAgent":
                return _DummyResult(
                    RoutingResult(
                        relevant_tables=["retail_transactions_typed"],
                        datasource="retail",
                        reasoning="Revenue maps to retail transactions.",
                        schema_subset="",
                    )
                )
            if agent.name == "NLQAgent":
                raise RuntimeError("stop-after-router")
            return _DummyResult(SQLCandidate(sql="SELECT 1"))

    monkeypatch.setattr(orchestrator, "Runner", _FakeRunner)
    monkeypatch.setattr(orchestrator, "load_csv_to_duckdb", lambda _csv_path: object())
    monkeypatch.setattr(orchestrator, "get_schema_summary", lambda _conn: SchemaSummary())
    monkeypatch.setattr(
        orchestrator,
        "load_datasource_config",
        lambda: {"datasources": []},
    )

    with pytest.raises(RuntimeError, match="stop-after-router"):
        asyncio.run(orchestrator.run_pipeline("Show monthly revenue trend", "dummy.csv"))

    assert call_order[:3] == [
        "ClarificationAgent",
        "QueryRouterAgent",
        "NLQAgent",
    ]
