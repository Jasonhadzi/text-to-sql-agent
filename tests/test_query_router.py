"""Tests for the Query Router Agent — config loading, wiring, and routing outputs."""

import pytest

from src.agents.query_router import _build_router_instructions, _load_datasource_tables, query_router_agent
from src.models.schemas import RoutingResult


class TestQueryRouterConfig:
    def test_load_datasource_tables_from_config(self) -> None:
        tables = _load_datasource_tables()
        assert isinstance(tables, list)
        assert all(isinstance(t, str) for t in tables)
        assert all(t.strip() for t in tables)
        assert "retail_transactions_typed" in tables

    def test_built_instructions_include_dynamic_section(self) -> None:
        instructions = _build_router_instructions()
        assert "Dynamically loaded table names" in instructions


class TestQueryRouterAgentWiring:
    def test_agent_name_and_prompt_wiring(self) -> None:
        assert query_router_agent.name == "QueryRouterAgent"
        assert "query routing" in query_router_agent.instructions.lower()

    def test_routing_result_defaults(self) -> None:
        result = RoutingResult(relevant_tables=["test_table"])
        assert result.datasource == "default"
        assert result.reasoning == ""
        assert result.schema_subset == ""


@pytest.fixture
def retail_routing() -> RoutingResult:
    return RoutingResult(
        relevant_tables=["retail_transactions_typed"],
        datasource="retail",
        reasoning="The question asks about sales revenue which maps to the retail transactions table.",
        schema_subset="",
    )


class TestRoutingResultSamples:
    def test_routes_to_retail_table(self, retail_routing: RoutingResult) -> None:
        assert "retail_transactions_typed" in retail_routing.relevant_tables

    def test_returns_at_least_one_table(self, retail_routing: RoutingResult) -> None:
        assert len(retail_routing.relevant_tables) >= 1

    def test_datasource_populated(self, retail_routing: RoutingResult) -> None:
        assert retail_routing.datasource != ""

    def test_reasoning_not_empty(self, retail_routing: RoutingResult) -> None:
        assert len(retail_routing.reasoning) > 0
