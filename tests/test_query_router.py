"""Tests for query router config loading and prompt wiring."""

from src.agents.query_router_agent import _build_router_instructions, _load_datasource_tables, query_router_agent
from src.models.schemas import QueryRoute


class TestQueryRouterConfig:
    def test_load_datasource_tables_from_config(self) -> None:
        tables = _load_datasource_tables()
        assert isinstance(tables, list)
        assert all(isinstance(t, str) for t in tables)
        assert all(t.strip() for t in tables)

    def test_built_instructions_include_available_tables_section(self) -> None:
        instructions = _build_router_instructions()
        assert "Available Tables (from datasource_config.json)" in instructions


class TestQueryRouterAgentWiring:
    def test_agent_name_and_prompt_wiring(self) -> None:
        assert query_router_agent.name == "QueryRouterAgent"
        assert "Query Router Agent" in query_router_agent.instructions

    def test_query_route_schema_defaults(self) -> None:
        route = QueryRoute()
        assert route.relevant_tables == []
        assert route.confidence == 0.0
