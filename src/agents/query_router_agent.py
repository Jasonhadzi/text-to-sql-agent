"""QueryRouterAgent — selects relevant tables from datasource config."""

from __future__ import annotations

import json
from pathlib import Path

from agents import Agent, ModelSettings
from agents.agent_output import AgentOutputSchema

from src.models.schemas import QueryRoute
from src.prompts import load_prompt


def _load_datasource_tables() -> list[str]:
    """Read available table names from config/datasource_config.json."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "datasource_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    tables = config.get("tables", [])
    names: list[str] = []
    for table in tables:
        if isinstance(table, dict):
            name = table.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return names


def _build_router_instructions() -> str:
    base_prompt = load_prompt("query_router_prompt")
    table_names = _load_datasource_tables()
    table_block = "\n".join(f"- {name}" for name in table_names) or "- (no tables found)"
    return f"{base_prompt}\n\n## Available Tables (from datasource_config.json)\n{table_block}\n"


query_router_agent = Agent(
    name="QueryRouterAgent",
    instructions=_build_router_instructions(),
    model="gpt-4o-mini",
    output_type=AgentOutputSchema(QueryRoute, strict_json_schema=False),
    model_settings=ModelSettings(temperature=0),
)
