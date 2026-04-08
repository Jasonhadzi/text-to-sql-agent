"""ClarificationAgent — determines if a user question is SQL-ready."""

from agents import Agent, ModelSettings
from agents.agent_output import AgentOutputSchema

from src.models.schemas import ClarificationDecision
from src.prompts import load_prompt

clarification_agent = Agent(
    name="ClarificationAgent",
    instructions=load_prompt("clarification_prompt"),
    model="gpt-4o-mini",
    output_type=AgentOutputSchema(ClarificationDecision, strict_json_schema=False),
    model_settings=ModelSettings(temperature=0),
)
