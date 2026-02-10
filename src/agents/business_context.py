"""BusinessContextAgent — interprets the business intent behind a user question."""

from agents import Agent, ModelSettings

from src.models.schemas import BusinessContext
from src.prompts import load_prompt

business_context_agent = Agent(
    name="BusinessContextAgent",
    instructions=load_prompt("business_context"),
    model="gpt-4o-mini",
    output_type=BusinessContext,
    model_settings=ModelSettings(temperature=0),
)
