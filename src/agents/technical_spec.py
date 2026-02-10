"""TechnicalSpecAgent — rewrites the business context into a precise SQL task spec."""

from agents import Agent, ModelSettings

from src.models.schemas import TechnicalSpec
from src.prompts import load_prompt

technical_spec_agent = Agent(
    name="TechnicalSpecAgent",
    instructions=load_prompt("technical_spec"),
    model="gpt-4o-mini",
    output_type=TechnicalSpec,
    model_settings=ModelSettings(temperature=0),
)
