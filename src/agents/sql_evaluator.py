"""SQLGuardrailEvaluatorAgent — advisory safety/correctness review of SQL."""

from agents import Agent, ModelSettings

from src.models.schemas import ValidationResult
from src.prompts import load_prompt

sql_evaluator_agent = Agent(
    name="SQLGuardrailEvaluatorAgent",
    instructions=load_prompt("sql_evaluator"),
    model="gpt-4o-mini",
    output_type=ValidationResult,
    model_settings=ModelSettings(temperature=0),
)
