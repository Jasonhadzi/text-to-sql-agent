"""AnalysisAgent — writes a structured analytical report from query results."""

from agents import Agent, ModelSettings

from src.models.schemas import AnalysisReport
from src.prompts import load_prompt

analysis_agent = Agent(
    name="AnalysisAgent",
    instructions=load_prompt("analysis"),
    model="gpt-4o",
    output_type=AnalysisReport,
    model_settings=ModelSettings(temperature=0),
)
