"""Tests for clarification decision contracts and agent wiring."""

import pytest
from pydantic import ValidationError

from src.agents.clarification import clarification_agent
from src.models.schemas import ClarificationDecision
from src.prompts import load_prompt


class TestClarificationDecision:
    def test_clear_question_routes_to_query_router(self) -> None:
        decision = ClarificationDecision(
            is_clear=True,
            route_to="anything_else",
            clarifying_question="this should be ignored",
            ambiguity_reasons=[],
        )
        assert decision.is_clear is True
        assert decision.route_to == "query_router"
        assert decision.clarifying_question == ""

    def test_unclear_question_requires_clarifying_question(self) -> None:
        with pytest.raises(ValidationError):
            ClarificationDecision(
                is_clear=False,
                route_to="clarification_agent",
                clarifying_question="   ",
                ambiguity_reasons=["Missing metric definition"],
            )

    def test_unclear_question_accepts_single_question(self) -> None:
        decision = ClarificationDecision(
            is_clear=False,
            route_to="clarification_agent",
            clarifying_question="Which metric should define performance: revenue, order count, or margin?",
            ambiguity_reasons=["'Performance' is ambiguous without a metric"],
        )
        assert decision.is_clear is False
        assert decision.clarifying_question != ""


class TestClarificationAgentWiring:
    def test_prompt_file_loads(self) -> None:
        prompt = load_prompt("clarification_prompt")
        assert "Clarification Agent" in prompt
        assert "query_router" in prompt

    def test_agent_name_and_prompt_wiring(self) -> None:
        assert clarification_agent.name == "ClarificationAgent"
        assert "Clarification Agent" in clarification_agent.instructions
