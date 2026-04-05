"""Tests for the Clarification Agent — schemas, wiring, and sample outputs."""

import pytest

from src.agents.clarification import clarification_agent
from src.models.schemas import ClarificationResult
from src.prompts import load_prompt


@pytest.fixture
def clear_result() -> ClarificationResult:
    return ClarificationResult(
        is_clear=True,
        clarifying_question="",
        original_question="What were total sales last month?",
        interpreted_intent="Sum of Total_Amount for the previous calendar month",
        confidence=0.95,
    )


@pytest.fixture
def unclear_result() -> ClarificationResult:
    return ClarificationResult(
        is_clear=False,
        clarifying_question="Could you specify which metric you're interested in — revenue, number of orders, or something else?",
        original_question="Tell me about the data",
        interpreted_intent="",
        confidence=0.3,
    )


class TestClarificationResult:
    def test_clear_result_fields(self, clear_result: ClarificationResult) -> None:
        assert clear_result.is_clear is True
        assert clear_result.clarifying_question == ""
        assert clear_result.confidence > 0.5

    def test_unclear_result_has_question(self, unclear_result: ClarificationResult) -> None:
        assert unclear_result.is_clear is False
        assert len(unclear_result.clarifying_question) > 0

    def test_original_question_echoed(self, clear_result: ClarificationResult) -> None:
        assert clear_result.original_question == "What were total sales last month?"

    def test_confidence_range(self) -> None:
        result = ClarificationResult(
            is_clear=True,
            original_question="test",
            confidence=0.85,
        )
        assert 0.0 <= result.confidence <= 1.0


class TestClarificationAgentWiring:
    def test_prompt_file_loads(self) -> None:
        prompt = load_prompt("clarification")
        assert "clarification" in prompt.lower()

    def test_agent_name_and_instructions(self) -> None:
        assert clarification_agent.name == "ClarificationAgent"
        assert len(clarification_agent.instructions) > 0
