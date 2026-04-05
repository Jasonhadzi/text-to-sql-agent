"""Clarification Agent — detects vague queries and asks follow-up questions.

Receives a user question and decides if it is clear enough to generate SQL.
If unclear, returns a clarifying question. If clear, passes to Query Router.

# TODO: Rachel — implement agent using the existing pattern (Agent + load_prompt + output_type)
"""

clarification_agent = None
