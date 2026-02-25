"""
Fallback node - graceful error handling and user guidance.

Invoked only when the "supervisor" produces no sub-queries (e.g. due to an LLM failure or an empty query).  
Partial failures within individual sub-queries (e.g. no data found for a property) are handled inline by the "synthesizer" node.

Uses Claude Haiku to produce a friendly, constructive response that explains what went wrong and 
suggests how the user can rephrase their question.
"""

import logging

from src.data_loader import load_data
from src.llm import get_haiku, invoke_with_backoff
from src.state import GraphState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful virtual real estate asset management assistant.

The user's query could not be processed.
Briefly explain what went wrong (1 sentence), then suggest how the user could rephrase or narrow their question (1-2 sentences).
Be friendly and constructive.

Available data covers: properties, tenants, profit/loss broken down by month, quarter, year, ledger type (revenue/expenses), ledger group, and ledger category.
"""

_llm = get_haiku(temperature=0.4)


def fallback(state: GraphState) -> dict:
    """
    Generate a helpful fallback response when no sub-queries could be produced.

    Parameters
    ----------
    state : GraphState
        Relevant fields:
        - "query" - the original user question.
        - "error" - error message from the supervisor, if any.

    Returns
    -------
    dict
        {"response": <str>} - the model's friendly error/redirect message.
    """
    df = load_data()
    available_properties = df["property_name"].dropna().unique().tolist()

    reason = state.get("error") or "The query could not be understood or decomposed."
    logger.warning("Fallback triggered — reason: %s", reason)

    user_message = (
        f"User query: {state['query']}\n\n"
        f"Reason: {reason}\n\n"
        f"Available properties: {available_properties}"
    )

    response = invoke_with_backoff(_llm, [("system", SYSTEM_PROMPT), ("user", user_message)])
    return {"response": response.content}
