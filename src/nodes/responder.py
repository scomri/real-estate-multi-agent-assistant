"""
Responder node - natural-language response generation.

Formats the pipeline's output into a polished, user-facing answer using Claude Sonnet (smart tier).  

Two system prompts are used:
- **DATA_SYSTEM_PROMPT** - for data-bound intents; instructs the model to present all numbers in currency format within 3-5 sentences.
- **GENERAL_SYSTEM_PROMPT** - for general knowledge queries that required no dataset lookup; a simpler, open-ended prompt.
"""

import logging

from src.llm import get_sonnet, invoke_with_backoff
from src.state import GraphState

logger = logging.getLogger(__name__)

DATA_SYSTEM_PROMPT = """\
You are a virtual real estate asset management assistant.
The user will provide you with a request related to asset management, such as comparing property prices, calculating profit and loss (P&L), or retrieving asset details.
Format the provided data result into a clear, concise, professional response.
- Use currency formatting for monetary values (e.g. $1,234,567).
- Be specific - include all numbers from the result.
- Keep the response to 3-5 sentences unless detail is genuinely needed.
"""

GENERAL_SYSTEM_PROMPT = """\
You are a knowledgeable virtual real estate assistant.
Answer the user's question clearly and concisely.
"""

_llm = get_sonnet(temperature=0.3)


def responder(state: GraphState) -> dict:
    """
    Generate a natural-language answer and write it to state.
    Selects the appropriate system prompt based on intent, 
    then invokes the LLM with the user's original query and (where applicable) the computed result dict.

    Parameters
    ----------
    state : GraphState
        Relevant fields:
        - ``intent`` - used to choose between data and general prompts.
        - ``query``  - the original user question, forwarded to the model.
        - ``result`` - structured aggregation dict (only for data intents).

    Returns
    -------
    dict
        ``{"response": <str>}`` - the model's formatted reply.
    """
    intent = state.get("intent", "general")
    logger.info("Generating response for intent: %s", intent)

    if intent == "general":
        messages = [
            ("system", GENERAL_SYSTEM_PROMPT),
            ("user", state["query"]),
        ]
    else:
        messages = [
            ("system", DATA_SYSTEM_PROMPT),
            ("user", f"User query: {state['query']}\n\nData result:\n{state.get('result')}"),
        ]

    response = invoke_with_backoff(_llm, messages)
    content = response.content
    logger.info("Response generated (%d chars)", len(content))
    logger.debug("Response preview: '%s...'", content[:80])
    return {"response": content}
