"""
Synthesizer node - multi-result response generation.

This node is the reduce step in the map-reduce pattern.
It receives the fully-merged "sub_results" list (one item per sub-query) and 
generates a single, coherent natural-language response that addresses all questions using Claude Sonnet.

Three categories of sub-result are handled:
- Data intent (success=True)     → structured result dict included in context
- General intent (success=True)  → sub-question answered from LLM's knowledge
- Failed intent (success=False)  → acknowledged in response with a brief note
"""

import logging

from src.llm import get_sonnet, invoke_with_backoff
from src.state import GraphState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a virtual real estate asset management assistant.

You will be given one or more sub-questions with their results (or failure reasons).
Write a single, coherent response that addresses every sub-question.

Formatting rules:
- If there are multiple sub-questions, use a short **bold header** for each answer (e.g. **P&L for Building 180** or **General: cap rate**).
- If there is only one sub-question, answer directly without a header.
- Use currency formatting for all monetary values (e.g. $1,234,567).
- Be specific and include all numbers from the result data.
- Keep each answer to 2-4 sentences unless more detail is genuinely needed.
- If a sub-question could not be answered, briefly acknowledge it and suggest how the user might rephrase (1 sentence).
"""


_llm = get_sonnet(temperature=0.3)


def synthesizer(state: GraphState) -> dict:
    """
    Generate a unified natural-language answer from all sub-query results.

    Builds a user message that lists each sub-question alongside its result (or failure reason), then 
    invokes the LLM to produce one coherent response.

    Parameters
    ----------
    state : GraphState
        Must contain "sub_results" - the merged list produced by all parallel "process_subquery" branches.  
        Also uses "query" for logging.

    Returns
    -------
    dict
        {"response": <str>} - the model's formatted reply.
    """
    sub_results = state.get("sub_results") or []
    logger.info(
        "Synthesizing response for %d sub-result(s) - query: '%s...'",
        len(sub_results),
        state.get("query", "")[:60],
    )

    # Build context blocks for each sub-result
    blocks = []
    for i, item in enumerate(sub_results, start=1):
        label = f"Sub-question {i}: {item['query']}"
        if item["success"]:
            if item["intent"] == "general":
                blocks.append(f"{label}\n[Answer from general knowledge - no dataset lookup]")
            else:
                blocks.append(f"{label}\nData result: {item['result']}")
        else:
            blocks.append(f"{label}\nFailed: {item.get('error', 'No data available')}")

    user_message = "\n\n".join(blocks)
    logger.debug("Synthesizer context:\n%s", user_message[:400])

    response = invoke_with_backoff(_llm, [("system", SYSTEM_PROMPT), ("user", user_message)])
    content = response.content
    logger.info("Synthesizer response generated (%d chars)", len(content))
    return {"response": content}
