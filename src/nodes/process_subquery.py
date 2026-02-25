"""
"process_subquery" node - single sub-query pipeline worker.

Called by LangGraph's Send API as a parallel worker node.
Receives one sub-query ({"query": str, "intent": str}) and runs the full
data pipeline ("extractor" → "retrieval" → "calculator") for data-bound intents,
or passes through directly to the "synthesizer" for general/unknown intents.

This node is the map step in the map-reduce pattern.  
Its output always contains a single-item list in "sub_results"; 
the "Annotated[list, operator.add]" reducer in GraphState concatenates results from all parallel branches.
"""

import logging

from src.nodes.calculator import calculator
from src.nodes.extractor import extractor
from src.nodes.retrieval import retrieval

logger = logging.getLogger(__name__)

_DATA_INTENTS = {"pnl", "comparison", "asset_details"}


def process_subquery(sub_state: dict) -> dict:
    """
    Run the appropriate pipeline for one sub-query and return a single result item.

    Parameters
    ----------
    sub_state : dict
        Must contain:
        - "query"  - the sub-question text.
        - "intent" - one of the five valid intent labels.

    Returns
    -------
    dict
        {"sub_results": [item]} where item is a dict with keys:
            - "query"   - the original sub-question text.
            - "intent"  - the classified intent.
            - "result"  - aggregated result dict from the calculator, or None.
            - "success" - True if the pipeline produced a usable result.
            - "error"   - descriptive error string on failure, None on success.
    """
    query = sub_state["query"]
    intent = sub_state["intent"]

    logger.info("Processing sub-query [%s]: '%s...'", intent, query[:60])

    if intent in _DATA_INTENTS:
        # Run: extractor → retrieval → calculator
        s = {"query": query, "intent": intent, "entities": {}, "error": None}

        extracted = extractor(s)
        s.update(extracted)

        retrieved = retrieval(s)
        s.update(retrieved)

        if s.get("data"):
            calculated = calculator(s)
            s.update(calculated)
            item = {
                "query": query,
                "intent": intent,
                "result": s.get("result"),
                "success": True,
                "error": None,
            }
            logger.info("Sub-query [%s] succeeded - %d data rows", intent, len(s["data"]))
        else:
            item = {
                "query": query,
                "intent": intent,
                "result": None,
                "success": False,
                "error": s.get("error", "No matching data found"),
            }
            logger.warning("Sub-query [%s] failed - no data: %s", intent, item["error"])

    elif intent == "general":
        item = {
            "query": query,
            "intent": "general",
            "result": None,
            "success": True,
            "error": None,
        }
        logger.info("Sub-query [general] passed through for open-ended response")

    else:
        item = {
            "query": query,
            "intent": intent,
            "result": None,
            "success": False,
            "error": "Query could not be classified into a supported intent",
        }
        logger.warning("Sub-query [%s] - unhandled intent", intent)

    return {"sub_results": [item]}
