"""
Shared state schema for the Cortex Real Estate LangGraph agent.

The single "GraphState" TypedDict travels through every node in the graph.
Each node reads the fields it needs and returns a dict containing only the fields it updates;
LangGraph merges that partial dict back into the running state automatically.
"""

import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict):
    """
    Typed dictionary that represents the full agent state at any graph step.

    Fields
    ------
    query : str
        Raw text input from the user.

    sub_queries : list[dict]
        Decomposed sub-questions with their classified intents.
        Each item: {"query": str, "intent": str}.
        Populated by the "supervisor" node. 
        Always contains 1-3 items.

    sub_results : Annotated[list[dict], operator.add]
        Results collected from each parallel "process_subquery" branch.
        The "operator.add" reducer concatenates contributions from all branches.
        Each item: {"query": str, "intent": str, "result": Any, "success": bool, "error": str|None}.
        Populated by the "process_subquery" node.

    response : str
        Final natural-language answer surfaced to the user.
        Populated by the "synthesizer" or "fallback" node.

    error : str | None
        Human-readable error message set when the "supervisor" node cannot produce any sub-queries.
        None during normal operation.
    """
    query: str
    sub_queries: list[dict]                            # [{"query": str, "intent": str}, ...]
    sub_results: Annotated[list[dict], operator.add]   # parallel-safe reducer
    response: str
    error: str | None
