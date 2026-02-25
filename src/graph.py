"""
LangGraph agent graph for the Cortex Real Estate AI assistant.

The graph implements a map-reduce pattern:

1. "supervisor": 
    Decomposes the user query into 1-3 sub-questions, each with a classified intent.

2. "fan_out" (Send API): 
    "_route_supervisor" returns a list of "Send("process_subquery", {...})" objects, one per sub-query.  
    LangGraph runs all "process_subquery" branches in parallel.

3. "process_subquery" (parallel map step): 
    Each branch runs the full data-based pipeline (extractor → retrieval → calculator) or 
    passes through directly for general/unknown intents. 
    Returns {"sub_results": [single_item]}.
    The "Annotated[list, operator.add]" reducer in GraphState concatenates all items.

4. "synthesizer" (reduce step):
    Runs once after all parallel branches complete.
    Generates one coherent response covering all sub-questions.

5. "fallback" (safety net):
    Only reached if the supervisor produces no sub-queries (e.g. LLM failure).

Entry point
-----------
Call "build_graph" to obtain a compiled, runnable LangGraph instance.
"""

import logging

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from src.nodes.fallback import fallback
from src.nodes.process_subquery import process_subquery
from src.nodes.supervisor import supervisor
from src.nodes.synthesizer import synthesizer
from src.state import GraphState

logger = logging.getLogger(__name__)


def _route_supervisor(state: GraphState):
    """
    Return routing decisions after the supervisor node.

    If the supervisor produced no sub-queries (edge case / LLM failure), route to fallback.
    Otherwise return a list of Send objects, one per sub-query, to fan out in parallel.

    Parameters
    ----------
    state : GraphState
        Must contain "sub_queries" populated by the supervisor.

    Returns
    -------
    str | list[Send]
        "fallback" when "sub_queries" is empty, otherwise a list of
        "Send("process_subquery", {"query": ..., "intent": ...})" objects.
    """
    sub_queries = state.get("sub_queries") or []
    if not sub_queries:
        logger.warning("Supervisor produced no sub-queries - routing to fallback")
        return "fallback"
    logger.debug("Fan-out: %d sub-quer%s", len(sub_queries), "y" if len(sub_queries) == 1 else "ies")
    return [
        Send("process_subquery", {"query": sq["query"], "intent": sq["intent"]})
        for sq in sub_queries
    ]


def build_graph():
    """
    Build and compile the Cortex Real Estate LangGraph agent.

    Graph structure
    ---------------
    START → supervisor
              ↓  _route_supervisor: Send x 1-3
            process_subquery x 1-3  (parallel)
              ↓  sub_results merged via operator.add reducer
            synthesizer → END
              (fallback → END  if supervisor produced no sub-queries)

    Returns
    -------
    CompiledGraph
        A LangGraph compiled graph that accepts a partial "GraphState" dict
        (only "query" is required) and returns the completed state.
    """
    g = StateGraph(GraphState)

    g.add_node("supervisor", supervisor)
    g.add_node("process_subquery", process_subquery)
    g.add_node("synthesizer", synthesizer)
    g.add_node("fallback", fallback)

    g.add_edge(START, "supervisor")

    g.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        ["process_subquery", "fallback"],
    )

    g.add_edge("process_subquery", "synthesizer")
    g.add_edge("synthesizer", END)
    g.add_edge("fallback", END)

    compiled = g.compile()
    logger.info("Graph compiled successfully")
    return compiled
