"""
Supervisor node - query decomposition and intent classification.

Uses a structured LLM call (Claude Haiku) to decompose the user's raw query into
1-3 focused sub-questions, each classified into one of five intent labels.
Single-intent queries produce a list with one item; compound queries are split into
independent, self-contained sub-questions so each can be processed in parallel.

Valid intents
-------------
pnl             Questions about profit, loss, revenue, or overall financial performance.
comparison      Comparing financial metrics across two or more properties or tenants.
asset_details   Details about a specific property or tenant (tenants, categories, etc.).
general         General real estate knowledge that does not require a data lookup.
unknown         Ambiguous, out-of-scope, or unmappable queries.
"""

import logging

from pydantic import BaseModel, Field

from src.data_loader import get_data_dict
from src.llm import get_haiku, invoke_with_backoff
from src.state import GraphState

logger = logging.getLogger(__name__)

VALID_INTENTS = ("pnl", "comparison", "asset_details", "general", "unknown")
MAX_SUB_QUERIES = 3

SYSTEM_PROMPT = """\
You are an intent classifier and query decomposer for a virtual real estate asset management assistant.

Your task: decompose the user's query into 1 to {max_sub_queries} focused, self-contained sub-questions, each classified into exactly one of these intents:

- pnl           : questions about profit, loss, revenue, expenses, or financial performance (total P&L, net income, revenue vs expenses, etc.)
- comparison    : comparing financial metrics across two or more entities (properties, tenants, ledgers, time periods, etc.)
- asset_details : requesting details about a specific entity (what tenants, what categories, what ledger items, etc.)
- general       : general real estate knowledge NOT requiring a data lookup
- unknown       : query is ambiguous, out-of-scope, or cannot be mapped to the above

Rules:
- Return exactly 1 sub-query for single-intent questions - do NOT split unnecessarily.
- Split compound queries into at most {max_sub_queries} sub-questions.
- If you detect more than {max_sub_queries} distinct questions, combine the most closely related ones.
- Each sub-question must be independently answerable - include entity names, dates, and context in each sub-question if the original query was split (do not rely on other sub-questions for context).

Available dataset columns:
{data_dict}
"""


class SubQuery(BaseModel):
    query: str = Field(description="Self-contained sub-question, independently answerable")
    intent: str = Field(description="One of: pnl, comparison, asset_details, general, unknown")


class SupervisorResult(BaseModel):
    sub_queries: list[SubQuery] = Field(
        min_length=1,
        max_length=MAX_SUB_QUERIES,
        description="Decomposed sub-questions with their classified intents",
    )


_llm = get_haiku()
_structured_llm = _llm.with_structured_output(SupervisorResult)


def supervisor(state: GraphState) -> dict:
    """
    Decompose the user query into sub-questions and classify each intent.

    Invokes the structured LLM with a system prompt that includes the dataset schema.
    Each returned sub-query is validated: invalid intent labels are replaced with "unknown".
    The result is always a non-empty list of at most MAX_SUB_QUERIES items.

    Parameters
    ----------
    state : GraphState
        Must contain "query" - the raw user input string.

    Returns
    -------
    dict
        {"sub_queries": [{"query": str, "intent": str}, ...]}
        Always contains 1-3 items.
    """
    query = state["query"]
    logger.info("Decomposing query: '%s...'", query[:60])

    prompt = SYSTEM_PROMPT.format(
        data_dict=get_data_dict(),
        max_sub_queries=MAX_SUB_QUERIES,
    )
    result: SupervisorResult = invoke_with_backoff(
        _structured_llm, [("system", prompt), ("user", query)]
    )

    sub_queries = []
    for sq in result.sub_queries:
        intent = sq.intent if sq.intent in VALID_INTENTS else "unknown"
        sub_queries.append({"query": sq.query, "intent": intent})

    logger.info(
        "Decomposed into %d sub-quer%s: %s",
        len(sub_queries),
        "y" if len(sub_queries) == 1 else "ies",
        [(sq["intent"], sq["query"][:40]) for sq in sub_queries],
    )
    return {"sub_queries": sub_queries}
