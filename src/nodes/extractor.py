"""
Extractor node - entity linking and fuzzy resolution.

Uses a structured LLM call (Claude Haiku) to resolve filter entities
(property names, tenant names, year, months, and financial metric) directly to
exact dataset values using the schema injected into the system prompt (primary resolver).

If the LLM cannot find an exact match it returns the as-heard string.
"_fuzzy_resolve" then acts as a high-confidence typo-correction fallback (cutoff=0.80, n=1).  

Entities that survive neither gate are dropped with a WARNING log rather 
than being mapped to the closest approximate value.
"""

import difflib
import logging
from typing import Optional

from pydantic import BaseModel, Field

from src.data_loader import get_data_dict, load_data
from src.llm import get_haiku, invoke_with_backoff
from src.state import GraphState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an entity extractor for a virtual real estate asset management assistant.
Extract relevant filters from the user query using the dataset schema below.

Dataset schema:
{data_dict}

Rules:
- properties: resolve to the exact property names from the schema above that best match what the user mentioned; 
    if no exact match exists but a property was clearly mentioned, include the name as heard (or [] if none mentioned)
- tenants: resolve to the exact tenant names from the schema above that best match what the user mentioned; 
    if no exact match exists but a tenant was clearly mentioned, include the name as heard (or [] if none mentioned)
- year: integer year if mentioned, else None
- months: list of month strings in the format found in the data (e.g. "2023-01") if mentioned, else []
- quarters: list of quarter strings in the format found in the data (e.g. "2024-Q1") if mentioned, else []
- ledger_groups: list of ledger group names mentioned (use exact values from the schema above, or [] if none)
- ledger_categories: list of ledger category names mentioned (use exact values from the schema above, or [] if none)
- metric: one of "profit", "revenue", or "expenses" (default "profit")
- group_by: the dimension the user wants to group or compare by.
    Use one of: "property" (default), "ledger_group", "ledger_category", "tenant", "month", "quarter", "year".
    Default "property" unless the user explicitly asks to group/break down by another dimension
"""


class Entities(BaseModel):

    properties: list[str] = Field(default_factory=list)
    tenants: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    months: list[str] = Field(default_factory=list)
    quarters: list[str] = Field(default_factory=list)
    ledger_groups: list[str] = Field(default_factory=list)
    ledger_categories: list[str] = Field(default_factory=list)
    metric: str = "profit"
    group_by: str = "property"


_llm = get_haiku()
_structured_llm = _llm.with_structured_output(Entities)


def _fuzzy_resolve(names: list[str], candidates: list[str], cutoff: float = 0.80) -> list[str]:
    """
    Resolve entity names to actual dataset values using a two-stage cascade.

    First attempts a case-insensitive exact match.  
    If that fails, uses fuzzy matching (difflib) as a typo-correction fallback.  
    Only the single best match is accepted (n=1) to prevent one ambiguous mention from 
    expanding into multiple false-positive candidates.  
    Entities with no match at or above the cutoff are dropped and a WARNING is emitted.

    Parameters
    ----------
    names : list[str]
        Names returned by the LLM extractor (already attempted schema linking).
    candidates : list[str]
        Authoritative list of values from the dataset.
    cutoff : float
        Minimum similarity ratio for the fuzzy fallback (default 0.80).

    Returns
    -------
    list[str]
        Resolved names, deduplicated and in original order.
    """
    resolved = []
    candidates_lower = {c.lower(): c for c in candidates}
    for name in names:
        # Exact match first (case-insensitive)
        if name.lower() in candidates_lower:
            resolved.append(candidates_lower[name.lower()])
            logger.info("Exact match: '%s' → '%s'", name, candidates_lower[name.lower()])
            continue
        # Fuzzy fallback: n=1 so only the best match is accepted
        matches = difflib.get_close_matches(name.lower(), candidates_lower.keys(), n=1, cutoff=cutoff)
        if matches:
            corrected = candidates_lower[matches[0]]
            logger.warning("Fuzzy correction: '%s' → '%s'", name, corrected)
            resolved.append(corrected)
        else:
            logger.warning("Entity dropped - no match found for: '%s'", name)
    return list(dict.fromkeys(resolved))  # deduplicate, preserve order


def extractor(state: GraphState) -> dict:
    """
    Extract entities from the user query and resolve them to dataset values.

    1. Calls the structured LLM to extract raw entity strings from "state["query"]".
    2. Loads the unique property and tenant names present in the dataset.
    3. Runs "_fuzzy_resolve" on both name lists to correct spelling and casing differences.

    Parameters
    ----------
    state : GraphState
        Must contain "query" - the raw user input string
            (if no "query" - this won't be reached because the 
            "supervisor" node will return "unknown" and the graph will end)

    Returns
    -------
    dict
        {"entities": <dict>} with keys:
            - "properties"        - resolved property name(s)
            - "tenants"           - resolved tenant name(s)
            - "year"              - integer year or None
            - "months"            - list of "YYYY-MM" strings
            - "quarters"          - list of "YYYY-Qn" strings
            - "ledger_groups"     - resolved ledger group name(s)
            - "ledger_categories" - resolved ledger category name(s)
            - "metric"            - one of "profit", "revenue", "expenses"
            - "group_by"          - grouping dimension: "property" (default), "tenant",
                                        "ledger_group", "ledger_category", 
                                        "month", "quarter", or "year"
    """
    logger.info("Extracting entities...")
    prompt = SYSTEM_PROMPT.format(data_dict=get_data_dict())
    result: Entities = invoke_with_backoff(
        _structured_llm, [("system", prompt), ("user", state["query"])]
    )
    logger.info("Raw LLM entities: %s", result)

    df = load_data()
    all_properties = df["property_name"].dropna().unique().tolist()
    all_tenants = df["tenant_name"].dropna().unique().tolist()
    all_ledger_groups = df["ledger_group"].dropna().unique().tolist()
    all_ledger_categories = df["ledger_category"].dropna().unique().tolist()

    entities = {
        "properties": _fuzzy_resolve(result.properties, all_properties),
        "tenants": _fuzzy_resolve(result.tenants, all_tenants),
        "year": result.year,
        "months": result.months,
        "quarters": result.quarters,
        "ledger_groups": _fuzzy_resolve(result.ledger_groups, all_ledger_groups),
        "ledger_categories": _fuzzy_resolve(result.ledger_categories, all_ledger_categories),
        "metric": result.metric,
        "group_by": result.group_by,
    }
    logger.info("Resolved entities: %s", entities)
    return {"entities": entities}
