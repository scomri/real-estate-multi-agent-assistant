"""
Retrieval node - DataFrame filtering.

Applies the entity filters extracted by the extractor node to the in-memory dataset and writes the matching rows to state.  
When no rows match, an informative error message is written instead so the fallback node can communicate the failure to the user.
"""

import logging

import pandas as pd

from src.data_loader import load_data
from src.state import GraphState

logger = logging.getLogger(__name__)


def retrieval(state: GraphState) -> dict:
    """
    Filter the dataset using the entities stored in state.

    Builds a boolean mask from whichever entity filters are non-empty (properties, tenants, year, months) and 
    applies it to the full DataFrame.
    All active filters are ANDed together.

    Parameters
    ----------
    state : GraphState
        Must contain ``entities`` - a dict produced by the extractor node.

    Returns
    -------
    dict
        On success: ``{"data": [<records>], "error": None}`` where *records* is a list of row dicts from the filtered DataFrame.

        On failure (no rows match):
        ``{"data": [], "error": "<descriptive message>"}``
    """
    df = load_data()
    entities: dict = state.get("entities") or {}

    mask = pd.Series([True] * len(df), index=df.index)

    active_filters = {k: v for k, v in entities.items() if v}
    logger.info("Filtering with: %s", active_filters)

    if entities.get("properties"):
        mask &= df["property_name"].isin(entities["properties"])

    if entities.get("tenants"):
        mask &= df["tenant_name"].isin(entities["tenants"])

    if entities.get("year"):
        mask &= df["year"] == entities["year"]

    if entities.get("months"):
        mask &= df["month"].astype(str).isin(entities["months"])

    if entities.get("quarters"):
        mask &= df["quarter"].isin(entities["quarters"])

    if entities.get("ledger_groups"):
        mask &= df["ledger_group"].isin(entities["ledger_groups"])

    if entities.get("ledger_categories"):
        mask &= df["ledger_category"].isin(entities["ledger_categories"])

    filtered = df[mask]

    if filtered.empty:
        filters_desc = ", ".join(f"{k}={v}" for k, v in entities.items() if v)
        logger.warning("No data found for: %s", filters_desc or "the specified filters")
        return {
            "data": [],
            "error": f"No data found for: {filters_desc or 'the specified filters'}.",
        }

    logger.info("Retrieved %d rows", len(filtered))
    return {"data": filtered.to_dict("records"), "error": None}
