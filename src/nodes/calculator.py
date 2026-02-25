"""
Calculator node - intent-specific, metric-aware aggregation.

Reads the filtered rows stored in "state["data"]" and computes a structured result dict whose 
shape depends on the current intent AND the metric requested (read from "state["entities"]["metric"]"):

- pnl
    - metric "profit" / absent : revenue + expenses + net breakdown
    - metric "revenue"         : revenue-only figure
    - metric "expenses"        : expenses-only figure
- comparison
    - metric "profit" / absent : net profit per group, sorted descending
    - metric "revenue"         : revenue per group, sorted descending
    - metric "expenses"        : expenses per group, sorted descending
    - "group_by" controls the grouping axis (default "property"; 
            also accepts "ledger_group", "ledger_category", "tenant", "month", "quarter", "year")
- asset_details
    - metric "profit" / absent : profit_total (sum of all rows)
    - metric "revenue"         : revenue_total (revenue rows only)
    - metric "expenses"        : expenses_total (expenses rows only)
- (fallback)                   : raw profit total, unchanged
"""

import logging

import pandas as pd

from src.state import GraphState

logger = logging.getLogger(__name__)

_GROUP_COL_MAP: dict[str, str] = {
    "property":        "property_name",
    "ledger_group":    "ledger_group",
    "ledger_category": "ledger_category",
    "tenant":          "tenant_name",
    "month":           "month",
    "quarter":         "quarter",
    "year":            "year",
}


def calculator(state: GraphState) -> dict:
    """
    Aggregate retrieved rows into a structured result based on intent and metric.

    Parameters
    ----------
    state : GraphState
        Must contain:
            - "data"       - list of row dicts from the retrieval node (non-empty)
            - "intent"     - classification from the "supervisor" node
            - "entities"   - dict produced by the "extractor" node; 
                             "metric" key may be "profit", "revenue", "expenses", or absent;
                             "group_by" key controls the grouping axis for comparison (and optional breakdown for "pnl").

    Returns
    -------
    dict
        {"result": <dict>} where the shape of the inner dict varies by intent and metric.
        Returns {"result": None} if state["data"] is empty or missing.
    """
    if not state.get("data"):
        return {"result": None}

    df = pd.DataFrame(state["data"])
    intent = state.get("intent", "pnl")
    # Read metric; absent / empty / None → "profit" (full aggregation, backward-compat)
    metric   = (state.get("entities") or {}).get("metric")   or "profit"
    # Read group_by; absent / empty / None → "property" (per-property, backward-compat)
    group_by = (state.get("entities") or {}).get("group_by") or "property"
    group_col = _GROUP_COL_MAP.get(group_by, "property_name")
    result: dict

    logger.info("Calculating result for intent=%s metric=%s group_by=%s", intent, metric, group_by)

    # ------------------------------------------------------------------
    # P&L intent
    # ------------------------------------------------------------------
    if intent == "pnl":
        period = {
            "from": str(df["month"].min()) if "month" in df.columns else None,
            "to":   str(df["month"].max()) if "month" in df.columns else None,
        }
        props = df["property_name"].unique().tolist()

        if metric == "revenue":
            rev_df  = df[df["ledger_type"] == "revenue"]
            revenue = float(rev_df["profit"].sum())
            result  = {
                "revenue":    revenue,
                "properties": props,
                "period":     period,
            }
            logger.debug("P&L (revenue only) - revenue=%.2f", revenue)
            pnl_subset = rev_df

        elif metric == "expenses":
            exp_df   = df[df["ledger_type"] == "expenses"]
            expenses = float(exp_df["profit"].sum())
            result   = {
                "expenses":   expenses,
                "properties": props,
                "period":     period,
            }
            logger.debug("P&L (expenses only) - expenses=%.2f", expenses)
            pnl_subset = exp_df

        else:  # "profit" or any unrecognised value → full breakdown
            by_type  = df.groupby("ledger_type")["profit"].sum()
            revenue  = float(by_type.get("revenue",  0))
            expenses = float(by_type.get("expenses", 0))
            net      = float(df["profit"].sum())
            result   = {
                "revenue":    revenue,
                "expenses":   expenses,
                "net":        net,
                "properties": props,
                "period":     period,
            }
            logger.debug(
                "P&L (full) - revenue=%.2f, expenses=%.2f, net=%.2f",
                revenue, expenses, net,
            )
            pnl_subset = df

        # Optional group-by breakdown (additive - scalar totals above are always present)
        if group_by != "property" and group_col in df.columns:
            by_group_series = (
                pnl_subset.groupby(group_col)["profit"]
                .sum()
                .sort_values(ascending=False)
            )
            result["by_group"]   = {k: float(v) for k, v in by_group_series.items()}
            result["grouped_by"] = group_col
            logger.debug("P&L breakdown by %s - %d groups", group_col, len(by_group_series))

    # ------------------------------------------------------------------
    # Comparison intent
    # ------------------------------------------------------------------
    elif intent == "comparison":
        if metric == "revenue":
            subset = df[df["ledger_type"] == "revenue"]
        elif metric == "expenses":
            subset = df[df["ledger_type"] == "expenses"]
        else:  # "profit" or absent → all rows (current behaviour)
            subset = df

        by_group_series = (
            subset.groupby(group_col)["profit"]
            .sum()
            .sort_values(ascending=False)
        )
        result = {
            "comparison": {k: float(v) for k, v in by_group_series.items()},
            "grouped_by": group_col,
        }
        logger.debug(
            "Comparison (metric=%s, grouped_by=%s) - %d groups ranked",
            metric, group_col, len(by_group_series),
        )

    # ------------------------------------------------------------------
    # Asset-details intent
    # ------------------------------------------------------------------
    elif intent == "asset_details":
        period = {
            "from": str(df["month"].min()) if "month" in df.columns else None,
            "to":   str(df["month"].max()) if "month" in df.columns else None,
        }
        base = {
            "properties":        df["property_name"].unique().tolist(),
            "tenants":           df["tenant_name"].dropna().unique().tolist(),
            "ledger_categories": df["ledger_category"].dropna().unique().tolist(),
            "period":            period,
        }

        if metric == "revenue":
            rev_df = df[df["ledger_type"] == "revenue"]
            base["revenue_total"] = float(rev_df["profit"].sum())
            logger.debug(
                "Asset details (revenue) - %d properties, %d tenants",
                len(base["properties"]), len(base["tenants"]),
            )
        elif metric == "expenses":
            exp_df = df[df["ledger_type"] == "expenses"]
            base["expenses_total"] = float(exp_df["profit"].sum())
            logger.debug(
                "Asset details (expenses) - %d properties, %d tenants",
                len(base["properties"]), len(base["tenants"]),
            )
        else:  # "profit" or absent → current behaviour
            base["profit_total"] = float(df["profit"].sum())
            logger.debug(
                "Asset details (profit) - %d properties, %d tenants",
                len(base["properties"]), len(base["tenants"]),
            )

        result = base

    # ------------------------------------------------------------------
    # Fallback - unchanged
    # ------------------------------------------------------------------
    else:
        result = {"raw_total": float(df["profit"].sum())}
        logger.debug("Raw total - %.2f", result["raw_total"])

    return {"result": result}
