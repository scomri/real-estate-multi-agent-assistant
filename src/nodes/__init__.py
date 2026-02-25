"""
Node implementations for the Cortex Real Estate LangGraph agent.

Each module in this package defines a single callable node function that
accepts a :class:`~src.state.GraphState` dict and returns a partial-state
dict with the fields it populates.

Nodes
-----
supervisor  – classifies user intent into one of five categories.
extractor   – extracts and fuzzy-resolves named entities from the query.
retrieval   – filters the dataset based on extracted entities.
calculator  – aggregates filtered rows into a structured result dict.
responder   – formats the result into a natural-language response.
fallback    – generates a helpful error/redirect message when the pipeline fails.
"""
