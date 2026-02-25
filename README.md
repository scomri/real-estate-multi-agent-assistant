# Cortex Real Estate Multi-Agent AI Assistant

> Conversational AI agent for real estate asset management and analytics

![Python](https://img.shields.io/badge/Python-3.13-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-Multi--agent-darkgreen) ![Claude](https://img.shields.io/badge/Claude-LLM-purple) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red) 

**[🎥 Watch Demo Video](demo/streamlit-app-2026-02-25-12-06-02.webm)**

---

Cortex Real Estate AI Assistant answers natural language questions about real estate property profit & loss data. 
Users can ask about financial performance, compare properties, explore asset details, or pose general real estate questions - and receive professional, data-grounded answers in plain English.

The system is built on a **LangGraph map-reduce workflow** that decomposes the user's query into up to three focused sub-questions, classifies each sub-question's intent, runs each sub-pipeline in parallel, and synthesises all results into a single coherent response using Claude.

---

## Setup Instructions
Follow these steps to get the assistant running locally.

### Prerequisites
- Python <= **3.13** (for Pydantic V2 support)
- An **Anthropic API key** (`ANTHROPIC_API_KEY`)

### Installation
```bash
# Clone the project
git clone <repo-url>
cd real-estate-multi-agent-assistant

# Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate        # Windows (Git Bash / PowerShell)
# source .venv/bin/activate          # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Create a .env file in the project root (see Configuration below)
```

### Configuration
Create a `.env` file in the project root:

```dotenv
# Required
ANTHROPIC_API_KEY=...

# Optional - path to the dataset (default: data/cortex.parquet)
DATA_PATH=data/cortex.parquet

# Optional - logging verbosity (default: INFO)
LOG_LEVEL=INFO
```

### Logging
Logging is configured once in `app.py` via `logging.basicConfig()` and applies to all modules.

**Format:** `HH:MM:SS.mmm | LEVEL    | module - message`

| Level | What is logged |
|---|---|
| `DEBUG` | Fuzzy name corrections, LLM attempt numbers, response content previews, synthesizer context |
| `INFO` | Query decomposition result, sub-query intents, extracted entities, retrieval row count, response length |
| `WARNING` | Entity dropped during fuzzy resolution; no data found for a sub-query; rate-limit backoff delays |
| `ERROR` | Rate-limit retries exhausted after all 6 attempts |

Set `LOG_LEVEL=DEBUG` in `.env` for full verbosity, or `LOG_LEVEL=WARNING` to see only data-miss and rate-limit events (Default = `INFO`).

### Usage
```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. 

Each response includes a collapsible **`Debug: agent state`** panel showing the decomposed sub-queries and sub-results.

---

## Architecture

### Approach

- **Multi-Agent System** 
    - The system uses a LangGraph-based multi-agent architecture with a `supervisor` agent and multiple worker agents.
    - The `supervisor` agent is responsible for decomposing the user's query into smaller sub-queries and assigning them to the appropriate worker agents.
    - The worker agents are responsible for processing the sub-queries and returning the results to the supervisor agent.

- **Map-Reduce Pipeline** 
    - The `supervisor` node decomposes the user's query into up to three focused, self-contained sub-questions, each classified into one of five intents (`pnl`, `comparison`, `asset_details`, `general`, `unknown`). 
    - LangGraph's Send API fans these out to parallel `process_subquery` workers, which each run the full data-based pipeline (`extractor` → `retrieval` → `calculator`); results are merged by the `synthesizer` into a single coherent response.
    - If intents are `general` or `unknown`, the `process_subquery` node will pass the sub-query directly to the `synthesizer` without running the data-based pipeline.

- **Partial-Failure Handling** 
    - Sub-queries that return no data are marked `success=False` and acknowledged inline by the `synthesizer` rather than failing the entire response. 
    - The `fallback` node is reserved only for the edge case where the `supervisor` itself produces zero sub-queries.

- **Two-Tier Model Strategy** 
    - Claude Haiku 4.5 handles the speed-sensitive, high-frequency tasks (decomposition, extraction, fallback).
    - Claude Sonnet 4.5 is reserved for the `synthesizer` where response quality and coherence matter most.

- **Data Schema-Driven Prompts** 
    - `get_data_dict()` in `data_loader.py` generates a compact schema string from the DataFrame at startup (cached via `lru_cache`). 
    - All LLM prompts inject this string at call time, so property names, tenant names, and date formats always reflect the actual dataset without hardcoding.

- **Two-Stage Entity Resolution** 
    - The LLM extractor is the primary resolver, prompted with the dataset schema to map user mentions directly to exact dataset values. 
    - A `difflib` fuzzy fallback (cutoff 0.80, `n=1`) corrects residual typos. 
    - Entities that pass neither gate are dropped with a `WARNING` log rather than silently forwarding to retrieval.

- **Dynamic `group_by` Calculation** 
    - The `extractor` surfaces a `group_by` field (default `property`) supporting seven axes: `property`, `ledger_group`, `ledger_category`, `tenant`, `month`, `quarter`, `year`. 
    - The `calculator` uses this to group comparisons or append a `by_group` breakdown to P&L results, letting users ask questions like "rank tenants by revenue" without any extra routing logic.


### Tech Stack
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.13 | Runtime |
| LangGraph | latest | Agentic workflow + map-reduce state machine |
| LangChain Anthropic | latest | Claude API wrapper |
| Claude Haiku 4.5 | `claude-haiku-4-5` | Fast tier - `supervisor`, `extractor`, `fallback` |
| Claude Sonnet 4.5 | `claude-sonnet-4-5` | Smart tier - `synthesizer` |
| Streamlit | latest | Chat UI |
| Pandas | latest | DataFrame filtering and aggregation |
| PyArrow / fastparquet | latest | Parquet file I/O |


### Project Structure
```
real-estate-multi-agent-assistant/
├── app.py                      # Entry point: Streamlit, logging, graph
├── requirements.txt            
├── tests.py                    # Full test suite (non-LLM and LLM/integration tests)
├── data/
│   └── cortex.parquet          # Real estate dataset
└── src/
    ├── graph.py                # LangGraph StateGraph + fan-out routing
    ├── state.py                # Graph State (shared schema + operator.add reducer)
    ├── data_loader.py          # Data loading + normalization + schema-string builder
    ├── llm.py                  # ChatAnthropic factories + invoke_with_backoff
    └── nodes/
        ├── supervisor.py       # Query decomposition + intent classification
        ├── process_subquery.py # Parallel map worker
        ├── extractor.py        # LLM-based entity extraction + fuzzy fallback
        ├── retrieval.py        # Pandas DataFrame filtering
        ├── calculator.py       # Intent-specific aggregation
        ├── synthesizer.py      # Natural language response generation
        └── fallback.py         # Graceful error message when no sub-queries
```


---

## LangGraph Multi-Agent Workflow

### Graph Overview

The graph is a **map-reduce pipeline** with one conditional routing point:

```
                                ┌─[1-3 sub-queries via Send API]──► process_subquery x1-3 ─────┐
                                │         (parallel fan-out)          (map step)               │
START ──► supervisor ───────────┤                                                              ▼
                                │                            sub_results merged via operator.add
                                │                                                              │
                                │                                                   synthesizer ──► END
                                │
                                └─[no sub-queries]───────────────────────────────────► fallback ──► END
```

Each `process_subquery` branch internally runs:

```
[pnl / comparison / asset_details]  extractor ──► retrieval ───┬──[data found]──► calculator ──► sub_results
                                                               └──[no data]────────────────────► sub_results (success=False)
[general]                           pass-through ──────────────────────────────────────────────► sub_results
[unknown]                           mark as failed ────────────────────────────────────────────► sub_results (success=False)
```


### Node Descriptions
| Node | Model | Role |
|---|---|---|
| **`supervisor`** | Haiku | Decomposes the user query into 1–3 focused, self-contained sub-questions; classifies each into one of five intent labels |
| **`process_subquery`** | - | Parallel map worker: runs `extractor` → `retrieval` → `calculator` for data intents; passes through for general intents; marks unknown intents as failed |
| **`extractor`** | Haiku | Extracts entities (property/tenant names, time periods, ledger groups/categories), metric, and `group_by` dimension from the query; resolves via LLM schema linking first, then fuzzy fallback |
| **`retrieval`** | - | Applies entity filters as AND-combined boolean masks on the Pandas DataFrame |
| **`calculator`** | - | Aggregates filtered rows into a structured result dict shaped by intent and `group_by` dimension; comparison groups along the requested axis; P&L adds an optional `by_group` breakdown when a non-default dimension is requested |
| **`synthesizer`** | Sonnet | Reduces all parallel sub-results into one coherent, currency-formatted natural language answer; handles partial failures inline |
| **`fallback`** | Haiku | Reached only when the supervisor produces zero sub-queries (LLM failure edge case); explains the failure and suggests rephrasing |

**Model Selection Rationale** 
- **Claude Haiku 4.5** handles the speed-sensitive, high-frequency tasks (decomposition/classification in `supervisor`, entity extraction in `extractor`, and error messaging in `fallback`) because its lower latency and cost are sufficient for structured output. 
- **Claude Sonnet 4.5** is used for the `synthesizer` node, where response quality, multi-result coherence, and formatting matter most.


### Shared State Schema
All nodes communicate through a single `GraphState` TypedDict (defined in `src/state.py`). 
Each node receives the full state and returns a partial dict containing only the fields it writes; LangGraph merges these updates automatically.

| Field | Type | Set by | Purpose |
|---|---|---|---|
| `query` | `str` | caller | Raw user input |
| `sub_queries` | `list[dict]` | `supervisor` | Decomposed sub-questions with classified intents: `[{"query": str, "intent": str}, ...]` |
| `sub_results` | `Annotated[list[dict], operator.add]` | `process_subquery` | Results from each parallel branch; `operator.add` reducer concatenates contributions. Each item: `{"query", "intent", "result", "success", "error"}` |
| `response` | `str` | `synthesizer` / `fallback` | Final user-facing answer |
| `error` | `str \| None` | `supervisor` / any node | Human-readable failure description when the supervisor cannot produce sub-queries; `None` during normal operation |

**Note:** `intent`, `entities`, `data`, and `result` are not top-level state fields.
They are local variables within each `process_subquery` branch and stored per-sub-query inside `sub_results`.


### Routing Logic
The graph has one conditional edge (implemented in `src/graph.py`):
- **`_route_supervisor(state)`** reads `state['sub_queries']` and returns:
    - A list of `Send("process_subquery", {"query": ..., "intent": ...})` objects (one per sub-query) - fans out in parallel when at least one sub-query was produced
    - `'fallback'` when `sub_queries` is empty - supervisor produced no output (LLM failure edge case)

**Note:** 
- Per-sub-query failures (empty retrieval results, unresolvable entities) are **not** routing decisions. 
- They are handled inline inside `process_subquery`: the item is appended to `sub_results` with `success=False` and an `error` string. 
- The `synthesizer` then acknowledges each failure and suggests how to rephrase, producing a mixed response that answers the successful parts and explains the unsuccessful ones.


---

## Challenges & Solutions

- **LLM API Rate Limits**
    - **Problem:** 
    Running multiple LLM calls within a short window (especially for testing) might exceed Anthropic's rate limit of 50 Requests Per Minute (RPM) and cause an HTTP 429 response which would crash the conversation.
    - **Solution:** 
    Every LLM call goes through `invoke_with_backoff` (`src/llm.py`). 
    The function catches `anthropic.RateLimitError` and retries with exponential backoff (up to 6 attempts).
    If all attempts are exhausted, the error is re-raised so the caller still fails loudly. 


- **Inconsistent Data Formats in Raw Dataset**
    - **Problem:** 
    Two columns in the data source file were stored in formats incompatible with downstream logic:
        - `year` was stored as a string rather than an integer. 
        The extractor's Pydantic model declares `year: Optional[int]`, so a string comparison in the `retrieval` mask would return zero rows.
        - `month` was stored as `"2024-M12"` instead of the ISO format `"2024-12"`. 
        Because prompts inject raw dataset examples, this format would propagate into entity extraction and break `retrieval` filters.
    - **Solution:** 
        The `_normalize` function in `src/data_loader.py` is applied after the data file is read and before the DataFrame is passed to `lru_cache`. 
        It casts `year` to an integer and rewrites `month` with a regex substitution to ISO format. 
        One-time normalization at load ensures all nodes and prompts use the cleaned, cached DataFrame, bypassing raw formats.
    

- **Entity Resolution Robustness**
    - **Problem:** 
        Users might misspell property or tenant names. 
        A query like "show me Bilding 180 revenue" would produce an empty retrieval result because `"Bilding 180"` does not appear anywhere in the dataset.
    - **Solution:** 
        Entity resolution uses a two-stage cascade in `_fuzzy_resolve` in `src/nodes/extractor.py`:
        1. **LLM as primary resolver:** 
            The `extractor`'s system prompt injects the full dataset schema (all unique entity names; properties, tenants, ledgers). 
            The model is instructed to resolve mentions directly to exact schema values. 
        2. **Fuzzy fallback for residual typos:** 
            If the LLM returns a name that still does not exactly match a dataset value, `_fuzzy_resolve` runs `difflib.get_close_matches` with a similarity cutoff of 0.80 and `n=1` (only the single best match is accepted), preventing one ambiguous mention from expanding into multiple false-positive candidates. 
            Exact case-insensitive matches are promoted before fuzzy matching is attempted.
        3. **Drop unresolvable entities:** 
            If a name passes neither stage, it is dropped, and a WARNING is logged rather than being silently forwarded to `retrieval` where it would produce zero rows. 
            This makes data-miss failures explicit and diagnosable.
        - Deduplication is applied to preserve insertion order.


- **Multi-Intent Queries**
    - **Problem:** 
        Users can ask questions that span multiple intents in a single message, for example: "What is Building 180's revenue in 2024, and which property had the highest profit overall, and what does NOI mean?" 
        A linear single-intent pipeline must either pick one intent and discard the rest, or fail with an `"unknown"` classification.
    - **Solution:** 
        The `supervisor` node was redesigned as a **query decomposer** (in addition to being a query classifier). 
        - Using structured output, it breaks the user's message into up to `MAX_SUB_QUERIES = 3` focused, self-contained sub-questions, each with its own intent label. 
        Each sub-question is written to be independently answerable - entity names, dates, and context are repeated in each sub-question, so no branch relies on another for context.
        - LangGraph's Send API (`_route_supervisor` in `src/graph.py`) fans these sub-queries out as parallel graph branches. 
        LangGraph executes all branches concurrently; each writes its result into `sub_results` using an `Annotated[list, operator.add]` reducer that safely concatenates contributions from all branches.
        - Once all branches complete, the `synthesizer` node receives the merged `sub_results` list and generates a single response with a header for each sub-question, addressing every part of the original query.


- **Handling Missing Data Without Blocking the Entire Response**
    - **Problem:** 
        A compound query such as "What is Building 180's revenue and what are the tenants of Building 999?" contains one valid sub-question and one that will return no data, because Building 999 does not exist.
        In a linear pipeline, an empty retrieval result would have routed the entire request to a generic fallback, losing the valid answer entirely.
    - **Solution:** 
        - Under the map-reduce architecture, each sub-query is processed independently in its own `process_subquery` branch. 
        - When `retrieval` returns zero rows, the branch records `success=False` and the `retrieval`'s descriptive `error` string in its `sub_results` item and returns normally (and no routing change occurs). 
        - The `synthesizer` node receives the full `sub_results` list and is prompted to:
            - Answer the successful sub-questions with the data result.
            - Acknowledge each failed sub-question and suggest how the user might rephrase.
        - This produces a partial but informative response rather than an all-or-nothing response.
        - The `fallback` node is reserved for the edge case where the `supervisor` itself fails to produce any sub-queries.


- **Dynamic Group-By Dimension for the Calculator**
    - **Problem:** 
        Comparison and P&L queries are often not simply about totals per property. 
        Users can also ask about tenants, ledgers, and time periods.
        With a fixed `property_name` grouping axis, the calculator could only answer the per-property variant, forcing the `synthesizer` to produce a generic answer that ignored the requested granularity.
    - **Solution:** 
        - `extractor.py`: 
            - A `group_by` field was added to the `Entities` Pydantic model (default `"property"`). 
            - The system prompt lists the valid dimension values and instructs the LLM to set this field when the user explicitly requests a different grouping axis.
        - `calculator.py`: 
            - Resolves `group_by` to the correct DataFrame column via a `_GROUP_COL_MAP` dict and applies it as follows:
                - **Comparison intent**: always groups by `group_col`. 
                The result dict includes a `"grouped_by"` key so the `synthesizer` can label the axis correctly in its response.
                - **P&L intent**: scalar totals are always computed and always present in the result. 
                When `group_by` is not `"property"`, an additional `"by_group"` dict and `"grouped_by"` key are appended to the result. 
                This additive design ensures the `synthesizer` receives aggregate totals regardless of the requested breakdown.
                - **Asset details and fallback**: unaffected by `group_by`.
        - Backward compatibility is preserved by defaulting to `"property"` when the `group_by` key is absent, empty, or `None`.

---


## Future Work

- **Clarification Loops (Multi-Turn Conversational Memory)**
    - **Current:** 
    The pipeline is a single-pass Directed Acyclic Graph (DAG). 
    If the data is missing, `fallback` generates an error message and the graph ends. 
    - **Enhancement:** 
    LangGraph requires a checkpointer (memory) and a Human-in-the-loop (interrupt) mechanism to actually ask the user a clarifying question, receive their response, and resume the graph rather than starting over from scratch.

- **Metric-Aware & Dynamic Calculation**
    - **Current:** 
    The `Entities` schema hardcodes metric to either "profit", "revenue", or "expenses", and `calculator` uses simple if/else branches for these.
    If a user asks for "Net Operating Income" or "Average Monthly Expenses" for example, the Python logic drops it.
    - **Enhancement:** 
    The calculator could be enhanced into a ReAct agent or a Tool-calling node that generates dynamic Pandas queries against the dataset instead of relying on hardcoded if/else statements.


---


## Testing

The test script is self-contained - it requires no pytest installation and mirrors the graph's pipeline paths.

```bash
# Full test suite - requires ANTHROPIC_API_KEY and network access
python tests.py

# Non-LLM tests only - runs in seconds, no API key required
python tests.py --fast
```

**Test structure:**

| Section | What is tested |
|---|---|
| Data Loader | DataFrame shape, column presence, `lru_cache` memoisation |
| Fuzzy resolve | Exact match, typo correction, below-cutoff rejection, deduplication, tenant match, empty input |
| Graph routing | `_route_supervisor` fan-out logic, empty sub-queries → `fallback`, Send object construction |
| `retrieval` node | Property / tenant / month / year filters, empty result, known format edge cases |
| `calculator` node | Result shape and numeric accuracy for all three intents; metric-aware keys; missing metric key safe fallback; group_by dimension |
| `supervisor` (LLM) | Each intent label, compound query decomposition ≤ 3 sub-queries, garbage input always yields valid intents |
| `extractor` (LLM) | Property, tenant, year extraction; metric default; fuzzy typo correction end-to-end; entities dict schema |
| `synthesizer` (LLM) | Non-empty string, currency formatting, general knowledge pass-through, multi-result synthesis, partial failure handled gracefully |
| `fallback` (LLM) | Helpful response when `supervisor` produces no sub-queries; with and without error field |
| Integration (LLM) | Full `graph.invoke()` for all pipeline paths; multi-intent decomposition; nonexistent property graceful response; fuzzy typo end-to-end; state schema completeness; response always populated |
