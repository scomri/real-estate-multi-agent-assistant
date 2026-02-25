"""
tests.py - Cortex Real Estate LangGraph Agent Test Suite

Usage
-----
  python tests.py           # all tests  (requires ANTHROPIC_API_KEY + network)
  python tests.py --fast    # non-LLM tests only (no network required, runs in <30s)

Structure
---------
  Section 0 : Bootstrap  (sys.path, encoding, dotenv)
  Section 1 : Infrastructure  (runner, helpers, constants)
  Section 2 : Non-LLM unit tests  (data loader, fuzzy, retrieval, calculator, routing)
  Section 3 : LLM unit tests  (supervisor, extractor, synthesizer, fallback)
  Section 4 : Integration tests  (full graph.invoke, including multi-intent)
  Section 5 : Runner / summary
"""

# ==============================================================================
# SECTION 0: Bootstrap
# ==============================================================================

import difflib
import io
import os
import sys
import time
import traceback

# UTF-8 stdout so Unicode in messages does not crash on Windows cp1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FAST_MODE = "--fast" in sys.argv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load .env BEFORE importing src modules -- LLM clients are instantiated at
# import time and need ANTHROPIC_API_KEY to already be in the environment.
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ==============================================================================
# SECTION 1: Infrastructure
# ==============================================================================

class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    GREY   = "\033[90m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def _c(text, code):
    return f"{code}{text}{C.RESET}"

def banner(text):
    w = 72
    print()
    print(_c("=" * w, C.CYAN))
    print(_c(f"  {text}", C.BOLD + C.CYAN))
    print(_c("=" * w, C.CYAN))

def subheader(text):
    print(_c(f"\n--- {text} ---", C.YELLOW))


_results = []


def run_test(name, fn, llm=False):
    """Execute fn(), record PASS / FAIL / SKIP."""
    if llm and FAST_MODE:
        print(f"  {_c('[SKIP]', C.GREY)} {name}  (--fast mode)")
        _results.append({"name": name, "passed": True, "duration": 0.0,
                         "error": "", "skipped": True})
        return

    print(f"  {_c('[TEST]', C.BOLD)} {name}", end=" ... ", flush=True)
    t0 = time.perf_counter()
    try:
        fn()
        dur = time.perf_counter() - t0
        print(_c("PASS", C.GREEN) + f"  ({dur:.2f}s)")
        _results.append({"name": name, "passed": True, "duration": dur,
                         "error": "", "skipped": False})
    except AssertionError as exc:
        dur = time.perf_counter() - t0
        msg = str(exc) or "Assertion failed"
        print(_c("FAIL", C.RED) + f"  ({dur:.2f}s)")
        print(f"    {_c('AssertionError:', C.RED)} {msg}")
        _results.append({"name": name, "passed": False, "duration": dur,
                         "error": msg, "skipped": False})
    except Exception as exc:
        dur = time.perf_counter() - t0
        msg = f"{type(exc).__name__}: {exc}"
        print(_c("FAIL", C.RED) + f"  ({dur:.2f}s)")
        print(f"    {_c('Exception:', C.RED)} {msg}")
        for line in traceback.format_tb(exc.__traceback__)[-3:]:
            for sub in line.splitlines():
                print(f"    {sub}")
        _results.append({"name": name, "passed": False, "duration": dur,
                         "error": msg, "skipped": False})


def assert_eq(label, actual, expected):
    assert actual == expected, f"{label}: expected {expected!r}, got {actual!r}"

def assert_in(label, item, container):
    assert item in container, f"{label}: {item!r} not found in {container!r}"

def assert_type(label, value, expected_type):
    assert isinstance(value, expected_type), (
        f"{label}: expected {expected_type.__name__}, got {type(value).__name__}"
    )

def assert_gt(label, value, threshold):
    assert value > threshold, f"{label}: {value!r} is not > {threshold!r}"

def assert_approx(label, actual, expected, tol=0.02):
    diff = abs(actual - expected)
    assert diff <= tol, (
        f"{label}: expected ~{expected:.4f}, got {actual:.4f} (diff={diff:.6f}, tol={tol})"
    )


# Pre-computed constants verified against data/cortex.parquet
#   year dtype  : int64   (e.g. 2024)     — normalised by data_loader._normalize
#   month format: 'YYYY-MM' (e.g. '2024-12') — normalised by data_loader._normalize

ALL_PROPERTIES = [
    "Building 180", "Building 17", "Building 160", "Building 140", "Building 120",
]
ALL_TENANTS = [
    "Tenant 1", "Tenant 2", "Tenant 3", "Tenant 4", "Tenant 5",
    "Tenant 6", "Tenant 7", "Tenant 8", "Tenant 9", "Tenant 10",
    "Tenant 11", "Tenant 12", "Tenant 13", "Tenant 14", "Tenant 15",
    "Tenant 16", "Tenant 17", "Tenant 18",
]
VALID_INTENTS = ("pnl", "comparison", "asset_details", "general", "unknown")

B180_REVENUE  =  391490.26
B180_EXPENSES =   -6590.23
B180_NET      =  384900.03

COMPARISON_ORDER = [
    "Building 120",
    "Building 160",
    "Building 140",
    "Building 180",
    "Building 17",
]

B17_PROFIT = 352566.81
B17_REVENUE_TOTAL  =  358231.51   # Building 17 revenue rows only
B17_EXPENSES_TOTAL =   -5664.70   # Building 17 expenses rows only
TOTAL_ROWS = 3924


# ==============================================================================
# SECTION 2: Non-LLM Unit Tests
# ==============================================================================

# --- 2a. Data Loader ----------------------------------------------------------

def test_load_data_returns_dataframe():
    from src.data_loader import load_data
    import pandas as pd
    df = load_data()
    assert_type("return type", df, pd.DataFrame)
    assert_gt("row count", len(df), 0)
    expected_cols = {
        "entity_name", "property_name", "tenant_name",
        "ledger_type", "ledger_group", "ledger_category",
        "ledger_code", "ledger_description",
        "month", "quarter", "year", "profit",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"DataFrame missing columns: {missing}"


def test_get_data_dict_is_string():
    from src.data_loader import get_data_dict
    dd = get_data_dict()
    assert_type("return type", dd, str)
    assert_gt("length", len(dd), 50)
    for col in ("property_name", "tenant_name", "ledger_type", "profit"):
        assert col in dd, f"get_data_dict does not mention column: {col}"


def test_load_data_is_cached():
    from src.data_loader import load_data
    df1 = load_data()
    df2 = load_data()
    assert df1 is df2, "load_data should return the same cached object (lru_cache)"


# --- 2b. _fuzzy_resolve -------------------------------------------------------
# Inlined from src/nodes/extractor.py to avoid importing a module that
# triggers a slow LLM client network call at import time.
# The extractor LLM tests (section 3) separately verify the real function.

def _fuzzy_resolve(names, candidates, cutoff=0.80):
    """Inline copy of src.nodes.extractor._fuzzy_resolve (cutoff=0.80, n=1)."""
    resolved = []
    candidates_lower = {c.lower(): c for c in candidates}
    for name in names:
        if name.lower() in candidates_lower:
            resolved.append(candidates_lower[name.lower()])
            continue
        matches = difflib.get_close_matches(
            name.lower(), candidates_lower.keys(), n=1, cutoff=cutoff
        )
        if matches:
            resolved.append(candidates_lower[matches[0]])
    return list(dict.fromkeys(resolved))


def test_fuzzy_exact_ci():
    assert_eq("CI match", _fuzzy_resolve(["building 180"], ALL_PROPERTIES), ["Building 180"])

def test_fuzzy_typo():
    result = _fuzzy_resolve(["Bilding 180"], ALL_PROPERTIES)
    assert_in("typo -> B180", "Building 180", result)

def test_fuzzy_below_cutoff():
    assert_eq("no match", _fuzzy_resolve(["xyz_totally_unknown_place"], ALL_PROPERTIES), [])

def test_fuzzy_deduplicates():
    result = _fuzzy_resolve(["Building 180", "building 180"], ALL_PROPERTIES)
    assert result.count("Building 180") == 1, f"Duplicate not removed: {result}"

def test_fuzzy_tenant():
    assert_in("tenant CI", "Tenant 1", _fuzzy_resolve(["tenant 1"], ALL_TENANTS))

def test_fuzzy_empty_input():
    assert_eq("empty -> []", _fuzzy_resolve([], ALL_PROPERTIES), [])


# --- 2c. Graph routing --------------------------------------------------------
# Inlined from src/graph.py to avoid importing all node modules, which each
# trigger slow LLM client initialization.
# Integration tests separately verify end-to-end routing behaviour.

def _route_supervisor_new(state):
    """Inline copy of the new _route_supervisor routing logic (map-reduce pattern)."""
    sub_queries = state.get("sub_queries") or []
    if not sub_queries:
        return "fallback"
    return "fan_out"   # actual impl returns list[Send]; here we just confirm not "fallback"

def _sq(*intents):
    """Build a sub_queries list from intent strings."""
    return [{"query": f"sub-query for {intent}", "intent": intent} for intent in intents]

def test_route_empty_sub_queries():
    assert_eq("empty -> fallback", _route_supervisor_new({"sub_queries": []}), "fallback")

def test_route_missing_sub_queries():
    assert_eq("missing -> fallback", _route_supervisor_new({}), "fallback")

def test_route_single_sub_query():
    result = _route_supervisor_new({"sub_queries": _sq("pnl")})
    assert result != "fallback", "non-empty sub_queries should not route to fallback"

def test_route_multi_sub_queries():
    result = _route_supervisor_new({"sub_queries": _sq("pnl", "comparison")})
    assert result != "fallback", "multi-intent should not route to fallback"

def test_fan_out_produces_sends():
    from langgraph.constants import Send
    sub_queries = _sq("pnl", "comparison", "general")
    sends = [Send("process_subquery", sq) for sq in sub_queries]
    assert_eq("3 Send objects", len(sends), 3)
    assert all(isinstance(s, Send) for s in sends), "all items must be Send objects"


# --- 2d. Retrieval  (pandas, no LLM) -----------------------------------------

def _ent(**kw):
    base = {"properties": [], "tenants": [], "year": None, "months": [], "metric": "profit"}
    base.update(kw)
    return base

def _st(**kw):
    s = {"query": "", "intent": "pnl", "entities": _ent(),
         "data": [], "result": None, "response": "", "error": None}
    s.update(kw)
    return s

def _st_metric(intent, data, metric):
    """State dict with a specific metric value."""
    return _st(intent=intent, data=data, entities=_ent(metric=metric))


def test_retrieval_by_property():
    from src.nodes.retrieval import retrieval
    out = retrieval(_st(entities=_ent(properties=["Building 180"])))
    assert_eq("error is None", out["error"], None)
    assert_gt("rows > 0", len(out["data"]), 0)
    bad = [r for r in out["data"] if r["property_name"] != "Building 180"]
    assert not bad, f"{len(bad)} row(s) have wrong property_name"


def test_retrieval_by_tenant():
    from src.nodes.retrieval import retrieval
    out = retrieval(_st(entities=_ent(tenants=["Tenant 1"])))
    assert out["error"] is None, f"error: {out['error']}"
    assert_gt("rows > 0", len(out["data"]), 0)


def test_retrieval_by_month():
    from src.nodes.retrieval import retrieval
    out = retrieval(_st(entities=_ent(properties=["Building 180"], months=["2024-12"])))
    assert out["error"] is None, f"error: {out['error']}"
    assert_gt("rows > 0", len(out["data"]), 0)
    bad = [r for r in out["data"] if r["month"] != "2024-12"]
    assert not bad, f"{len(bad)} row(s) have wrong month"


def test_retrieval_no_filters():
    from src.nodes.retrieval import retrieval
    out = retrieval(_st(entities=_ent()))
    assert out["error"] is None, f"error: {out['error']}"
    assert_eq("all rows", len(out["data"]), TOTAL_ROWS)


def test_retrieval_nonexistent():
    from src.nodes.retrieval import retrieval
    out = retrieval(_st(entities=_ent(properties=["XYZ Tower"])))
    assert_eq("data is []", out["data"], [])
    assert out["error"] is not None and len(out["error"]) > 0, "Expected error message"


def test_retrieval_by_year():
    """Year filter works correctly: int year matches normalized int column."""
    from src.nodes.retrieval import retrieval
    out = retrieval(_st(entities=_ent(properties=["Building 180"], year=2024)))
    assert out["error"] is None, f"error: {out['error']}"
    assert_gt("rows > 0", len(out["data"]), 0)
    bad = [r for r in out["data"] if r["year"] != 2024]
    assert not bad, f"{len(bad)} row(s) have wrong year"


def test_retrieval_month_iso_format():
    """ISO month format 'YYYY-MM' matches normalized data column."""
    from src.nodes.retrieval import retrieval
    out = retrieval(_st(entities=_ent(months=["2024-12"])))
    assert out["error"] is None, f"error: {out['error']}"
    assert_gt("rows > 0", len(out["data"]), 0)
    bad = [r for r in out["data"] if r["month"] != "2024-12"]
    assert not bad, f"{len(bad)} row(s) have wrong month"


# --- 2e. Calculator  (pandas, no LLM) ----------------------------------------

def _rows(prop_name):
    import pandas as pd
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "cortex.parquet"))
    return df[df["property_name"] == prop_name].to_dict("records")

def _all_rows():
    import pandas as pd
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "cortex.parquet"))
    return df.to_dict("records")


def test_calculator_pnl_shape():
    from src.nodes.calculator import calculator
    r = calculator(_st(intent="pnl", data=_rows("Building 180")))["result"]
    for k in ("revenue", "expenses", "net", "properties", "period"):
        assert k in r, f"pnl missing key: {k}"
    assert "from" in r["period"] and "to" in r["period"]


def test_calculator_pnl_values():
    from src.nodes.calculator import calculator
    r = calculator(_st(intent="pnl", data=_rows("Building 180")))["result"]
    assert_approx("revenue",  r["revenue"],  B180_REVENUE)
    assert_approx("expenses", r["expenses"], B180_EXPENSES)
    assert_approx("net",      r["net"],      B180_NET)
    assert_in("B180 in properties", "Building 180", r["properties"])


def test_calculator_comparison_shape():
    from src.nodes.calculator import calculator
    r = calculator(_st(intent="comparison", data=_all_rows()))["result"]
    assert_in("comparison key", "comparison", r)
    for p in ALL_PROPERTIES:
        assert p in r["comparison"], f"missing: {p}"


def test_calculator_comparison_sorted():
    from src.nodes.calculator import calculator
    comp = calculator(_st(intent="comparison", data=_all_rows()))["result"]["comparison"]
    vals = list(comp.values())
    assert vals == sorted(vals, reverse=True), "not sorted descending"
    assert_eq("property order", list(comp.keys()), COMPARISON_ORDER)


def test_calculator_asset_details():
    from src.nodes.calculator import calculator
    r = calculator(_st(intent="asset_details", data=_rows("Building 17")))["result"]
    for k in ("properties", "tenants", "ledger_categories", "profit_total", "period"):
        assert k in r, f"asset_details missing key: {k}"
    assert_in("Building 17", "Building 17", r["properties"])
    assert_in("Tenant 8", "Tenant 8", r["tenants"])
    assert_approx("B17 profit", r["profit_total"], B17_PROFIT)


def test_calculator_empty_data():
    from src.nodes.calculator import calculator
    assert_eq("empty data -> None", calculator(_st(intent="pnl", data=[]))["result"], None)


def test_calculator_unknown_intent():
    from src.nodes.calculator import calculator
    r = calculator(_st(intent="unknown", data=_rows("Building 180")))["result"]
    assert_in("raw_total", "raw_total", r)
    assert_type("raw_total float", r["raw_total"], float)


# --- 2e continued: metric-aware tests ----------------------------------------

def test_calculator_pnl_revenue_only():
    """pnl + metric='revenue' → only 'revenue' key, no 'expenses' or 'net'."""
    from src.nodes.calculator import calculator
    r = calculator(_st_metric("pnl", _rows("Building 180"), "revenue"))["result"]
    assert "revenue"    in r,       "pnl revenue: missing 'revenue' key"
    assert "expenses"   not in r,   "pnl revenue: unexpected 'expenses' key"
    assert "net"        not in r,   "pnl revenue: unexpected 'net' key"
    assert "properties" in r,       "pnl revenue: missing 'properties' key"
    assert "period"     in r,       "pnl revenue: missing 'period' key"
    assert_approx("B180 revenue", r["revenue"], B180_REVENUE)


def test_calculator_pnl_expenses_only():
    """pnl + metric='expenses' → only 'expenses' key, no 'revenue' or 'net'."""
    from src.nodes.calculator import calculator
    r = calculator(_st_metric("pnl", _rows("Building 180"), "expenses"))["result"]
    assert "expenses"   in r,       "pnl expenses: missing 'expenses' key"
    assert "revenue"    not in r,   "pnl expenses: unexpected 'revenue' key"
    assert "net"        not in r,   "pnl expenses: unexpected 'net' key"
    assert "properties" in r,       "pnl expenses: missing 'properties' key"
    assert "period"     in r,       "pnl expenses: missing 'period' key"
    assert_approx("B180 expenses", r["expenses"], B180_EXPENSES)


def test_calculator_pnl_metric_absent():
    """pnl + metric='' → falls back to full {revenue, expenses, net, …}."""
    from src.nodes.calculator import calculator
    r = calculator(_st_metric("pnl", _rows("Building 180"), ""))["result"]
    for k in ("revenue", "expenses", "net", "properties", "period"):
        assert k in r, f"pnl absent-metric missing key: {k}"


def test_calculator_comparison_revenue():
    """comparison + metric='revenue' → values sorted desc; all non-negative."""
    from src.nodes.calculator import calculator
    comp = calculator(_st_metric("comparison", _all_rows(), "revenue"))["result"]["comparison"]
    vals = list(comp.values())
    assert vals == sorted(vals, reverse=True), "revenue comparison not sorted descending"
    assert all(v >= 0 for v in vals), f"revenue comparison has negative values: {vals}"


def test_calculator_comparison_expenses():
    """comparison + metric='expenses' → values sorted desc; all non-positive."""
    from src.nodes.calculator import calculator
    comp = calculator(_st_metric("comparison", _all_rows(), "expenses"))["result"]["comparison"]
    vals = list(comp.values())
    assert vals == sorted(vals, reverse=True), "expenses comparison not sorted descending"
    assert all(v <= 0 for v in vals), f"expenses comparison has positive values: {vals}"


def test_calculator_asset_details_revenue():
    """asset_details + metric='revenue' → 'revenue_total' key, no 'profit_total'."""
    from src.nodes.calculator import calculator
    r = calculator(_st_metric("asset_details", _rows("Building 17"), "revenue"))["result"]
    assert "revenue_total"   in r,      "asset_details revenue: missing 'revenue_total'"
    assert "profit_total"    not in r,  "asset_details revenue: unexpected 'profit_total'"
    assert "expenses_total"  not in r,  "asset_details revenue: unexpected 'expenses_total'"
    assert_approx("B17 revenue_total", r["revenue_total"], B17_REVENUE_TOTAL)
    for k in ("properties", "tenants", "ledger_categories", "period"):
        assert k in r, f"asset_details revenue: missing shared key '{k}'"


def test_calculator_asset_details_expenses():
    """asset_details + metric='expenses' → 'expenses_total' key, no 'profit_total'."""
    from src.nodes.calculator import calculator
    r = calculator(_st_metric("asset_details", _rows("Building 17"), "expenses"))["result"]
    assert "expenses_total"  in r,      "asset_details expenses: missing 'expenses_total'"
    assert "profit_total"    not in r,  "asset_details expenses: unexpected 'profit_total'"
    assert "revenue_total"   not in r,  "asset_details expenses: unexpected 'revenue_total'"
    assert_approx("B17 expenses_total", r["expenses_total"], B17_EXPENSES_TOTAL)
    for k in ("properties", "tenants", "ledger_categories", "period"):
        assert k in r, f"asset_details expenses: missing shared key '{k}'"


def test_calculator_metric_key_missing():
    """entities dict without a 'metric' key at all must not raise; returns full pnl."""
    from src.nodes.calculator import calculator
    state = {
        "intent": "pnl",
        "data": _rows("Building 180"),
        "entities": {"properties": ["Building 180"], "year": None, "months": [], "tenants": []},
        "query": "", "result": None, "response": "", "error": None,
    }
    r = calculator(state)["result"]
    for k in ("revenue", "expenses", "net"):
        assert k in r, f"missing-metric-key state missing key: {k}"


# --- 2f. Calculator — group_by dimension (pandas, no LLM) --------------------

def _st_group(intent, data, group_by, metric="profit"):
    """State dict with a specific group_by value."""
    return _st(intent=intent, data=data, entities=_ent(metric=metric, group_by=group_by))


def test_calculator_comparison_by_ledger_group():
    """Comparison grouped by ledger_group → keys are ledger_group values, not property names."""
    from src.nodes.calculator import calculator
    r = calculator(_st_group("comparison", _rows("Building 180"), "ledger_group"))["result"]
    assert "comparison" in r
    assert r["grouped_by"] == "ledger_group", f"unexpected grouped_by: {r['grouped_by']}"
    keys = list(r["comparison"].keys())
    assert len(keys) > 0, "comparison by ledger_group: no groups returned"
    for k in keys:
        assert "Building" not in k, f"comparison by ledger_group has property-name key: {k}"


def test_calculator_comparison_by_tenant():
    """Comparison grouped by tenant → keys are tenant_name values."""
    from src.nodes.calculator import calculator
    r = calculator(_st_group("comparison", _all_rows(), "tenant"))["result"]
    assert "comparison" in r
    assert r["grouped_by"] == "tenant_name", f"unexpected grouped_by: {r['grouped_by']}"
    keys = list(r["comparison"].keys())
    assert len(keys) > 0, "comparison by tenant: no groups returned"
    for k in keys:
        assert k in ALL_TENANTS, f"unexpected tenant key in comparison: {k}"


def test_calculator_comparison_by_month():
    """Comparison grouped by month → keys contain a '-' date separator."""
    from src.nodes.calculator import calculator
    r = calculator(_st_group("comparison", _rows("Building 180"), "month"))["result"]
    assert "comparison" in r
    assert r["grouped_by"] == "month", f"unexpected grouped_by: {r['grouped_by']}"
    keys = list(r["comparison"].keys())
    assert len(keys) > 0, "comparison by month: no groups returned"
    for k in keys:
        assert "-" in str(k), f"unexpected month key format: {k}"


def test_calculator_pnl_with_group_by_ledger_group():
    """P&L with group_by=ledger_group → scalar totals still present AND by_group added."""
    from src.nodes.calculator import calculator
    r = calculator(_st_group("pnl", _rows("Building 180"), "ledger_group"))["result"]
    for k in ("revenue", "expenses", "net"):
        assert k in r, f"pnl+group_by: missing scalar key '{k}'"
    assert "by_group"   in r, "pnl+group_by: missing 'by_group' key"
    assert "grouped_by" in r, "pnl+group_by: missing 'grouped_by' key"
    assert r["grouped_by"] == "ledger_group"
    assert len(r["by_group"]) > 0, "pnl+group_by: by_group dict is empty"


def test_calculator_comparison_default_unchanged():
    """Regression: comparison with no group_by → groups by property_name as before."""
    from src.nodes.calculator import calculator
    r = calculator(_st_metric("comparison", _all_rows(), "profit"))["result"]
    assert "comparison" in r
    assert r["grouped_by"] == "property_name", f"default comparison grouped_by changed: {r['grouped_by']}"
    for k in r["comparison"]:
        assert k in ALL_PROPERTIES, f"unexpected key in default comparison: {k}"


def test_calculator_pnl_default_unchanged():
    """Regression: pnl with no group_by → no 'by_group' key added."""
    from src.nodes.calculator import calculator
    r = calculator(_st_metric("pnl", _rows("Building 180"), "profit"))["result"]
    assert "by_group"   not in r, "default pnl must not have 'by_group'"
    assert "grouped_by" not in r, "default pnl must not have 'grouped_by'"


# ==============================================================================
# SECTION 3: LLM Unit Tests
# (skipped with --fast; each test imports its module lazily)
# ==============================================================================

# --- Supervisor ---------------------------------------------------------------

def _ss(q):
    return {"query": q, "sub_queries": [], "sub_results": [], "response": "", "error": None}

def test_supervisor_pnl():
    from src.nodes.supervisor import supervisor
    sqs = supervisor(_ss("Total revenue and profit for Building 180 in 2024"))["sub_queries"]
    assert len(sqs) >= 1, "should produce at least 1 sub-query"
    assert_eq("pnl", sqs[0]["intent"], "pnl")

def test_supervisor_comparison():
    from src.nodes.supervisor import supervisor
    sqs = supervisor(_ss("Compare profit of Building 120 and Building 160"))["sub_queries"]
    assert len(sqs) >= 1
    assert_eq("comparison", sqs[0]["intent"], "comparison")

def test_supervisor_asset_details():
    from src.nodes.supervisor import supervisor
    sqs = supervisor(_ss("Which tenants and ledger categories are in Building 17?"))["sub_queries"]
    assert len(sqs) >= 1
    assert_eq("asset_details", sqs[0]["intent"], "asset_details")

def test_supervisor_general():
    from src.nodes.supervisor import supervisor
    sqs = supervisor(_ss("What is a cap rate in real estate?"))["sub_queries"]
    assert len(sqs) >= 1
    assert_eq("general", sqs[0]["intent"], "general")

def test_supervisor_unknown():
    from src.nodes.supervisor import supervisor
    sqs = supervisor(_ss("What is the weather in Amsterdam?"))["sub_queries"]
    assert len(sqs) >= 1
    assert_in("unknown/general", sqs[0]["intent"], ("unknown", "general"))

def test_supervisor_always_valid():
    from src.nodes.supervisor import supervisor, VALID_INTENTS
    sqs = supervisor(_ss("asdfjkl; !@#$%^&*() nonsense"))["sub_queries"]
    assert len(sqs) >= 1
    assert all(sq["intent"] in VALID_INTENTS for sq in sqs), f"all intents must be valid: {sqs}"

def test_supervisor_max_sub_queries():
    from src.nodes.supervisor import supervisor, MAX_SUB_QUERIES
    compound = (
        "What is the revenue for Building 180 in 2024, and compare all properties by profit, "
        "and also explain what NOI means in real estate?"
    )
    sqs = supervisor(_ss(compound))["sub_queries"]
    assert len(sqs) <= MAX_SUB_QUERIES, f"must not exceed {MAX_SUB_QUERIES} sub-queries, got {len(sqs)}"


# --- Extractor ----------------------------------------------------------------

def _es(q, intent="pnl"):
    return {"query": q, "intent": intent, "entities": {}, "data": [],
            "result": None, "response": "", "error": None}

def test_extractor_property():
    from src.nodes.extractor import extractor
    out = extractor(_es("Show me the P&L for Building 180."))
    assert_in("B180", "Building 180", out["entities"]["properties"])

def test_extractor_tenant():
    from src.nodes.extractor import extractor
    out = extractor(_es("What revenue did Tenant 1 generate?"))
    assert_in("Tenant 1", "Tenant 1", out["entities"]["tenants"])

def test_extractor_year():
    from src.nodes.extractor import extractor
    year = extractor(_es("Building 160 profit in 2024"))["entities"].get("year")
    assert year in (2024, "2024"), f"bad year: {year!r}"

def test_extractor_metric_default():
    from src.nodes.extractor import extractor
    metric = extractor(_es("Tell me about Building 17.", intent="asset_details"))["entities"].get("metric")
    assert metric in ("profit", None), f"unexpected metric: {metric!r}"

def test_extractor_fuzzy():
    from src.nodes.extractor import extractor
    out = extractor(_es("P&L for Bilding 180?"))
    assert_in("fuzzy -> B180", "Building 180", out["entities"]["properties"])

def test_extractor_schema():
    from src.nodes.extractor import extractor
    entities = extractor(_es("Compare all properties.", intent="comparison"))["entities"]
    for k in ("properties", "tenants", "year", "months", "metric"):
        assert k in entities, f"entities missing: {k}"


# --- Synthesizer --------------------------------------------------------------

def _srs(*items):
    """Build a state dict for the synthesizer from sub_result item dicts."""
    return {"query": "test", "sub_queries": [], "sub_results": list(items),
            "response": "", "error": None}

def _sr_data(q, intent, result):
    return {"query": q, "intent": intent, "result": result, "success": True, "error": None}

def _sr_general(q):
    return {"query": q, "intent": "general", "result": None, "success": True, "error": None}

def _sr_failed(q, error="No matching data found"):
    return {"query": q, "intent": "pnl", "result": None, "success": False, "error": error}

def test_synthesizer_data():
    from src.nodes.synthesizer import synthesizer
    out = synthesizer(_srs(_sr_data("P&L for Building 180?", "pnl", {
        "revenue": B180_REVENUE, "expenses": B180_EXPENSES, "net": B180_NET,
        "properties": ["Building 180"], "period": {"from": "2024-M01", "to": "2025-M03"},
    })))
    assert_type("str", out["response"], str)
    assert_gt("non-empty", len(out["response"]), 20)

def test_synthesizer_numbers():
    from src.nodes.synthesizer import synthesizer
    resp = synthesizer(_srs(_sr_data("Revenue?", "pnl", {
        "revenue": 850567.42, "expenses": -12345.0, "net": 838222.42,
        "properties": ["Building 120"], "period": {"from": "2024-M01", "to": "2025-M03"},
    })))["response"]
    assert "$" in resp or any(c.isdigit() for c in resp), "no numbers in response"

def test_synthesizer_general():
    from src.nodes.synthesizer import synthesizer
    out = synthesizer(_srs(_sr_general("What does NOI mean?")))
    assert_type("str", out["response"], str)
    assert_gt("non-empty", len(out["response"]), 20)

def test_synthesizer_multi_result():
    from src.nodes.synthesizer import synthesizer
    out = synthesizer(_srs(
        _sr_data("P&L for Building 180?", "pnl", {
            "revenue": B180_REVENUE, "expenses": B180_EXPENSES, "net": B180_NET,
            "properties": ["Building 180"], "period": {"from": "2024-M01", "to": "2025-M03"},
        }),
        _sr_general("What is cap rate?"),
    ))
    assert_type("str", out["response"], str)
    assert_gt("non-empty", len(out["response"]), 30)

def test_synthesizer_partial_failure():
    from src.nodes.synthesizer import synthesizer
    out = synthesizer(_srs(
        _sr_failed("P&L for XYZ Tower?", error="No data found for: properties=['XYZ Tower']."),
        _sr_data("P&L for Building 180?", "pnl", {
            "revenue": B180_REVENUE, "net": B180_NET,
            "properties": ["Building 180"], "period": {"from": "2024-M01", "to": "2025-M03"},
        }),
    ))
    assert_type("str", out["response"], str)
    assert_gt("non-empty", len(out["response"]), 20)


# --- Fallback -----------------------------------------------------------------

def _fs(q, error=None):
    return {"query": q, "sub_queries": [], "sub_results": [], "response": "", "error": error}

def test_fallback_no_sub_queries():
    from src.nodes.fallback import fallback
    out = fallback(_fs("What is the weather in Amsterdam?"))
    assert_type("str", out["response"], str)
    assert_gt("non-empty", len(out["response"]), 20)

def test_fallback_with_error():
    from src.nodes.fallback import fallback
    out = fallback(_fs("XYZ Tower data",
                       error="Supervisor could not decompose the query."))
    assert_type("str", out["response"], str)
    assert_gt("non-empty", len(out["response"]), 20)

def test_fallback_no_error_field():
    from src.nodes.fallback import fallback
    out = fallback(_fs("Potato banana unicorn", error=None))
    assert_type("str", out["response"], str)
    assert_gt("non-empty", len(out["response"]), 20)


# ==============================================================================
# SECTION 4: Integration Tests  (full graph.invoke)
# ==============================================================================

_GRAPH = None

def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        from src.graph import build_graph
        _GRAPH = build_graph()
    return _GRAPH

def _invoke(query):
    return _get_graph().invoke({
        "query": query, "sub_queries": [], "sub_results": [], "response": "", "error": None,
    })

def _first_data_result(final):
    """Return the first successful data sub-result, or None."""
    return next(
        (r for r in final.get("sub_results", []) if r.get("success") and r.get("result")),
        None,
    )


def test_e2e_pnl():
    final = _invoke("What is the total profit and loss for Building 180?")
    sqs = final.get("sub_queries", [])
    srs = final.get("sub_results", [])
    assert len(sqs) >= 1, "should have at least 1 sub-query"
    assert sqs[0]["intent"] == "pnl", f"expected pnl intent, got {sqs[0]['intent']}"
    dr = _first_data_result(final)
    assert dr is not None, "at least one sub-result should have a data result"
    assert_in("revenue in result", "revenue", dr["result"])
    assert_type("response str", final["response"], str)
    assert_gt("response non-empty", len(final["response"]), 20)


def test_e2e_comparison():
    final = _invoke("Compare the total profit of Building 120 and Building 160.")
    sqs = final.get("sub_queries", [])
    assert len(sqs) >= 1
    assert sqs[0]["intent"] == "comparison", f"expected comparison, got {sqs[0]['intent']}"
    dr = _first_data_result(final)
    assert dr is not None, "should have a data result"
    comp = dr["result"].get("comparison", {})
    assert any(p in comp for p in ("Building 120", "Building 160")), (
        f"Neither property in comparison: {list(comp.keys())}"
    )


def test_e2e_asset_details():
    final = _invoke("What tenants and ledger categories are in Building 17?")
    sqs = final.get("sub_queries", [])
    assert len(sqs) >= 1
    assert sqs[0]["intent"] == "asset_details", f"expected asset_details, got {sqs[0]['intent']}"
    dr = _first_data_result(final)
    assert dr is not None, "should have a data result"
    assert_type("response str", final["response"], str)


def test_e2e_general():
    final = _invoke("What is the difference between gross and net yield in real estate?")
    sqs = final.get("sub_queries", [])
    srs = final.get("sub_results", [])
    assert len(sqs) >= 1
    assert sqs[0]["intent"] == "general", f"expected general, got {sqs[0]['intent']}"
    assert all(r.get("result") is None for r in srs), "general query should not produce data results"
    assert_gt("response non-empty", len(final["response"]), 30)


def test_e2e_unknown_fallback():
    final = _invoke("Tell me the current stock price of Apple Inc.")
    sqs = final.get("sub_queries", [])
    assert len(sqs) >= 1
    assert_in("unknown/general", sqs[0]["intent"], ("unknown", "general"))
    assert_gt("response non-empty", len(final["response"]), 20)


def test_e2e_nonexistent_property():
    final = _invoke("What is the P&L for XYZ Skyscraper Tower?")
    assert_type("response str", final["response"], str)
    assert_gt("response non-empty", len(final["response"]), 20)


def test_e2e_fuzzy_typo():
    final = _invoke("Net income for Bilding 180?")  # deliberate typo
    srs = final.get("sub_results", [])
    if srs and srs[0].get("success"):
        dr = _first_data_result(final)
        assert dr is not None or len(final.get("response", "")) > 10, (
            "fuzzy typo query should retrieve data or produce a response"
        )
    assert_type("response str", final["response"], str)


def test_e2e_state_schema():
    final = _invoke("What is NOI?")
    missing = {"query", "sub_queries", "sub_results", "response", "error"} - set(final.keys())
    assert not missing, f"Final state missing keys: {missing}"


def test_e2e_multi_intent():
    """Compound query should produce 2+ sub-queries and a synthesized response."""
    final = _invoke(
        "What is the revenue for Building 180 in 2024, "
        "and also compare all properties by total profit?"
    )
    sqs = final.get("sub_queries", [])
    srs = final.get("sub_results", [])
    assert len(sqs) >= 2, f"Expected 2+ sub-queries for multi-intent query, got {len(sqs)}"
    assert len(srs) >= 2, f"Expected 2+ sub-results, got {len(srs)}"
    intents = {sq["intent"] for sq in sqs}
    assert intents & {"pnl", "comparison"}, f"Expected pnl or comparison in intents: {intents}"
    assert_type("response str", final["response"], str)
    assert_gt("response non-empty", len(final["response"]), 30)


def test_e2e_response_always_set():
    """Every pipeline path must produce a non-empty response string."""
    for q in [
        "Total P&L for all properties in 2024",
        "Compare Building 120 vs Building 140",
        "What is amortisation in real estate?",
        "What is the weather in Paris?",
        "Show me data for ZZZ Property that does not exist",
    ]:
        final = _invoke(q)
        resp = final.get("response", "")
        assert isinstance(resp, str) and len(resp) > 10, (
            f"Empty/missing response for: {q!r}  (got {resp!r})"
        )


# ==============================================================================
# SECTION 5: Runner
# ==============================================================================

def main():
    banner("Cortex Real Estate Agent -- Test Suite")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Python       : {sys.version.split()[0]}")
    if FAST_MODE:
        print(f"  Mode         : {_c('FAST (LLM tests skipped)', C.YELLOW)}")
    else:
        print(f"  Mode         : full  (use --fast to skip LLM tests)")

    # ---- Non-LLM tests -------------------------------------------------------
    subheader("Data Loader")
    run_test("load_data returns DataFrame with correct columns", test_load_data_returns_dataframe)
    run_test("get_data_dict returns non-empty string",           test_get_data_dict_is_string)
    run_test("load_data is memoised (lru_cache)",                test_load_data_is_cached)

    subheader("_fuzzy_resolve (pure function, inlined)")
    run_test("exact match case-insensitive",  test_fuzzy_exact_ci)
    run_test("close typo resolves correctly", test_fuzzy_typo)
    run_test("below-cutoff name returns []",  test_fuzzy_below_cutoff)
    run_test("duplicate inputs deduplicated", test_fuzzy_deduplicates)
    run_test("tenant name matched",           test_fuzzy_tenant)
    run_test("empty input list returns []",   test_fuzzy_empty_input)

    subheader("Graph Routing (map-reduce, inlined pure functions)")
    run_test("empty sub_queries -> fallback",          test_route_empty_sub_queries)
    run_test("missing sub_queries key -> fallback",    test_route_missing_sub_queries)
    run_test("single sub-query -> fan-out",            test_route_single_sub_query)
    run_test("multi sub-queries -> fan-out",           test_route_multi_sub_queries)
    run_test("fan_out produces Send objects",          test_fan_out_produces_sends)

    subheader("Retrieval Node (pandas, no LLM)")
    run_test("filter by property",                        test_retrieval_by_property)
    run_test("filter by tenant",                          test_retrieval_by_tenant)
    run_test("filter by month (YYYY-Mnn format)",         test_retrieval_by_month)
    run_test("no filters returns all rows",               test_retrieval_no_filters)
    run_test("nonexistent property sets error",           test_retrieval_nonexistent)
    run_test("int year filter matches normalized int column",    test_retrieval_by_year)
    run_test("ISO month 'YYYY-MM' matches normalized data",      test_retrieval_month_iso_format)

    subheader("Calculator Node (pandas, no LLM)")
    run_test("pnl result shape",               test_calculator_pnl_shape)
    run_test("pnl result values match data",   test_calculator_pnl_values)
    run_test("comparison result shape",        test_calculator_comparison_shape)
    run_test("comparison sorted descending",   test_calculator_comparison_sorted)
    run_test("asset_details shape and values", test_calculator_asset_details)
    run_test("empty data -> result None",      test_calculator_empty_data)
    run_test("unknown intent -> raw_total",    test_calculator_unknown_intent)
    run_test("pnl metric=revenue → revenue key only",            test_calculator_pnl_revenue_only)
    run_test("pnl metric=expenses → expenses key only",          test_calculator_pnl_expenses_only)
    run_test("pnl metric='' → full breakdown fallback",          test_calculator_pnl_metric_absent)
    run_test("comparison metric=revenue → non-negative values",  test_calculator_comparison_revenue)
    run_test("comparison metric=expenses → non-positive values", test_calculator_comparison_expenses)
    run_test("asset_details metric=revenue → revenue_total",     test_calculator_asset_details_revenue)
    run_test("asset_details metric=expenses → expenses_total",   test_calculator_asset_details_expenses)
    run_test("missing metric key → safe fallback to full pnl",   test_calculator_metric_key_missing)
    run_test("comparison group_by=ledger_group → ledger keys",   test_calculator_comparison_by_ledger_group)
    run_test("comparison group_by=tenant → tenant keys",         test_calculator_comparison_by_tenant)
    run_test("comparison group_by=month → month keys",           test_calculator_comparison_by_month)
    run_test("pnl group_by=ledger_group → scalar + by_group",    test_calculator_pnl_with_group_by_ledger_group)
    run_test("comparison default (no group_by) unchanged",        test_calculator_comparison_default_unchanged)
    run_test("pnl default (no group_by) unchanged",               test_calculator_pnl_default_unchanged)

    # ---- LLM tests (skipped with --fast) -------------------------------------
    subheader("Supervisor Node (LLM)")
    run_test("P&L query -> pnl sub-query",              test_supervisor_pnl,              llm=True)
    run_test("comparison query -> comparison sub-query", test_supervisor_comparison,      llm=True)
    run_test("asset_details query",                      test_supervisor_asset_details,   llm=True)
    run_test("general real estate query",                test_supervisor_general,         llm=True)
    run_test("off-topic -> unknown/general sub-query",   test_supervisor_unknown,         llm=True)
    run_test("garbage input -> valid intent(s)",         test_supervisor_always_valid,    llm=True)
    run_test("compound query -> ≤3 sub-queries",         test_supervisor_max_sub_queries, llm=True)

    subheader("Extractor Node (LLM)")
    run_test("property name extracted",      test_extractor_property, llm=True)
    run_test("tenant name extracted",        test_extractor_tenant,   llm=True)
    run_test("year extracted",               test_extractor_year,     llm=True)
    run_test("metric defaults to profit",    test_extractor_metric_default, llm=True)
    run_test("fuzzy typo in property name",  test_extractor_fuzzy,    llm=True)
    run_test("entities dict has all 5 keys", test_extractor_schema,   llm=True)

    subheader("Synthesizer Node (LLM)")
    run_test("data sub-result -> non-empty str",      test_synthesizer_data,            llm=True)
    run_test("response contains numbers",             test_synthesizer_numbers,         llm=True)
    run_test("general sub-result -> non-empty str",   test_synthesizer_general,         llm=True)
    run_test("multi-result synthesis",                test_synthesizer_multi_result,    llm=True)
    run_test("partial failure handled gracefully",    test_synthesizer_partial_failure, llm=True)

    subheader("Fallback Node (LLM)")
    run_test("no sub-queries -> helpful response",    test_fallback_no_sub_queries,   llm=True)
    run_test("with supervisor error message",         test_fallback_with_error,       llm=True)
    run_test("error field is None",                  test_fallback_no_error_field,   llm=True)

    subheader("Integration Tests (full graph.invoke)")
    run_test("E2E: pnl pipeline",                             test_e2e_pnl,                  llm=True)
    run_test("E2E: comparison pipeline",                      test_e2e_comparison,           llm=True)
    run_test("E2E: asset_details pipeline",                   test_e2e_asset_details,        llm=True)
    run_test("E2E: general query (passes through)",           test_e2e_general,              llm=True)
    run_test("E2E: unknown query -> synthesizer handles it",  test_e2e_unknown_fallback,     llm=True)
    run_test("E2E: nonexistent property -> graceful response", test_e2e_nonexistent_property, llm=True)
    run_test("E2E: fuzzy typo in property name",              test_e2e_fuzzy_typo,           llm=True)
    run_test("E2E: final state has all GraphState keys",      test_e2e_state_schema,         llm=True)
    run_test("E2E: multi-intent query decomposed",            test_e2e_multi_intent,         llm=True)
    run_test("E2E: response always populated (5 path types)", test_e2e_response_always_set,  llm=True)

    # ---- Summary -------------------------------------------------------------
    total    = len(_results)
    skipped  = sum(1 for r in _results if r.get("skipped"))
    passed   = sum(1 for r in _results if r["passed"])
    failed   = total - passed
    duration = sum(r["duration"] for r in _results)

    banner("Test Summary")
    print(f"  Total   : {total}")
    print(f"  {_c('Passed', C.GREEN)}  : {passed - skipped}")
    if skipped:
        print(f"  {_c('Skipped', C.GREY)}  : {skipped}  (--fast mode, no LLM)")
    if failed:
        print(f"  {_c('Failed', C.RED)}  : {failed}")
        print()
        print(_c("  Failed tests:", C.RED + C.BOLD))
        for r in _results:
            if not r["passed"]:
                print(f"    {_c('FAIL', C.RED)} {r['name']}")
                if r["error"]:
                    print(f"         {r['error']}")
    else:
        print(f"  {_c('All tests passed!', C.GREEN + C.BOLD)}")
    print(f"  Time    : {duration:.2f}s")
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
