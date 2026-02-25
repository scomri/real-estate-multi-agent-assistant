import logging
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.getLevelName(os.getenv("LOG_LEVEL", "DEBUG")),
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.graph import build_graph
from src.state import GraphState

st.set_page_config(page_title="Cortex Real Estate AI Assistant", page_icon="🏢", layout="centered")
st.title("🏢 Cortex Real Estate AI Assistant")
st.caption("Ask about property P&L, comparisons, asset details, or general real estate questions.")


@st.cache_resource
def get_graph():
    g = build_graph()
    logger.info("Graph built successfully")
    return g


graph = get_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("$", r"\$"))

# Input
if prompt := st.chat_input("e.g. What is the total P&L for all properties this year?"):
    if not prompt.strip():
        st.warning("Please enter a non-empty message.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    initial_state: GraphState = {
        "query": prompt.strip(),
        "sub_queries": [],
        "sub_results": [],
        "response": "",
        "error": None,
    }

    logger.info("Query received: '%s'", prompt.strip()[:120])

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            logger.info("Invoking graph")
            try:
                final_state = graph.invoke(initial_state)
            except Exception:
                logger.error("Graph invocation failed", exc_info=True)
                raise

        response = final_state.get("response", "Sorry, I could not generate a response.")
        logger.info("Response received (%d chars)", len(response))
        # Escape bare $ signs so Streamlit doesn't treat them as LaTeX delimiters,
        # which breaks bold/italic markdown rendering around currency values.
        safe_response = response.replace("$", r"\$")
        st.markdown(safe_response)

        with st.expander("Debug: agent state"):
            sub_results = final_state.get("sub_results") or []
            # Top-level metadata
            meta = {k: v for k, v in final_state.items() if k not in ("sub_results", "response")}
            meta["sub_results_count"] = len(sub_results)
            st.json(meta)

            # Per-subquery results
            if sub_results:
                st.markdown("**Sub-query results**")
                for i, item in enumerate(sub_results, 1):
                    intent = item.get("intent", "?")
                    success = item.get("success", False)
                    status_icon = "✅" if success else "❌"
                    label = f"{status_icon} [{i}] `{intent}` — {item.get('query', '')}"
                    with st.expander(label, expanded=False):
                        if item.get("error"):
                            st.error(item["error"])
                        result = item.get("result")
                        if result is not None:
                            st.json(result)
                        elif success:
                            st.caption("No structured result (general intent)")

    st.session_state.messages.append({"role": "assistant", "content": response})
