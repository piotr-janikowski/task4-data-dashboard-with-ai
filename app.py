from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ai_chat import answer_data_question
from src.analytics import fill_missing_days
from src.dashboard_control import run_dashboard_command
from src.pipeline import run_store

st.set_page_config(page_title="Bookstore Analytics", page_icon="📚", layout="wide")

DATA_DIR = Path("data")
STORES = sorted(p.name for p in DATA_DIR.glob("STORE*"))


@st.cache_data
def get_results(store_name):
    return run_store(DATA_DIR / store_name)


def revenue_chart(daily):
    series = fill_missing_days(daily)
    fig = go.Figure()
    fig.add_scatter(x=series.index, y=series.values, name="Daily", line=dict(width=1), opacity=0.5)
    fig.add_scatter(x=series.index, y=series.rolling(7, center=True, min_periods=1).mean(), name="7-day average", line=dict(width=3))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Revenue (USD)", hovermode="x unified", legend=dict(orientation="h", y=1.1))
    return fig


def render_store(result):
    c1, c2 = st.columns(2)
    c1.metric("Unique users", f"{result['unique_users']:,}")
    c2.metric("Unique author sets", f"{result['unique_author_sets']:,}")
    left, right = st.columns(2)
    with left:
        st.subheader("Top 5 days by revenue")
        st.dataframe(pd.DataFrame(result["top_days"], columns=["Date", "Revenue (USD)"]), hide_index=True, width="stretch")
    with right:
        st.subheader("Most popular author(s)")
        for authors in result["top_authors"]:
            st.markdown("**" + ", ".join(authors) + "**")
        st.caption(f"{result['top_authors_sold']} copies sold")
        st.subheader("Best buyer (with aliases)")
        for ids, total in result["best_buyers"]:
            st.code(str(ids), language="python")
            st.caption(f"Total spent: ${total:,.2f}")
    st.subheader("Daily revenue")
    st.plotly_chart(revenue_chart(result["daily_revenue"]), width="stretch")


st.title("📚 Bookstore analytics")
store_tabs = st.tabs(STORES)
for tab, store in zip(store_tabs, STORES):
    with tab:
        render_store(get_results(store))

st.divider()
chat_tab, control_tab = st.tabs(["Part 2 · Ask your data", "Part 3 · Control the dashboard"])

with chat_tab:
    st.subheader("Ask questions about your data")
    selected_store = st.selectbox("Store for questions", STORES, key="qa_store")
    qa_key = f"qa_history_{selected_store}"
    if qa_key not in st.session_state:
        st.session_state[qa_key] = []
    for message in st.session_state[qa_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask about revenue, books, authors, or users", key="qa_input")
    if question:
        st.session_state[qa_key].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Checking the data..."):
                try:
                    answer = answer_data_question(question, DATA_DIR / selected_store, st.session_state[qa_key][:-1])
                except Exception as exc:
                    answer = f"Nie udało się uzyskać odpowiedzi: {exc}"
            st.markdown(answer)
        st.session_state[qa_key].append({"role": "assistant", "content": answer})
    st.caption("The agent interprets your question and uses read-only Python/Pandas analysis on the selected store data. It does not execute arbitrary code or use external data.")

with control_tab:
    st.subheader("Control the dashboard with natural language")
    selected_control_store = st.selectbox("Store for dashboard commands", STORES, key="control_store")
    control_history_key = f"control_history_{selected_control_store}"
    if control_history_key not in st.session_state:
        st.session_state[control_history_key] = []

    command = st.text_input("Command", placeholder="Show users whose email starts with s or p", key="control_command")
    if st.button("Run command", type="primary", key="run_dashboard_command"):
        if not command.strip():
            st.warning("Enter a command first.")
        else:
            history = [
                {"role": item["role"], "content": item.get("context", item.get("display", ""))}
                for item in st.session_state[control_history_key]
                if item.get("role") in {"user", "assistant"}
            ]
            st.session_state[control_history_key].append({
                "role": "user", "display": command.strip(), "context": command.strip()
            })
            try:
                with st.spinner("Analyzing the data..."):
                    result = run_dashboard_command(command.strip(), DATA_DIR / selected_control_store, history=history)
                table_context = ""
                for table in result.get("tables", []):
                    table_context += f"\n\n{table['title']}:\n" + pd.DataFrame(table["data"]).head(20).to_json(orient="records", force_ascii=False)
                assistant_context = result.get("description", "") + table_context
                st.session_state[control_history_key].append({
                    "role": "assistant",
                    "display": result.get("description", "Analysis completed."),
                    "context": assistant_context,
                    "result": result,
                })
            except Exception as exc:
                error_message = f"I could not complete the command: {exc}"
                st.session_state[control_history_key].append({
                    "role": "assistant", "display": error_message, "context": error_message,
                    "result": {"operation": "analysis", "description": error_message, "tables": []},
                })

    for index, message in enumerate(st.session_state[control_history_key]):
        with st.chat_message(message["role"]):
            st.markdown(message.get("display", ""))
            result = message.get("result")
            if message["role"] == "assistant" and result:
                for table_index, table in enumerate(result.get("tables", [])):
                    st.caption(table.get("title", "Analysis results"))
                    if table.get("data"):
                        st.dataframe(pd.DataFrame(table["data"]), hide_index=True, width="stretch")
                    else:
                        st.info("No matching rows were found.")

    if st.button("Clear conversation", key="clear_dashboard_conversation"):
        st.session_state[control_history_key] = []
        st.rerun()

    st.caption("Part 3 uses the same read-only Groq/Python/Pandas analytical agent as Part 2. Follow-up commands use the conversation history. Data access is limited to the project's STORE folders.")
