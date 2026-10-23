import json
import os
import re
from pathlib import Path

import streamlit as st

import pandas as pd
from groq import Groq

from src.books import add_author_sets, load_books
from src.dates import add_date
from src.prices import add_paid_price
from src.users import add_real_user, load_users

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_ROWS = 100


def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Set it as an environment variable or in .streamlit/secrets.toml.")
    return Groq(api_key=api_key)


@st.cache_data(show_spinner=False)
def load_store_data(store_path):
    store_path = Path(store_path)
    orders = pd.read_parquet(store_path / "orders.parquet")
    orders = add_date(add_paid_price(orders))
    books = add_author_sets(load_books(store_path / "books.yaml"))
    users = add_real_user(load_users(store_path / "users.csv"))
    return orders, books, users


@st.cache_data(show_spinner=False)
def _analysis_frames(store_path):
    orders, books, users = load_store_data(store_path)
    book_columns = [column for column in ["id", "title", "author", "genre", "publisher", "year"] if column in books]
    user_columns = [column for column in ["id", "name", "address", "phone", "email", "real_user"] if column in users]
    book_sales = orders.merge(books[book_columns], left_on="book_id", right_on="id", how="left", suffixes=("", "_book"))
    user_sales = orders.merge(users[user_columns], left_on="user_id", right_on="id", how="left", suffixes=("", "_user"))
    author_sales = book_sales.copy()
    author_sales["author"] = author_sales["author"].fillna("").astype(str).str.split(",")
    author_sales = author_sales.explode("author")
    author_sales["author"] = author_sales["author"].astype(str).str.strip()
    author_sales = author_sales[author_sales["author"] != ""]
    author_sales = author_sales.groupby("author", as_index=False).agg(
        copies_sold=("quantity", "sum"),
        revenue_usd=("paid_price", "sum"),
        book_count=("book_id", "nunique"),
    )
    book_sales_summary = book_sales.groupby([column for column in ["book_id", "title", "author", "genre", "publisher", "year"] if column in book_sales], dropna=False, as_index=False).agg(
        copies_sold=("quantity", "sum"),
        revenue_usd=("paid_price", "sum"),
        order_count=("book_id", "size"),
    )
    customer_sales = user_sales.groupby([column for column in ["user_id", "name", "email", "real_user"] if column in user_sales], dropna=False, as_index=False).agg(
        revenue_usd=("paid_price", "sum"),
        copies_bought=("quantity", "sum"),
        order_count=("user_id", "size"),
    )
    revenue_by_date = orders.groupby("date", as_index=False).agg(revenue_usd=("paid_price", "sum"), copies_sold=("quantity", "sum"), order_count=("date", "size"))
    revenue_by_date["date"] = revenue_by_date["date"].astype(str)
    return {
        "orders": orders,
        "books": books,
        "users": users,
        "book_sales": book_sales_summary,
        "author_sales": author_sales,
        "customer_sales": customer_sales,
        "revenue_by_date": revenue_by_date,
    }



def request_tool_plan(tool, system_prompt, user_message):
    """Use the same Groq client and tool-calling setup for structured agent plans."""
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": tool["function"]["name"]}},
        temperature=0,
    )
    calls = response.choices[0].message.tool_calls or []
    if not calls:
        raise ValueError("I could not interpret that command. Please rephrase it.")
    return json.loads(calls[0].function.arguments or "{}")


def _tool_specs():
    return [{
        "type": "function",
        "function": {
            "name": "analyze_store_data",
            "description": "Flexible, read-only analysis of the selected store's project data. Choose a dataset and task. Use author_sales for individual author rankings, book_sales for book rankings, customer_sales for customer rankings, revenue_by_date for date/revenue analysis, or inspect the schema first with overview. For a revenue question without dates, use task=date_range with empty dates to get the full available sales period, then aggregate or rank as needed. All calculations are executed by Python/Pandas, not guessed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "string", "enum": ["orders", "books", "users", "book_sales", "author_sales", "customer_sales", "revenue_by_date"]},
                    "task": {"type": "string", "enum": ["overview", "search", "rank", "aggregate", "sample", "date_range"]},
                    "stores": {"type": "string", "description": "Optional comma-separated store names, such as STORE1,STORE2. Leave empty to use only the selected store. Only use multiple stores when the user asks to compare or analyze them together."},
                    "field": {"type": "string", "description": "Column to search, e.g. title, author, email. Empty when unused."},
                    "query": {"type": "string", "description": "Text to search for. Empty when unused."},
                    "metric": {"type": "string", "description": "Numeric column to aggregate or rank, e.g. paid_price, quantity, revenue_usd, copies_sold, year. Empty when unused."},
                    "group_by": {"type": "string", "description": "Column to group by for aggregate/rank tasks. Empty when unused."},
                    "sort": {"type": "string", "enum": ["desc", "asc"], "description": "Ranking direction."},
                    "limit": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Maximum number of returned rows. Use 0 when the task does not need a row limit; Python will apply a safe default where relevant."},
                    "filter_field": {"type": "string", "description": "Optional column to filter by. Empty when unused."},
                    "filter_value": {"type": "string", "description": "Optional exact or partial filter value. Empty when unused."},
                    "start_date": {"type": "string", "description": "Optional inclusive start date YYYY-MM-DD. Empty if not specified."},
                    "end_date": {"type": "string", "description": "Optional inclusive end date YYYY-MM-DD. Empty if not specified."}
                },
                "required": ["dataset", "task"],
                "additionalProperties": False
            }
        }
    }]


def _clean_records(frame, limit=MAX_ROWS):
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10
    if limit <= 0:
        return []
    result = frame.head(min(limit, MAX_ROWS)).copy()
    result = result.astype(object).where(pd.notna(result), None)
    return result.to_dict("records")


def _normalize_tool_args(args):
    defaults = {
        "dataset": "orders",
        "task": "overview",
        "stores": "",
        "field": "",
        "query": "",
        "metric": "",
        "group_by": "",
        "sort": "desc",
        "limit": 10,
        "filter_field": "",
        "filter_value": "",
        "start_date": "",
        "end_date": "",
    }
    normalized = defaults | {key: value for key, value in args.items() if key in defaults}
    for key in ("stores", "field", "query", "metric", "group_by", "filter_field", "filter_value", "start_date", "end_date"):
        if normalized[key] is None:
            normalized[key] = ""
        else:
            normalized[key] = str(normalized[key]).strip()
    if normalized["sort"] not in {"asc", "desc"}:
        normalized["sort"] = "desc"
    try:
        normalized["limit"] = max(0, min(int(normalized["limit"]), MAX_ROWS))
    except (TypeError, ValueError):
        normalized["limit"] = 10
    return normalized


def _available_store_paths(selected_store_path):
    selected_store_path = Path(selected_store_path).resolve()
    data_dir = selected_store_path.parent
    return {path.name: path for path in sorted(data_dir.glob("STORE*")) if path.is_dir()}


def _run_analysis(args, store_path):
    args = _normalize_tool_args(args)
    selected_store_path = Path(store_path).resolve()
    available_stores = _available_store_paths(selected_store_path)
    requested_stores = [name.strip() for name in args["stores"].split(",") if name.strip()]
    if not requested_stores:
        requested_stores = [selected_store_path.name]
    invalid_stores = [name for name in requested_stores if name not in available_stores]
    if invalid_stores:
        return {"error": f"Unknown store(s): {', '.join(invalid_stores)}.", "available_stores": list(available_stores)}

    per_store_frames = [(name, _analysis_frames(available_stores[name])) for name in requested_stores]
    dataset = args["dataset"]
    task = args["task"]
    if dataset not in per_store_frames[0][1]:
        return {"error": f"Unknown dataset '{dataset}'.", "available_datasets": list(per_store_frames[0][1])}
    if len(per_store_frames) == 1:
        frame = per_store_frames[0][1][dataset].copy()
        frame.insert(0, "store", per_store_frames[0][0])
    else:
        frames_to_combine = []
        for store_name, store_frames in per_store_frames:
            if dataset not in store_frames:
                return {"error": f"Dataset '{dataset}' is unavailable in {store_name}."}
            current = store_frames[dataset].copy()
            current.insert(0, "store", store_name)
            frames_to_combine.append(current)
        frame = pd.concat(frames_to_combine, ignore_index=True, sort=False)
    start_date = args["start_date"]
    end_date = args["end_date"]

    date_column = "date" if "date" in frame.columns else "timestamp" if "timestamp" in frame.columns else None
    if start_date or end_date:
        if not date_column:
            return {"error": f"Dataset '{dataset}' has no date column to filter."}
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        if start_date:
            dates_start = pd.to_datetime(start_date, errors="raise")
            frame = frame[dates >= dates_start]
            dates = pd.to_datetime(frame[date_column], errors="coerce")
        if end_date:
            dates_end = pd.to_datetime(end_date, errors="raise") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            frame = frame[dates <= dates_end]

    filter_field = args.get("filter_field", "").strip()
    filter_value = args.get("filter_value", "").strip()
    if filter_field and filter_value:
        if filter_field not in frame.columns:
            return {"error": f"Column '{filter_field}' is not available in dataset '{dataset}'.", "available_columns": list(frame.columns)}
        frame = frame[frame[filter_field].astype("string").str.contains(filter_value, case=False, na=False, regex=False)]

    if task == "date_range":
        if "date" not in frame.columns:
            return {"error": f"Dataset '{dataset}' has no date column."}
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        if dates.empty:
            return {"dataset": dataset, "start_date": None, "end_date": None, "message": "No dates are available."}
        return {"dataset": dataset, "start_date": str(dates.min().date()), "end_date": str(dates.max().date()), "row_count": int(len(frame))}

    if task == "overview":
        columns = {}
        for column in frame.columns:
            values = frame[column]
            columns[column] = {"dtype": str(values.dtype), "non_null": int(values.notna().sum()), "unique": int(values.nunique(dropna=True))}
        return {"dataset": dataset, "row_count": int(len(frame)), "columns": columns, "date_range": _date_range(frame)}

    if task == "search":
        field = args.get("field", "").strip()
        query = args.get("query", "").strip()
        if field not in frame.columns:
            return {"error": f"Column '{field}' is not available in dataset '{dataset}'.", "available_columns": list(frame.columns)}
        if not query:
            return {"error": "A non-empty query is required for search."}
        found = frame[frame[field].astype("string").str.contains(query, case=False, na=False, regex=False)]
        return {"dataset": dataset, "matching_rows": int(len(found)), "rows": _clean_records(found, args.get("limit", 10))}

    if task == "sample":
        return {"dataset": dataset, "row_count": int(len(frame)), "rows": _clean_records(frame, args.get("limit", 10))}

    if task == "aggregate":
        metric = args.get("metric", "").strip()
        group_by = args.get("group_by", "").strip()
        group_columns = [column.strip() for column in group_by.split(",") if column.strip()]
        if metric not in frame.columns or not pd.api.types.is_numeric_dtype(frame[metric]):
            return {"error": f"Numeric metric '{metric}' is not available in dataset '{dataset}'.", "available_columns": list(frame.columns)}
        if group_columns:
            missing = [column for column in group_columns if column not in frame.columns]
            if missing:
                return {"error": f"Group column(s) {missing} are not available.", "available_columns": list(frame.columns)}
            result = frame.groupby(group_columns, dropna=False)[metric].agg(total="sum", average="mean", count="count").reset_index()
            result = result.sort_values("total", ascending=args.get("sort", "desc") == "asc")
            return {"dataset": dataset, "metric": metric, "group_by": group_columns, "groups": int(len(result)), "rows": _clean_records(result, args.get("limit", 10))}
        return {"dataset": dataset, "metric": metric, "row_count": int(len(frame)), "sum": float(frame[metric].sum()), "average": float(frame[metric].mean()) if len(frame) else None, "min": float(frame[metric].min()) if len(frame) else None, "max": float(frame[metric].max()) if len(frame) else None}

    if task == "rank":
        metric = args.get("metric", "").strip()
        group_by = args.get("group_by", "").strip()
        group_columns = [column.strip() for column in group_by.split(",") if column.strip()]
        if metric not in frame.columns or not pd.api.types.is_numeric_dtype(frame[metric]):
            return {"error": f"Numeric metric '{metric}' is not available in dataset '{dataset}'.", "available_columns": list(frame.columns)}
        if group_columns:
            missing = [column for column in group_columns if column not in frame.columns]
            if missing:
                return {"error": f"Rank column(s) {missing} are not available.", "available_columns": list(frame.columns)}
            frame = frame.groupby(group_columns, dropna=False, as_index=False)[metric].sum()
        ranked = frame.sort_values(metric, ascending=args.get("sort", "desc") == "asc")
        return {"dataset": dataset, "ranked_by": metric, "group_by": group_columns or None, "available_results": int(len(ranked)), "rows": _clean_records(ranked, args.get("limit", 10))}

    return {"error": f"Unsupported task: {task}"}


def _date_range(frame):
    column = "date" if "date" in frame.columns else "timestamp" if "timestamp" in frame.columns else None
    if not column:
        return None
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    if dates.empty:
        return None
    return {"start": str(dates.min().date()), "end": str(dates.max().date())}


def _run_tool(name, args, store_path):
    if name == "analyze_store_data":
        return _run_analysis(args, store_path)
    return {"error": "Unknown tool"}


def _format_table(frame, columns):
    if frame.empty:
        return "No matching records were found in this store’s data."
    visible = frame[columns].head(10).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in visible.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def _try_fast_answer(question, store_path):
    """Answer common, deterministic questions without an LLM round-trip."""
    text = question.strip()
    normalized = re.sub(r"\s+", " ", text.lower())
    frames = None

    if re.search(r"\b(total|overall|entire|full)\b.*\b(revenue|sales)\b|\b(revenue|sales)\b.*\b(total|overall)\b", normalized):
        if not re.search(r"\b( november | january | february | march | april | may | june | july | august | september | october | december | between | from \d{4}-\d{2}-\d{2})", f" {normalized} "):
            frames = _analysis_frames(store_path)
            daily = frames["revenue_by_date"]
            if daily.empty:
                return "No sales records are available for the selected store."
            dates = pd.to_datetime(daily["date"], errors="coerce").dropna()
            total = pd.to_numeric(daily["revenue_usd"], errors="coerce").sum()
            if not dates.empty:
                return f"Total revenue for {Path(store_path).name} over the full available period ({dates.min().date()}–{dates.max().date()}) is ${total:,.2f} USD."

    author_match = re.search(r"\bbooks?\s+by\s+(.+?)[?.!]*$", text, flags=re.IGNORECASE)
    if author_match and not re.search(r"\b(top|best[- ]selling|most sold|copies sold|revenue)\b", normalized):
        author = author_match.group(1).strip(" \t.,?!\"'")
        if author:
            frames = frames or _analysis_frames(store_path)
            books = frames["books"].copy()
            if "author" in books.columns and "title" in books.columns:
                found = books[books["author"].astype("string").str.contains(author, case=False, na=False, regex=False)]
                columns = [column for column in ["title", "author", "year", "genre"] if column in found.columns]
                if found.empty:
                    return f"No books matching author '{author}' were found in {Path(store_path).name}."
                return f"Books by author matching '{author}' in {Path(store_path).name}:\n\n" + _format_table(found, columns)

    top_match = re.search(r"\btop\s+(\d+)\b", normalized)
    requested = min(int(top_match.group(1)), 20) if top_match else (5 if re.search(r"\btop\b", normalized) else None)
    if requested and not re.search(r"\bauthor(s)?\b", normalized) and not re.search(r"\b(include|along with|and also|their best-selling)\b", normalized) and re.search(r"\b(book|books|title|titles)\b", normalized) and re.search(r"\b(top|best[- ]selling|most sold|copies sold)\b", normalized):
        frames = frames or _analysis_frames(store_path)
        frame = frames["book_sales"].sort_values("copies_sold", ascending=False).head(requested)
        columns = [column for column in ["title", "author", "copies_sold", "revenue_usd"] if column in frame.columns]
        return f"Top {len(frame)} best-selling books in {Path(store_path).name}:\n\n" + _format_table(frame, columns)

    if requested and re.search(r"\bauthor(s)?\b", normalized) and re.search(r"\b(top|best[- ]selling|most sold|copies sold|highest revenue)\b", normalized):
        frames = frames or _analysis_frames(store_path)
        metric = "revenue_usd" if "revenue" in normalized else "copies_sold"
        frame = frames["author_sales"].sort_values(metric, ascending=False).head(requested)
        columns = [column for column in ["author", metric, "book_count"] if column in frame.columns]
        return f"Top {len(frame)} authors in {Path(store_path).name}, ranked by {'revenue' if metric == 'revenue_usd' else 'copies sold'}:\n\n" + _format_table(frame, columns)

    if requested and re.search(r"\bcustomer(s)?\b", normalized) and re.search(r"\b(top|best|highest|most)\b", normalized):
        frames = frames or _analysis_frames(store_path)
        metric = "copies_bought" if re.search(r"\b(copies|books|quantity)\b", normalized) else "revenue_usd"
        frame = frames["customer_sales"].sort_values(metric, ascending=False).head(requested)
        columns = [column for column in ["user_id", "name", "email", metric] if column in frame.columns]
        return f"Top {len(frame)} customers in {Path(store_path).name}:\n\n" + _format_table(frame, columns)

    return None


def answer_dashboard_command(question, store_path, history=None):
    """Run a dashboard command through the same analytical agent as Part 2.

    Returns an English explanation and structured table data for Streamlit.
    The agent can use multiple read-only Pandas tool calls and conversation history.
    """
    client = get_client()
    store_name = Path(store_path).name
    available_stores = _available_store_paths(store_path)
    system_prompt = f"""You are the analytical agent powering both Part 2 and Part 3 of a bookstore analytics dashboard.
You have read-only access to the project data for these stores: {', '.join(available_stores)}.
The currently selected store is {store_name}. Use only that store unless the user explicitly names or asks to compare other stores.
Always answer in English. Never use outside data, invent values, or execute arbitrary code. All calculations must be performed through the analyze_store_data tool using Python/Pandas.

Available datasets:
- orders: order-level sales records
- books: book metadata
- users: user metadata
- book_sales: sales aggregated by book, with title, author and copies sold
- author_sales: sales aggregated per individual author
- customer_sales: sales aggregated per customer
- revenue_by_date: daily revenue, quantities and order counts

Interpret natural and informal commands flexibly. For a request to filter or find records, choose the relevant dataset and use search, filter_field/filter_value, rank, aggregate or sample as appropriate. For people names in book titles, do not treat every capitalized word as a person's name; use actual metadata or a conservative interpretation and explain any limitation. For duplicate emails, use users and inspect/group the email field. For longest author names, search or sample books and use available author metadata; do not claim a sales condition unless the data supports it. For ranking requests without a number, use 5. For revenue without a date range, use the full available sales period. For comparisons, use the stores parameter and include store in grouping where appropriate.

Use conversation history to interpret follow-ups such as 'show only the top five', 'add their revenue', 'now compare with STORE2', and 'do the same for books'. Make the smallest reasonable assumptions and state them briefly. You may make multiple tool calls when needed, but keep the plan efficient. Do not stop after an intermediate result if another tool call is necessary to answer the full question. Return concise conclusions and relevant tables. Never claim a top five if fewer than five eligible records were returned. If the data or available tools cannot support a request, state that clearly instead of inventing a result."""
    messages = [{"role": "system", "content": system_prompt}]
    for item in history or []:
        if item.get("role") in {"user", "assistant"} and item.get("content"):
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": question})
    tools = _tool_specs()
    collected_results = []
    max_rounds = 8

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            answer = message.content or "The analysis completed, but no text explanation was returned."
            tables = []
            for result in collected_results:
                if not isinstance(result, dict) or result.get("error"):
                    continue
                rows = result.get("rows")
                if isinstance(rows, list) and rows:
                    tables.append({
                        "title": _table_title(result),
                        "data": rows,
                    })
                elif isinstance(result.get("groups"), int) and result.get("groups") == 0:
                    tables.append({"title": _table_title(result), "data": []})
            # Keep only the latest copy of an identical table to avoid duplicate rendering.
            unique_tables = []
            seen = set()
            for table in tables:
                key = (table["title"], json.dumps(table["data"], sort_keys=True, default=str))
                if key not in seen:
                    seen.add(key)
                    unique_tables.append(table)
            return {"operation": "analysis", "description": answer, "tables": unique_tables}

        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
                result = _run_tool(call.function.name, args, store_path)
            except Exception as exc:
                args = {}
                result = {"error": str(exc)}
            collected_results.append(result)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    messages.append({
        "role": "user",
        "content": "Prepare the most accurate answer now using only the tool results already collected. Do not call more tools. State any limitation instead of inventing data.",
    })
    response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
    answer = response.choices[0].message.content or "The analysis could not be summarized from the collected results."
    tables = []
    for result in collected_results:
        if isinstance(result, dict) and not result.get("error") and isinstance(result.get("rows"), list) and result["rows"]:
            tables.append({"title": _table_title(result), "data": result["rows"]})
    return {"operation": "analysis", "description": answer, "tables": tables}


def _table_title(result):
    dataset = result.get("dataset", "Analysis results")
    if result.get("ranked_by"):
        return f"{dataset.replace('_', ' ').title()} ranked by {result['ranked_by'].replace('_', ' ')}"
    if result.get("group_by"):
        groups = result["group_by"]
        if isinstance(groups, list):
            groups = ", ".join(groups)
        return f"{dataset.replace('_', ' ').title()} grouped by {groups}"
    if result.get("matching_rows") is not None:
        return f"Matching {dataset.replace('_', ' ')} ({result['matching_rows']} rows)"
    return f"{dataset.replace('_', ' ').title()} results"


def answer_data_question(question, store_path, history=None):
    explicit_store_comparison = bool(re.search(r"\b(compare|comparison|versus|vs\.?|between)\b.*\bSTORE\s*\d+\b|\bSTORE\s*\d+\b.*\b(and|versus|vs\.?)\b.*\bSTORE\s*\d+\b", question, flags=re.IGNORECASE))
    fast_answer = None if explicit_store_comparison else _try_fast_answer(question, store_path)
    if fast_answer is not None:
        return fast_answer

    client = get_client()
    store_name = Path(store_path).name
    system_prompt = f"""You are a helpful analytical data agent for a bookstore dashboard. You have read-only access to the project data files for all available stores: {", ".join(_available_store_paths(store_path))}. The currently selected store is {store_name}. By default, analyze only the selected store. If the user explicitly asks to compare stores or names a different store, use the stores parameter to access the named store(s). Never combine stores unless the question asks for it. Do not use outside facts, access external data, or invent results. Always answer in English, regardless of the language used in the question. Keep all user-facing messages, explanations, errors, and table introductions in English.

Use the analyze_store_data tool to inspect and analyze the data. All calculations must be performed by Python/Pandas through this tool. The optional stores argument accepts a comma-separated list of available stores. Leave it empty for the selected store only. For cross-store comparisons, explicitly name the stores and group by the store column as well as the relevant entity, e.g. group_by="store,author". You may call the tool multiple times when genuinely needed, but plan efficient analyses and avoid repeating the same query.

Available datasets:
- orders: order-level records
- books: book metadata
- users: user metadata
- book_sales: sales aggregated by book, including title and author
- author_sales: sales aggregated per individual author
- customer_sales: sales aggregated per customer
- revenue_by_date: daily revenue and quantities

Interpret natural, informal, incomplete questions reasonably. For revenue questions without a date range, first determine the full available sales date range, then calculate total revenue across that entire range. For a top-N request without a number, use 5. For tasks that return rows, use a limit of at least 1; for overview and date_range tasks, use 0 or 1. For top-selling authors, rank author_sales by copies_sold. For top-selling books, rank book_sales by copies_sold. For highest-revenue questions, rank or aggregate by revenue_usd. For "authors of the best-selling books", return the titles and authors of the best-selling books unless the user clearly asks for total sales grouped by author.

Use the conversation history to understand follow-ups such as "in that period", "those books", and "what about STORE3". All stores named STORE1 through STORE5 are available if present in the project folder. If the user asks to compare two or more stores, analyze each named store, not just the currently selected store. Do not carry over a previous date range when the user explicitly asks for the full available period. If a request is ambiguous but has a sensible, low-risk interpretation, proceed with that interpretation and state your assumption briefly. Ask a clarifying question only when needed.

Never present one result as a top five if at least five eligible results are available. If fewer results exist, say how many were found. Do not dump all records; use a reasonable result limit. Revenue values are in USD according to the project's price conversion logic. Do not execute arbitrary code or use tools other than analyze_store_data. If a tool call fails, try a corrected call when possible and explain any remaining limitation."""
    messages = [{"role": "system", "content": system_prompt}]
    for item in history or []:
        if item.get("role") in {"user", "assistant"} and item.get("content"):
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": question})
    tools = _tool_specs()

    max_rounds = 6
    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content or "Unable to prepare an answer."
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
                result = _run_tool(call.function.name, args, store_path)
            except Exception as exc:
                result = {"error": str(exc)}
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False, default=str)})
    messages.append({"role": "user", "content": "Prepare the best accurate answer now using only the tool results already collected. Do not call tools again. If something could not be calculated, clearly state the limitation instead of inventing data."})
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content or "Unable to prepare an answer from the collected results."
    except Exception:
        return "The analysis could not be completed within the available step limit. Try narrowing the question or splitting it into smaller parts."
