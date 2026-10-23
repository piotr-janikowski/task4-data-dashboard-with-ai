import re
from functools import lru_cache

import pandas as pd

from src.ai_chat import MODEL, request_tool_plan, load_store_data, answer_dashboard_command
MAX_ROWS = 500

DATASET_FIELDS = {
    "books": {"id", "title", "author", "genre", "publisher", "year", "authors", "sold"},
    "users": {"id", "name", "address", "phone", "email", "real_user"},
    "orders": {"user_id", "book_id", "timestamp", "date", "quantity", "unit_price", "unit_price_usd", "paid_price"},
    "sales": {"id", "title", "author", "genre", "publisher", "year", "authors", "sold", "user_id", "book_id", "timestamp", "date", "quantity", "unit_price", "unit_price_usd", "paid_price", "customer_name", "customer_email"},
}

OPERATORS = {
    "contains", "not_contains", "starts_with", "ends_with", "equals", "not_equals",
    "in", "not_in", "gt", "gte", "lt", "lte", "between", "is_missing", "not_missing",
    "regex", "contains_person_name", "has_duplicate_value",
}

AGGREGATIONS = {"count", "sum", "mean", "median", "min", "max", "nunique"}




def run_dashboard_command(command, store_path, history=None):
    """Run Part 3 through the same analytical agent used by Part 2."""
    return answer_dashboard_command(command, store_path, history=history)


def interpret_dashboard_command(command):
    tool = {
        "type": "function",
        "function": {
            "name": "dashboard_plan",
            "description": "Translate a natural-language dashboard command into a safe sequence of dataframe operations. Do not calculate or invent data. Use only available datasets and columns. Prefer the simplest valid plan. For ambiguous requests, choose the most likely interpretation and state it in explanation. Use contains_person_name only when the user asks about people/person names in book titles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["query", "clear"]},
                    "dataset": {"type": "string", "enum": ["books", "users", "orders", "sales"]},
                    "filters": {"type": "array", "items": {"type": "object", "properties": {
                        "field": {"type": "string"},
                        "operator": {"type": "string", "enum": sorted(OPERATORS)},
                        "value": {"type": "string"},
                        "logic": {"type": "string", "enum": ["and", "or"]}
                    }, "required": ["field", "operator", "value", "logic"], "additionalProperties": False}},
                    "sort_by": {"type": "string"},
                    "sort_order": {"type": "string", "enum": ["asc", "desc"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ROWS},
                    "group_by": {"type": "array", "items": {"type": "string"}},
                    "aggregate_field": {"type": "string"},
                    "aggregation": {"type": "string", "enum": [""] + sorted(AGGREGATIONS)},
                    "show_duplicates_by": {"type": "string"},
                    "explanation": {"type": "string"}
                },
                "required": ["operation", "dataset", "filters", "sort_by", "sort_order", "limit", "group_by", "aggregate_field", "aggregation", "show_duplicates_by", "explanation"],
                "additionalProperties": False
            }
        }
    }
    system_prompt = """You translate English or Polish dashboard commands into a JSON tool plan. Respond through the tool only. Never write Python or SQL.
Available datasets: books (book metadata), users (user records), orders (order rows), sales (orders joined with book and user fields).
Book fields: id,title,author,genre,publisher,year,authors,sold. User fields: id,name,address,phone,email,real_user. Order fields: id,user_id,book_id,timestamp,date,quantity,unit_price,unit_price_usd,paid_price. Sales combines these fields.
For 'emails starting with s or p', use users.email starts_with value 's,p' and logic or. For 'filter out ...', use not_contains or negation only if a literal term is provided; for people names in titles use books.title contains_person_name. For longest author set/name, use books.author or books.authors and set sort_by to length:author or length:authors with descending order; if asking about books that sold, filter sold > 0. For sold copies, use sold or quantity as appropriate, not row counts. For 'top N' or 'bottom N', sort by the requested metric and set limit. For aggregate questions, use group_by, aggregate_field, aggregation. For duplicate requests, use show_duplicates_by. Use multiple filters for combined conditions; each filter's logic indicates how it joins to the previous filter. Default limit is 100. Do not assume a specific store beyond the selected store. Clear all filters means operation clear. Keep explanation short and in English."""
    plan = request_tool_plan(tool, system_prompt, command)
    _validate_plan(plan)
    return plan


def _validate_plan(plan):
    if plan.get("operation") == "clear":
        return
    dataset = plan.get("dataset")
    if dataset not in DATASET_FIELDS:
        raise ValueError("The requested dataset is not supported.")
    fields = DATASET_FIELDS[dataset]
    for item in plan.get("filters", []):
        if item["field"] not in fields:
            raise ValueError(f"Column '{item['field']}' is not available in {dataset}.")
        if item["operator"] not in OPERATORS:
            raise ValueError(f"Operator '{item['operator']}' is not supported.")
    for key in ("sort_by", "aggregate_field", "show_duplicates_by"):
        value = plan.get(key, "")
        if value and value not in fields and not (key == "sort_by" and value.startswith("length:") and value.split(":", 1)[1] in fields):
            raise ValueError(f"Column '{value}' is not available in {dataset}.")
    for field in plan.get("group_by", []):
        if field not in fields:
            raise ValueError(f"Column '{field}' is not available in {dataset}.")
    if plan.get("aggregation") and plan["aggregation"] not in AGGREGATIONS:
        raise ValueError("The requested aggregation is not supported.")
    if not 1 <= int(plan.get("limit", 100) or 100) <= MAX_ROWS:
        raise ValueError(f"Limit must be between 1 and {MAX_ROWS}.")


@lru_cache(maxsize=10)
def _load_cached_frames(store_path, file_signature):
    orders, books, users = load_store_data(store_path)
    sales = orders.merge(books, left_on="book_id", right_on="id", how="left", suffixes=("", "_book"))
    user_columns = [column for column in ["id", "name", "email"] if column in users.columns]
    sales = sales.merge(users[user_columns].rename(columns={"id": "user_id", "name": "customer_name", "email": "customer_email"}), on="user_id", how="left", suffixes=("", "_user"))
    if "authors" in books.columns and "author" not in books.columns:
        books = books.copy()
        books["author"] = books["authors"].map(lambda names: ", ".join(sorted(names)) if isinstance(names, (set, frozenset, list, tuple)) else str(names))
    if "authors" in books.columns:
        books = books.copy()
        books["authors"] = books["authors"].map(lambda names: ", ".join(sorted(names)) if isinstance(names, (set, frozenset, list, tuple)) else str(names))
    return orders, books, users, sales


def _frames(store_path):
    from pathlib import Path
    path = Path(store_path)
    names = ["orders.parquet", "books.yaml", "users.csv"]
    signature = tuple((name, (path / name).stat().st_mtime_ns, (path / name).stat().st_size) for name in names)
    return _load_cached_frames(str(path.resolve()), signature)


def _as_text(series):
    return series.astype("string").fillna("")


def _person_name_mask(series):
    # Conservative fallback: recognize multi-token proper-name phrases rather than every capitalized word.
    # This avoids treating normal title-case words as personal names, but is not full NER.
    text = _as_text(series)
    patterns = [
        r"\b(?:Mr|Mrs|Ms|Dr|Sir|Lady|President|King|Queen)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b",
        r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",
    ]
    mask = pd.Series(False, index=series.index)
    for pattern in patterns:
        mask |= text.str.contains(pattern, regex=True, na=False)
    return mask


def _filter_mask(frame, item):
    field, operator, value = item["field"], item["operator"], item.get("value", "")
    series = frame[field]
    text = _as_text(series)
    lowered = text.str.casefold()
    target = str(value).casefold()
    if operator == "contains":
        return lowered.str.contains(target, regex=False, na=False)
    if operator == "not_contains":
        return ~lowered.str.contains(target, regex=False, na=False)
    if operator == "starts_with":
        prefixes = [part.strip().casefold() for part in str(value).split(",") if part.strip()]
        if not prefixes:
            raise ValueError("Provide at least one prefix.")
        return lowered.map(lambda val: any(val.startswith(prefix) for prefix in prefixes))
    if operator == "ends_with":
        suffixes = [part.strip().casefold() for part in str(value).split(",") if part.strip()]
        if not suffixes:
            raise ValueError("Provide at least one suffix.")
        return lowered.map(lambda val: any(val.endswith(suffix) for suffix in suffixes))
    if operator in {"equals", "not_equals"}:
        mask = lowered == target
        return ~mask if operator == "not_equals" else mask
    if operator in {"in", "not_in"}:
        values = {part.strip().casefold() for part in str(value).split(",") if part.strip()}
        mask = lowered.isin(values)
        return ~mask if operator == "not_in" else mask
    if operator in {"is_missing", "not_missing"}:
        missing = series.isna() | text.str.strip().str.casefold().isin({"", "nan", "none", "null", "n/a", "na", "-"})
        return missing if operator == "is_missing" else ~missing
    if operator in {"gt", "gte", "lt", "lte", "between"}:
        numeric = pd.to_numeric(series, errors="coerce")
        if operator == "between":
            parts = [part.strip() for part in str(value).split(",", 1)]
            if len(parts) != 2:
                raise ValueError("For a range, provide two comma-separated values, for example 2000,2020.")
            low, high = map(float, parts)
            return numeric.between(low, high, inclusive="both")
        target_num = float(value)
        return {"gt": numeric > target_num, "gte": numeric >= target_num, "lt": numeric < target_num, "lte": numeric <= target_num}[operator]
    if operator == "regex":
        try:
            return text.str.contains(str(value), case=False, regex=True, na=False)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression: {exc}") from exc
    if operator == "contains_person_name":
        return _person_name_mask(series)
    if operator == "has_duplicate_value":
        return series.duplicated(keep=False)
    raise ValueError(f"Unsupported operator: {operator}")


def apply_dashboard_action(action, store_path):
    if action.get("operation") == "clear":
        return {"operation": "clear", "message": "Displayed results cleared. This does not change the source data."}
    _validate_plan(action)
    orders, books, users, sales = _frames(str(store_path))
    dataset = action["dataset"]
    frame = {"books": books, "users": users, "orders": orders, "sales": sales}[dataset].copy()
    original_count = len(frame)

    combined_mask = None
    for item in action.get("filters", []):
        current = _filter_mask(frame, item)
        if combined_mask is None:
            combined_mask = current
        elif item.get("logic", "and") == "or":
            combined_mask = combined_mask | current
        else:
            combined_mask = combined_mask & current
    if combined_mask is not None:
        frame = frame.loc[combined_mask].copy()

    duplicate_field = action.get("show_duplicates_by", "")
    if duplicate_field:
        frame = frame.loc[frame[duplicate_field].duplicated(keep=False)].copy()

    group_by = action.get("group_by", [])
    aggregate_field = action.get("aggregate_field", "")
    aggregation = action.get("aggregation", "")
    if group_by and aggregation:
        target = aggregate_field or group_by[0]
        if aggregation == "count":
            result = frame.groupby(group_by, dropna=False).size().reset_index(name="count")
        elif aggregation == "nunique":
            result = frame.groupby(group_by, dropna=False)[target].nunique().reset_index(name=f"{target}_nunique")
        else:
            if not pd.api.types.is_numeric_dtype(frame[target]):
                frame[target] = pd.to_numeric(frame[target], errors="coerce")
            result = frame.groupby(group_by, dropna=False)[target].agg(aggregation).reset_index(name=f"{target}_{aggregation}")
        frame = result

    sort_by = action.get("sort_by", "")
    if sort_by:
        if sort_by.startswith("length:"):
            length_field = sort_by.split(":", 1)[1]
            if length_field not in frame.columns:
                raise ValueError(f"Cannot calculate text length for '{length_field}'.")
            frame = frame.assign(_sort_length=_as_text(frame[length_field]).str.len())
            frame = frame.sort_values("_sort_length", ascending=action.get("sort_order", "asc") != "desc", na_position="last").drop(columns=["_sort_length"])
        else:
            if sort_by not in frame.columns:
                raise ValueError(f"Cannot sort by '{sort_by}' after this operation.")
            frame = frame.sort_values(sort_by, ascending=action.get("sort_order", "asc") != "desc", na_position="last")

    limit = int(action.get("limit", 100) or 100)
    limit = max(1, min(limit, MAX_ROWS))
    displayed = frame.head(limit).copy()
    displayed = displayed.replace({pd.NA: None})
    displayed = displayed.astype(object).where(pd.notna(displayed), None)
    description = action.get("explanation", "Dashboard query completed.").strip() or "Dashboard query completed."
    if group_by and aggregation:
        description += f" Grouped by {', '.join(group_by)} using {aggregation}."
    return {
        "operation": "filter",
        "dataset": dataset,
        "description": description,
        "total_rows": original_count,
        "matching_rows": len(frame),
        "data": displayed.to_dict("records"),
    }
