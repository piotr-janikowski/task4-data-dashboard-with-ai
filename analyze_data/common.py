from pathlib import Path

import pandas as pd
import yaml


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STORES = sorted(path.name for path in DATA_DIR.glob("STORE*"))


def load_orders_raw(store):
    return pd.read_parquet(DATA_DIR / store / "orders.parquet")


def load_users_raw(store):
    return pd.read_csv(DATA_DIR / store / "users.csv", dtype=str, keep_default_na=False)


def load_users_pandas_default(store):
    return pd.read_csv(DATA_DIR / store / "users.csv")


def load_books_records(store):
    with open(DATA_DIR / store / "books.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_books_raw(store):
    books = pd.DataFrame(load_books_records(store))
    books.columns = [column.lstrip(":") for column in books.columns]
    return books


def looks_empty(series, tokens):
    as_text = series.fillna("").astype("string").str.strip().str.lower()
    return series.isna() | as_text.isin(tokens)
