from pathlib import Path

import pandas as pd

from src.analytics import daily_revenue, top_days
from src.books import add_author_sets, load_books, sold_per_author_set, top_author_sets
from src.dates import add_date
from src.prices import add_paid_price
from src.users import add_real_user, load_users, top_customers


def run_store(store_path):
    store_path = Path(store_path)

    orders = pd.read_parquet(store_path / "orders.parquet")
    orders = add_date(add_paid_price(orders))
    books = add_author_sets(load_books(store_path / "books.yaml"))
    users = add_real_user(load_users(store_path / "users.csv"))

    daily = daily_revenue(orders)
    by_set = sold_per_author_set(orders, books)
    top_authors, top_authors_sold = top_author_sets(by_set)

    return {
        "store": store_path.name,
        "daily_revenue": daily,
        "top_days": [(d.strftime("%Y-%m-%d"), float(v)) for d, v in top_days(daily).items()], # type: ignore
        "unique_users": int(users["real_user"].nunique()),
        "unique_author_sets": int(books["authors"].nunique()),
        "top_authors": top_authors,
        "top_authors_sold": top_authors_sold,
        "best_buyers": top_customers(orders, users),
    }