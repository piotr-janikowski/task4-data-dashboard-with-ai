import pandas as pd
import yaml

MISSING_TOKENS = ["", "null", "nan", "none", "n/a", "na", "-"]
TEXT_COLUMNS = ["title", "author", "genre", "publisher"]
MIN_VALID_YEAR = 1000


def clean_text_column(column):
    column = column.str.strip()
    is_missing = column.str.lower().isin(MISSING_TOKENS)
    return column.mask(is_missing)


def load_books(path):
    with open(path, encoding="utf-8") as f:
        books = pd.DataFrame(yaml.safe_load(f))
    books.columns = [c.lstrip(":") for c in books.columns]

    for column in TEXT_COLUMNS:
        books[column] = clean_text_column(books[column])

    years = pd.to_numeric(books["year"], errors="coerce")
    years = years.where(years >= MIN_VALID_YEAR)
    books["year"] = years.astype("Int64")
    return books


def parse_authors(raw):
    return frozenset(name.strip() for name in str(raw).split(",") if name.strip())


def add_author_sets(books):
    books = books.copy()
    books["authors"] = books["author"].map(parse_authors)
    return books


def sold_per_author_set(orders, books):
    sold = orders.groupby("book_id")["quantity"].sum()
    books = books.assign(sold=books["id"].map(sold).fillna(0).astype(int))
    return books.groupby("authors")["sold"].sum().sort_values(ascending=False)


def top_author_sets(by_set):
    top = by_set[by_set == by_set.max()]
    return [sorted(s) for s in top.index], int(by_set.max())
