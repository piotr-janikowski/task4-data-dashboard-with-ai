import pandas as pd

from common import (
    STORES,
    load_books_raw,
    load_books_records,
    load_orders_raw,
    looks_empty,
)

EMPTY_TOKENS = ["", "null", "nan", "none", "n/a", "na", "-"]
TEXT_COLUMNS = ["title", "author", "genre", "publisher"]


def split_authors(text):
    names = []
    for part in text.split(","):
        names.append(part.strip())
    return names


def count_name_variants(books):
    names = set()
    for text in books["author"]:
        for name in split_authors(text):
            names.add(name)

    variants = 0
    for name in names:
        for other in names:
            if other == name:
                continue
            if other.startswith(name + " ") or other.endswith(" " + name):
                variants += 1
    return variants


def make_sets(books):
    return books["author"].map(lambda text: frozenset(split_authors(text)))


def sold_by_set(orders, books):
    sold_per_book = orders.groupby("book_id")["quantity"].sum()
    table = pd.DataFrame({
        "authors": make_sets(books),
        "sold": books["id"].map(sold_per_book).fillna(0).astype(int),
    })
    return table.groupby("authors")["sold"].sum().sort_values(ascending=False)


def show_structure(records, books):
    print("1. STRUKTURA BOOKS.YAML (STORE1)")
    print("liczba rekordów", len(records))
    print("kolumny", books.columns.tolist())

    print("każdy rekord ma ten sam zestaw pól")
    key_sets = set()
    for record in records:
        key_sets.add(tuple(sorted(record.keys())))
    print("różne zestawy pól", len(key_sets))
    print("liczba pól w rekordzie", sorted({len(record) for record in records}))

    print("w pliku klucze zaczynają się od dwukropka")
    print("klucze pierwszego rekordu", list(records[0].keys()))

    print("\nWNIOSEK: struktura jest regularna, trzeba tylko usunąć dwukropki z nazw kolumn.")
    print("src/books.py load_books: books.columns = [c.lstrip(':') for c in books.columns]")


def check_types_and_missing(books):
    print("2. TYPY I BRAKI W BOOKS")

    print("POZIOM 1: pandas widzi braki tylko w publisher i year")
    print(books.isna().sum().to_string())

    print("POZIOM 2: brak może być ukryty jako tekst, więc szukamy go w każdej kolumnie")
    hidden = {}
    for column in books.columns:
        hidden[column] = looks_empty(books[column], EMPTY_TOKENS).sum()
    print(pd.Series(hidden).to_string())

    print("POZIOM 3: jakie typy wartości są w każdej kolumnie")
    for column in books.columns:
        types = books[column].map(lambda value: type(value).__name__).value_counts().to_dict()
        print("  " + column + ": " + str(types))

    print("POZIOM 4: jak wyglądają teksty udające brak w publisher")
    publisher_empty = looks_empty(books["publisher"], EMPTY_TOKENS) & books["publisher"].notna()
    print(books.loc[publisher_empty, "publisher"].map(repr).value_counts().to_string())

    print("\nWNIOSEK: ukryte braki są w publisher (NULL, spacje, myślnik, pusty tekst) i w year (patrz sekcja 5).")
    print("src/books.py clean_text_column: strip() i zamiana tokenów braku na brak (MISSING_TOKENS)")


def check_duplicates(books):
    print("3. DUPLIKATY W BOOKS")

    print("każda książka ma unikalne id")
    print("rekordy", len(books))
    print("unikalne id", books["id"].nunique())

    print("żadne dwa rekordy nie są identyczne")
    print("w pełni zduplikowane rekordy", books.drop(columns="id").duplicated().sum())

    print("ten sam tytuł może mieć kilka id")
    print("powtórzone tytuły", books["title"].duplicated().sum())
    repeated_title = books.loc[books["title"].duplicated(keep=False), ["id", "title", "author", "year"]]
    print(repeated_title.sort_values("title").head(6).to_string(index=False))

    print("\nWNIOSEK: powtórzony tytuł to inna książka (inny autor lub rok), więc łączymy po id, nie po tytule.")
    print("src/books.py sold_per_author_set: sprzedaż sumowana po book_id, tytuły nie są używane")


def check_whitespace(books):
    print("4. BIAŁE ZNAKI W TEKSTACH")

    print("teksty nie mają spacji na brzegach ani podwójnych spacji")
    rows = []
    for column in TEXT_COLUMNS:
        values = books[column].dropna().astype(str)
        rows.append({
            "column": column,
            "edge_spaces": int((values != values.str.strip()).sum()),
            "double_spaces": int(values.str.contains("  ").sum()),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nWNIOSEK: spacje na brzegach są tylko w publisher i wynikają z ukrytych braków.")
    print("src/books.py clean_text_column: strip() stosowany do wszystkich kolumn tekstowych")


def check_year():
    print("5. ROK WYDANIA (year), WSZYSTKIE SKLEPY RAZEM")
    all_years = pd.concat([load_books_raw(store)["year"] for store in STORES], ignore_index=True)

    print("POZIOM 0: year jest liczbą całkowitą")
    types = all_years.map(lambda value: type(value).__name__).value_counts()
    print(types.rename_axis("typ").rename("rekordy").to_string())

    print("POZIOM 1: wartości tekstowe w year")
    text_years = all_years[all_years.map(lambda value: isinstance(value, str))]
    print(text_years.map(repr).value_counts().rename_axis("wartość").rename("rekordy").to_string())

    print("POZIOM 2: co się stanie po zamianie na liczby (błędne zapisy staną się brakiem)")
    numbers = pd.to_numeric(all_years, errors="coerce")
    print("braki po zamianie", numbers.isna().sum())

    print("POZIOM 3: wartości liczbowe poniżej 1000 to zapełniacze, nie lata")
    small = numbers[numbers < 1000]
    print(small.value_counts().rename_axis("wartość").rename("rekordy").to_string())
    print("lata z przedziału 1-999", ((numbers > 0) & (numbers < 1000)).sum())
    print("najmniejszy rok >= 1000", numbers[numbers >= 1000].min())
    print("największy rok", numbers.max())

    print("\nWNIOSEK: prawdziwe lata to 1822-2026, a 0 i -1 oraz teksty (puste, tabulator, NULL, myślnik) to zapełniacze.")
    print("src/books.py load_books: pd.to_numeric(errors='coerce') zamienia teksty na brak")
    print("src/books.py MIN_VALID_YEAR = 1000: wartości 0 i -1 zamieniane na brak")
    print("src/books.py load_books: kolumna year jako Int64, bo dopuszcza braki")


def check_authors(books):
    print("6. AUTORZY (author)")
    authors = books["author"]

    print("POZIOM 0: pole author to jedno nazwisko")
    print("rekordy z przecinkiem w author", authors.str.contains(",", regex=False).sum())
    print(authors[authors.str.contains(",", regex=False)].head(4).to_string())

    print("POZIOM 1: jakie znaki poza literami występują w author i czy są inne separatory")
    characters = authors.str.cat(sep="")
    others = pd.Series(list(characters))
    others = others[~others.str.match(r"[A-Za-z\s.'\-]")]
    print(others.value_counts().rename_axis("znak").rename("wystąpienia").to_string())
    for separator in [";", "&", "/", " and ", " i "]:
        print("rekordy z separatorem '" + separator + "'", authors.str.contains(separator, regex=False).sum())

    print("POZIOM 2: ilu autorów ma jedna książka")
    counts = authors.map(lambda text: len(split_authors(text)))
    print(counts.value_counts().sort_index().rename_axis("autorów").rename("książki").to_string())
    empty_names = authors.map(lambda text: sum(1 for name in split_authors(text) if name == ""))
    print("puste fragmenty po rozdzieleniu", empty_names.sum())
    raw_parts = authors.map(lambda text: sum(1 for part in text.split(",") if part != part.strip()))
    print("fragmenty ze spacją na brzegu przed strip()", raw_parts.sum())

    print("POZIOM 3: przed imieniem bywają tytuły, a po nazwisku przyrostki")
    first_words = []
    last_words = []
    for text in authors:
        for name in split_authors(text):
            words = name.split()
            first_words.append(words[0])
            last_words.append(words[-1])
    print("  najczęstsze pierwsze słowa: " + str(pd.Series(first_words).value_counts().head(10).to_dict()))
    print("  najczęstsze ostatnie słowa: " + str(pd.Series(last_words).value_counts().head(10).to_dict()))

    print("POZIOM 4: żadna osoba nie występuje raz w krótszej, a raz w dłuższej formie (z tytułem lub przyrostkiem)")
    print("nazwy będące początkiem lub końcem innej nazwy", count_name_variants(books))

    print("POZIOM 5: ten sam zestaw autorów zapisany w różnej kolejności")
    sets = make_sets(books)
    print("różne napisy w author", authors.nunique())
    print("różne zestawy autorów", sets.nunique())
    print("różnica (zestawy zapisane na kilka sposobów)", authors.nunique() - sets.nunique())
    table = pd.DataFrame({"author": authors, "set": sets})
    spellings = table.groupby("set")["author"].nunique()
    example_sets = spellings[spellings > 1].head(3).index
    for example in example_sets:
        print("  przykład: " + str(table.loc[table["set"] == example, "author"].unique().tolist()))

    print("\nWNIOSEK: autorzy są rozdzieleni przecinkiem, tytuły i przyrostki zostają bez zmian, a kolejność autorów nie ma znaczenia.")
    print("src/books.py parse_authors: split(',') i strip() każdego fragmentu")
    print("src/books.py parse_authors: tytuły i przyrostki zostają, bo żadna osoba nie jest zapisana w dwóch formach")
    print("src/books.py parse_authors: frozenset, bo kolejność autorów nie ma znaczenia")
    print("src/books.py add_author_sets: liczba zestawów = books['authors'].nunique()")


def check_sales(books):
    print("7. SPRZEDAŻ WEDŁUG ZESTAWÓW AUTORÓW (STORE1)")
    orders = load_orders_raw("STORE1")
    by_set = sold_by_set(orders, books)

    print("suma sprzedaży po zestawach równa się sumie quantity w zamówieniach")
    print("suma po zestawach", by_set.sum())
    print("suma quantity", orders["quantity"].sum())

    print("sprzedaż liczymy w egzemplarzach (suma quantity), nie w wierszach zamówień")
    print("liczba wierszy zamówień", len(orders))
    print("liczba sprzedanych egzemplarzy", orders["quantity"].sum())

    print("najpopularniejsze zestawy autorów")
    print(by_set.head(5).to_string())
    top_value = by_set.max()
    print("zestawy z maksymalną sprzedażą", (by_set == top_value).sum())

    print("\nWNIOSEK: sumy się zgadzają; remisu nie ma, ale treść zadania mówi o autorach w liczbie mnogiej, więc obsługujemy remis.")
    print("src/books.py sold_per_author_set: suma quantity po book_id, potem suma po zestawie")
    print("src/books.py top_author_sets: zwraca listę wszystkich zestawów z maksymalną sprzedażą, na wypadek remisu")


def compare_stores():
    print("8. TE SAME TESTY DLA WSZYSTKICH SKLEPÓW (książki)")
    rows = []
    for store in STORES:
        records = load_books_records(store)
        books = load_books_raw(store)
        orders = load_orders_raw(store)
        sets = make_sets(books)
        by_set = sold_by_set(orders, books)
        key_sets = {tuple(sorted(record.keys())) for record in records}
        characters = pd.Series(list(books["author"].str.cat(sep="")))
        other_characters = characters[~characters.str.match(r"[A-Za-z\s.'\-,]")]
        rows.append({
            "store": store,
            "books": len(books),
            "key_sets": len(key_sets),
            "dup_id": int(books["id"].duplicated().sum()),
            "hidden_publisher": int((looks_empty(books["publisher"], EMPTY_TOKENS)).sum()),
            "year_text_values": int(books["year"].map(lambda value: isinstance(value, str)).sum()),
            "year_numbers_below_1000": int((pd.to_numeric(books["year"], errors="coerce") < 1000).sum()),
            "other_author_chars": len(other_characters),
            "name_variants": count_name_variants(books),
            "author_strings": books["author"].nunique(),
            "author_sets": sets.nunique(),
            "sum_check": int(by_set.sum() - orders["quantity"].sum()),
            "top_sets": int((by_set == by_set.max()).sum()),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    records = load_books_records("STORE1")
    books = load_books_raw("STORE1")
    show_structure(records, books)
    check_types_and_missing(books)
    check_duplicates(books)
    check_whitespace(books)
    check_year()
    check_authors(books)
    check_sales(books)
    compare_stores()
