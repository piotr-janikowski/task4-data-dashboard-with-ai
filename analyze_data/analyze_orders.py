import pandas as pd

from common import (
    STORES,
    load_books_raw,
    load_orders_raw,
    load_users_raw,
    looks_empty,
    print_claim,
    print_decision,
    print_header,
    print_result,
)

EMPTY_TOKENS = ["", "null", "nan", "none", "n/a", "na", "-"]

PRICE_SYMBOLS = ["$", "USD", "EUR", "€", "$¢", "€¢"]

DATE_KINDS = {
    "iso": r"\d{4}-\d{2}-\d{2}",
    "slash": r"\d{2}/\d{2}/\d{2}(?!\d)",
    "dot": r"\d{1,2}\.\d{1,2}\.\d{4}",
    "mon": r"\d{1,2}-[A-Za-z]+-\d{4}",
    "ctime": r"[A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}",
}
TIME_PATTERN = r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:\s*[AaPp]\.?[Mm]\.?)?"
SEPARATORS = " ,;T"


def make_price_layout(series):
    layout = series.str.replace(r"\d", "9", regex=True)
    layout = layout.str.replace(r"9+", "9", regex=True)
    return layout


def make_date_layout(series):
    layout = series.str.replace(r"\d", "9", regex=True)
    layout = layout.str.replace(r"[A-Za-z]+", "A", regex=True)
    return layout


def classify_dates(timestamps):
    result = pd.DataFrame()
    for kind in DATE_KINDS:
        result[kind] = timestamps.str.contains(DATE_KINDS[kind], regex=True)
    return result


def cut_out_date(timestamps, kinds):
    rest = timestamps.copy()
    for kind in DATE_KINDS:
        rows = kinds[kind]
        if kind == "ctime":
            rest[rows] = timestamps[rows].str.extract(r"(\d{2}:\d{2}:\d{2})")[0]
        else:
            rest[rows] = timestamps[rows].str.replace(DATE_KINDS[kind], "", n=1, regex=True)
    return rest


def extract_am_pm(rest):
    return rest.str.extract(r"\d\s*([AaPp]\.?[Mm]\.?)$")[0]


def show_overview(orders):
    print_header("1. PRZEGLĄD TABELI ORDERS")
    print_result("kształt (wiersze, kolumny)", orders.shape)
    print(orders.dtypes)
    print(orders.head(5).to_string())


def check_duplicates(orders):
    print_header("2. DUPLIKATY")

    print_claim("każde zamówienie ma unikalne id")
    print_result("wiersze", len(orders))
    print_result("unikalne id", orders["id"].nunique())
    print_result("powtórzone id", orders["id"].duplicated().sum())

    print_claim("żadne dwa wiersze nie są identyczne we wszystkich kolumnach")
    print_result("w pełni zduplikowane wiersze", orders.duplicated().sum())

    print_claim("ten sam zakup nie został zapisany dwa razy pod różnymi id")
    columns = ["user_id", "book_id", "quantity", "unit_price", "timestamp"]
    print_result("wiersze powtórzone w " + str(columns), orders.duplicated(subset=columns, keep=False).sum())

    print("\nWNIOSEK: w zamówieniach nie ma duplikatów, więc nic nie usuwamy.")
    print_decision("src/pipeline.py nie ma kroku usuwania duplikatów zamówień")


def check_missing_values(orders):
    print_header("3. BRAKI DANYCH W ORDERS")

    print_claim("POZIOM 1: pandas widzi braki tylko w kolumnie shipping")
    print(orders.isna().sum().to_string())

    print_claim("POZIOM 2: brak może być ukryty jako tekst, więc szukamy go w każdej kolumnie")
    hidden = {}
    for column in orders.columns:
        hidden[column] = looks_empty(orders[column], EMPTY_TOKENS).sum()
    print(pd.Series(hidden).to_string())

    print_claim("POZIOM 3: na ile sposobów zapisano 'nic' w shipping")
    empty_rows = looks_empty(orders["shipping"], EMPTY_TOKENS)
    print(orders.loc[empty_rows, "shipping"].map(repr).value_counts().to_string())
    print_result("wiersze bez adresu dostawy", str(empty_rows.sum()) + " z " + str(len(orders)))

    print_claim("POZIOM 4: pozostałe wartości shipping wyglądają jak adresy")
    print(orders.loc[~empty_rows, "shipping"].head(3).to_string())

    print("\nWNIOSEK: ukryte braki są tylko w shipping, a ta kolumna nie jest potrzebna do żadnego wyniku.")
    print_decision("src nie czyści kolumny shipping")


def check_numeric_columns(orders):
    print_header("4. KOLUMNY LICZBOWE")

    print_claim("id, user_id, book_id i quantity są liczbami całkowitymi")
    print(orders[["id", "user_id", "book_id", "quantity"]].dtypes.to_string())

    print_claim("quantity ma tylko małe, dodatnie wartości")
    print(orders["quantity"].value_counts().sort_index().rename_axis("quantity").rename("wiersze").to_string())
    print_result("quantity <= 0", (orders["quantity"] <= 0).sum())

    print("\nWNIOSEK: quantity nie wymaga czyszczenia, można go mnożyć przez cenę.")
    print_decision("src/prices.py add_paid_price: paid_price = quantity * unit_price_usd, bez czyszczenia quantity")


def check_prices(orders):
    print_header("5. CENY (unit_price)")
    price = orders["unit_price"]
    total = len(price)

    print_claim("POZIOM 0: unit_price to zwykła liczba")
    as_number = pd.to_numeric(price, errors="coerce")
    print_result("wiersze wczytane jako liczba", str(as_number.notna().sum()) + " z " + str(total))
    print(price.sample(8, random_state=1).map(repr).to_string())

    print_claim("POZIOM 1: po usunięciu cyfr i kropek zostają symbole wokół liczby")
    leftovers = price.str.replace(r"[\d.]", "", regex=True)
    print_result("różne pozostałości", leftovers.nunique())
    print(leftovers.map(repr).value_counts().head(12).to_string())

    print_claim("POZIOM 2: po usunięciu spacji zostaje kilka symboli")
    symbols = leftovers.str.replace(r"\s", "", regex=True)
    print(symbols.value_counts().to_string())
    unknown = ~symbols.isin(PRICE_SYMBOLS)
    print_result("wiersze z symbolem spoza listy " + str(PRICE_SYMBOLS), unknown.sum())

    print_claim("POZIOM 3: z symbolu wynika waluta i sposób zapisu centów")
    is_euro = symbols.str.contains("EUR|€", regex=True)
    has_cent_sign = symbols.str.contains("¢", regex=False)
    print_result("ceny w euro", str(is_euro.sum()) + " (" + format(is_euro.mean(), ".0%") + ")")
    print_result("ceny w dolarach", str((~is_euro).sum()) + " (" + format((~is_euro).mean(), ".0%") + ")")
    print_result("ceny zapisane ze znakiem ¢", has_cent_sign.sum())

    print_claim("POZIOM 4: zapis ze znakiem ¢ ma dwie liczby, a centy mają dwie cyfry")
    numbers = price[has_cent_sign].str.findall(r"\d+")
    two_numbers = (numbers.map(len) == 2).sum()
    two_digit_cents = numbers.map(lambda parts: len(parts[1]) == 2).sum()
    print_result("zapisy z dwiema liczbami", str(two_numbers) + " z " + str(len(numbers)))
    print_result("zapisy z centami o dwóch cyfrach", str(two_digit_cents) + " z " + str(len(numbers)))
    print(price[has_cent_sign].head(4).map(repr).to_string())

    print_claim("POZIOM 5: zapis bez ¢ ma jedną liczbę z opcjonalną częścią dziesiętną")
    plain = price[~has_cent_sign]
    one_number = plain.str.fullmatch(r"[^\d]*\d+(?:\.\d*)?[^\d]*")
    print_result("zapisy pasujące do wzorca jednej liczby", str(one_number.sum()) + " z " + str(len(plain)))
    decimals = plain.str.extract(r"\d+\.(\d*)")[0].dropna().str.len()
    print(decimals.value_counts().sort_index().rename_axis("cyfr po kropce").rename("wiersze").to_string())
    print_result("zapisy bez kropki", (~plain.str.contains(".", regex=False)).sum())

    print_claim("POZIOM 6: symbol stoi przed liczbą, po niej, ze spacją lub bez")
    layouts = make_price_layout(price)
    print_result("różne układy zapisu", layouts.nunique())
    print(layouts.map(repr).value_counts().head(8).to_string())

    print("\nWNIOSEK: cena to jedna liczba i symbol waluty w dowolnym układzie, a zapis z ¢ ma dwie liczby.")
    print_decision("src/prices.py parse_price: waluta z symbolu (€ lub EUR to euro, reszta to dolary)")
    print_decision("src/prices.py parse_price: gałąź dla ¢ składa jednostki i centy (centy zawsze 2 cyfry)")
    print_decision("src/prices.py parse_price: wyrażenie \\d+(?:\\.\\d+)? bierze liczbę niezależnie od pozycji symbolu")
    print_decision("src/prices.py price_in_usd: euro mnożymy przez EUR_TO_USD = 1.2 i zaokrąglamy do centów")


def check_dates(orders):
    print_header("6. DATY (timestamp)")
    timestamps = orders["timestamp"]
    total = len(timestamps)

    print_claim("POZIOM 0: pandas sam poradzi sobie z datami")
    strict = pd.to_datetime(timestamps, format="ISO8601", errors="coerce", utc=True)
    guessed = pd.to_datetime(timestamps, format="mixed", errors="coerce", utc=True)
    print_result("wiersze zgodne ze ścisłym formatem ISO 8601", str(strict.notna().sum()) + " z " + str(total))
    print_result("wiersze, które pandas zgaduje (format='mixed')", str(guessed.notna().sum()) + " z " + str(total))

    print_claim("POZIOM 1: ile jest różnych układów zapisu (cyfry jako 9, litery jako A)")
    layouts = make_date_layout(timestamps)
    print_result("różne układy", layouts.nunique())
    print(layouts.map(repr).value_counts().head(6).to_string())

    print_claim("POZIOM 2: da się wyróżnić 5 rodzajów daty, każdy wiersz ma dokładnie jeden")
    kinds = classify_dates(timestamps)
    print(kinds.sum().rename_axis("rodzaj").rename("wiersze").to_string())
    print(kinds.sum(axis=1).value_counts().rename_axis("ile rodzajów pasuje").rename("wiersze").to_string())
    for kind in DATE_KINDS:
        print("  przykłady " + kind + ": " + str(timestamps[kinds[kind]].head(3).tolist()))

    print_claim("POZIOM 3: po wycięciu daty zostaje tylko godzina i separatory")
    rest_with_separators = cut_out_date(timestamps, kinds)
    rest = rest_with_separators.str.strip(SEPARATORS)
    good_time = rest.str.fullmatch(TIME_PATTERN)
    print_result("wiersze z poprawną godziną", str(good_time.sum()) + " z " + str(total))
    if (~good_time).any():
        print(rest[~good_time].head(5).map(repr).to_string())
    separators = rest_with_separators.str.replace(TIME_PATTERN, "", regex=True)
    print("  separatory wokół godziny:")
    print(separators.map(repr).value_counts().rename_axis("separator").rename("wiersze").to_string())

    print_claim("POZIOM 4: znaczniki AM/PM")
    markers = extract_am_pm(rest)
    print(markers.map(repr).value_counts(dropna=False).rename_axis("znacznik").rename("wiersze").to_string())

    print_claim("POZIOM 5: zapis ze slashami, np. 10/28/25 (miesiąc/dzień czy dzień/miesiąc)")
    slash = timestamps[kinds["slash"]].str.extract(r"(\d{2})/(\d{2})/(\d{2})(?!\d)").astype(int)
    print_result("wiersze ze slashami", len(slash))
    print_result("pierwsza liczba > 12", (slash[0] > 12).sum())
    print_result("druga liczba > 12", (slash[1] > 12).sum())
    print_result("obie liczby <= 12", ((slash[0] <= 12) & (slash[1] <= 12)).sum())
    print_result("występujące dwucyfrowe lata", sorted(slash[2].unique().tolist()))
    month_first = pd.to_datetime(timestamps[kinds["slash"]], format="mixed", dayfirst=False, errors="coerce", utc=True)
    day_first = pd.to_datetime(timestamps[kinds["slash"]], format="mixed", dayfirst=True, errors="coerce", utc=True)
    print_result("daty, które zmieniłyby się przy zgadywaniu dayfirst", str((month_first != day_first).sum()) + " z " + str(len(slash)))

    print_claim("POZIOM 6: zapis z kropkami, np. 10.12.2025 (dzień.miesiąc czy miesiąc.dzień)")
    dot = timestamps[kinds["dot"]].str.extract(r"(\d{1,2})\.(\d{1,2})\.(\d{4})").astype(int)
    print_result("wiersze z kropkami", len(dot))
    print_result("pierwsza liczba > 12", (dot[0] > 12).sum())
    print_result("druga liczba > 12", (dot[1] > 12).sum())

    print_claim("POZIOM 7: rodzaje jednoznaczne mają sensowne zakresy")
    iso = timestamps[kinds["iso"]].str.extract(r"(\d{4})-(\d{2})-(\d{2})").astype(int)
    print_result("iso: miesiąc min/max", str(iso[1].min()) + "/" + str(iso[1].max()))
    print_result("iso: dzień min/max", str(iso[2].min()) + "/" + str(iso[2].max()))
    names = timestamps[kinds["mon"]].str.extract(r"\d{1,2}-([A-Za-z]+)-\d{4}")[0]
    print_result("mon: różne zapisy nazw miesięcy", names.nunique())
    print_result("mon: różne nazwy po pierwszych 3 literach", names.str[:3].str.lower().nunique())
    print_result("mon: przykłady", sorted(names.unique().tolist())[:8])

    print_claim("POZIOM 8: godziny w zapisach z AM/PM")
    hours_ampm = rest[markers.notna()].str.extract(r"^(\d{1,2}):")[0].astype(int)
    hours_24 = rest[markers.isna()].str.extract(r"^(\d{1,2}):")[0].astype(int)
    print_result("wiersze z AM/PM", len(hours_ampm))
    print_result("godzina z AM/PM min/max", str(hours_ampm.min()) + "/" + str(hours_ampm.max()))
    print_result("godzina bez AM/PM min/max", str(hours_24.min()) + "/" + str(hours_24.max()))

    print("\nWNIOSEK: timestamp to jeden z 5 rodzajów daty plus godzina, w dowolnej kolejności.")
    print_decision("src/dates.py DATE_PATTERNS: pięć osobnych wzorców (iso, slash, dot, mon, ctime)")
    print_decision("src/dates.py parse_timestamp: slash to miesiąc/dzień/rok, rok = 2000 + dwie cyfry")
    print_decision("src/dates.py parse_timestamp: dot to dzień.miesiąc.rok")
    print_decision("src/dates.py MONTHS: nazwa miesiąca rozpoznawana po pierwszych 3 literach, bez wielkości liter")
    print_decision("src/dates.py parse_timestamp: AM/PM zamieniane na 24 godziny (godziny 1-12)")
    print_decision("src/dates.py TIME_RE: sekundy i milisekundy opcjonalne, godzina szukana po wycięciu daty")
    print_decision("src/dates.py add_date: błąd ValueError, gdy jakakolwiek data się nie sparsuje")


def check_links(orders):
    print_header("7. POWIĄZANIA ORDERS Z USERS I BOOKS (STORE1)")
    users = load_users_raw("STORE1")
    books = load_books_raw("STORE1")

    print_claim("każde user_id z zamówień istnieje w users")
    user_ids = set(users["id"].astype(int))
    missing_users = set(orders["user_id"]) - user_ids
    print_result("użytkownicy z zamówień", orders["user_id"].nunique())
    print_result("user_id bez rekordu w users", len(missing_users))
    print_result("użytkownicy bez żadnego zamówienia", len(user_ids - set(orders["user_id"])))

    print_claim("każde book_id z zamówień istnieje w books")
    book_ids = set(books["id"])
    missing_books = set(orders["book_id"]) - book_ids
    print_result("książki z zamówień", orders["book_id"].nunique())
    print_result("book_id bez rekordu w books", len(missing_books))
    print_result("książki bez żadnego zamówienia", len(book_ids - set(orders["book_id"])))

    print("\nWNIOSEK: wszystkie powiązania są kompletne.")
    print_decision("src/users.py top_customers: merge z validate='many_to_one' i assert, że każde zamówienie ma właściciela")
    print_decision("src/books.py sold_per_author_set: książki bez zamówień dostają 0 (fillna), reszta pasuje po id")


def compare_stores_basic():
    print_header("8. TE SAME TESTY DLA WSZYSTKICH SKLEPÓW (zamówienia)")
    rows = []
    for store in STORES:
        orders = load_orders_raw(store)
        without_shipping = orders.drop(columns="shipping")
        users = load_users_raw(store)
        books = load_books_raw(store)
        rows.append({
            "store": store,
            "rows": len(orders),
            "dup_id": orders["id"].duplicated().sum(),
            "dup_rows": orders.duplicated().sum(),
            "hidden_na_outside_shipping": int(without_shipping.apply(lambda c: looks_empty(c, EMPTY_TOKENS)).sum().sum()),
            "shipping_empty": int(looks_empty(orders["shipping"], EMPTY_TOKENS).sum()),
            "quantity_min": orders["quantity"].min(),
            "quantity_max": orders["quantity"].max(),
            "user_id_missing_in_users": len(set(orders["user_id"]) - set(users["id"].astype(int))),
            "book_id_missing_in_books": len(set(orders["book_id"]) - set(books["id"])),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def compare_stores_prices():
    print_header("9. TE SAME TESTY CEN DLA WSZYSTKICH SKLEPÓW")
    base_layouts = set(make_price_layout(load_orders_raw(STORES[0])["unit_price"]))
    rows = []
    for store in STORES:
        price = load_orders_raw(store)["unit_price"]
        symbols = price.str.replace(r"[\d.\s]", "", regex=True)
        has_cent_sign = symbols.str.contains("¢", regex=False)
        numbers = price[has_cent_sign].str.findall(r"\d+")
        layouts = make_price_layout(price)
        rows.append({
            "store": store,
            "numeric_prices": int(pd.to_numeric(price, errors="coerce").notna().sum()),
            "unknown_symbols": int((~symbols.isin(PRICE_SYMBOLS)).sum()),
            "euro_share": format(symbols.str.contains("EUR|€", regex=True).mean(), ".0%"),
            "cent_sign_rows": int(has_cent_sign.sum()),
            "cents_not_2_digits": int((numbers.map(lambda parts: len(parts[1])) != 2).sum()),
            "layouts": layouts.nunique(),
            "layouts_not_in_STORE1": len(set(layouts) - base_layouts),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def compare_stores_dates():
    print_header("10. TE SAME TESTY DAT DLA WSZYSTKICH SKLEPÓW")
    rows = []
    for store in STORES:
        timestamps = load_orders_raw(store)["timestamp"]
        kinds = classify_dates(timestamps)
        rest = cut_out_date(timestamps, kinds).str.strip(SEPARATORS)
        markers = extract_am_pm(rest)
        slash = timestamps[kinds["slash"]].str.extract(r"(\d{2})/(\d{2})/\d{2}(?!\d)").astype(int)
        dot = timestamps[kinds["dot"]].str.extract(r"(\d{1,2})\.(\d{1,2})\.\d{4}").astype(int)
        hours_ampm = rest[markers.notna()].str.extract(r"^(\d{1,2}):")[0].astype(int)
        rows.append({
            "store": store,
            "layouts": make_date_layout(timestamps).nunique(),
            "not_exactly_1_kind": int((kinds.sum(axis=1) != 1).sum()),
            "bad_time": int((~rest.str.fullmatch(TIME_PATTERN)).sum()),
            "slash_1st>12": int((slash[0] > 12).sum()),
            "slash_2nd>12": int((slash[1] > 12).sum()),
            "dot_1st>12": int((dot[0] > 12).sum()),
            "dot_2nd>12": int((dot[1] > 12).sum()),
            "ampm_hours": str(hours_ampm.min()) + "-" + str(hours_ampm.max()),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    orders = load_orders_raw("STORE1")
    show_overview(orders)
    check_duplicates(orders)
    check_missing_values(orders)
    check_numeric_columns(orders)
    check_prices(orders)
    check_dates(orders)
    check_links(orders)
    compare_stores_basic()
    compare_stores_prices()
    compare_stores_dates()
