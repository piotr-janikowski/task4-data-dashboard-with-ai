import pandas as pd

from common import (
    STORES,
    load_orders_raw,
    load_users_pandas_default,
    load_users_raw,
)

EMPTY_TOKENS = ["", "null", "nan", "none", "n/a", "na", "-"]
FIELDS = ["name", "address", "phone", "email"]


def normalize_for_matching(users):
    normalized = pd.DataFrame()
    normalized["name"] = users["name"].str.strip().str.lower()
    normalized["address"] = users["address"].str.strip().str.lower()
    normalized["phone"] = users["phone"].str.replace(r"\D", "", regex=True)
    normalized["email"] = users["email"].str.strip().str.lower()
    is_empty = normalized["address"].isin(EMPTY_TOKENS)
    normalized["address"] = normalized["address"].mask(is_empty)
    return normalized


def count_shared_fields(normalized):
    shared_counts = {}
    for field in FIELDS:
        values = normalized[field].dropna()
        for value, group in values.groupby(values):
            ids = list(group.index)
            for first in range(len(ids)):
                for second in range(first + 1, len(ids)):
                    pair = (ids[first], ids[second])
                    shared_counts[pair] = shared_counts.get(pair, 0) + 1
    return shared_counts


def find_different_field(normalized, pair):
    first, second = pair
    different = []
    for field in FIELDS:
        value_first = normalized.at[first, field]
        value_second = normalized.at[second, field]
        same = pd.notna(value_first) and value_first == value_second
        if not same:
            different.append(field)
    return different


def find_chains(shared_counts):
    neighbours = {}
    for pair, count in shared_counts.items():
        if count == 3:
            first, second = pair
            neighbours.setdefault(first, set()).add(second)
            neighbours.setdefault(second, set()).add(first)

    chains = []
    for pair, count in shared_counts.items():
        if count == 2:
            first, second = pair
            middle = neighbours.get(first, set()) & neighbours.get(second, set())
            if len(middle) > 0:
                chains.append((first, sorted(middle)[0], second))
    return chains


def count_pairs_by_shared_fields(shared_counts):
    result = {1: 0, 2: 0, 3: 0, 4: 0}
    for count in shared_counts.values():
        result[count] += 1
    return result


def count_blank_address_pairs(users):
    total = 0
    forms = users["address"].map(repr).value_counts()
    for form in ["''", "' '", "'NULL'"]:
        if form in forms.index:
            rows = forms[form]
            total += rows * (rows - 1) // 2
    return total


def show_overview(users):
    print("1. PRZEGLĄD TABELI USERS (STORE1)")
    print("kształt (wiersze, kolumny)", users.shape)
    print("kolumny", users.columns.tolist())
    print(users.head(5).to_string())


def check_ids_and_duplicates(users):
    print("2. IDENTYFIKATORY I DUPLIKATY")

    print("każdy rekord ma unikalne id")
    print("rekordy", len(users))
    print("unikalne id", users["id"].nunique())

    print("żadne dwa rekordy nie mają identycznych wszystkich czterech pól")
    print("rekordy z identycznym name, address, phone i email", users.drop(columns="id").duplicated().sum())

    print("pojedyncze pola powtarzają się między rekordami (to ślad zmian danych tej samej osoby)")
    rows = []
    for field in FIELDS:
        rows.append({
            "field": field,
            "rows_sharing_value": int(users[field].duplicated(keep=False).sum()),
            "distinct_values": users[field].nunique(),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("(w address liczą się też puste zapisy, patrz sekcja 4)")

    print("\nWNIOSEK: nie ma dosłownych duplikatów, ale pola się powtarzają, więc osoby trzeba rozpoznawać po zgodnych polach.")
    print("src/users.py: nie usuwamy rekordów, tylko grupujemy je w osoby (kolumna real_user)")


def check_text_quality(users):
    print("3. JAKOŚĆ TEKSTÓW W USERS")

    print("pola nie mają spacji na brzegach, podwójnych spacji ani wartości różniących się tylko wielkością liter")
    rows = []
    for field in FIELDS:
        values = users[field]
        rows.append({
            "field": field,
            "empty_text": int((values == "").sum()),
            "only_spaces": int((values.str.strip().eq("") & values.ne("")).sum()),
            "null_text": int(values.str.strip().str.lower().isin(["null", "nan", "none", "n/a"]).sum()),
            "edge_spaces": int(((values != values.str.strip()) & values.str.strip().ne("")).sum()),
            "double_spaces": int(values.str.contains("  ").sum()),
            "case_variants": int(values.nunique() - values.str.lower().nunique()),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print("e-maile mają poprawną budowę")
    print("bez znaku @", (~users["email"].str.contains("@", regex=False)).sum())
    parts = users["email"].str.split("@", n=1, expand=True)
    print("pusta część przed lub po @", ((parts[0] == "") | (parts[1] == "")).sum())
    print("e-maile z wielkimi literami", (users["email"] != users["email"].str.lower()).sum())

    print("\nWNIOSEK: braki występują tylko w address, a name, phone i email są kompletne i czyste.")
    print("src/users.py normalize_users: strip, lower i zwinięcie wielu spacji (zabezpieczenie, w tych danych nie jest konieczne)")


def check_address_missing(users):
    print("4. BRAKI W ADRESACH")

    print("POZIOM 1: ile braków widzi pandas przy domyślnym wczytaniu pliku")
    default_users = load_users_pandas_default("STORE1")
    print("braki w address (pd.read_csv bez opcji)", default_users["address"].isna().sum())

    print("POZIOM 2: jak naprawdę zapisano 'brak adresu' w pliku")
    forms = users["address"].map(repr)
    hidden_forms = forms[forms.isin(["''", "' '", "'NULL'"])]
    print(hidden_forms.value_counts().rename_axis("zapis").rename("rekordy").to_string())
    print("razem rekordów bez adresu", len(hidden_forms))

    print("POZIOM 3: pandas pomija część z nich")
    print("różnica (rekordy widoczne tylko po ręcznym sprawdzeniu)", len(hidden_forms) - default_users["address"].isna().sum())
    print("  (pandas zamienia na brak pusty tekst i NULL, ale pojedyncza spacja zostaje zwykłym tekstem)")

    print("POZIOM 4: gdyby brak traktować jak zwykłą wartość, powstałyby fałszywe pary osób")
    for form in ["''", "' '", "'NULL'"]:
        rows = (forms == form).sum()
        print("pary z takim samym adresem " + form, str(rows) + " rekordów daje " + str(rows * (rows - 1) // 2) + " par")
    print("fałszywe pary razem", count_blank_address_pairs(users))

    print("\nWNIOSEK: brak adresu ma trzy zapisy, a brak nie może łączyć rekordów w jedną osobę.")
    print("src/users.py load_users: dtype=str i keep_default_na=False, żeby pandas niczego nie zgadywał")
    print("src/users.py MISSING_TOKENS: puste, null, nan, n/a, na, none i '-' zamieniane na brak")
    print("src/users.py count_matching_fields: col.dropna(), więc brak nigdy nie liczy się jako zgodność")


def check_phone_formats(users):
    print("5. TELEFONY")

    print("POZIOM 1: telefony mają kilka układów zapisu")
    layouts = users["phone"].str.replace(r"\d", "9", regex=True)
    print(layouts.value_counts().rename_axis("układ").rename("rekordy").to_string())

    print("POZIOM 2: po usunięciu wszystkiego poza cyframi każdy numer ma tyle samo cyfr")
    digits = users["phone"].str.replace(r"\D", "", regex=True)
    print(digits.str.len().value_counts().rename_axis("liczba cyfr").rename("rekordy").to_string())

    print("POZIOM 3: ten sam numer w różnych układach wyglądałby jak różne numery")
    print("różne zapisy telefonów", users["phone"].nunique())
    print("różne numery po usunięciu znaków niebędących cyframi", digits.nunique())

    print("\nWNIOSEK: wszystkie numery mają 10 cyfr, więc same cyfry jednoznacznie identyfikują numer.")
    print("src/users.py normalize_users: phone = tylko cyfry (replace(r'\\D', ''))")


def check_shared_fields(users):
    print("6. KTÓRE REKORDY TO TA SAMA OSOBA")
    normalized = normalize_for_matching(users)
    shared_counts = count_shared_fields(normalized)

    print("POZIOM 1: ile pól wspólnych mają pary rekordów, które mają cokolwiek wspólnego")
    by_count = count_pairs_by_shared_fields(shared_counts)
    print(pd.Series(by_count).rename_axis("wspólnych pól").rename("pary").to_string())
    print("pary z czterema wspólnymi polami (dosłowny duplikat)", by_count[4])
    print("pary z jednym wspólnym polem", by_count[1])

    print("POZIOM 2: w parach z trzema wspólnymi polami zmieniło się dokładnie jedno pole, które?")
    different_counts = {"name": 0, "address": 0, "phone": 0, "email": 0}
    for pair, count in shared_counts.items():
        if count == 3:
            different = find_different_field(normalized, pair)
            different_counts[different[0]] += 1
    print(pd.Series(different_counts).rename_axis("zmienione pole").rename("pary").to_string())

    print("POZIOM 3: pary z dwoma wspólnymi polami to końce łańcuchów A - B - C")
    chains = find_chains(shared_counts)
    print("pary z dwoma wspólnymi polami", by_count[2])
    print("z nich z pośrednikiem połączonym z obu stron trzema polami", len(chains))
    first, middle, second = chains[0]
    print("  przykład łańcucha (rekordy A, B, C):")
    print(users.loc[[first, middle, second]].to_string())
    print("  A i B różnią się w polu: " + str(find_different_field(normalized, (first, middle))))
    print("  B i C różnią się w polu: " + str(find_different_field(normalized, (middle, second))))
    print("  A i C różnią się w polach: " + str(find_different_field(normalized, (first, second))))

    print("POZIOM 4: czy łańcuchy są dłuższe niż trzy rekordy")
    ends_by_record = {}
    for pair, count in shared_counts.items():
        if count >= 3:
            for record in pair:
                ends_by_record[record] = ends_by_record.get(record, 0) + 1
    print(pd.Series(ends_by_record).value_counts().sort_index().rename_axis("powiązań 3-polowych").rename("rekordy").to_string())

    print("\nWNIOSEK: ta sama osoba to rekordy zgodne w co najmniej 3 z 4 pól, a dwa jej rekordy mogą zgadzać się w 2 polach i być połączone przez trzeci.")
    print("src/users.py find_user_clusters: min_matches=3 (zmienione jedno pole z czterech)")
    print("src/users.py find_user_clusters: łączenie przechodnie (union-find), bo A i C zgadzają się tylko w 2 polach")
    

def check_orders_and_users(users):
    print("7. UŻYTKOWNICY A ZAMÓWIENIA (STORE1)")
    orders = load_orders_raw("STORE1")
    user_ids = set(users["id"].astype(int))
    ordering_ids = set(orders["user_id"])

    print("każde user_id z zamówień istnieje w users")
    print("user_id bez rekordu w users", len(ordering_ids - user_ids))

    print("część rekordów users nie ma żadnych zamówień")
    print("rekordy bez zamówień", len(user_ids - ordering_ids))

    print("czasem jeden z rekordów tej samej osoby nie ma zamówień")
    normalized = normalize_for_matching(users)
    shared_counts = count_shared_fields(normalized)
    pairs_with_unused_record = 0
    for pair, count in shared_counts.items():
        if count >= 3:
            first_id = int(users.at[pair[0], "id"])
            second_id = int(users.at[pair[1], "id"])
            if first_id not in ordering_ids or second_id not in ordering_ids:
                pairs_with_unused_record += 1
    print("pary rekordów tej samej osoby z jednym rekordem bez zamówień", pairs_with_unused_record)

    print("\nWNIOSEK: zamówienia można bezpiecznie dołączyć do users po id, a lista aliasów osoby może zawierać id bez zamówień.")
    print("src/users.py top_customers: merge(orders, users, on='user_id', validate='many_to_one')")
    print("src/users.py top_customers: lista id obejmuje wszystkie rekordy osoby z tabeli users, nie tylko te z zamówieniami")


def compare_stores():
    print("8. TE SAME TESTY DLA WSZYSTKICH SKLEPÓW (użytkownicy)")
    rows = []
    for store in STORES:
        users = load_users_raw(store)
        orders = load_orders_raw(store)
        forms = users["address"].map(repr)
        digits = users["phone"].str.replace(r"\D", "", regex=True)
        normalized = normalize_for_matching(users)
        shared_counts = count_shared_fields(normalized)
        by_count = count_pairs_by_shared_fields(shared_counts)
        different_counts = {"name": 0, "address": 0, "phone": 0, "email": 0}
        for pair, count in shared_counts.items():
            if count == 3:
                different_counts[find_different_field(normalized, pair)[0]] += 1
        rows.append({
            "store": store,
            "users": len(users),
            "dup_id": int(users["id"].duplicated().sum()),
            "addr_empty": int((forms == "''").sum()),
            "addr_space": int((forms == "' '").sum()),
            "addr_NULL": int((forms == "'NULL'").sum()),
            "blank_pairs": count_blank_address_pairs(users),
            "phone_layouts": users["phone"].str.replace(r"\d", "9", regex=True).nunique(),
            "phone_digit_lengths": str(sorted(digits.str.len().unique().tolist())),
            "email_no_at": int((~users["email"].str.contains("@", regex=False)).sum()),
            "pairs_1": by_count[1],
            "pairs_2": by_count[2],
            "pairs_3": by_count[3],
            "pairs_4": by_count[4],
            "chains": len(find_chains(shared_counts)),
            "diff_name": different_counts["name"],
            "diff_address": different_counts["address"],
            "diff_phone": different_counts["phone"],
            "diff_email": different_counts["email"],
            "ids_missing_in_users": len(set(orders["user_id"]) - set(users["id"].astype(int))),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    users = load_users_raw("STORE1")
    show_overview(users)
    check_ids_and_duplicates(users)
    check_text_quality(users)
    check_address_missing(users)
    check_phone_formats(users)
    check_shared_fields(users)
    check_orders_and_users(users)
    compare_stores()
