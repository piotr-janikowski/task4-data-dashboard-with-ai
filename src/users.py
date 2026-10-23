import pandas as pd
import itertools
from collections import Counter
import numpy as np


missing_tokens = {"", "null", "nan", "n/a", "na", "none", "-"}
user_fields = ["name", "address", "phone", "email"]

def load_users(path):
    users = pd.read_csv(path, dtype=str, keep_default_na=False)
    users["id"] = users["id"].astype(int)
    return users

def normalize_users(users):
    norm = pd.DataFrame({"id": users["id"]})
    for col in ["name", "address", "email"]:
        norm[col] = (
            users[col]
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )
    norm["phone"] = users["phone"].str.replace(r"\D", "", regex=True)
    for col in user_fields:
        norm.loc[norm[col].isin(missing_tokens), col] = np.nan
    return norm

def count_matching_fields(norm):
    matches = Counter()
    for field in user_fields:
        col = norm[field].dropna()
        for _, group in col.groupby(col):
            for i, j in itertools.combinations(group.index, 2):
                matches[(i, j)] += 1
    return matches

def find_user_clusters(norm, min_matches=3):
    parent = list(range(len(norm)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), count in count_matching_fields(norm).items():
        if count >= min_matches:
            parent[find(i)] = find(j)
    return pd.Series([find(i) for i in range(len(norm))], index=norm.index)

def add_real_user(users):
    users = users.copy()
    users["real_user"] = find_user_clusters(normalize_users(users))
    return users

def top_customers(orders, users):
    link = users[["id", "real_user"]].rename(columns={"id": "user_id"})
    merged = orders.merge(
        link, on="user_id", how="left", validate="many_to_one"
    )
    assert merged["real_user"].notna().all()
    spending = (
        merged.groupby("real_user")["paid_price"]
        .sum()
        .round(2)
        .sort_values(ascending=False)
    )
    best = spending[spending == spending.max()]
    return [
        (
            sorted(
                users.loc[users["real_user"] == c, "id"].tolist()
            ),
            float(total),
        )
        for c, total in best.items()
    ]