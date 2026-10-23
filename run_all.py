from pathlib import Path

from src.analytics import plot_daily_revenue
from src.pipeline import run_store

Path("output").mkdir(exist_ok=True)

for store in sorted(Path("data").glob("STORE*")):
    r = run_store(store)
    print(f"\n=== {r['store']} ===")
    print("Top 5 days:   ", r["top_days"])
    print("Unique users: ", r["unique_users"])
    print("Author sets:  ", r["unique_author_sets"])
    print("Top author(s):", r["top_authors"], "-", r["top_authors_sold"], "copies")
    print("Best buyer(s):", r["best_buyers"])
    plot_daily_revenue(r["daily_revenue"], f"{r['store']} – daily revenue",
                       path=f"output/{r['store']}_daily_revenue.png")