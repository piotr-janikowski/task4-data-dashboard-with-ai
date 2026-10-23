import pandas as pd
import matplotlib.pyplot as plt

def daily_revenue(orders):
    return orders.groupby("date")["paid_price"].sum().round(2)

def top_days(daily, n=5):
    return daily.sort_values(ascending=False).head(n)

def plot_daily_revenue(daily, title="Daily revenue", path=None):
    series = daily.copy()
    series.index = pd.to_datetime(series.index)
    series = series.asfreq("D", fill_value=0)          # dni bez sprzedaży = 0 USD

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(series.index, series.values, linewidth=0.8, alpha=0.6, label="Daily")
    ax.plot(series.index, series.rolling(7, center=True, min_periods=1).mean(),
            linewidth=2, label="7-day average")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue (USD)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig

def fill_missing_days(daily):
    series = daily.copy()
    series.index = pd.to_datetime(series.index)
    return series.asfreq("D", fill_value=0) 