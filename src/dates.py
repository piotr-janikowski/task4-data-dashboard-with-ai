import re
import pandas as pd


months = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
date_patterns = [
    ("iso", re.compile(r"(\d{4})-(\d{2})-(\d{2})")),
    ("slash", re.compile(r"(\d{2})/(\d{2})/(\d{2})(?!\d)")),
    ("dot", re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")),
    ("mon", re.compile(r"(\d{1,2})-([A-Za-z]+)-(\d{4})")),
    ("ctime", re.compile(r"[A-Za-z]{3}\s+([A-Za-z]{3})\s+(\d{1,2})\s+\d{2}:\d{2}:\d{2}\s+(\d{4})")),
]
time_re = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?\s*([AaPp])?")

def parse_timestamp(raw):
    s = str(raw).strip()

    for kind, pattern in date_patterns:
        m = pattern.search(s)
        if m:
            break
    else:
        return pd.NaT

    a, b, c = m.groups()
    if kind == "iso":
        year, month, day = int(a), int(b), int(c)
    elif kind == "slash":
        month, day, year = int(a), int(b), 2000 + int(c)
    elif kind == "dot":
        day, month, year = int(a), int(b), int(c)
    elif kind == "mon":
        day, month, year = int(a), months[b[:3].lower()], int(c)
    else:
        month, day, year = months[a.lower()], int(b), int(c)

    rest = s if kind == "ctime" else s[:m.start()] + " " + s[m.end():]
    hour, minute, second, ampm = time_re.search(rest).groups() # type: ignore
    hour, minute, second = int(hour), int(minute), int(second or 0)

    if ampm:
        ampm = ampm.lower()
        if ampm == "p" and hour != 12:
            hour += 12
        if ampm == "a" and hour == 12:
            hour = 0
    return pd.Timestamp(year, month, day, hour, minute, second)


def add_date(orders):
    orders = orders.copy()
    orders["timestamp"] = orders["timestamp"].map(parse_timestamp)

    if orders["timestamp"].isna().any():
        raise ValueError("Nie sparsowano części dat – sprawdź formaty.")

    orders["timestamp"] = pd.to_datetime(orders["timestamp"])
    orders["date"] = orders["timestamp"].dt.date
    return orders