import re

eur_to_usd = 1.2

def parse_price(raw):
    s = str(raw).strip()
    currency = "EUR" if ("€" in s or "EUR" in s) else "USD"

    if "¢" in s:
        numbers = re.findall(r"\d+", s)
        value = int(numbers[0]) + int(numbers[1]) / 100
    else:
        value = float(re.search(r"\d+(?:\.\d+)?", s).group()) # type: ignore

    return value, currency


def price_in_usd(raw):
    value, currency = parse_price(raw)
    if currency == "EUR":
        return round(value * eur_to_usd, 2)
    return value


def add_paid_price(orders):
    orders = orders.copy() #lepiej działać na kopii
    orders["unit_price_usd"] = orders["unit_price"].map(price_in_usd)
    orders["paid_price"] = (orders["quantity"] * orders["unit_price_usd"]).round(2)
    return orders