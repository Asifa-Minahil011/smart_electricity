"""Time-of-use tariff. Edit TARIFF to change peak hours and rates (same numbers go in the report)."""
import numpy as np

# (start_hour inclusive, end_hour exclusive, price per kWh)
TARIFF = [
    (0, 7, 0.10),    # off-peak
    (7, 18, 0.15),   # standard
    (18, 22, 0.25),  # peak
    (22, 24, 0.15),  # standard
]
CURRENCY = "€"


def hourly_rates(tariff=TARIFF) -> np.ndarray:
    """Return 24 prices (one per hour of day)."""
    rates = np.zeros(24)
    for start, end, price in tariff:
        rates[start:end] = price
    return rates


def daily_cost(profile_kw, rates=None) -> float:
    """Cost of a 24-value kW profile (each value = average kW over that hour = kWh)."""
    rates = hourly_rates() if rates is None else rates
    return float(np.dot(np.asarray(profile_kw, dtype=float), rates))
