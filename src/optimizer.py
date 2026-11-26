"""Shift flexible appliances to the cheapest hours inside their allowed window."""
from dataclasses import dataclass
import numpy as np
from src.tariff import hourly_rates


@dataclass
class Appliance:
    name: str
    power_kw: float
    duration_h: int
    preferred_start: int      # when the user would normally run it
    earliest: int = 0         # allowed window [earliest, latest_end]
    latest_end: int = 24


def _block_cost(rates, start, duration, power):
    return float(rates[start:start + duration].sum() * power)


def optimize(appliances, rates=None):
    """Return (schedule DataFrame-like list of dicts, baseline_cost, optimized_cost) for the appliances only."""
    rates = hourly_rates() if rates is None else rates
    rows, base_total, opt_total = [], 0.0, 0.0
    for a in appliances:
        base = _block_cost(rates, a.preferred_start, a.duration_h, a.power_kw)
        starts = range(a.earliest, a.latest_end - a.duration_h + 1)
        best = min(starts, key=lambda s: (_block_cost(rates, s, a.duration_h, a.power_kw), abs(s - a.preferred_start)))
        opt = _block_cost(rates, best, a.duration_h, a.power_kw)
        rows.append({"appliance": a.name, "power_kw": a.power_kw, "duration_h": a.duration_h,
                     "preferred_start": a.preferred_start, "optimal_start": best,
                     "cost_before": round(base, 3), "cost_after": round(opt, 3),
                     "saving": round(base - opt, 3)})
        base_total += base
        opt_total += opt
    return rows, base_total, opt_total


def load_profile(appliances, starts):
    """24-value kW profile of the appliances given chosen start hours."""
    p = np.zeros(24)
    for a, s in zip(appliances, starts):
        p[s:s + a.duration_h] += a.power_kw
    return p
