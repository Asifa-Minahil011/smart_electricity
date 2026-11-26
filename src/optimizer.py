"""Shift flexible appliances to the cheapest hours inside their allowed window,
without pushing total household load above an optional power limit."""
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


def optimize(appliances, rates=None, base_load=None, limit_kw=None):
    """Greedy scheduler.

    base_load: 24 kW values already in the house (the forecast). limit_kw: max total kW per hour.
    Appliances are placed biggest-energy first; each takes the cheapest start hour that keeps
    every hour <= limit_kw. If no start fits, the one with the lowest peak is used and flagged.
    Returns (rows, baseline_cost, optimized_cost) for the appliances only.
    """
    rates = hourly_rates() if rates is None else rates
    load = np.zeros(24) if base_load is None else np.array(base_load, dtype=float)
    order = sorted(range(len(appliances)),
                   key=lambda i: -appliances[i].power_kw * appliances[i].duration_h)
    chosen, notes = {}, {}
    for i in order:
        a = appliances[i]
        d = a.duration_h
        starts = list(range(a.earliest, a.latest_end - d + 1))
        if not starts:
            chosen[i], notes[i] = a.preferred_start, "window too short"
            continue
        ok = [s for s in starts
              if limit_kw is None or (load[s:s + d] + a.power_kw <= limit_kw + 1e-9).all()]
        if ok:
            best = min(ok, key=lambda s: (_block_cost(rates, s, d, a.power_kw), abs(s - a.preferred_start)))
            notes[i] = ""
        else:
            best = min(starts, key=lambda s: (load[s:s + d].max() + a.power_kw,
                                              _block_cost(rates, s, d, a.power_kw)))
            notes[i] = "limit exceeded"
        chosen[i] = best
        load[best:best + d] += a.power_kw

    rows, base_total, opt_total = [], 0.0, 0.0
    for i, a in enumerate(appliances):
        base = _block_cost(rates, a.preferred_start, a.duration_h, a.power_kw)
        opt = _block_cost(rates, chosen[i], a.duration_h, a.power_kw)
        rows.append({"appliance": a.name, "power_kw": a.power_kw, "duration_h": a.duration_h,
                     "preferred_start": a.preferred_start, "optimal_start": chosen[i],
                     "cost_before": round(base, 3), "cost_after": round(opt, 3),
                     "saving": round(base - opt, 3), "note": notes[i]})
        base_total += base
        opt_total += opt
    return rows, base_total, opt_total


def load_profile(appliances, starts):
    """24-value kW profile of the appliances given chosen start hours."""
    p = np.zeros(24)
    for a, s in zip(appliances, starts):
        p[s:s + a.duration_h] += a.power_kw
    return p
