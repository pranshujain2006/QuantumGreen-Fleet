"""Independent validation of an optimised plan against operational constraints."""

import pandas as pd

from .config import DEFAULTS
from .data_generator import CARGO_COMPATIBILITY, fuel_flags
from .green_fuel import range_ok


def validate_plan(plan: pd.DataFrame, fleet: pd.DataFrame, orders: pd.DataFrame,
                  congestion_window_hr=DEFAULTS["congestion_window_hr"]):
    vessels = fleet.set_index("vessel_id")
    checks = []

    def add(name, bad, severity="FAIL", ok_msg="All satisfied"):
        checks.append({"constraint": name, "status": "PASS" if not bad else severity,
                       "violations": len(bad), "detail": "; ".join(bad) if bad else ok_msg})

    unserved = sorted(set(orders.order_id) - set(plan.order_id))
    add("Every order assigned", [f"{o} unserved" for o in unserved])

    add("Cargo capacity", [f"{r.order_id}: {r.cargo_t:.0f}t > {r.vessel_id} {vessels.loc[r.vessel_id].capacity_t:.0f}t"
                           for r in plan.itertuples() if r.cargo_t > vessels.loc[r.vessel_id].capacity_t])

    add("Cargo / vessel type compatibility",
        [f"{r.order_id} ({r.cargo_type}) on {r.vessel_type}" for r in plan.itertuples()
         if r.vessel_type not in CARGO_COMPATIBILITY[r.cargo_type]])

    add("Vessel availability", [f"{r.vessel_id} not available" for r in plan.itertuples()
                                if not vessels.loc[r.vessel_id].available])

    dup = plan.vessel_id[plan.vessel_id.duplicated()].unique()
    add("One voyage per vessel", [f"{v} double-booked" for v in dup])

    add("Fuel / engine compatibility",
        [f"{r.vessel_id} cannot burn {r.fuel}" for r in plan.itertuples()
         if r.fuel not in fuel_flags(vessels.loc[r.vessel_id])])

    add("Bunker range for chosen fuel",
        [f"{r.vessel_id} {r.fuel} range short" for r in plan.itertuples()
         if not range_ok(vessels.loc[r.vessel_id], r.energy_gj, r.fuel)])

    add("Delivery deadline", [f"{r.order_id} late by {r.late_hr:.1f}h" for r in plan.itertuples()
                              if r.late_hr > 0], severity="WARN")

    clashes = []
    p = plan.reset_index(drop=True)
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p.destination[i] == p.destination[j] and abs(p.eta_hr[i] - p.eta_hr[j]) < congestion_window_hr:
                clashes.append(f"{p.order_id[i]} & {p.order_id[j]} at {p.destination[i]}")
    add("Berth congestion", clashes, severity="WARN")
    return pd.DataFrame(checks)
