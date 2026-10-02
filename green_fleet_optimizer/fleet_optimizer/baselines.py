"""Traditional methods used to validate the quantum-inspired optimiser."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import BASELINE_FUEL


def business_as_usual(options: pd.DataFrame, orders: pd.DataFrame):
    """Typical dispatcher: earliest-deadline-first, send the vessel that arrives soonest,
    at full design speed on the direct route, burning conventional fuel."""
    bau = options[(options.speed_level == options.speed_level.max())
                  & (options.route == "Direct") & (options.fuel == BASELINE_FUEL)]
    used, picked = set(), []
    for oid in orders.sort_values("deadline_hr").order_id:
        cand = bau[(bau.order_id == oid) & ~bau.vessel_id.isin(used)]
        if cand.empty:
            continue
        best = cand.sort_values("eta_hr").iloc[0]
        used.add(best.vessel_id)
        picked.append(best.name)
    return picked


def hungarian(options: pd.DataFrame, unserved_cost: dict):
    """Exact linear assignment (Kuhn-Munkres) on the best option per (order, vessel).
    Optimal when there is no port congestion; ignores the QUBO's quadratic terms."""
    best = options.loc[options.groupby(["order_id", "vessel_id"]).objective.idxmin()]
    orders = list(unserved_cost)
    vessels = sorted(best.vessel_id.unique())
    big = 1e12
    n_o, n_v = len(orders), len(vessels)
    cost = np.full((n_o, n_v + n_o), big)
    ids = {}
    for _, r in best.iterrows():
        i, j = orders.index(r.order_id), vessels.index(r.vessel_id)
        cost[i, j] = r.objective
        ids[(i, j)] = r.name
    for i, oid in enumerate(orders):
        cost[i, n_v + i] = unserved_cost[oid]
    rows, cols = linear_sum_assignment(cost)
    return [ids[(i, j)] for i, j in zip(rows, cols) if j < n_v]
