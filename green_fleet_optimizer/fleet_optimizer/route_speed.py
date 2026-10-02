"""Route & speed planning: enumerate every feasible (order, vessel, route, speed, fuel)
option and cost it with the ML predictions. These options become the QUBO variables."""

import numpy as np
import pandas as pd

from .config import DEFAULTS, DIESEL_LHV_GJ_PER_T, ROUTE_OPTIONS, SPEED_LEVELS
from .data_generator import CARGO_COMPATIBILITY, PORT_HANDLING_HR, fuel_flags
from .green_fuel import berth_power, fuel_outcome, range_ok

VESSEL_COLS = ["vessel_id", "vessel_type", "capacity_t", "design_speed_kn", "mcr_kw",
               "sfoc_g_kwh", "aux_kw", "age_yr"]
BALLAST_SPEED = 0.80


def eligible_pairs(fleet, orders):
    """(order, vessel) pairs passing hard pre-filters: availability, cargo type, capacity."""
    pairs = []
    for _, o in orders.iterrows():
        ok = fleet[fleet.available
                   & fleet.vessel_type.isin(CARGO_COMPATIBILITY[o.cargo_type])
                   & (fleet.capacity_t >= o.cargo_t)]
        pairs += [(o.order_id, v) for v in ok.vessel_id]
    return pd.DataFrame(pairs, columns=["order_id", "vessel_id"])


def build_options(data, predictor, carbon_price=DEFAULTS["carbon_price"],
                  emission_weight=DEFAULTS["emission_weight"],
                  late_penalty=DEFAULTS["late_penalty_per_hr"], speed_levels=SPEED_LEVELS):
    fleet, orders, weather, dist = data["fleet"], data["orders"], data["weather"], data["distances"]
    pairs = eligible_pairs(fleet, orders)
    if pairs.empty:
        return pd.DataFrame()
    base = (pairs.merge(orders, on="order_id").merge(weather, on="order_id")
                 .merge(fleet, on="vessel_id"))

    # --- ballast (repositioning) leg: vessel's current port -> cargo origin, empty ---
    base["ballast_nm"] = [dist.loc[a, b] for a, b in zip(base.current_port, base.origin)]
    ballast = base[VESSEL_COLS].copy()
    ballast["speed_kn"] = base.design_speed_kn * BALLAST_SPEED
    ballast["load_frac"] = 0.0
    ballast["distance_nm"] = base.ballast_nm.clip(lower=1.0)
    ballast["wind_kn"] = base.wind_kn * 0.7
    ballast["wave_m"] = base.wave_m * 0.7
    ballast["current_kn"] = 0.0
    bp = predictor.predict(ballast)
    moving = base.ballast_nm > 0
    base["ballast_fuel_t"] = np.where(moving, bp.fuel_t, 0.0)
    base["ballast_hr"] = np.where(moving, bp.sailing_hr, 0.0)

    # --- laden leg for each route option x speed level ---
    grid = []
    for route, r in ROUTE_OPTIONS.items():
        for s in speed_levels:
            g = base.copy()
            g["route"] = route
            g["speed_level"] = s
            g["speed_kn"] = g.design_speed_kn * s
            g["route_nm"] = g.distance_nm * r["distance_factor"]
            g["route_wind_kn"] = g.wind_kn * r["weather_factor"]
            g["route_wave_m"] = g.wave_m * r["weather_factor"]
            grid.append(g)
    opt = pd.concat(grid, ignore_index=True)
    laden = opt[VESSEL_COLS].copy()
    laden["speed_kn"] = opt.speed_kn
    laden["load_frac"] = (opt.cargo_t / opt.capacity_t).clip(0, 1)
    laden["distance_nm"] = opt.route_nm
    laden["wind_kn"] = opt.route_wind_kn
    laden["wave_m"] = opt.route_wave_m
    laden["current_kn"] = opt.current_kn
    lp = predictor.predict(laden)
    opt["laden_fuel_t"] = lp.fuel_t.values
    opt["sailing_hr"] = lp.sailing_hr.values
    opt["exp_delay_hr"] = lp.delay_hr.values

    # --- timeline & deadline ---
    opt["start_hr"] = opt.available_from_hr + opt.ballast_hr + PORT_HANDLING_HR
    opt["eta_hr"] = opt.start_hr + opt.sailing_hr + opt.exp_delay_hr
    opt["late_hr"] = (opt.eta_hr - opt.deadline_hr).clip(lower=0)
    opt["on_time_prob"] = predictor.on_time_probability(
        (opt.deadline_hr - opt.start_hr - opt.sailing_hr - opt.exp_delay_hr).values)
    opt["voyage_hr"] = opt.eta_hr - opt.available_from_hr + PORT_HANDLING_HR
    opt["fuel_t_vlsfo_eq"] = opt.ballast_fuel_t + opt.laden_fuel_t
    opt["energy_gj"] = opt.fuel_t_vlsfo_eq * DIESEL_LHV_GJ_PER_T

    # --- expand by feasible fuel & add costs ---
    vessels = fleet.set_index("vessel_id")
    rows = []
    for rec in opt.to_dict("records"):
        v = vessels.loc[rec["vessel_id"]]
        berth = berth_power(v, 2 * PORT_HANDLING_HR, carbon_price)
        for fuel in fuel_flags(v):
            if not range_ok(v, rec["energy_gj"], fuel):
                continue
            fo = fuel_outcome(rec["energy_gj"], fuel, carbon_price)
            rows.append({**rec, "fuel": fuel, **fo, **berth})
    out = pd.DataFrame(rows)
    out["opex_cost"] = out.opex_per_day * out.voyage_hr / 24.0
    out["late_cost"] = out.late_hr * late_penalty
    out["total_co2e_t"] = out.co2e_t + out.berth_co2e_t
    out["carbon_cost"] = out.total_co2e_t * carbon_price
    out["operating_cost"] = out.fuel_cost + out.opex_cost + out.berth_cost
    out["objective"] = (out.operating_cost + out.late_cost
                        + out.carbon_cost * emission_weight)
    keep = ["order_id", "vessel_id", "vessel_type", "origin", "destination", "cargo_type", "cargo_t",
            "capacity_t", "route", "route_nm", "ballast_nm", "speed_level", "speed_kn", "fuel",
            "fuel_t_vlsfo_eq", "energy_gj", "fuel_energy_gj", "sailing_hr", "exp_delay_hr",
            "available_from_hr", "start_hr", "eta_hr", "deadline_hr", "late_hr", "on_time_prob", "berth_source",
            "fuel_cost", "opex_cost", "berth_cost", "late_cost", "co2e_t", "berth_co2e_t",
            "total_co2e_t", "carbon_cost", "operating_cost", "objective"]
    return out[keep].reset_index(drop=True)
