"""Green technology evaluation: alternative fuels and shore power on cost, lifecycle
(well-to-wake) emissions and feasibility."""

import pandas as pd

from .config import AUX_MGO, BASELINE_FUEL, FUELS, SHORE_POWER
from .data_generator import fuel_flags


def fuel_outcome(energy_gj, fuel, carbon_price):
    """Cost and emissions of delivering `energy_gj` of diesel-equivalent work with `fuel`."""
    spec = FUELS[fuel]
    gj = energy_gj * spec["efficiency"]
    emissions_t = gj * spec["wtw_gco2e_mj"] / 1000.0
    fuel_cost = gj * spec["price_per_gj"]
    return {"fuel_energy_gj": gj, "fuel_cost": fuel_cost, "co2e_t": emissions_t,
            "carbon_cost": emissions_t * carbon_price}


def range_ok(vessel, energy_gj, fuel):
    if FUELS[fuel]["tank_gj"] is None:
        return True
    tank = FUELS[fuel]["tank_gj"] * vessel["capacity_t"] / 1000.0
    return energy_gj * FUELS[fuel]["efficiency"] <= tank


def berth_power(vessel, berth_hr, carbon_price):
    """Hotel load in port: shore power if the vessel is fitted, otherwise auxiliary engines."""
    kwh = vessel["aux_kw"] * berth_hr
    aux = {"berth_source": "Aux engine (MGO)", "berth_cost": kwh * AUX_MGO["price_per_kwh"],
           "berth_co2e_t": kwh * AUX_MGO["kgco2e_kwh"] / 1000}
    if not vessel["shore_power_ready"]:
        return aux
    shore = {"berth_source": "Shore power", "berth_cost": kwh * SHORE_POWER["price_per_kwh"],
             "berth_co2e_t": kwh * SHORE_POWER["grid_kgco2e_kwh"] / 1000}
    total = lambda o: o["berth_cost"] + o["berth_co2e_t"] * carbon_price
    return shore if total(shore) <= total(aux) else aux


def evaluate_fuels_for_voyage(vessel, energy_gj, carbon_price):
    """Every fuel for one voyage, flagged by compatibility and range feasibility."""
    compatible = set(fuel_flags(vessel))
    base = fuel_outcome(energy_gj, BASELINE_FUEL, carbon_price)
    rows = []
    for fuel, spec in FUELS.items():
        o = fuel_outcome(energy_gj, fuel, carbon_price)
        abated = base["co2e_t"] - o["co2e_t"]
        extra = o["fuel_cost"] - base["fuel_cost"]
        rows.append({
            "fuel": fuel, **o, "total_cost": o["fuel_cost"] + o["carbon_cost"],
            "readiness": spec["readiness"], "engine_compatible": fuel in compatible,
            "range_ok": range_ok(vessel, energy_gj, fuel),
            "abatement_cost_per_t": (extra / abated) if abated > 1e-6 else None,
        })
    df = pd.DataFrame(rows)
    df["feasible"] = df.engine_compatible & df.range_ok
    return df


def fleet_fuel_scenarios(energy_gj_total, carbon_prices):
    """Fleet-wide cost of each fuel across a carbon-price sweep (policy what-if)."""
    rows = []
    for cp in carbon_prices:
        for fuel in FUELS:
            o = fuel_outcome(energy_gj_total, fuel, cp)
            rows.append({"carbon_price": cp, "fuel": fuel,
                         "total_cost": o["fuel_cost"] + o["carbon_cost"], "co2e_t": o["co2e_t"]})
    return pd.DataFrame(rows)


def shore_power_comparison(fleet, berth_hr, carbon_price):
    rows = []
    for _, v in fleet.iterrows():
        kwh = v.aux_kw * berth_hr
        rows.append({
            "vessel_id": v.vessel_id, "shore_power_ready": v.shore_power_ready,
            "aux_cost": kwh * AUX_MGO["price_per_kwh"],
            "aux_co2e_t": kwh * AUX_MGO["kgco2e_kwh"] / 1000,
            "shore_cost": kwh * SHORE_POWER["price_per_kwh"],
            "shore_co2e_t": kwh * SHORE_POWER["grid_kgco2e_kwh"] / 1000,
        })
    df = pd.DataFrame(rows)
    df["saving_incl_carbon"] = (df.aux_cost + df.aux_co2e_t * carbon_price) - \
                               (df.shore_cost + df.shore_co2e_t * carbon_price)
    return df
