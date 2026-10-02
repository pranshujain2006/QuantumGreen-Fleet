"""Synthetic but physically grounded data: fleet, ports, cargo orders, weather and
historical voyages. Replace these generators with real AIS / noon-report / weather
feeds in production; the rest of the pipeline only depends on the column names."""

import numpy as np
import pandas as pd

from .config import FUELS, PORT_DISTANCES_NM, VESSEL_CLASSES

CARGO_COMPATIBILITY = {
    "Containers": ["Container Feeder", "General Cargo"],
    "Dry Bulk": ["Bulk Carrier", "General Cargo"],
    "Liquid Bulk": ["Product Tanker"],
    "Vehicles": ["Ro-Ro"],
    "Break Bulk": ["General Cargo", "Bulk Carrier"],
}
PORT_HANDLING_HR = 24.0
AUX_SFOC = 220.0  # g/kWh for auxiliary engines


def port_list():
    return sorted({p for pair in PORT_DISTANCES_NM for p in pair})


def distance_matrix():
    """All-pairs shortest sea distance (Floyd-Warshall over the known legs)."""
    ports = port_list()
    idx = {p: i for i, p in enumerate(ports)}
    n = len(ports)
    d = np.full((n, n), np.inf)
    np.fill_diagonal(d, 0.0)
    for (a, b), nm in PORT_DISTANCES_NM.items():
        d[idx[a], idx[b]] = d[idx[b], idx[a]] = nm
    for k in range(n):
        d = np.minimum(d, d[:, [k]] + d[[k], :])
    return pd.DataFrame(d, index=ports, columns=ports)


def physics_fuel(mcr, design_speed, sfoc, aux_kw, age, speed, load_frac,
                 distance, wind, wave, current):
    """Ground-truth fuel model (tonnes) used to simulate history.
    Propulsion power follows the cube law with weather and hull-fouling added resistance."""
    v_eff = np.maximum(speed + current, 3.0)
    hours = distance / v_eff
    added = 0.010 * wind + 0.06 * np.power(wave, 1.5)
    fouling = 1.0 + 0.008 * age
    power = mcr * np.power(speed / design_speed, 3) * (0.75 + 0.25 * load_frac) * (1 + added) * fouling
    power = np.minimum(power, 1.10 * mcr)
    main = power * sfoc * hours / 1e6
    aux = aux_kw * AUX_SFOC * hours / 1e6
    return main + aux, hours


def generate_fleet(rng, n_per_class=None):
    n_per_class = n_per_class or {"Container Feeder": 5, "Bulk Carrier": 4, "Product Tanker": 4,
                                  "Ro-Ro": 3, "General Cargo": 4}
    ports = port_list()
    rows = []
    for cls, n in n_per_class.items():
        spec = VESSEL_CLASSES[cls]
        for _ in range(n):
            u = lambda key: rng.uniform(*spec[key])
            age = int(rng.integers(1, 25))
            rows.append({
                "vessel_type": cls,
                "capacity_t": round(u("capacity"), -2),
                "design_speed_kn": round(u("speed"), 1),
                "mcr_kw": round(u("mcr"), -1),
                "sfoc_g_kwh": round(u("sfoc"), 1),
                "aux_kw": round(u("aux_kw"), -1),
                "opex_per_day": round(u("opex"), -2),
                "age_yr": age,
                # Newer ships are more likely to be dual-fuel ready.
                "lng_ready": bool(rng.random() < (0.45 if age < 10 else 0.15)),
                "methanol_ready": bool(rng.random() < (0.40 if age < 8 else 0.10)),
                "ammonia_ready": bool(rng.random() < (0.15 if age < 5 else 0.03)),
                "hydrogen_ready": bool(rng.random() < 0.08 and cls in ("Ro-Ro", "General Cargo")),
                "shore_power_ready": bool(rng.random() < 0.55),
                "available": bool(rng.random() > 0.08),
                "available_from_hr": float(rng.choice([0, 0, 0, 12, 24, 48])),
                "current_port": str(rng.choice(ports)),
            })
    fleet = pd.DataFrame(rows)
    fleet.insert(0, "vessel_id", [f"V{i + 1:02d}" for i in range(len(fleet))])
    return fleet


def generate_orders(rng, fleet, n_orders=10):
    """Each order is sized for a reference vessel that is then reserved, so at least one
    feasible plan always exists and every deadline is reachable by that vessel."""
    ports = port_list()
    dist = distance_matrix()
    rows, reserved = [], set()
    cargo_types = list(CARGO_COMPATIBILITY)
    for _ in range(n_orders * 50):
        if len(rows) == n_orders:
            break
        cargo = str(rng.choice(cargo_types))
        eligible = fleet[fleet.vessel_type.isin(CARGO_COMPATIBILITY[cargo]) & fleet.available
                         & ~fleet.vessel_id.isin(reserved)]
        if eligible.empty:
            continue
        origin, dest = rng.choice(ports, size=2, replace=False)
        nm = dist.loc[origin, dest]
        ref = eligible.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        reserved.add(ref.vessel_id)
        tonnes = round(ref.capacity_t * rng.uniform(0.45, 0.95), -2)
        ballast_hr = dist.loc[ref.current_port, origin] / (0.8 * ref.design_speed_kn)
        transit_hr = nm / (0.85 * ref.design_speed_kn)
        deadline = (ref.available_from_hr + ballast_hr + PORT_HANDLING_HR
                    + transit_hr * rng.uniform(1.1, 1.6) + rng.uniform(12, 72))
        rows.append({
            "origin": origin, "destination": dest, "distance_nm": float(nm),
            "cargo_type": cargo, "cargo_t": tonnes, "deadline_hr": round(deadline, 1),
            "freight_value_usd": round(tonnes * rng.uniform(18, 45), -2),
        })
    orders = pd.DataFrame(rows)
    orders.insert(0, "order_id", [f"O{i + 1:02d}" for i in range(len(orders))])
    return orders


def generate_weather(rng, orders):
    """Forecast sea state per order on the direct route (monsoon-season spread)."""
    return pd.DataFrame({
        "order_id": orders.order_id,
        "wind_kn": rng.gamma(4.0, 4.0, len(orders)).round(1).clip(2, 40),
        "wave_m": rng.gamma(3.0, 0.7, len(orders)).round(2).clip(0.3, 6),
        "current_kn": rng.normal(0, 0.6, len(orders)).round(2),
    })


def generate_history(rng, fleet, n=3000):
    """Historical voyages with realised fuel and delay, the ML training set."""
    v = fleet.iloc[rng.integers(0, len(fleet), n)].reset_index(drop=True)
    speed = v.design_speed_kn * rng.uniform(0.62, 1.02, n)
    load = rng.uniform(0.0, 1.0, n)
    distance = rng.uniform(150, 2600, n)
    wind = rng.gamma(4.0, 4.0, n).clip(1, 45)
    wave = rng.gamma(3.0, 0.7, n).clip(0.2, 7)
    current = rng.normal(0, 0.6, n)
    fuel, hours = physics_fuel(v.mcr_kw, v.design_speed_kn, v.sfoc_g_kwh, v.aux_kw, v.age_yr,
                               speed, load, distance, wind, wave, current)
    fuel = fuel * rng.lognormal(0, 0.06, n)
    delay = np.maximum(0, 0.35 * np.power(wave, 1.6) + 0.03 * wind + 0.004 * distance / 10
                       + rng.exponential(1.5, n) - 1.0)
    return pd.DataFrame({
        "vessel_id": v.vessel_id, "vessel_type": v.vessel_type, "capacity_t": v.capacity_t,
        "design_speed_kn": v.design_speed_kn, "mcr_kw": v.mcr_kw, "sfoc_g_kwh": v.sfoc_g_kwh,
        "aux_kw": v.aux_kw, "age_yr": v.age_yr, "speed_kn": speed.round(2),
        "load_frac": load.round(3), "distance_nm": distance.round(1), "wind_kn": wind.round(1),
        "wave_m": wave.round(2), "current_kn": current.round(2), "sailing_hr": hours.round(2),
        "fuel_t": fuel.round(3), "delay_hr": delay.round(2),
    })


def make_dataset(seed=42, n_orders=10, n_history=3000, missing_rate=0.02):
    rng = np.random.default_rng(seed)
    fleet = generate_fleet(rng)
    orders = generate_orders(rng, fleet, n_orders)
    weather = generate_weather(rng, orders)
    history = generate_history(rng, fleet, n_history)
    # Inject realistic noise for the preprocessing stage to clean.
    for col in ["wind_kn", "wave_m", "current_kn"]:
        mask = rng.random(len(history)) < missing_rate
        history.loc[mask, col] = np.nan
    outliers = rng.choice(len(history), size=max(1, n_history // 200), replace=False)
    history.loc[outliers, "fuel_t"] *= 8  # sensor/entry errors
    return {"fleet": fleet, "orders": orders, "weather": weather, "history": history,
            "distances": distance_matrix()}


def fuel_flags(vessel):
    """Fuels a vessel can burn given its retrofit flags."""
    return [f for f, spec in FUELS.items() if spec["retrofit"] is None or bool(vessel[spec["retrofit"]])]


if __name__ == "__main__":
    data = make_dataset()
    for k, df in data.items():
        print(k, df.shape)
