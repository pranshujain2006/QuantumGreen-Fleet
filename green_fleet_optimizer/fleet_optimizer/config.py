"""Static reference data: ports, vessel classes, fuels and optimisation defaults."""

# Approximate sea distances (nautical miles) between Indian Ocean hub ports.
PORT_DISTANCES_NM = {
    ("Mumbai (JNPT)", "Chennai"): 1480,
    ("Mumbai (JNPT)", "Kochi"): 590,
    ("Mumbai (JNPT)", "Kandla"): 400,
    ("Mumbai (JNPT)", "Colombo"): 890,
    ("Mumbai (JNPT)", "Jebel Ali"): 1060,
    ("Mumbai (JNPT)", "Singapore"): 2440,
    ("Chennai", "Visakhapatnam"): 340,
    ("Chennai", "Kolkata"): 790,
    ("Chennai", "Colombo"): 590,
    ("Chennai", "Singapore"): 1590,
    ("Kochi", "Colombo"): 320,
    ("Kochi", "Jebel Ali"): 1450,
    ("Kolkata", "Singapore"): 1650,
    ("Visakhapatnam", "Kolkata"): 470,
    ("Kandla", "Jebel Ali"): 820,
}

# Alternative (weather-routed) path: extra distance vs. calmer sea state.
ROUTE_OPTIONS = {
    "Direct": {"distance_factor": 1.00, "weather_factor": 1.00},
    "Weather-routed": {"distance_factor": 1.06, "weather_factor": 0.55},
}

# Vessel classes: capacity (t), design speed (kn), MCR (kW), SFOC (g/kWh),
# auxiliary load at berth (kW), charter/operating cost ($/day).
VESSEL_CLASSES = {
    "Container Feeder": {"capacity": (8000, 18000), "speed": (16, 19), "mcr": (7000, 12000),
                         "sfoc": (175, 190), "aux_kw": (600, 900), "opex": (9000, 14000)},
    "Bulk Carrier":     {"capacity": (25000, 60000), "speed": (12, 14.5), "mcr": (6000, 9500),
                         "sfoc": (170, 185), "aux_kw": (400, 700), "opex": (8000, 12000)},
    "Product Tanker":   {"capacity": (15000, 45000), "speed": (13, 15), "mcr": (6500, 10000),
                         "sfoc": (172, 188), "aux_kw": (700, 1100), "opex": (10000, 15000)},
    "Ro-Ro":            {"capacity": (5000, 12000), "speed": (17, 20), "mcr": (9000, 14000),
                         "sfoc": (180, 195), "aux_kw": (900, 1400), "opex": (11000, 16000)},
    "General Cargo":    {"capacity": (4000, 12000), "speed": (12, 14), "mcr": (3000, 5500),
                         "sfoc": (185, 200), "aux_kw": (250, 450), "opex": (5000, 8000)},
}

# Fuel properties.
#   price_per_gj : delivered energy price ($/GJ)
#   wtw_gco2e_mj : well-to-wake GHG intensity (gCO2e/MJ), incl. methane slip for LNG
#   efficiency   : relative energy needed vs. diesel engine (1.0 = same)
#   tank_gj      : usable bunker energy per 1000 t of vessel capacity (range limit);
#                  None = bunkering available at every port, so range never binds
#   readiness    : 0-1 maturity score (engines, bunkering availability, safety rules)
#   retrofit     : vessel flag required to burn it (None = every vessel can)
FUELS = {
    "VLSFO":          {"price_per_gj": 14.8, "wtw_gco2e_mj": 91.0, "efficiency": 1.00,
                       "tank_gj": None, "readiness": 1.00, "retrofit": None},
    "LNG":            {"price_per_gj": 13.5, "wtw_gco2e_mj": 79.0, "efficiency": 0.98,
                       "tank_gj": 1200, "readiness": 0.80, "retrofit": "lng_ready"},
    "Grey Methanol":  {"price_per_gj": 21.0, "wtw_gco2e_mj": 97.0, "efficiency": 1.00,
                       "tank_gj": 1000, "readiness": 0.70, "retrofit": "methanol_ready"},
    "Bio-Methanol":   {"price_per_gj": 42.0, "wtw_gco2e_mj": 22.0, "efficiency": 1.00,
                       "tank_gj": 1000, "readiness": 0.55, "retrofit": "methanol_ready"},
    "Green Ammonia":  {"price_per_gj": 36.0, "wtw_gco2e_mj": 12.0, "efficiency": 1.03,
                       "tank_gj": 750, "readiness": 0.35, "retrofit": "ammonia_ready"},
    "Green Hydrogen": {"price_per_gj": 45.0, "wtw_gco2e_mj": 8.0, "efficiency": 0.92,
                       "tank_gj": 350, "readiness": 0.25, "retrofit": "hydrogen_ready"},
}
BASELINE_FUEL = "VLSFO"
DIESEL_LHV_GJ_PER_T = 41.0          # energy content of VLSFO

# Shore power at berth vs. running auxiliary engines on MGO.
SHORE_POWER = {"price_per_kwh": 0.14, "grid_kgco2e_kwh": 0.71}
AUX_MGO = {"price_per_kwh": 0.21, "kgco2e_kwh": 0.70 * 1.10}   # incl. upstream share

SPEED_LEVELS = [0.70, 0.80, 0.90, 1.00]   # fraction of design speed

DEFAULTS = {
    "carbon_price": 100.0,        # $ per tCO2e
    "emission_weight": 1.0,       # extra multiplier on carbon cost in the objective
    "late_penalty_per_hr": 4000,  # $ per hour late (demurrage, SLA penalties)
    "congestion_window_hr": 12,   # arrivals at same port within this window collide
    "congestion_cost": 25000,     # $ waiting cost per colliding pair
    "unserved_multiplier": 3.0,   # unserved order cost = multiplier x worst option
    "sa_sweeps": 1500,
    "sa_restarts": 4,
    "seed": 42,
}
