"""End-to-end workflow: raw data -> preprocessing -> ML -> QUBO + SA -> green fuels -> validation."""

import pandas as pd

from .baselines import business_as_usual, hungarian
from .config import DEFAULTS
from .constraints import validate_plan
from .data_generator import make_dataset
from .fuel_model import FleetPredictor
from .preprocessing import clean_history
from .qubo_optimizer import FleetQUBO
from .route_speed import build_options


def plan_kpis(name, result, qubo):
    plan = result["plan"]
    return {
        "method": name,
        "orders_served": len(plan),
        "unserved": len(result["unserved"]),
        "fuel_t_vlsfo_eq": plan.fuel_t_vlsfo_eq.sum(),
        "fuel_cost": plan.fuel_cost.sum(),
        "operating_cost": plan.operating_cost.sum(),
        "co2e_t": plan.total_co2e_t.sum(),
        "late_hr": plan.late_hr.sum(),
        "avg_on_time_prob": plan.on_time_prob.mean() if len(plan) else 0.0,
        "qubo_energy": result["energy"],
        "objective_usd": result["energy"] * qubo.scale,
    }


def train_models(seed=DEFAULTS["seed"]):
    """Preprocess history and fit the ML models. Depends only on the seed."""
    history = make_dataset(seed=seed)["history"]
    clean, prep_report = clean_history(history)
    predictor = FleetPredictor(seed=seed).fit(clean)
    return clean, prep_report, predictor


def train(seed=DEFAULTS["seed"], n_orders=10):
    clean, prep_report, predictor = train_models(seed)
    return make_dataset(seed=seed, n_orders=n_orders), clean, prep_report, predictor


def optimise(data, predictor, carbon_price=DEFAULTS["carbon_price"],
             emission_weight=DEFAULTS["emission_weight"],
             late_penalty=DEFAULTS["late_penalty_per_hr"],
             congestion_window_hr=DEFAULTS["congestion_window_hr"],
             congestion_cost=DEFAULTS["congestion_cost"],
             sweeps=DEFAULTS["sa_sweeps"], restarts=DEFAULTS["sa_restarts"], seed=DEFAULTS["seed"],
             hybrid=True):
    options = build_options(data, predictor, carbon_price, emission_weight, late_penalty)
    qubo = FleetQUBO(options, data["orders"].order_id, congestion_window_hr, congestion_cost)
    bau_ids = business_as_usual(qubo.options, data["orders"])
    hun_ids = hungarian(qubo.options, qubo.unserved_cost)
    bau, hun = qubo.evaluate_plan(bau_ids), qubo.evaluate_plan(hun_ids)
    warm = [qubo.choice_from_option_ids(hun_ids)] if hybrid else []
    sa = qubo.solve(sweeps=sweeps, restarts=restarts, seed=seed, warm_starts=warm)
    kpis = pd.DataFrame([plan_kpis("Business-as-usual (greedy)", bau, qubo),
                         plan_kpis("Hungarian (exact, linear only)", hun, qubo),
                         plan_kpis("Quantum-inspired QUBO + SA", sa, qubo)])
    return {
        "options": qubo.options, "qubo": qubo, "sa": sa, "bau": bau, "hungarian": hun,
        "kpis": kpis,
        "validation": validate_plan(sa["plan"], data["fleet"], data["orders"], congestion_window_hr),
        "bau_validation": validate_plan(bau["plan"], data["fleet"], data["orders"], congestion_window_hr),
    }
