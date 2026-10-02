"""Run the full workflow from the command line and print a summary.

    python run_pipeline.py --carbon-price 100 --sweeps 400
"""

import argparse

import pandas as pd

from fleet_optimizer.config import DEFAULTS
from fleet_optimizer.pipeline import optimise, train

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--orders", type=int, default=10)
    ap.add_argument("--carbon-price", type=float, default=DEFAULTS["carbon_price"])
    ap.add_argument("--emission-weight", type=float, default=DEFAULTS["emission_weight"])
    ap.add_argument("--sweeps", type=int, default=DEFAULTS["sa_sweeps"])
    ap.add_argument("--restarts", type=int, default=DEFAULTS["sa_restarts"])
    ap.add_argument("--export-qubo", help="save the QUBO matrix to this .npy file")
    args = ap.parse_args()

    print("[1/4] Generating data, preprocessing and training ML models ...")
    data, clean, prep, predictor = train(args.seed, args.orders)
    print(f"      preprocessing: {prep}")
    print(predictor.metrics.round(3).to_string(index=False))
    print(f"      selected fuel model: {predictor.fuel_model_name}; delay MAE {predictor.delay_mae:.2f} h")

    print("[2/4] Building route/speed/fuel options and QUBO, running simulated annealing ...")
    res = optimise(data, predictor, carbon_price=args.carbon_price,
                   emission_weight=args.emission_weight, sweeps=args.sweeps,
                   restarts=args.restarts, seed=args.seed)
    q = res["qubo"]
    print(f"      {len(res['options'])} options -> {q.n} binary variables, "
          f"{q.n_clash_pairs} vessel-clash pairs, {q.n_congestion_pairs} congestion pairs, "
          f"SA {res['sa']['runtime_s']:.2f}s")

    print("[3/4] Plan comparison")
    print(res["kpis"].round(2).to_string(index=False))
    bau, sa = res["kpis"].iloc[0], res["kpis"].iloc[2]
    for col, label in [("fuel_t_vlsfo_eq", "fuel"), ("operating_cost", "operating cost"), ("co2e_t", "CO2e")]:
        if bau[col]:
            print(f"      {label}: {100 * (bau[col] - sa[col]) / bau[col]:.1f}% saving vs business-as-usual")

    cols = ["order_id", "vessel_id", "route", "speed_kn", "fuel", "eta_hr", "deadline_hr",
            "on_time_prob", "fuel_cost", "total_co2e_t"]
    print("\n      Optimised plan:")
    print(res["sa"]["plan"][cols].round(1).to_string(index=False))

    print("\n[4/4] Constraint validation")
    print(res["validation"].to_string(index=False))

    if args.export_qubo:
        import numpy as np
        np.save(args.export_qubo, q.Q)
        print(f"QUBO saved to {args.export_qubo}")


if __name__ == "__main__":
    main()
