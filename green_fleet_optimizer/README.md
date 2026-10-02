# 🚢 QuantumGreen Fleet

**AI-powered fuel prediction and quantum-inspired optimisation for efficient, cost-effective and sustainable fleet operations.**
Smart India Hackathon 2026 prototype.

QuantumGreen Fleet predicts voyage fuel consumption with machine learning. It then formulates fleet planning (vessel allocation, route, speed and fuel choice) as a **QUBO** and solves it with **simulated annealing**. Alternative fuels (LNG, methanol, ammonia, hydrogen) and shore power are evaluated on cost and lifecycle emissions. Every plan is checked against operational constraints, and the results are shown in an interactive Streamlit dashboard.

## Workflow

```
Raw data (vessels, routes, weather, fuel, cargo)
  └─ Preprocessing ── clean · impute · remove outliers · feature engineering
       ├─ ML prediction ── fuel · time · delay · on-time probability
       └─ QUBO + simulated annealing ── vessel × route × speed × fuel
            └─ Route & speed planning
                 └─ Green fuel analysis ── cost · emissions · feasibility · shore power
                      └─ Constraint validation ── capacity · availability · compatibility · deadlines
                           └─ Dashboard ── compare · decide · optimise
```

| Stage | Module |
|---|---|
| Data (synthetic, physics-based) | `fleet_optimizer/data_generator.py` |
| Clean + transform + prepare | `fleet_optimizer/preprocessing.py` |
| Fuel / delay prediction (Ridge, Random Forest, Gradient Boosting, XGBoost) | `fleet_optimizer/fuel_model.py` |
| Route & speed option generation | `fleet_optimizer/route_speed.py` |
| Green fuel & shore-power evaluation | `fleet_optimizer/green_fuel.py` |
| QUBO formulation + simulated annealing | `fleet_optimizer/qubo_optimizer.py` |
| Traditional baselines (greedy dispatch, Hungarian) | `fleet_optimizer/baselines.py` |
| Constraint validation | `fleet_optimizer/constraints.py` |
| End-to-end pipeline | `fleet_optimizer/pipeline.py` |
| Dashboard | `app.py` |

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py            # interactive dashboard
python run_pipeline.py          # command-line run with summary
python run_pipeline.py --carbon-price 250 --export-qubo fleet_qubo.npy
pytest -q tests                 # tests
```

## The QUBO model

Each binary variable `x_i` is one option: *(order, vessel, route, speed level, fuel)*. Each order also gets an *unserved* slack variable. The optimiser minimises `E(x) = xᵀQx + c`:

| Term | Meaning |
|---|---|
| diagonal `Q_ii` | ML-predicted cost of the option: fuel + opex + berth power + late penalty + weighted carbon cost |
| `A(Σ_{i∈order} x_i − 1)²` | every order served exactly once, or explicitly left unserved |
| `B·x_i·x_j` for the same vessel | one voyage per vessel in the planning horizon |
| `C·x_i·x_j` for the same destination within the berth window | port-congestion waiting cost |

The congestion coupling makes this a true quadratic problem rather than a linear assignment problem. `Q` is exported as a NumPy matrix, so it can be sent to a quantum annealer (for example D-Wave) unchanged. In this prototype it is solved with **simulated annealing**, using order-reassignment moves followed by a zero-temperature descent. With **hybrid mode** on, one restart is warm-started from the classical assignment solution.

## Validation against traditional methods

The dashboard and CLI compare three plans on the same objective:

1. **Business-as-usual**: earliest deadline first, the soonest-arriving vessel, full speed, direct route, VLSFO.
2. **Hungarian algorithm**: exact linear assignment; optimal when there is no port congestion.
3. **QUBO + SA**.

On the default scenario (seed 42, 10 orders), QUBO + SA uses **~31% less fuel**, costs **~15% less to operate** and emits **~33% less GHG** than business-as-usual. It passes every constraint. Across the seeds we tested, SA matched the Hungarian optimum when congestion was light and beat it by up to ~13% when congestion was heavy. A unit test also checks SA against brute-force enumeration on a small instance.

## Data

All data is **synthetic**. Voyage fuel comes from a cube-law propulsion model with weather-added resistance, hull fouling and noise. Fuel prices and well-to-wake emission factors are indicative figures. For real deployments, replace the functions in `data_generator.py` with AIS, noon-report, weather, bunker-price and port feeds. The rest of the pipeline only depends on the column names.

## Future scope

- Real-time AIS, weather, fuel-price and port data.
- Multi-voyage scheduling per vessel.
- Running the exported QUBO on quantum annealing hardware.
- Physics-informed ML (Karniadakis et al., 2021).

## References

- Chen et al. (2023). ML-based prediction of harbour vessel fuel consumption. *Ocean Engineering*.
- Zhou et al. (2023). Ship fuel prediction using metocean and onboard data. *Ocean Engineering*.
- Lucas (2014). Ising formulations of many NP problems. *Frontiers in Physics*.
- Karniadakis et al. (2021). Physics-informed machine learning. *Nature Reviews Physics*.
