import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fleet_optimizer.pipeline import optimise, train  # noqa: E402
from fleet_optimizer.qubo_optimizer import FleetQUBO  # noqa: E402


@pytest.fixture(scope="module")
def trained():
    return train(seed=42, n_orders=8)


@pytest.fixture(scope="module")
def result(trained):
    data, _, _, predictor = trained
    return optimise(data, predictor, sweeps=300, restarts=2)


def test_fuel_model_is_accurate(trained):
    _, _, _, predictor = trained
    best = predictor.metrics.iloc[0]
    assert best["R2"] > 0.9 and best["MAPE_%"] < 15


def test_qubo_energy_matches_plan_cost(result):
    """With no violated penalties, E(x) = sum of chosen costs + congestion pairs."""
    q, sa = result["qubo"], result["sa"]
    plan = sa["plan"]
    linear = plan.objective.sum() / q.scale + sum(q.unserved_cost[o] / q.scale for o in sa["unserved"])
    ids = list(plan.index)
    pairs = sum(2 * q.Q[i, j] for i, j in itertools.combinations(ids, 2))
    assert sa["energy"] == pytest.approx(linear + pairs, rel=1e-9)


def test_one_hot_and_no_vessel_clash(result):
    plan = result["sa"]["plan"]
    assert plan.order_id.is_unique
    assert plan.vessel_id.is_unique
    assert not (result["validation"].status == "FAIL").any()


def test_sa_not_worse_than_classical(result):
    k = result["kpis"].set_index("method").qubo_energy
    assert k["Quantum-inspired QUBO + SA"] <= k["Hungarian (exact, linear only)"] + 1e-9
    assert k["Quantum-inspired QUBO + SA"] <= k["Business-as-usual (greedy)"] + 1e-9


def test_sa_finds_brute_force_optimum_on_tiny_problem(result):
    """Exhaustive check on a 3-order sub-problem."""
    opts = result["options"]
    keep = opts.order_id.unique()[:3]
    sub = opts[opts.order_id.isin(keep)]
    sub = sub.loc[sub.groupby(["order_id", "vessel_id"]).objective.nsmallest(3).index.get_level_values(-1)]
    q = FleetQUBO(sub)
    best = min(q.energy(q.x_from_choice(c)) for c in itertools.product(*[range(len(g)) for g in q.groups]))
    sa = q.solve(sweeps=400, restarts=3, seed=0)
    assert sa["energy"] == pytest.approx(best, rel=1e-9)
    assert np.isfinite(best)


def test_qubo_accepts_arrow_backed_strings(result):
    """pandas 3 stores text as Arrow arrays; building the QUBO must not rely on .values."""
    opts = result["options"].copy()
    for col in ["order_id", "vessel_id", "destination"]:
        opts[col] = opts[col].astype("string[pyarrow]")
    q = FleetQUBO(opts)
    assert q.n == len(opts) + opts.order_id.nunique()
