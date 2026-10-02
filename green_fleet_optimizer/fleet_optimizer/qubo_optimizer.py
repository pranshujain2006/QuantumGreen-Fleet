"""Quantum-inspired fleet optimisation.

Every candidate option (order, vessel, route, speed, fuel) is a binary variable x_i,
plus one "unserved" slack variable per order. The plan's energy is E(x) = x^T Q x + c:

  * diagonal     : scaled cost of the option (fuel + opex + berth + lateness + weighted carbon)
  * one-hot      : A * (sum_{i in order} x_i - 1)^2            -> every order handled exactly once
  * vessel clash : B * x_i x_j  for two orders on the same vessel -> one voyage per vessel
  * congestion   : C * x_i x_j  for arrivals at the same port within the berth window

The quadratic congestion term is what makes this a true QUBO rather than a plain
assignment problem. Q can be handed as-is to a quantum annealer (e.g. D-Wave); here it
is solved with simulated annealing, the classical analogue of quantum annealing.
"""

import time

import numpy as np
import pandas as pd

from .config import DEFAULTS


class FleetQUBO:
    def __init__(self, options: pd.DataFrame, order_ids=None, congestion_window_hr=DEFAULTS["congestion_window_hr"],
                 congestion_cost=DEFAULTS["congestion_cost"],
                 unserved_multiplier=DEFAULTS["unserved_multiplier"]):
        self.options = options.reset_index(drop=True)
        # Orders with no feasible option still get a slack variable (reported unserved).
        self.orders = list(dict.fromkeys(list(order_ids if order_ids is not None else [])
                                         + list(self.options.order_id)))
        n_opt = len(self.options)
        self.n = n_opt + len(self.orders)                   # options + slack vars
        self.scale = float(self.options.objective.median())

        cost = self.options.objective.values / self.scale
        self.groups = []
        linear = np.zeros(self.n)
        linear[:n_opt] = cost
        self.unserved_cost = {}
        for k, oid in enumerate(self.orders):
            members = np.flatnonzero(self.options.order_id.values == oid)
            slack = n_opt + k
            worst = cost[members].max() if len(members) else cost.max()
            linear[slack] = unserved_multiplier * worst
            self.unserved_cost[oid] = linear[slack] * self.scale
            self.groups.append(np.append(members, slack))

        self.A = self.B = 2.0 * linear.max()   # penalties dominate any cost gain
        self.C = congestion_cost / self.scale
        Q = np.zeros((self.n, self.n))
        Q[np.diag_indices(self.n)] = linear
        self.constant = 0.0
        for g in self.groups:                  # A (sum x - 1)^2 = A(1 - sum x + 2 sum_{i<j} x_i x_j)
            Q[g[:, None], g[None, :]] += self.A
            Q[g, g] -= 2 * self.A              # diag gets A (from x_i^2) - 2A (linear) = -A
            self.constant += self.A

        vessel = self.options.vessel_id.values
        order = self.options.order_id.values
        dest = self.options.destination.values
        eta = self.options.eta_hr.values
        diff_order = order[:, None] != order[None, :]
        clash = diff_order & (vessel[:, None] == vessel[None, :])
        congest = diff_order & ~clash & (dest[:, None] == dest[None, :]) & \
            (np.abs(eta[:, None] - eta[None, :]) < congestion_window_hr)
        # symmetric storage: pair weight w -> Q_ij = Q_ji = w / 2
        Q[:n_opt, :n_opt] += clash * (self.B / 2) + congest * (self.C / 2)
        self.Q = Q
        self.n_clash_pairs = int(clash.sum() // 2)
        self.n_congestion_pairs = int(congest.sum() // 2)

    # ---- helpers ------------------------------------------------------------
    def energy(self, x):
        return float(x @ self.Q @ x + self.constant)

    def x_from_choice(self, choice):
        x = np.zeros(self.n)
        for g, c in zip(self.groups, choice):
            x[g[c]] = 1.0
        return x

    def choice_from_option_ids(self, option_ids):
        """Map a set of selected option row ids (at most one per order) to a group choice."""
        chosen = set(option_ids)
        choice = []
        for g in self.groups:
            hit = [k for k, idx in enumerate(g[:-1]) if idx in chosen]
            choice.append(hit[0] if hit else len(g) - 1)
        return choice

    def decode(self, x):
        sel = np.flatnonzero(x[:len(self.options)] > 0.5)
        plan = self.options.loc[sel].copy()
        served = set(plan.order_id)
        unserved = [o for o in self.orders if o not in served]
        return plan, unserved

    # ---- simulated annealing -----------------------------------------------
    def solve(self, sweeps=DEFAULTS["sa_sweeps"], restarts=DEFAULTS["sa_restarts"],
              seed=DEFAULTS["seed"], t_start=None, t_end=None, warm_starts=()):
        """Simulated annealing with one-hot preserving moves: each move reassigns one order
        to another of its variables (a paired flip x_a: 1->0, x_b: 0->1), so the search
        stays on valid one-hot states while the energy is always evaluated from Q.

        warm_starts: optional group choices (e.g. from a classical solver) used as the
        initial state of the first restarts - a hybrid quantum-classical strategy. The
        remaining restarts start from random states."""
        rng = np.random.default_rng(seed)
        Q = self.Q
        diag = np.diag(Q).copy()
        off = Q - np.diag(diag)
        if t_start is None:
            t_start = max(np.std(diag[:len(self.options)]), 1e-3) * 2.0
        if t_end is None:
            t_end = t_start * 1e-3
        n_moves = sweeps * len(self.groups)
        temps = t_start * (t_end / t_start) ** (np.arange(n_moves) / max(n_moves - 1, 1))

        started = time.perf_counter()
        best = {"energy": np.inf}
        traces = []
        for r in range(restarts):
            if r < len(warm_starts):
                choice = list(warm_starts[r])
            else:
                choice = [int(rng.integers(len(g))) for g in self.groups]
            x = self.x_from_choice(choice)
            field = off @ x
            e = self.energy(x)
            trace = []
            run_best_e, run_best_choice = e, list(choice)
            for m in range(n_moves):
                k = int(rng.integers(len(self.groups)))
                g = self.groups[k]
                if len(g) < 2:
                    continue
                new = int(rng.integers(len(g) - 1))
                if new >= choice[k]:
                    new += 1
                a, b = g[choice[k]], g[new]
                delta = -(diag[a] + 2 * field[a]) + (diag[b] + 2 * (field[b] - off[b, a]))
                if delta <= 0 or rng.random() < np.exp(-delta / temps[m]):
                    field += off[:, b] - off[:, a]
                    choice[k] = new
                    e += delta
                    if e < run_best_e - 1e-12:
                        run_best_e, run_best_choice = e, list(choice)
                if m % len(self.groups) == 0:
                    trace.append(e)
            # zero-temperature polish from the run's best state
            run_best_choice, run_best_e = self._descend(run_best_choice, diag, off)
            trace.append(run_best_e)
            traces.append(trace)
            if run_best_e < best["energy"]:
                best = {"energy": run_best_e, "choice": run_best_choice, "restart": r,
                        "warm": r < len(warm_starts)}

        x = self.x_from_choice(best["choice"])
        plan, unserved = self.decode(x)
        return {"x": x, "energy": self.energy(x), "plan": plan, "unserved": unserved,
                "traces": traces, "runtime_s": time.perf_counter() - started,
                "n_vars": self.n, "best_restart": best["restart"],
                "best_from_warm_start": best["warm"], "restart_best": [min(t) for t in traces]}

    def _descend(self, choice, diag, off):
        """Steepest descent: repeatedly apply the best single-order reassignment."""
        choice = list(choice)
        x = self.x_from_choice(choice)
        field = off @ x
        while True:
            best_delta, best_move = -1e-12, None
            for k, g in enumerate(self.groups):
                a = g[choice[k]]
                deltas = -(diag[a] + 2 * field[a]) + diag[g] + 2 * (field[g] - off[g, a])
                deltas[choice[k]] = 0.0
                j = int(np.argmin(deltas))
                if deltas[j] < best_delta:
                    best_delta, best_move = deltas[j], (k, j)
            if best_move is None:
                break
            k, j = best_move
            field += off[:, self.groups[k][j]] - off[:, self.groups[k][choice[k]]]
            choice[k] = j
        return choice, self.energy(self.x_from_choice(choice))

    def evaluate_plan(self, option_ids):
        """Energy and decoded plan for an externally produced plan (for baselines)."""
        x = self.x_from_choice(self.choice_from_option_ids(option_ids))
        plan, unserved = self.decode(x)
        return {"x": x, "energy": self.energy(x), "plan": plan, "unserved": unserved}
