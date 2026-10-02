"""QuantumGreen Fleet - interactive dashboard.   Run:  streamlit run app.py"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fleet_optimizer.config import DEFAULTS, FUELS, SPEED_LEVELS
from fleet_optimizer.data_generator import PORT_HANDLING_HR
from fleet_optimizer.green_fuel import (evaluate_fuels_for_voyage, fleet_fuel_scenarios,
                                        shore_power_comparison)
from fleet_optimizer.pipeline import optimise, train
from fleet_optimizer.preprocessing import FEATURES

st.set_page_config(page_title="QuantumGreen Fleet", page_icon="🚢", layout="wide")

# Validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_COLORS = {"Business-as-usual (greedy)": PALETTE[1],
                 "Hungarian (exact, linear only)": PALETTE[0],
                 "Quantum-inspired QUBO + SA": PALETTE[2]}
FUEL_COLORS = dict(zip(FUELS, PALETTE))
STATUS_ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}


def style(fig, height=360, title=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
                      title=title, legend=dict(orientation="h", y=-0.2), hovermode="closest")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme="streamlit")


@st.cache_resource(show_spinner="Generating data and training ML models ...")
def get_trained(seed, n_orders):
    return train(seed, n_orders)


@st.cache_resource(show_spinner="Building QUBO and running simulated annealing ...", max_entries=8)
def get_optimised(seed, n_orders, carbon_price, emission_weight, late_penalty,
                  window, cong_cost, sweeps, restarts, hybrid):
    data, _, _, predictor = get_trained(seed, n_orders)
    return optimise(data, predictor, carbon_price=carbon_price, emission_weight=emission_weight,
                    late_penalty=late_penalty, congestion_window_hr=window,
                    congestion_cost=cong_cost, sweeps=sweeps, restarts=restarts, seed=seed,
                    hybrid=hybrid)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Scenario")
    seed = st.number_input("Scenario seed", 0, 9999, DEFAULTS["seed"],
                           help="Regenerates fleet, cargo orders and weather")
    n_orders = st.slider("Cargo orders", 4, 16, 10)
    st.header("🌍 Economics & policy")
    carbon_price = st.slider("Carbon price ($/tCO₂e)", 0, 400, int(DEFAULTS["carbon_price"]), 10)
    emission_weight = st.slider("Emission priority (× carbon cost)", 0.0, 5.0,
                                DEFAULTS["emission_weight"], 0.25)
    late_penalty = st.slider("Late delivery penalty ($/h)", 0, 20000,
                             DEFAULTS["late_penalty_per_hr"], 500)
    st.header("⚓ Port congestion")
    window = st.slider("Berth window (h)", 0, 72, DEFAULTS["congestion_window_hr"])
    cong_cost = st.slider("Waiting cost per clash ($)", 0, 300000, DEFAULTS["congestion_cost"], 5000)
    st.header("⚛️ Simulated annealing")
    sweeps = st.slider("Sweeps per restart", 100, 5000, DEFAULTS["sa_sweeps"], 100)
    restarts = st.slider("Restarts", 1, 10, DEFAULTS["sa_restarts"])
    hybrid = st.toggle("Hybrid warm start", True,
                       help="First restart starts from the classical assignment solution")

data, clean, prep_report, predictor = get_trained(seed, n_orders)
res = get_optimised(seed, n_orders, float(carbon_price), float(emission_weight), float(late_penalty),
                    float(window), float(cong_cost), sweeps, restarts, hybrid)
kpis = res["kpis"].set_index("method")
plan = res["sa"]["plan"].sort_values("order_id")
bau_plan = res["bau"]["plan"]
qubo = res["qubo"]

st.title("🚢 QuantumGreen Fleet")
st.caption("ML fuel prediction · quantum-inspired (QUBO + simulated annealing) fleet optimisation · "
           "green fuel evaluation")

tabs = st.tabs(["📊 Overview", "🗂️ Data", "🤖 Fuel prediction", "⚛️ Optimisation",
                "🧭 Route & speed", "🌱 Green fuels", "✅ Constraints"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    b, q = kpis.loc["Business-as-usual (greedy)"], kpis.loc["Quantum-inspired QUBO + SA"]

    def pct(col):
        return f"{-100 * (b[col] - q[col]) / b[col]:.1f}%" if b[col] else None

    c = st.columns(5)
    c[0].metric("Orders served", f"{int(q.orders_served)}/{len(data['orders'])}",
                f"{int(q.orders_served - b.orders_served):+d} vs BAU")
    c[1].metric("Fuel (t VLSFO-eq)", f"{q.fuel_t_vlsfo_eq:,.0f}", pct("fuel_t_vlsfo_eq"), delta_color="inverse")
    c[2].metric("Operating cost", f"${q.operating_cost / 1e6:,.2f}M", pct("operating_cost"), delta_color="inverse")
    c[3].metric("GHG (tCO₂e, well-to-wake)", f"{q.co2e_t:,.0f}", pct("co2e_t"), delta_color="inverse")
    c[4].metric("Mean on-time probability", f"{q.avg_on_time_prob:.0%}",
                f"{100 * (q.avg_on_time_prob - b.avg_on_time_prob):+.0f} pts")
    st.caption("Deltas compare the optimised plan with business-as-usual dispatch "
               "(earliest deadline first, nearest vessel, full speed, direct route, VLSFO). "
               "When the plans serve different numbers of orders, compare the total objective below.")

    cols = st.columns(3)
    for col, (measure, label) in zip(cols, [("objective_usd", "Total objective ($, incl. penalties)"),
                                            ("operating_cost", "Operating cost ($)"),
                                            ("co2e_t", "GHG emissions (tCO₂e)")]):
        fig = go.Figure(go.Bar(
            x=[m.split(" (")[0] for m in kpis.index], y=kpis[measure],
            marker=dict(color=[METHOD_COLORS[m] for m in kpis.index], cornerradius=4),
            text=[f"{v:,.0f}" for v in kpis[measure]], textposition="outside",
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>"))
        with col:
            show(style(fig, 320, label))

    st.subheader("Optimised fleet plan")
    st.dataframe(plan[["order_id", "vessel_id", "vessel_type", "origin", "destination", "cargo_type",
                       "cargo_t", "route", "speed_kn", "fuel", "berth_source", "eta_hr", "deadline_hr",
                       "on_time_prob", "fuel_t_vlsfo_eq", "operating_cost", "total_co2e_t"]]
                 .round(1), hide_index=True, width="stretch")
    if res["sa"]["unserved"]:
        st.warning(f"Unserved orders: {', '.join(res['sa']['unserved'])}")
    st.download_button("⬇️ Download plan (CSV)", plan.to_csv(index=False), "optimised_plan.csv", "text/csv")

    st.subheader("Method comparison")
    st.dataframe(res["kpis"].round(2), hide_index=True, width="stretch")

# ---------------------------------------------------------------- data
with tabs[1]:
    st.subheader("Raw data")
    st.markdown("**Fleet**")
    st.dataframe(data["fleet"], hide_index=True, width="stretch")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Cargo orders**")
        st.dataframe(data["orders"], hide_index=True, width="stretch")
    with c2:
        st.markdown("**Weather forecast (direct route)**")
        st.dataframe(data["weather"], hide_index=True, width="stretch")
    st.subheader("Preprocessing (clean + transform + prepare)")
    c = st.columns(4)
    c[0].metric("Historical voyages", prep_report["rows_in"])
    c[1].metric("Missing weather values imputed", prep_report["missing_imputed"])
    c[2].metric("Outliers removed", prep_report["outliers_removed"])
    c[3].metric("Training rows", prep_report["rows_out"])
    st.dataframe(clean.head(200), hide_index=True, width="stretch")
    st.caption(f"Model features ({len(FEATURES)}): " + ", ".join(FEATURES))

# ---------------------------------------------------------------- ML
with tabs[2]:
    st.subheader("Model comparison (20% held-out voyages)")
    st.dataframe(predictor.metrics.round(3), hide_index=True, width="stretch")
    st.caption(f"Selected: **{predictor.fuel_model_name}** (lowest MAPE). "
               f"Delay model MAE: {predictor.delay_mae:.2f} h. "
               "Install `xgboost` to add XGBoost to the comparison.")
    c1, c2 = st.columns(2)
    tf = predictor.test_frame
    with c1:
        lim = float(tf[["actual_fuel_t", "predicted_fuel_t"]].max().max())
        fig = go.Figure([
            go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="Perfect prediction",
                       line=dict(color="gray", dash="dot", width=1), hoverinfo="skip"),
            go.Scatter(x=tf.actual_fuel_t, y=tf.predicted_fuel_t, mode="markers", name="Voyage",
                       marker=dict(size=8, color=PALETTE[0], opacity=0.55,
                                   line=dict(width=1, color="white")),
                       customdata=tf.vessel_type,
                       hovertemplate="%{customdata}<br>actual %{x:.1f} t<br>predicted %{y:.1f} t<extra></extra>")])
        fig.update_xaxes(title="Actual fuel (t)")
        fig.update_yaxes(title="Predicted fuel (t)")
        show(style(fig, 400, "Predicted vs actual fuel"))
    with c2:
        imp = predictor.feature_importance().head(12)[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                               marker=dict(color=PALETTE[0], cornerradius=4),
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        show(style(fig, 400, "Feature importance"))

    st.subheader("What-if voyage predictor")
    fleet = data["fleet"]
    c = st.columns(4)
    vid = c[0].selectbox("Vessel", fleet.vessel_id,
                         format_func=lambda v: f"{v} · {fleet.set_index('vessel_id').loc[v, 'vessel_type']}")
    v = fleet.set_index("vessel_id").loc[vid]
    distance = c[1].number_input("Distance (nm)", 50, 5000, 1000, 50)
    load = c[2].slider("Load factor", 0.0, 1.0, 0.8, 0.05)
    speed = c[3].slider("Speed (kn)", 6.0, float(v.design_speed_kn * 1.05), float(v.design_speed_kn * 0.85), 0.1)
    c = st.columns(3)
    wind = c[0].slider("Wind (kn)", 0.0, 45.0, 15.0)
    wave = c[1].slider("Wave height (m)", 0.0, 7.0, 2.0, 0.1)
    current = c[2].slider("Current (kn, + = following)", -2.0, 2.0, 0.0, 0.1)

    speeds = np.round(np.linspace(0.6 * v.design_speed_kn, 1.05 * v.design_speed_kn, 30), 2)
    base = {k: v[k] for k in ["vessel_type", "capacity_t", "design_speed_kn", "mcr_kw", "sfoc_g_kwh",
                              "aux_kw", "age_yr"]}
    sweep = pd.DataFrame([{**base, "speed_kn": s, "load_frac": load, "distance_nm": distance,
                           "wind_kn": wind, "wave_m": wave, "current_kn": current}
                          for s in list(speeds) + [speed]])
    pred = predictor.predict(sweep)
    one = pred.iloc[-1]
    energy = one.fuel_t * 41.0
    co2 = energy * FUELS["VLSFO"]["wtw_gco2e_mj"] / 1000
    cost = energy * FUELS["VLSFO"]["price_per_gj"] + v.opex_per_day * (one.sailing_hr + one.delay_hr) / 24
    m = st.columns(5)
    m[0].metric("Fuel", f"{one.fuel_t:,.1f} t")
    m[1].metric("Sailing time", f"{one.sailing_hr:,.1f} h")
    m[2].metric("Expected delay", f"{one.delay_hr:,.1f} h")
    m[3].metric("Voyage cost (VLSFO)", f"${cost:,.0f}")
    m[4].metric("GHG", f"{co2:,.1f} tCO₂e")
    curve = pred.iloc[:-1]
    fig = go.Figure([
        go.Scatter(x=speeds, y=curve.fuel_t, mode="lines", name="Predicted fuel",
                   line=dict(color=PALETTE[0], width=2),
                   customdata=curve.sailing_hr,
                   hovertemplate="%{x:.1f} kn<br>%{y:.1f} t · %{customdata:.0f} h<extra></extra>"),
        go.Scatter(x=[speed], y=[one.fuel_t], mode="markers", name="Selected speed",
                   marker=dict(size=12, color=PALETTE[1], line=dict(width=2, color="white")),
                   hovertemplate="%{x:.1f} kn: %{y:.1f} t<extra></extra>")])
    fig.update_xaxes(title="Speed (kn)")
    fig.update_yaxes(title="Fuel for voyage (t)")
    show(style(fig, 340, "Speed-fuel curve (slow steaming saves fuel roughly with the square of speed)"))

# ---------------------------------------------------------------- optimisation
with tabs[3]:
    sa = res["sa"]
    c = st.columns(5)
    c[0].metric("Binary variables", qubo.n)
    c[1].metric("Candidate options", len(res["options"]))
    c[2].metric("Vessel-clash couplings", f"{qubo.n_clash_pairs:,}")
    c[3].metric("Congestion couplings", f"{qubo.n_congestion_pairs:,}")
    c[4].metric("SA runtime", f"{sa['runtime_s']:.2f} s")
    st.markdown(r"""
**Formulation.** Each option *i* = (order, vessel, route, speed, fuel) is a binary variable $x_i$,
plus one *unserved* slack variable per order. The planner minimises $E(x)=x^\top Q x + c$ with

* **diagonal** – ML-predicted voyage cost: fuel + opex + berth power + lateness + weighted carbon cost
* **one-hot penalty** $A\,(\sum_{i\in o}x_i-1)^2$ – every order is served exactly once (or explicitly left unserved)
* **vessel clash** $B\,x_ix_j$ – a vessel makes one voyage in the planning horizon
* **port congestion** $C\,x_ix_j$ – two arrivals at one port inside the berth window pay a waiting cost

The congestion term is quadratic, so the problem is a true QUBO rather than a linear assignment.
$Q$ can be sent directly to a quantum annealer; here it is solved by **simulated annealing**
with order-reassignment moves, followed by a zero-temperature descent.
""")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for r, t in enumerate(sa["traces"]):
            warm = hybrid and r == 0
            fig.add_trace(go.Scatter(
                y=t, mode="lines", name=f"Restart {r + 1}" + (" (warm)" if warm else ""),
                line=dict(width=2, color=PALETTE[r % len(PALETTE)]),
                hovertemplate="sweep %{x}<br>E = %{y:.3f}<extra></extra>"))
        hun_e = kpis.loc["Hungarian (exact, linear only)", "qubo_energy"]
        fig.add_hline(y=hun_e, line=dict(color="gray", dash="dot", width=1),
                      annotation_text="Hungarian", annotation_position="top right")
        fig.update_xaxes(title="Sweep")
        fig.update_yaxes(title="QUBO energy", type="log")
        show(style(fig, 380, "Annealing convergence"))
    with c2:
        k = min(qubo.n, 160)
        fig = go.Figure(go.Heatmap(z=np.sign(qubo.Q[:k, :k]) * np.log1p(np.abs(qubo.Q[:k, :k])),
                                   colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]],
                                   zmid=0, showscale=False,
                                   hovertemplate="Q[%{y},%{x}]<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(style(fig, 380, f"QUBO matrix (first {k} variables, signed log scale)"))
    buf = io.BytesIO()
    np.save(buf, qubo.Q)
    st.download_button("⬇️ Download QUBO matrix (.npy)", buf.getvalue(), "fleet_qubo.npy")

    st.subheader("Voyage timeline")
    fig = go.Figure()
    p = plan.sort_values("vessel_id")
    fig.add_trace(go.Bar(y=p.vessel_id + " · " + p.order_id, x=p.start_hr - p.available_from_hr,
                         base=p.available_from_hr, orientation="h",
                         name="Reposition + loading", marker=dict(color="#b7d3f6", cornerradius=4),
                         customdata=p.start_hr,
                         hovertemplate="%{y}<br>laden leg starts at %{customdata:.0f} h<extra></extra>"))
    fig.add_trace(go.Bar(y=p.vessel_id + " · " + p.order_id, x=p.eta_hr - p.start_hr, base=p.start_hr,
                         orientation="h", name="Laden voyage", marker=dict(color=PALETTE[0], cornerradius=4),
                         customdata=np.stack([p.origin, p.destination, p.eta_hr], axis=1),
                         hovertemplate="%{customdata[0]} → %{customdata[1]}<br>ETA %{customdata[2]:.0f} h<extra></extra>"))
    fig.add_trace(go.Scatter(y=p.vessel_id + " · " + p.order_id, x=p.deadline_hr, mode="markers",
                             name="Deadline", marker=dict(symbol="line-ns", size=16, line=dict(width=3, color=PALETTE[7])),
                             hovertemplate="deadline %{x:.0f} h<extra></extra>"))
    fig.update_layout(barmode="overlay", bargap=0.35)
    fig.update_xaxes(title="Hours from now")
    show(style(fig, 60 + 34 * len(p)))

# ---------------------------------------------------------------- route & speed
with tabs[4]:
    opts = res["options"]
    oid = st.selectbox("Order", data["orders"].order_id,
                       format_func=lambda o: "{} · {} → {} ({})".format(
                           o, *data["orders"].set_index("order_id").loc[o, ["origin", "destination", "cargo_type"]]))
    chosen = plan[plan.order_id == oid]
    o_opts = opts[opts.order_id == oid]
    if o_opts.empty:
        st.info("No feasible option for this order.")
    else:
        vessel = chosen.vessel_id.iloc[0] if len(chosen) else o_opts.vessel_id.iloc[0]
        vsel = st.selectbox("Vessel", sorted(o_opts.vessel_id.unique()),
                            index=sorted(o_opts.vessel_id.unique()).index(vessel))
        vo = o_opts[o_opts.vessel_id == vsel]
        best_fuel = vo.loc[vo.groupby(["route", "speed_level"]).objective.idxmin()]
        c1, c2 = st.columns(2)
        for col, measure, title in [(c1, "objective", "Voyage objective ($) by speed"),
                                    (c2, "eta_hr", "ETA (h) by speed")]:
            fig = go.Figure()
            for i, (route, g) in enumerate(best_fuel.groupby("route")):
                g = g.sort_values("speed_kn")
                fig.add_trace(go.Scatter(x=g.speed_kn, y=g[measure], mode="lines+markers", name=route,
                                         line=dict(width=2, color=PALETTE[i]),
                                         marker=dict(size=8, line=dict(width=2, color="white")),
                                         customdata=g.fuel,
                                         hovertemplate=route + "<br>%{x:.1f} kn · %{customdata}<br>%{y:,.0f}<extra></extra>"))
            if measure == "eta_hr":
                fig.add_hline(y=vo.deadline_hr.iloc[0], line=dict(color=PALETTE[7], dash="dot", width=1),
                              annotation_text="deadline")
            if len(chosen) and chosen.vessel_id.iloc[0] == vsel:
                fig.add_trace(go.Scatter(x=chosen.speed_kn, y=chosen[measure], mode="markers", name="Chosen",
                                         marker=dict(size=16, symbol="star", color=PALETTE[3],
                                                     line=dict(width=2, color="white")),
                                         hovertemplate="chosen<extra></extra>"))
            fig.update_xaxes(title="Speed (kn)")
            with col:
                show(style(fig, 360, title))
        if len(chosen):
            r = chosen.iloc[0]
            st.success(f"Chosen: **{r.vessel_id}**, **{r.route}** route ({r.route_nm:,.0f} nm), "
                       f"**{r.speed_kn:.1f} kn** ({r.speed_level:.0%} of design), fuel **{r.fuel}**, "
                       f"ETA {r.eta_hr:.0f} h vs deadline {r.deadline_hr:.0f} h "
                       f"(on-time probability {r.on_time_prob:.0%}).")
        st.dataframe(vo[["route", "route_nm", "speed_kn", "fuel", "sailing_hr", "exp_delay_hr", "eta_hr",
                         "late_hr", "on_time_prob", "fuel_t_vlsfo_eq", "fuel_cost", "total_co2e_t", "objective"]]
                     .sort_values("objective").round(1), hide_index=True, width="stretch")

# ---------------------------------------------------------------- green fuels
with tabs[5]:
    st.subheader("Fuel options for one voyage")
    oid2 = st.selectbox("Planned voyage", plan.order_id,
                        format_func=lambda o: f"{o} · {plan.set_index('order_id').loc[o, 'vessel_id']}",
                        key="green_order")
    r = plan.set_index("order_id").loc[oid2]
    vrow = data["fleet"].set_index("vessel_id").loc[r.vessel_id]
    ev = evaluate_fuels_for_voyage(vrow, r.energy_gj, carbon_price)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure([
            go.Bar(x=ev.fuel, y=ev.fuel_cost, name="Fuel cost", marker=dict(color=PALETTE[0], cornerradius=4),
                   hovertemplate="%{x}<br>fuel $%{y:,.0f}<extra></extra>"),
            go.Bar(x=ev.fuel, y=ev.carbon_cost, name=f"Carbon cost @ ${carbon_price}/t",
                   marker=dict(color=PALETTE[1], cornerradius=4),
                   hovertemplate="%{x}<br>carbon $%{y:,.0f}<extra></extra>")])
        fig.update_layout(barmode="stack")
        show(style(fig, 360, "Cost ($)"))
    with c2:
        fig = go.Figure(go.Bar(x=ev.fuel, y=ev.co2e_t, marker=dict(color=PALETTE[2], cornerradius=4),
                               text=[f"{x:,.0f}" for x in ev.co2e_t], textposition="outside",
                               hovertemplate="%{x}: %{y:,.0f} tCO₂e<extra></extra>"))
        show(style(fig, 360, "Well-to-wake GHG (tCO₂e)"))
    show_ev = ev.assign(engine_compatible=ev.engine_compatible.map({True: "✅", False: "❌ retrofit"}),
                        range_ok=ev.range_ok.map({True: "✅", False: "❌ bunker stop"}),
                        feasible=ev.feasible.map({True: "✅", False: "❌"}))
    st.dataframe(show_ev.round(1), hide_index=True, width="stretch")
    st.caption("Abatement cost = extra fuel spend per tonne CO₂e avoided vs VLSFO. "
               "Readiness reflects engine maturity, bunkering availability and safety rules.")

    st.subheader("Fleet-wide: which fuel wins at what carbon price?")
    total_gj = plan.energy_gj.sum()
    sc = fleet_fuel_scenarios(total_gj, list(range(0, 501, 25)))
    fig = go.Figure()
    for fuel, g in sc.groupby("fuel", sort=False):
        fig.add_trace(go.Scatter(x=g.carbon_price, y=g.total_cost / 1e6, mode="lines", name=fuel,
                                 line=dict(width=2, color=FUEL_COLORS[fuel]),
                                 hovertemplate=fuel + "<br>$%{x}/t → $%{y:.2f}M<extra></extra>"))
    fig.add_vline(x=carbon_price, line=dict(color="gray", dash="dot", width=1), annotation_text="current")
    fig.update_xaxes(title="Carbon price ($/tCO₂e)")
    fig.update_yaxes(title="Fuel + carbon cost ($M)")
    show(style(fig, 400))

    st.subheader("Shore power at berth")
    sp = shore_power_comparison(data["fleet"], 2 * PORT_HANDLING_HR, carbon_price)
    c = st.columns(3)
    ready = sp[sp.shore_power_ready]
    c[0].metric("Vessels fitted for shore power", f"{len(ready)}/{len(sp)}")
    c[1].metric("GHG saved per port call (fitted fleet)",
                f"{(ready.aux_co2e_t - ready.shore_co2e_t).sum():,.1f} t")
    c[2].metric("Saving per call incl. carbon (fitted fleet)", f"${ready.saving_incl_carbon.sum():,.0f}")
    st.dataframe(sp.round(2), hide_index=True, width="stretch")

# ---------------------------------------------------------------- constraints
with tabs[6]:
    c1, c2 = st.columns(2)
    for col, name, val in [(c1, "Optimised plan (QUBO + SA)", res["validation"]),
                           (c2, "Business-as-usual plan", res["bau_validation"])]:
        with col:
            st.subheader(name)
            fails = (val.status == "FAIL").sum()
            warns = (val.status == "WARN").sum()
            (st.success if not fails and not warns else st.warning if not fails else st.error)(
                f"{len(val) - fails - warns} passed · {warns} warnings · {fails} failed")
            st.dataframe(val.assign(status=val.status.map(lambda s: f"{STATUS_ICON[s]} {s}")),
                         hide_index=True, width="stretch")
