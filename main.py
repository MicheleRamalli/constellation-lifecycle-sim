"""Run the nominal case of the constellation and save the figures into figures/.

Constellation: Walker N_PLANES/N_SLOTS (coverage.py), planes simulated as
independent processes (lifecycle.simulate_constellation). At each time the
revisit at REVISIT_LAT_DEG comes from the lookup table revisit(n_1, n_2)
(coverage.py, quasi-stationarity ASSUMPTION). Two alternative requirements:
max revisit <= REVISIT_MAX_HOURS, mean revisit <= REVISIT_MEAN_HOURS.
Availability = fraction of time meeting the requirement, averaged from the end
of the deployment (last plane launch + commissioning) to the end of the horizon.

Strategies, applied to every plane:
    a)   launch on failure
    b)   in-orbit spares, (s, S) policy, S spares per plane
    c)   early ordering of the retirement replacement
    b+c) spares and early ordering
    d)   fixed launch cadence per plane, plane 2 shifted by T/2, with the search
         of the minimum cadence

Figures:
- revisit_vs_latitude.png         : max and mean revisit vs latitude of the reference Walker
- revisit_table.png               : lookup table revisit(n_1, n_2), max and mean
- active_vs_time.png              : operational satellites vs time, one panel per strategy
- availability_vs_order_time.png  : availability vs ORDER_TO_LAUNCH_DAYS sweep
- cost_vs_availability.png        : launch rate vs fraction of runs meeting the
                                    mission criterion, one panel per revisit requirement
- iride_scenario.png              : IRIDE Eaglet II from public TLEs: in-plane phases and
                                    revisit vs latitude (real, ideal, reference Walker)
- decay.png                       : Eaglet II orbital lifetime vs altitude; decay rate from
                                    the IRIDE TLEs vs the drag model
- end_of_life.png                 : residual lifetime and collision probability (> 1 cm,
                                    MASTER-8) of a failed satellite vs altitude
- solar_cycle.png                 : solar activity scenarios (strong cycle = ECSS Table A-1,
                                    ESA forecast 2021 = fap_mon.dat) and Eaglet II residual
                                    lifetime vs the phase of the end of life
- casualty_risk.png               : on-ground casualty risk (DRAMA SARA): E_c of one re-entry
                                    vs population year and expected casualties of the
                                    re-entries of each configuration
- altitude_trade.png              : altitude trade-off 400-800 km: launch rate of the cheapest
                                    strategy vs debris indicators, constant activity and
                                    solar cycle (only with --trade)

Second scenario: IRIDE Eaglet II, planes reconstructed from the derived TLE table
(inputs/iride_eaglet2_elements.csv, written by fetch_tle.py; planes.py).

Usage:
    python main.py          fast mode (default): the check of T* with the extra seeds
                            params.SEEDS_CHECK is skipped and the altitude trade-off is not
                            run; every other result is identical
    python main.py --full   also runs the T* check with the extra seeds
    python main.py --trade  also runs the altitude trade-off (altitude_trade.png)
    Run time on the development laptop (2026-11-11): default ~8 min, --trade ~53 min
    (solar cycle tables at 700-800 km take most of the extra time).
    --full adds ~4 min (measured before v0.3, not re-measured).
    To follow the progress when the output goes to a file: python -u main.py > run.log
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from scipy.integrate import quad
from scipy.optimize import brentq

import casualty
import coverage
import debris
import decay
import params
import planes
import tle
from analytic import (binomial_interval, reference_launch_rate, slot_availability_pre_order,
                      slot_availability_renewal_reward)
from lifecycle import (LAUNCH_FAILURE, RETIREMENT, constellation_availability,
                       constellation_revisit_series, simulate_constellation, simulate_fixed_cadence, simulate_plane,
                       time_to_requirement)
from reliability import reliability_analytic
from search import bisect_interval, search_max_interval

FIG_DIR = "figures"
COVERAGE_CACHE = os.path.join("data", "revisit_table.npz")
DAY_S = params.SECONDS_PER_DAY
YEAR_S = params.SECONDS_PER_YEAR
RELIABILITY = {"model": "weibull", "beta": params.WEIBULL_BETA,
               "theta_s": params.WEIBULL_THETA_YEARS * YEAR_S}
ORDER_SWEEP_DAYS = np.arange(90.0, 731.0, 80.0)  # 90, 170, ..., 730 days
PLANE_LAUNCH_S = np.arange(params.N_PLANES) * params.DEPLOY_INTERVAL_DAYS * DAY_S  # D1
DEPLOY_END_S = PLANE_LAUNCH_S[-1] + params.COMMISSIONING_DAYS * DAY_S

STRATEGIES = {
    "a) Launch on failure": dict(n_spares=0, reorder_level=0, pre_order_retirement=False),
    f"b) {params.N_SPARES} in-orbit spare(s) per plane": dict(
        n_spares=params.N_SPARES, reorder_level=params.REORDER_LEVEL, pre_order_retirement=False),
    "c) Early ordering": dict(n_spares=0, reorder_level=0, pre_order_retirement=True),
    f"b+c) {params.N_SPARES} spare(s) per plane + early ordering": dict(
        n_spares=params.N_SPARES, reorder_level=params.REORDER_LEVEL, pre_order_retirement=True),
}

# Revisit requirements: limit and text
REVISIT_REQS = {
    "max": {"limit_s": params.REVISIT_MAX_HOURS * 3600,
            "text": f"max revisit <= {params.REVISIT_MAX_HOURS:.0f} h"},
    "mean": {"limit_s": params.REVISIT_MEAN_HOURS * 3600,
             "text": f"mean revisit <= {params.REVISIT_MEAN_HOURS:.0f} h"},
}

_COVERAGE = {}


def coverage_data():
    """Coverage results from the cache in data/ (built once, see coverage.py)."""
    if not _COVERAGE:
        _COVERAGE.update(coverage.load_or_build_coverage_data(COVERAGE_CACHE))
    return _COVERAGE


def reference_tables():
    """Lookup tables of the reference Walker: {"max": [s], "mean": [s]}."""
    data = coverage_data()
    return {"max": data["table_planes_max_s"], "mean": data["table_planes_mean_s"]}


# A scenario = what the constellation runs need: slots per plane, launch date of
# each plane, end of deployment, lookup tables of the revisit.
REFERENCE = {"name": f"Walker {params.N_PLANES}/{params.N_SLOTS}/0", "n_slots": params.N_SLOTS,
             "plane_launch_s": PLANE_LAUNCH_S, "deploy_end_s": DEPLOY_END_S,
             "tables": reference_tables}


def hours(x_s):
    """Seconds -> hours as text, 'inf' when never revisited."""
    return "   inf" if not np.isfinite(x_s) else f"{x_s / 3600:6.2f}"


# ---------------------------------------------------------------------------
# Constellation runs
# ---------------------------------------------------------------------------

def run_strategy(strategy, order_to_launch_days, n_runs=params.N_RUNS, seed=params.SEED,
                 scenario=REFERENCE):
    """Strategies a/b/c/b+c on every plane: one launch per plane at its launch date.

    Inputs:
        strategy             : dict with n_spares, reorder_level, pre_order_retirement
        order_to_launch_days : order-to-launch time [days]
        n_runs, seed         : Monte Carlo runs and master seed
        scenario             : REFERENCE or the IRIDE scenario (slots, launch dates, tables)
    Output: constellation output (lifecycle.simulate_constellation)
    """
    inputs = []
    for t_launch_s in scenario["plane_launch_s"]:
        inputs.append(dict(
            n_slots=scenario["n_slots"], n_runs=n_runs,
            sim_duration_s=params.SIM_DURATION_YEARS * YEAR_S, dt_s=params.DT_DAYS * DAY_S,
            initial_launch_s=np.full(scenario["n_slots"] + strategy["n_spares"], t_launch_s),
            order_to_launch_s=order_to_launch_days * DAY_S,
            commissioning_s=params.COMMISSIONING_DAYS * DAY_S,
            design_life_s=params.DESIGN_LIFE_YEARS * YEAR_S,
            launch_success_prob=params.LAUNCH_SUCCESS_PROB,
            sats_per_launch=params.SATS_PER_LAUNCH, reliability=RELIABILITY, **strategy))
    return simulate_constellation(simulate_plane, inputs, seed)


def run_cadence(k, interval_days, seed, n_runs, scenario=REFERENCE):
    """Strategy d) on every plane: k satellites every interval_days.

    Plane p is deployed in one launch at plane_launch_s[p] carrying n_slots + k
    satellites: the k cadence satellites of that date fly with the deployment and
    act at once as spares (no operator would leave the plane without spares in
    the infant-mortality period; consistent with b). The cadence dates of plane p
    are p * T/2 + j * T (plane 2 shifted by T/2, ASSUMPTION in params.py); the
    first cadence launch is the first of those dates after the deployment.
    """
    interval_s = interval_days * DAY_S
    inputs = []
    n_slots = scenario["n_slots"]
    for p, t_launch_s in enumerate(scenario["plane_launch_s"]):
        phase_s = p * interval_s / 2.0
        j_first = int(np.floor((t_launch_s - phase_s) / interval_s)) + 1
        inputs.append(dict(
            n_slots=n_slots, cadence_sats=k, cadence_interval_s=interval_s,
            cadence_start_s=phase_s + j_first * interval_s, n_runs=n_runs,
            sim_duration_s=params.SIM_DURATION_YEARS * YEAR_S, dt_s=params.DT_DAYS * DAY_S,
            initial_launch_s=np.full(n_slots + k, t_launch_s),
            commissioning_s=params.COMMISSIONING_DAYS * DAY_S,
            design_life_s=params.DESIGN_LIFE_YEARS * YEAR_S,
            launch_success_prob=params.LAUNCH_SUCCESS_PROB, reliability=RELIABILITY))
    return simulate_constellation(simulate_fixed_cadence, inputs, seed)


def availability(out, req, scenario=REFERENCE):
    """Availability of each run for one revisit requirement ("max" or "mean") [-]."""
    revisit_s = constellation_revisit_series(out["n_active"], scenario["tables"]()[req])
    return constellation_availability(revisit_s, REVISIT_REQS[req]["limit_s"], out["t_s"],
                                      scenario["deploy_end_s"])


def summary(out, scenario=REFERENCE):
    """Key numbers of one constellation simulation.

    Output: dict with, for each requirement, mean and 5/95 percentiles of the
    availability; satellites launched per run; time to full operation [days]
    (first time the max-revisit requirement is met) and runs never reaching it.
    """
    result = {}
    for req in REVISIT_REQS:
        avail = availability(out, req, scenario)
        result[req] = (avail.mean(), *np.percentile(avail, [5, 95]))
    launched = out["n_launched"].sum(axis=1)
    revisit_s = constellation_revisit_series(out["n_active"], scenario["tables"]()["max"])
    t_full_days = time_to_requirement(revisit_s, REVISIT_REQS["max"]["limit_s"], out["t_s"]) / DAY_S
    reached = ~np.isnan(t_full_days)
    result["launched"] = (launched.mean(), *np.percentile(launched, [5, 95]))
    result["t_full_days"] = (t_full_days[reached].mean(), *np.percentile(t_full_days[reached], [5, 95]))
    result["never_full"] = int(np.sum(~reached))
    return result


def print_table(results, scenario=REFERENCE):
    """Print the comparison table of the strategies (both revisit requirements)."""
    print(f"{'Strategy':46s} {'Avail. max<=12h (5-95 %)':26s} {'Avail. mean<=6h (5-95 %)':26s} "
          f"{'Launched/run':14s} {'Full op. [d] (5-95 %)':22s} Never")
    for label, out in results.items():
        s = summary(out, scenario)
        print(f"{label:46s} {s['max'][0]:.4f} ({s['max'][1]:.3f}-{s['max'][2]:.3f})      "
              f"{s['mean'][0]:.4f} ({s['mean'][1]:.3f}-{s['mean'][2]:.3f})      "
              f"{s['launched'][0]:6.2f} ({s['launched'][1]:.0f}-{s['launched'][2]:.0f})  "
              f"{s['t_full_days'][0]:6.1f} ({s['t_full_days'][1]:.0f}-{s['t_full_days'][2]:.0f})"
              f"      {s['never_full']}")


def plot_active_vs_time(results, path):
    """2x2 panels: operational satellites of the constellation vs time (mean, 5-95 %)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    n_total = params.N_PLANES * params.N_SLOTS
    for ax, (label, out) in zip(axes.flat, results.items()):
        t_years = out["t_s"] / YEAR_S
        total = out["n_active"].sum(axis=0)
        p5, p95 = np.percentile(total, [5, 95], axis=0)
        ax.fill_between(t_years, p5, p95, color="C0", alpha=0.25, label="5-95 % band")
        ax.plot(t_years, total.mean(axis=0), color="C0", label="mean")
        ax.axhline(n_total, color="k", linestyle="--", label=f"all {n_total} slots")
        ax.axvline(DEPLOY_END_S / YEAR_S, color="gray", linestyle=":", label="end of deployment")
        s = summary(out)
        ax.set_title(f"{label}\navailability: max<=12h {s['max'][0]:.3f}, mean<=6h {s['mean'][0]:.3f}",
                     fontsize=9)
        ax.set_ylim(-0.2, n_total + 0.5)
        ax.grid(True, alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel("Time since first launch [years]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Operational satellites [-]")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"Walker {params.N_PLANES}/{params.N_SLOTS}, {params.N_RUNS} runs | DL = "
        f"{params.DESIGN_LIFE_YEARS:.0f} yr, order-to-launch = {params.ORDER_TO_LAUNCH_DAYS:.0f} d, "
        f"commissioning = {params.COMMISSIONING_DAYS:.0f} d, planes launched "
        f"{params.DEPLOY_INTERVAL_DAYS:.0f} d apart", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep of the order-to-launch time
# ---------------------------------------------------------------------------

def slot_analytic(strategy, order_days):
    """Steady-state slot availability of one plane for a) and c) without spares, else NaN."""
    if strategy["n_spares"] > 0:
        return np.nan
    f = slot_availability_pre_order if strategy["pre_order_retirement"] else slot_availability_renewal_reward
    return f(RELIABILITY, params.DESIGN_LIFE_YEARS * YEAR_S, order_days * DAY_S,
             params.COMMISSIONING_DAYS * DAY_S)


def sweep_order_time(strategy):
    """Slot and constellation availability for each value of ORDER_SWEEP_DAYS."""
    keys = ["slot_mc", "slot_rr", "con_mean", "con_p5", "con_p95"]
    sweep = {k: np.zeros(len(ORDER_SWEEP_DAYS)) for k in keys}
    for i, order_days in enumerate(ORDER_SWEEP_DAYS):
        out = run_strategy(strategy, order_days, n_runs=params.N_RUNS_SWEEP)
        after = out["t_s"] >= DEPLOY_END_S
        avail = availability(out, "max")
        sweep["slot_mc"][i] = out["n_active"][:, :, after].mean() / params.N_SLOTS
        sweep["slot_rr"][i] = slot_analytic(strategy, order_days)
        sweep["con_mean"][i] = avail.mean()
        sweep["con_p5"][i], sweep["con_p95"][i] = np.percentile(avail, [5, 95])
    return sweep


def plot_availability_vs_order_time(sweeps, path):
    """(a) slot availability per plane, MC vs analytic; (b) constellation availability (max req.)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for label, sw in sweeps.items():
        line, = ax1.plot(ORDER_SWEEP_DAYS, sw["slot_mc"], "o-",
                         label=f"{label}: MC, {params.SIM_DURATION_YEARS:.0f} yr")
        if not np.all(np.isnan(sw["slot_rr"])):
            ax1.plot(ORDER_SWEEP_DAYS, sw["slot_rr"], "--", color=line.get_color(),
                     label=f"{label}: analytic steady state (t -> inf)")
        ax2.plot(ORDER_SWEEP_DAYS, sw["con_mean"], "o-", color=line.get_color(),
                 label=f"{label}: MC mean")
        ax2.fill_between(ORDER_SWEEP_DAYS, sw["con_p5"], sw["con_p95"],
                         color=line.get_color(), alpha=0.15)
    ax1.set_title("(a) Mean fraction of occupied slots per plane after deployment")
    ax2.set_title(f"(b) Constellation availability ({REVISIT_REQS['max']['text']} at "
                  f"{params.REVISIT_LAT_DEG:.0f} deg N), band 5-95 %")
    for ax in (ax1, ax2):
        ax.set_xlabel("Order-to-launch time [days]")
        ax.set_ylabel("Availability [-]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Strategy d): search of the minimum cadence and cost comparison
# ---------------------------------------------------------------------------

# Mission criterion, for each revisit requirement: at least AVAIL_CONFIDENCE of
# the runs have availability >= AVAIL_TARGET.
CADENCE_CACHE = {}  # {(scenario name, k, seed, T_days): {req: fraction of runs meeting the criterion}}


def criterion_fractions(out, scenario=REFERENCE):
    """Fraction of runs with availability >= AVAIL_TARGET, for each revisit requirement."""
    return {req: np.mean(availability(out, req, scenario) >= params.AVAIL_TARGET)
            for req in REVISIT_REQS}


def cadence_metrics(k, seed, t_days, scenario=REFERENCE):
    """Criterion fractions of strategy d) for (k, seed, T), N_RUNS_SWEEP runs (cached)."""
    key = (scenario["name"], k, seed, int(t_days))
    if key not in CADENCE_CACHE:
        CADENCE_CACHE[key] = criterion_fractions(
            run_cadence(k, int(t_days), seed, params.N_RUNS_SWEEP, scenario), scenario)
    return CADENCE_CACHE[key]


def cadence_search(k, seed, req, t_ref_days=None, scenario=REFERENCE):
    """Largest interval T [days] per plane meeting the criterion for one revisit requirement.

    Coarse sweep + bisection (search.py); with t_ref_days (other seeds) a bisection
    between t_ref -/+ 2 grid steps after checking the bracket, else a full sweep.
    """
    def evaluate(t_days):
        return cadence_metrics(k, seed, t_days, scenario)[req]

    start, stop, step = (np.array(params.CADENCE_GRID_DAYS_PER_SAT) * k).astype(int)
    if t_ref_days is not None:
        lo, hi = t_ref_days - 2 * step, t_ref_days + 2 * step
        if lo > 0 and evaluate(lo) >= params.AVAIL_CONFIDENCE and evaluate(hi) < params.AVAIL_CONFIDENCE:
            return {"t_star_days": bisect_interval(evaluate, lo, hi, params.AVAIL_CONFIDENCE)}
    return search_max_interval(evaluate, np.arange(start, stop + 1, step), params.AVAIL_CONFIDENCE)


def mc_launch_rate_per_year(out, scenario=REFERENCE):
    """Mean satellites launched per year by the constellation after deployment (MC) [1/yr]."""
    after = out["t_s"] >= scenario["deploy_end_s"]
    years = (out["t_s"][-1] - scenario["deploy_end_s"]) / YEAR_S
    return out["n_launched"][:, after].sum(axis=1).mean() / years


def steady_rate_pre_order(n_spares, scenario=REFERENCE):
    """Steady-state launch rate of c) / b+c), one reorder per loss [satellites/yr].

    Each of the N + S units of a plane renews independently with mean cycle
    E[cycle_c] = ORDER + integral_0^(DL - ORDER - C) R dt; constellation rate
    P (N + S) / E[cycle_c].
    """
    order_s = params.ORDER_TO_LAUNCH_DAYS * DAY_S
    a_s = params.DESIGN_LIFE_YEARS * YEAR_S - order_s - params.COMMISSIONING_DAYS * DAY_S
    integral_s = YEAR_S * quad(lambda ty: reliability_analytic(ty * YEAR_S, **RELIABILITY),
                               0.0, a_s / YEAR_S, limit=200)[0]
    n_planes = len(scenario["plane_launch_s"])
    return n_planes * (scenario["n_slots"] + n_spares) / (order_s + integral_s) * YEAR_S


def failure_breakdown(out, req="max", scenario=REFERENCE):
    """Runs failing availability >= AVAIL_TARGET, by number of unplanned losses.

    Output: (fraction of runs failing [-], {number of losses: fraction of ALL runs [-]})
    """
    failing = availability(out, req, scenario) < params.AVAIL_TARGET
    losses = out["n_failures"] + out["n_launch_failures"]
    counts = {int(n): np.sum(failing & (losses == n)) / len(failing)
              for n in np.unique(losses[failing])}
    return failing.mean(), counts


def cost_study(full, scenario=REFERENCE, plot=True):
    """Search of T* for d) and comparison with c) and b+c), for both revisit requirements.

    full: if True, T* is also searched with the extra seeds params.SEEDS_CHECK
    (sensitivity check only: nothing else depends on it).
    scenario: REFERENCE or an altitude scenario of the trade-off (altitude_scenario)
    plot: save cost_vs_availability.png (reference scenario only)
    Output: dict with "rows" (one per configuration: family, label, rate, fraction of
    runs meeting the criterion per requirement and its binomial interval "<req>_ci"),
    "outputs" ({label: constellation output}), "cheapest" ({req: cheapest row meeting
    the criterion by point estimate, or None}) and "equivalent" ({req: rows not more
    expensive than the cheapest whose interval contains AVAIL_CONFIDENCE; more than
    one row = no single winner})
    """
    n_planes, n_slots = len(scenario["plane_launch_s"]), scenario["n_slots"]
    ref_rate = n_planes * reference_launch_rate(
        n_slots, RELIABILITY, params.DESIGN_LIFE_YEARS * YEAR_S,
        params.COMMISSIONING_DAYS * DAY_S) * YEAR_S
    print(f"{scenario['name']}: reference rate P N / E[up] = {ref_rate:.3f} satellites/yr "
          "(every satellite gives its whole useful life, no gaps)")
    print(f"Mission criterion: {params.AVAIL_CONFIDENCE:.0%} of the runs with availability "
          f">= {params.AVAIL_TARGET}")

    d_best = {req: {} for req in REVISIT_REQS}
    for req, spec in REVISIT_REQS.items():
        print(f"Requirement '{spec['text']}':")
        for k in params.CADENCE_SATS_OPTIONS:
            result = cadence_search(k, params.SEED, req, scenario=scenario)
            t_star = result["t_star_days"]
            if np.isnan(t_star):
                print(f"  d) k = {k}: no interval of the grid meets the criterion")
                continue
            if not result.get("bracketed", True):
                print(f"  d) k = {k}: every interval of the grid meets it (T* >= {t_star} d)")
            if full:
                checks = [cadence_search(k, seed, req, t_ref_days=t_star, scenario=scenario)["t_star_days"]
                          for seed in params.SEEDS_CHECK]
                check_text = f"seeds {params.SEEDS_CHECK}: {checks}"
            else:
                check_text = "seed check skipped, run with --full"
            d_best[req][k] = t_star
            print(f"  d) k = {k}: T* = {t_star} d ({check_text}) -> "
                  f"{n_planes * k * 365.25 / t_star:.3f} satellites/yr; "
                  f"non monotone: {result.get('non_monotone')}")

    d_curves = {}
    for k in params.CADENCE_SATS_OPTIONS:
        t_grid = np.array(sorted(t for (name, kk, seed, t) in CADENCE_CACHE
                                 if name == scenario["name"] and kk == k and seed == params.SEED))
        d_curves[k] = {"rate": n_planes * k * 365.25 / t_grid,
                       **{req: np.array([CADENCE_CACHE[(scenario["name"], k, params.SEED, t)][req]
                                         for t in t_grid])
                          for req in REVISIT_REQS}}

    configs = []
    for req in REVISIT_REQS:
        for k, t_star in d_best[req].items():
            if ("d", k, t_star) not in configs:
                configs.append(("d", k, t_star))
    for n_spares in range(0, 4):
        configs.append(("c" if n_spares == 0 else "b+c", n_spares, None))

    print(f"{'Configuration':26s} {'sats/yr steady':>14s} {'sats/yr MC':>10s} "
          f"{'avail max<=12h (5-95 %) | runs ok [binomial interval]':>52s} "
          f"{'avail mean<=6h (5-95 %) | runs ok [binomial interval]':>52s}")
    rows, outputs = [], {}
    for family, value, t_star in configs:
        if family == "d":
            out = run_cadence(value, t_star, params.SEED, params.N_RUNS, scenario)
            rate = n_planes * value * 365.25 / t_star
            label = f"d) k={value}, T={t_star} d"
        else:
            strategy = dict(n_spares=value, reorder_level=value, pre_order_retirement=True)
            out = run_strategy(strategy, params.ORDER_TO_LAUNCH_DAYS, scenario=scenario)
            rate = steady_rate_pre_order(value, scenario)
            label = "c) early ordering" if value == 0 else f"b+c) S={value} per plane"
        outputs[label] = out
        row = {"family": family, "label": label, "rate": rate}
        text = []
        for req in REVISIT_REQS:
            avail = availability(out, req, scenario)
            n_ok = int(np.sum(avail >= params.AVAIL_TARGET))
            row[req] = n_ok / len(avail)
            row[req + "_ci"] = binomial_interval(n_ok, len(avail), params.MC_CI_LEVEL)
            p5, p95 = np.percentile(avail, [5, 95])
            text.append(f"{avail.mean():.4f} ({p5:.3f}-{p95:.3f}) | {row[req]:.3f} "
                        f"[{row[req + '_ci'][0]:.3f}, {row[req + '_ci'][1]:.3f}]")
        rows.append(row)
        print(f"{label:26s} {rate:14.3f} {mc_launch_rate_per_year(out, scenario):10.3f}   "
              f"{text[0]:>50s}   {text[1]:>50s}")

    cheapest, equivalent = {}, {}
    for req, spec in REVISIT_REQS.items():
        print(f"Cheapest configuration per family, requirement '{spec['text']}':")
        for family in ["d", "c", "b+c"]:
            ok = [r for r in rows if r["family"] == family and r[req] >= params.AVAIL_CONFIDENCE]
            if ok:
                best = min(ok, key=lambda r: r["rate"])
                print(f"  {family:4s}: {best['label']} -> {best['rate']:.3f} satellites/yr")
            else:
                print(f"  {family:4s}: none of the configurations tested meets it")
        ok = [r for r in rows if r[req] >= params.AVAIL_CONFIDENCE]
        cheapest[req] = min(ok, key=lambda r: r["rate"]) if ok else None
        equivalent[req] = equivalent_rows(rows, cheapest[req], req)
        if len(equivalent[req]) > 1:
            print(f"  NO SINGLE WINNER: the {params.MC_CI_LEVEL:.0%} binomial interval of these "
                  f"configurations contains the criterion {params.AVAIL_CONFIDENCE}; they are "
                  "statistically equivalent to the requirement with this number of runs:")
            for r in equivalent[req]:
                print(f"    {r['label']}: {r['rate']:.3f} satellites/yr, runs ok {r[req]:.3f} "
                      f"[{r[req + '_ci'][0]:.3f}, {r[req + '_ci'][1]:.3f}]")

    failing, counts = failure_breakdown(outputs["c) early ordering"], scenario=scenario)
    print(f"c) alone, requirement 'max': {failing:.3f} of the runs have availability < "
          f"{params.AVAIL_TARGET}; by number of unplanned losses in the run: "
          + ", ".join(f"{n}: {f:.3f}" for n, f in counts.items()))

    if plot:
        plot_cost_vs_availability(d_curves, d_best, rows, ref_rate,
                                  os.path.join(FIG_DIR, "cost_vs_availability.png"))
    return {"rows": rows, "outputs": outputs, "cheapest": cheapest, "equivalent": equivalent}


def equivalent_rows(rows, best, req):
    """Configurations that cannot be told apart from the criterion, up to the cost of best.

    A fraction of runs measured with n runs has a binomial interval (row[req + "_ci"]).
    If the interval contains AVAIL_CONFIDENCE, the Monte Carlo cannot say whether the
    configuration meets the criterion. Output: rows with such an interval and a launch
    rate <= that of best (the cheapest by point estimate), sorted by rate; best itself
    is included when its own interval contains the criterion. Empty if best is None.
    """
    if best is None:
        return []
    near = [r for r in rows if r["rate"] <= best["rate"]
            and r[req + "_ci"][0] < params.AVAIL_CONFIDENCE <= r[req + "_ci"][1]]
    return sorted(near, key=lambda r: r["rate"])


def plot_cost_vs_availability(d_curves, d_best, rows, ref_rate, path):
    """Launch rate vs fraction of runs meeting the criterion, one panel per revisit requirement."""
    markers = {0: "s", 1: "D", 2: "^", 3: "v"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, (req, spec) in zip(axes, REVISIT_REQS.items()):
        for k, curve in d_curves.items():
            order = np.argsort(curve["rate"])
            line, = ax.plot(curve["rate"][order], curve[req][order], ".-", alpha=0.7,
                            label=f"d) fixed cadence, k = {k} per plane")
            if k in d_best[req]:
                t_star = d_best[req][k]
                ax.plot(params.N_PLANES * k * 365.25 / t_star, params.AVAIL_CONFIDENCE, "*",
                        ms=14, color=line.get_color(), label=f"d) k = {k}: T* = {t_star} d")
        for i, row in enumerate(r for r in rows if r["family"] in ("c", "b+c")):
            ax.plot(row["rate"], row[req], markers[i], ms=8, color="k",
                    mfc="none" if row["family"] == "c" else "k", label=row["label"])
        ax.axhline(params.AVAIL_CONFIDENCE, color="r", linestyle="--",
                   label=f"criterion: {params.AVAIL_CONFIDENCE:.0%} of runs")
        ax.axvline(ref_rate, color="gray", linestyle=":",
                   label=f"reference rate P N/E[up] = {ref_rate:.2f}/yr (whole useful life, no gaps)")
        ax.set_xlabel("Satellites launched per year by the constellation, steady state [1/yr]")
        ax.set_title(f"({'a' if req == 'max' else 'b'}) Requirement: {spec['text']} at "
                     f"{params.REVISIT_LAT_DEG:.0f} deg N", fontsize=10)
        ax.set_xlim(0.75 * ref_rate, 1.8 * ref_rate)  # the comparison is around the reference rate
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel(f"Fraction of runs with availability >= {params.AVAIL_TARGET} [-]")
    axes[0].set_ylim(-0.02, 1.02)
    fig.suptitle(f"Cost vs availability | Walker {params.N_PLANES}/{params.N_SLOTS}, "
                 f"DL = {params.DESIGN_LIFE_YEARS:.0f} yr, order-to-launch = "
                 f"{params.ORDER_TO_LAUNCH_DAYS:.0f} d, commissioning = {params.COMMISSIONING_DAYS:.0f} d",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Coverage of the reference Walker constellation
# ---------------------------------------------------------------------------

def coverage_study():
    """Walker choice, lookup table, minimum-access sensitivity, figures."""
    data = coverage_data()
    a_m = coverage.R_E + params.ALT_KM * 1e3
    inc = coverage.sso_inclination(a_m)
    print(f"Reference SSO: h = {params.ALT_KM:.0f} km, i = {np.degrees(inc):.3f} deg, "
          f"nodal period = {coverage.nodal_period(a_m, inc) / 60:.3f} min; "
          f"min access {params.MIN_ACCESS_S:.0f} s")
    for eta_deg, walker in zip(data["eta_sweep_deg"], data["best_walker"]):
        lam, _ = coverage.footprint_from_sensor(params.ALT_KM * 1e3, np.radians(eta_deg))
        print(f"  eta = {eta_deg:4.0f} deg (lambda = {np.degrees(lam):.2f} deg): smallest Walker "
              f"meeting max revisit <= {params.REVISIT_MAX_HOURS:.0f} h at "
              f"{params.REVISIT_LAT_DEG:.0f} deg: {walker[0]}/{walker[1]}/{walker[2]} "
              f"(T = {walker[0] * walker[1]})")
    ref = data["best_walker"][list(data["eta_sweep_deg"]).index(params.ETA_MAX_DEG)]
    if (ref[0], ref[1]) != (params.N_PLANES, params.N_SLOTS):
        print(f"  WARNING: params N_PLANES/N_SLOTS = {params.N_PLANES}/{params.N_SLOTS} "
              f"differ from the Walker search {ref[0]}/{ref[1]}")

    print(f"Minimum access sensitivity (Walker {ref[0]}/{ref[1]}/{ref[2]}, max / mean revisit [h]):")
    for ma, mx, mn in zip(data["min_access_sweep_s"], data["min_access_sweep_max_s"],
                          data["min_access_sweep_mean_s"]):
        print(f"  {ma:4.0f} s: {hours(mx)} / {hours(mn)}")
    w = data["best_walker_longest_access"]
    print(f"  smallest Walker with min access {max(params.MIN_ACCESS_SWEEP_S):.0f} s: "
          f"{w[0]}/{w[1]}/{w[2]} (T = {w[0] * w[1]})")

    for name, key in [("max", "table_planes_max_s"), ("mean", "table_planes_mean_s")]:
        table = data[key]
        print(f"Lookup table {name} revisit [h] (rows n_1, columns n_2):")
        print("      " + "".join(f"  n2={j}" for j in range(table.shape[1])))
        for i in range(table.shape[0]):
            print(f"  n1={i}" + "".join(f" {hours(x)}" for x in table[i]))
    plot_revisit_vs_latitude(data, os.path.join(FIG_DIR, "revisit_vs_latitude.png"))
    plot_revisit_table(data, os.path.join(FIG_DIR, "revisit_table.png"))


def plot_revisit_vs_latitude(data, path):
    """Max and mean revisit vs latitude of the reference Walker, for each eta of the sweep."""
    p, s, f = data["best_walker"][list(data["eta_sweep_deg"]).index(params.ETA_MAX_DEG)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for eta_deg in data["eta_sweep_deg"]:
        ax1.plot(data["lat_grid_deg"], data[f"profile_max_s_eta{eta_deg:.0f}"] / 3600, "o-",
                 label=f"eta = {eta_deg:.0f} deg")
        ax2.plot(data["lat_grid_deg"], data[f"profile_mean_s_eta{eta_deg:.0f}"] / 3600, "o-",
                 label=f"eta = {eta_deg:.0f} deg")
    ax1.axhline(params.REVISIT_MAX_HOURS, color="r", linestyle="--", label="requirement (max)")
    ax2.axhline(params.REVISIT_MEAN_HOURS, color="r", linestyle="--", label="requirement (mean)")
    ticks = {ax1: [6, 10, 20, 40, 60, 100], ax2: [0.5, 1, 2, 5, 10, 20]}
    for ax, title in [(ax1, "(a) Max revisit"), (ax2, "(b) Mean revisit")]:
        ax.axvline(params.REVISIT_LAT_DEG, color="gray", linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("Latitude [deg]")
        ax.set_ylabel("Revisit [h]")
        ax.set_yscale("log")
        ax.set_yticks(ticks[ax])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))  # 6, 10, 20 instead of 6x10^0
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Walker {p}/{s}/{f} (star, RAAN spread {params.RAAN_SPREAD_DEG:.0f} deg), "
                 f"SSO {params.ALT_KM:.0f} km, min access {params.MIN_ACCESS_S:.0f} s, "
                 "geometric coverage (no illumination)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_revisit_table(data, path):
    """Lookup table revisit(n_1, n_2) as two annotated grids (max and mean)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (key, title, limit_h) in zip(axes, [
            ("table_planes_max_s", "(a) Max revisit [h]", params.REVISIT_MAX_HOURS),
            ("table_planes_mean_s", "(b) Mean revisit [h]", params.REVISIT_MEAN_HOURS)]):
        table_h = data[key] / 3600
        shown = np.where(np.isfinite(table_h), table_h, np.nan)
        im = ax.imshow(shown, origin="lower", cmap="viridis_r")
        for i in range(table_h.shape[0]):
            for j in range(table_h.shape[1]):
                finite = np.isfinite(table_h[i, j])
                text = f"{table_h[i, j]:.1f}" if finite else "inf"  # (0, 0): no satellite at all
                ok = finite and table_h[i, j] <= limit_h
                # dark text on light cells (short revisit, and the blank inf cell), white on dark
                light = not finite or im.norm(table_h[i, j]) < 0.5
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="k" if light else "w", fontweight="bold" if ok else "normal")
        ax.set_xlabel("n_2 operational in plane 2 [-]")
        ax.set_ylabel("n_1 operational in plane 1 [-]")
        ax.set_title(f"{title}; bold = meets <= {limit_h:.0f} h")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"Revisit at {params.REVISIT_LAT_DEG:.0f} deg N vs operational satellites per plane "
                 "(R1 rephasing, quasi-stationary)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Second scenario: IRIDE Eaglet II from public TLEs
# ---------------------------------------------------------------------------

IRIDE_COVERAGE_CACHE = os.path.join("data", "iride_coverage.npz")


def iride_scenario():
    """Planes, coverage and simulation scenario of IRIDE Eaglet II (16 satellites).

    The launch date of each plane comes from its launch: the launches (prefix of
    the international designator) are ordered and mapped on IRIDE_LAUNCH_DAYS.
    Output: (planes scenario, coverage data, simulation scenario dict)
    """
    sc = planes.iride_scenario(params.IRIDE_ELEMENTS_CSV, params.PLANE_GROUP_THRESHOLD_DEG)
    cov = planes.load_or_build_iride_coverage(IRIDE_COVERAGE_CACHE, sc)
    launches = sorted({p["intdes"][0][:5] for p in sc["planes"]})
    plane_launch_s = np.array([params.IRIDE_LAUNCH_DAYS[launches.index(p["intdes"][0][:5])]
                               for p in sc["planes"]]) * DAY_S
    tables = {"max": cov["table_max_s"], "mean": cov["table_mean_s"]}
    scenario = {"name": "IRIDE Eaglet II (16 satellites from TLE)",
                "n_slots": int(cov["sats_per_plane"]), "plane_launch_s": plane_launch_s,
                "deploy_end_s": plane_launch_s.max() + params.COMMISSIONING_DAYS * DAY_S,
                "tables": lambda: tables}
    return sc, cov, scenario


def iride_study():
    """Objects, planes, comparison with the reference Walker, revisit and strategies."""
    sc, cov, scenario = iride_scenario()
    el = sc["elements"]
    print(f"Objects (derived CSV {params.IRIDE_ELEMENTS_CSV}), elements at the common epoch:")
    for j in np.argsort(sc["labels"] * 1000 + el["norad"] % 1000):
        print(f"  {el['name'][j]:24s} {el['norad'][j]:6d} {el['intdes'][j]:8s} "
              f"alt {(el['a_m'][j] - coverage.R_E) / 1e3:7.2f} km  i {np.degrees(el['inc_rad'][j]):8.4f}  "
              f"RAAN {np.degrees(sc['raan_rad'][j]):8.3f}  u {np.degrees(sc['u_rad'][j]):7.2f} deg  "
              f"plane {sc['labels'][j]}")
    for threshold in params.PLANE_GROUP_SWEEP_DEG:
        labels = planes.group_planes(sc["raan_rad"], np.radians(threshold))
        print(f"  grouping threshold {threshold:4.1f} deg -> {labels.max() + 1} planes, "
              f"sizes {np.bincount(labels).tolist()}")

    a_ref = coverage.R_E + params.ALT_KM * 1e3
    print(f"{'':28s} {'reference Walker':>18s} " + " ".join(f"{'IRIDE plane ' + str(p):>16s}"
                                                       for p in range(len(sc['planes']))))
    rows = [("launch", "-", [p["intdes"][0][:5] for p in sc["planes"]]),
            ("satellites per plane", f"{params.N_SLOTS}", [str(p["n"]) for p in sc["planes"]]),
            ("mean altitude [km]", f"{params.ALT_KM:.2f}", [f"{p['alt_m'] / 1e3:.2f}" for p in sc["planes"]]),
            ("inclination [deg]", f"{np.degrees(coverage.sso_inclination(a_ref)):.4f}",
             [f"{np.degrees(p['inc_rad']):.4f}" for p in sc["planes"]]),
            ("SSO inclination at that a", "-",
             [f"{np.degrees(coverage.sso_inclination(p['a_m'])):.4f}" for p in sc["planes"]]),
            ("RAAN [deg]", "0 / 90", [f"{np.degrees(p['raan_rad']):.3f}" for p in sc["planes"]]),
            ("in-plane spacing min-max [deg]", f"{360 / params.N_SLOTS:.0f}",
             [f"{np.degrees(p['spacing_rad'].min()):.1f}-{np.degrees(p['spacing_rad'].max()):.1f}"
              for p in sc["planes"]])]
    for name, ref, values in rows:
        print(f"  {name:26s} {ref:>18s} " + " ".join(f"{v:>16s}" for v in values))
    d_raan = abs(np.degrees(sc["planes"][0]["raan_rad"] - sc["planes"][1]["raan_rad"])) % 360
    d_raan = min(d_raan, 360 - d_raan)
    print(f"  RAAN separation of the planes: {d_raan:.2f} deg (= {d_raan / 15:.2f} h of local "
          f"time of the node, both SSO); reference star 2/4: {params.RAAN_SPREAD_DEG / params.N_PLANES:.0f} deg")

    i45 = list(cov["lat_grid_deg"]).index(params.REVISIT_LAT_DEG)
    ref = coverage_data()
    print(f"Revisit at {params.REVISIT_LAT_DEG:.0f} deg N (eta {params.ETA_MAX_DEG:.0f} deg, "
          f"min access {params.MIN_ACCESS_S:.0f} s), max / mean [h]:")
    print(f"  IRIDE 16, real phases     : {hours(cov['observed_max_s'][i45])} / {hours(cov['observed_mean_s'][i45])}")
    print(f"  IRIDE 2x8, evenly spaced  : {hours(cov['ideal_max_s'][i45])} / {hours(cov['ideal_mean_s'][i45])}")
    print(f"  reference Walker 2/4/0    : {hours(ref['profile_max_s_eta30'][i45])} / "
          f"{hours(ref['profile_mean_s_eta30'][i45])}")
    for name, key in [("max", "table_max_s"), ("mean", "table_mean_s")]:
        table = cov[key]
        print(f"IRIDE lookup table {name} revisit [h] (rows n_1 = plane 0, columns n_2 = plane 1):")
        print("      " + "".join(f"  n2={j}" for j in range(table.shape[1])))
        for i in range(table.shape[0]):
            print(f"  n1={i}" + "".join(f" {hours(x)}" for x in table[i]))

    print(f"Strategies on the IRIDE scenario (planes launched on days "
          f"{(scenario['plane_launch_s'] / DAY_S).astype(int).tolist()}, {scenario['n_slots']} slots per plane):")
    results = {label: run_strategy(s, params.ORDER_TO_LAUNCH_DAYS, scenario=scenario)
               for label, s in STRATEGIES.items()}
    print_table(results, scenario)
    plot_iride(sc, cov, os.path.join(FIG_DIR, "iride_scenario.png"))


def plot_iride(sc, cov, path):
    """(a) in-plane phases at the common epoch; (b) revisit vs latitude, real vs ideal vs reference."""
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    for p, plane in enumerate(sc["planes"]):
        ax1.plot(plane["u_sorted_rad"], np.full(plane["n"], p + 1.0), "o", ms=8,
                 label=f"plane {p}: launch {plane['intdes'][0][:2]}-{plane['intdes'][0][2:5]}, "
                       f"RAAN {np.degrees(plane['raan_rad']):.1f} deg")
    ax1.set_rticks([])
    ax1.set_title("(a) Argument of latitude u at the common epoch", fontsize=10)
    ax1.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.1, -0.2))

    ax2 = fig.add_subplot(1, 2, 2)
    ref = coverage_data()
    lat = cov["lat_grid_deg"]
    ax2.plot(lat, cov["observed_max_s"] / 3600, "o-", color="C0", label="IRIDE 16, real phases: max")
    ax2.plot(lat, cov["observed_mean_s"] / 3600, "o--", color="C0", label="IRIDE 16, real phases: mean")
    ax2.plot(lat, cov["ideal_max_s"] / 3600, "s-", color="C1", label="IRIDE 2x8 evenly spaced: max")
    ax2.plot(lat, cov["ideal_mean_s"] / 3600, "s--", color="C1", label="IRIDE 2x8 evenly spaced: mean")
    ax2.plot(ref["lat_grid_deg"], ref["profile_max_s_eta30"] / 3600, "^-", color="C2",
             label="reference Walker 2/4/0: max")
    ax2.plot(ref["lat_grid_deg"], ref["profile_mean_s_eta30"] / 3600, "^--", color="C2",
             label="reference Walker 2/4/0: mean")
    ax2.axhline(params.REVISIT_MAX_HOURS, color="r", linestyle="--", lw=1)
    ax2.axhline(params.REVISIT_MEAN_HOURS, color="r", linestyle=":", lw=1)
    ax2.axvline(params.REVISIT_LAT_DEG, color="gray", linestyle=":")
    ax2.set_yscale("log")
    ax2.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel("Latitude [deg]")
    ax2.set_ylabel("Revisit [h]")
    ax2.set_title(f"(b) Revisit, eta = {params.ETA_MAX_DEG:.0f} deg, min access "
                  f"{params.MIN_ACCESS_S:.0f} s (red: requirements)", fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=7)
    with open(params.IRIDE_ELEMENTS_CSV) as f:
        downloaded = next(l for l in f if l.startswith("# Downloaded")).split()[2]
    fig.suptitle(f"IRIDE Eaglet II, {len(sc['labels'])} satellites from public TLEs (CelesTrak, "
                 f"downloaded {downloaded}); geometric coverage", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orbital decay (v0.3): reference case Sentinel-3B (OSCAR) and Eaglet II
# ---------------------------------------------------------------------------

def eaglet_ballistic_coefficient():
    """B of Eaglet II [m^2/kg]: C_D x (box surface / 4) / mass (params.py, ASSUMPTIONs)."""
    area_m2 = decay.box_mean_area(*params.EAGLET_BOX_M)
    return decay.ballistic_coefficient(params.DRAG_COEFFICIENT, area_m2, params.EAGLET_MASS_KG)


def decay_study():
    """Lifetime of Sentinel-3B vs OSCAR, lifetime of Eaglet II, TLE decay rate vs model."""
    models = (params.ATM_MODEL, params.ATM_MODEL_SENSITIVITY)
    tables = {m: decay.read_density_table(params.ATM_DENSITY_CSV, m) for m in models}
    end_m = params.DECAY_END_ALT_KM * 1e3

    oscar_s3b_comparison()

    b = eaglet_ballistic_coefficient()
    print(f"Eaglet II: m = {params.EAGLET_MASS_KG} kg, box {params.EAGLET_BOX_M} m, "
          f"A = {decay.box_mean_area(*params.EAGLET_BOX_M):.3f} m^2, B = {b:.4f} m^2/kg; "
          f"lifetime from {params.ALT_KM:.0f} km [yr] for area x {params.EAGLET_AREA_FACTORS} "
          f"(limit {params.LEO_LIFETIME_LIMIT_YEARS:.0f} yr, ESSB-ST-U-007 5.4.2.3a.1)")
    for model, (alt_m, rho) in tables.items():
        for level in params.SOLAR_ACTIVITY_LEVELS:
            t_years = [decay.lifetime(params.ALT_KM * 1e3, end_m, alt_m, rho[level], b * f) / YEAR_S
                       for f in params.EAGLET_AREA_FACTORS]
            print(f"  {model:8s} {level:8s}: " + " / ".join(f"{t:6.2f}" for t in t_years))

    el = tle.read_elements_csv(params.IRIDE_ELEMENTS_CSV)
    alt_sat_m = el["a_m"] - coverage.R_E
    observed = decay.sma_rate_from_ndot(el["a_m"], el["n_rad_s"], el["ndot_rad_s2"]) * DAY_S
    print("IRIDE TLE: da/dt from ndot [m/day] vs model (area x 1):")
    predicted = {}
    for model, (alt_m, rho) in tables.items():
        predicted[model] = {level: decay.decay_rate(el["a_m"], decay.density(alt_sat_m, alt_m, rho[level]), b)
                            * DAY_S for level in params.SOLAR_ACTIVITY_LEVELS}
    for j in np.argsort(el["norad"]):
        print(f"  {el['name'][j]:24s} {el['norad'][j]:6d} alt {alt_sat_m[j] / 1e3:6.1f} km: observed "
              f"{observed[j]:7.1f}; {params.ATM_MODEL} low/mod/high "
              + " / ".join(f"{predicted[params.ATM_MODEL][lv][j]:6.1f}" for lv in params.SOLAR_ACTIVITY_LEVELS))
    decaying = observed < 0
    print(f"  {decaying.sum()} of {len(observed)} decaying, median {np.median(observed[decaying]):.1f} m/day; "
          f"rising (not possible with drag alone): {', '.join(el['name'][~decaying])}")
    plot_decay(tables, b, alt_sat_m, observed, os.path.join(FIG_DIR, "decay.png"))


def plot_decay(tables, b, alt_sat_m, observed, path):
    """(a) Eaglet II lifetime vs initial altitude; (b) TLE decay rate vs model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    alt0_m = np.arange(300e3, 801e3, 10e3)
    alt_plot_m = np.linspace(alt_sat_m.min() - 10e3, alt_sat_m.max() + 10e3, 50)
    for model, style in zip(tables, ("-", "--")):
        alt_m, rho = tables[model]
        for level, color in zip(params.SOLAR_ACTIVITY_LEVELS, ("C0", "C1", "C3")):
            t_years = {f: np.array([decay.lifetime(h, params.DECAY_END_ALT_KM * 1e3, alt_m, rho[level], b * f)
                                    for h in alt0_m]) / YEAR_S for f in params.EAGLET_AREA_FACTORS}
            ax1.plot(alt0_m / 1e3, t_years[1.0], style, color=color, label=f"{model}, {level}")
            if model == params.ATM_MODEL:
                ax1.fill_between(alt0_m / 1e3, t_years[min(params.EAGLET_AREA_FACTORS)],
                                 t_years[max(params.EAGLET_AREA_FACTORS)], color=color, alpha=0.12)
            rate = decay.decay_rate(coverage.R_E + alt_plot_m,
                                    decay.density(alt_plot_m, alt_m, rho[level]), b) * DAY_S
            ax2.plot(alt_plot_m / 1e3, -rate, style, color=color, label=f"{model}, {level}")
    ax1.axhline(params.LEO_LIFETIME_LIMIT_YEARS, color="r", linestyle="--", lw=1,
                label=f"{params.LEO_LIFETIME_LIMIT_YEARS:.0f} yr (ESSB-ST-U-007 5.4.2.3a.1)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Initial altitude [km]")
    ax1.set_ylabel("Orbital lifetime [yr]")
    ax1.set_title(f"(a) Eaglet II lifetime, constant solar activity (band: area x "
                  f"{min(params.EAGLET_AREA_FACTORS)}-{max(params.EAGLET_AREA_FACTORS)})", fontsize=10)
    decaying = observed < 0
    ax2.plot(alt_sat_m[decaying] / 1e3, -observed[decaying], "ko", label="IRIDE TLE, from ndot")
    ax2.set_yscale("log")
    ax2.set_xlabel("Altitude [km]")
    ax2.set_ylabel("Decay rate -da/dt [m/day]")
    ax2.set_title("(b) Decay rate: TLE vs model (area x 1)", fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Orbital decay by drag, densities ECSS-E-ST-10-04C Rev.1 Annex G", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Time-varying solar activity (v0.3): strong cycle (ECSS Table A-1), ESA forecast 2021
# ---------------------------------------------------------------------------

STRONG = "strong cycle"                     # ECSS Table A-1 (solar cycle 23)
WEAK = "ESA forecast 2021 (fap_mon.dat)"    # weak forecast cycle 25 (conservative scenario)
MONTH_ECSS_S = params.SOLAR_MONTH_DAYS * DAY_S   # one month of Table A-1 [s]
MONTH_CAL_S = YEAR_S / 12.0                      # one calendar month, mean length [s]


def months_since(date, epoch_month):
    """Calendar months from the first day of epoch_month ("YYYY-MM") to date ("YYYY-MM-DD" or
    "YYYY-MM-DDThh:mm"), with the fraction of the month elapsed (to the minute)."""
    d = np.datetime64(date, "m")
    m = d.astype("datetime64[M]")
    minutes_in_month = ((m + np.timedelta64(1, "M")).astype("datetime64[m]") - m.astype("datetime64[m]")).astype(int)
    whole = (m - np.datetime64(epoch_month, "M")).astype(int)
    return whole + (d - m.astype("datetime64[m]")).astype(int) / minutes_in_month


def seconds_between(start, end):
    """Seconds from date start to date end ("YYYY-MM-DD[Thh:mm[:ss]]")."""
    return (np.datetime64(end, "s") - np.datetime64(start, "s")).astype(int) * 1.0


def solar_scenarios(index=params.SOLAR_INDEX):
    """The two time-varying solar activity scenarios.

    Each scenario: dict with
      "f107"         : F10.7 of each month [sfu]
      "step_s"       : length of one month [s]
      "periodic"     : the series repeats (strong cycle) or not (ESA forecast 2021)
      "period_months": length of the repeated cycle, or None
      "range_months" : (first, last) start phase of one full cycle (uniform phase draws)
      "phase_of"     : function date -> phase (index of the series, fractional)
    WEAK is missing if the DRAMA file is not installed (message printed).
    Keys: STRONG, WEAK.
    """
    f_cycle = decay.read_solar_cycle(params.SOLAR_CYCLE_CSV, index)

    def strong_phase(date, cycle25_start=params.CYCLE25_START):
        start = cycle25_start if np.datetime64(date) >= np.datetime64(cycle25_start) else params.CYCLE24_START
        return seconds_between(start, date) / MONTH_ECSS_S

    scen = {"strong cycle": {"f107": f_cycle, "step_s": MONTH_ECSS_S, "periodic": True,
                             "period_months": float(len(f_cycle)),
                             "range_months": (0.0, float(len(f_cycle))), "phase_of": strong_phase,
                             "source": f"ECSS-E-ST-10-04C Rev.1 Table A-1 (cycle 23, {index})"}}
    if not os.path.exists(params.OSCAR_FORECAST_PATH):
        print(f"  NOTE: {params.OSCAR_FORECAST_PATH} not found: scenario '{WEAK}' skipped")
        return scen
    year, month, f_oscar = decay.read_oscar_forecast(params.OSCAR_FORECAST_PATH)
    epoch = f"{year[0]:04d}-{month[0]:02d}"
    first = months_since(params.CYCLE25_START, epoch)
    # end of cycle 25 in the forecast: minimum of F10.7 between 5 and 15 yr after its start
    window = np.arange(int(first) + 60, int(first) + 180)
    last = float(window[np.argmin(f_oscar[window])])
    scen[WEAK] = {"f107": f_oscar, "step_s": MONTH_CAL_S, "periodic": False,
                          "period_months": None, "range_months": (first, last),
                          "phase_of": lambda date: months_since(date, epoch),
                          "source": f"{params.OSCAR_FORECAST_INFO}, weak forecast cycle 25"}
    return scen


def rho_levels(model):
    """Altitude nodes [m] and density tables (3, n_alt) of one model at SOLAR_F107_NODES."""
    alt_m, rho = decay.read_density_table(params.ATM_DENSITY_CSV, model)
    return alt_m, np.vstack([rho[lv] for lv in params.SOLAR_ACTIVITY_LEVELS])


def lifetime_scenario(alt_start_m, b, scen, phase, model=params.ATM_MODEL, interp=params.SOLAR_INTERP_REF):
    """Lifetime [s] from alt_start_m to DECAY_END_ALT_KM in one solar scenario, start phase [months]."""
    alt_m, levels = rho_levels(model)
    return decay.lifetime_variable(alt_start_m, params.DECAY_END_ALT_KM * 1e3, alt_m,
                                   params.SOLAR_F107_NODES, levels, b, scen["f107"], scen["step_s"],
                                   phase, scen["periodic"], interp)


def s3b_ballistic_coefficient():
    """B of Sentinel-3B [m^2/kg], same C_D, area and mass as the OSCAR runs."""
    return decay.ballistic_coefficient(params.DRAG_COEFFICIENT, params.S3B_AREA_M2, params.S3B_MASS_KG)


def oscar_output_path(run):
    """Path of output/dmf_output.json of one OSCAR run (read at run time, never copied)."""
    return os.path.join(os.path.expanduser(run["folder"]), params.OSCAR_OUTPUT_FILE)


def oscar_series_scenario(run):
    """Our solar input from the series of an OSCAR run: monthly mean of its daily F3M.

    Output: scenario dict as solar_scenarios (calendar months, not periodic), or None if
    the run folder is missing.
    """
    path = oscar_output_path(run)
    if not os.path.exists(path):
        return None
    act = decay.read_oscar_activity(path)
    first, f3m = decay.monthly_mean(act["date"], act["F3M"])
    epoch = str(first)
    return {"f107": f3m, "step_s": MONTH_CAL_S, "periodic": False,
            "phase_of": lambda date: months_since(date, epoch),
            "source": f"OSCAR PyOscar.osg F3M, monthly mean ({run['folder']})"}


def s3b_lifetime_function(run, model, interp):
    """Function circular altitude [m] -> our Sentinel-3B lifetime [s] for the input of one run.

    run["activity"]: "constant moderate" (ECSS table, closed form), "strong cycle" (Table A-1
    from S3B_START) or "OSCAR series" (the run's own F3M). None if that input is missing.
    """
    b = s3b_ballistic_coefficient()
    end_m = params.DECAY_END_ALT_KM * 1e3
    alt_m, levels = rho_levels(model)
    if run["activity"] == "constant moderate":
        rho = levels[params.SOLAR_ACTIVITY_LEVELS.index("moderate")]
        return lambda h_m: decay.lifetime(h_m, end_m, alt_m, rho, b)
    scen = solar_scenarios()[STRONG] if run["activity"] == "strong cycle" else oscar_series_scenario(run)
    if scen is None:
        return None
    phase = scen["phase_of"](params.S3B_START)
    return lambda h_m: decay.lifetime_variable(h_m, end_m, alt_m, params.SOLAR_F107_NODES, levels, b,
                                               scen["f107"], scen["step_s"], phase, scen["periodic"], interp)


def equivalent_altitude(life_fn, target_s, lo_m, hi_m):
    """Circular altitude [m] in [lo_m, hi_m] where our lifetime equals target_s (brentq, like
    fzero, on the lifetime that grows monotonically with the altitude)."""
    return brentq(lambda h: life_fn(h) - target_s, lo_m, hi_m, xtol=1.0)


def oscar_s3b_comparison():
    """Like-for-like comparison with the three OSCAR runs on Sentinel-3B (no tuning).

    Each run is compared from ITS circular disposal altitude with ITS solar input. Reference =
    lifetime propagated by OSCAR (final date - start). Also the altitude that gives OSCAR's
    lifetime with our model, and the delta-v / propellant check of the OSCAR outputs.
    Output: {run: {(model, interp): {"life_s", "oscar_s", "h_eq_m"}}}
    """
    print(f"Sentinel-3B vs DRAMA 4.1.4 OSCAR, three runs, start {params.S3B_START} UTC, circular orbit, "
          f"B = {s3b_ballistic_coefficient():.4f} m^2/kg; reference = OSCAR propagated lifetime:")
    results = {}
    for name, run in params.OSCAR_S3B_RUNS.items():
        h_osc_m = run["perigee_km"] * 1e3
        oscar_s = run["lifetime_years"] * YEAR_S
        check_s = seconds_between(params.S3B_START, run["reentry"].replace(" ", "T"))
        print(f"  {name} ({run['solar_option']}): OSCAR altitude {run['perigee_km']} km, lifetime "
              f"{run['lifetime_years']} yr (re-entry {run['reentry']} - start = {check_s / YEAR_S:.4f} yr)")
        rows = {}
        interps = ("-",) if run["activity"] == "constant moderate" else (params.SOLAR_INTERP_REF,
                                                                        params.SOLAR_INTERP_SENSITIVITY)
        for model in (params.ATM_MODEL, params.ATM_MODEL_SENSITIVITY):
            for interp in interps:
                life_fn = s3b_lifetime_function(run, model, interp if interp != "-" else params.SOLAR_INTERP_REF)
                if life_fn is None:
                    break
                life_s = life_fn(h_osc_m)
                h_eq_m = equivalent_altitude(life_fn, oscar_s, h_osc_m - 150e3, h_osc_m + 150e3)
                rows[(model, interp)] = {"life_s": life_s, "oscar_s": oscar_s, "h_eq_m": h_eq_m}
                print(f"    {model:8s} {interp:6s}: ours {life_s / YEAR_S:6.2f} yr ({life_s / oscar_s - 1:+6.1%}); "
                      f"altitude giving OSCAR's lifetime {h_eq_m / 1e3:7.2f} km ({(h_eq_m - h_osc_m) / 1e3:+6.2f} km)")
        if not rows:
            print(f"    SKIPPED: {oscar_output_path(run)} not found")
        results[name] = rows
        r1, r2 = params.S3B_SMA_INITIAL_KM * 1e3, params.R_EARTH + h_osc_m
        dv = decay.hohmann_delta_v(r1, r2)
        mp_final = decay.propellant_mass(run["delta_v_m_s"], params.S3B_ISP_S, params.S3B_MASS_KG, True)
        mp_initial = decay.propellant_mass(run["delta_v_m_s"], params.S3B_ISP_S, params.S3B_MASS_KG, False)
        print(f"    check: Hohmann a = {params.S3B_SMA_INITIAL_KM} km -> R_eq + {run['perigee_km']} km: "
              f"{dv:.2f} m/s (OSCAR {run['delta_v_m_s']}); propellant, Isp {params.S3B_ISP_S:.0f} s, "
              f"{params.S3B_MASS_KG:.0f} kg as FINAL mass {mp_final:.2f} kg, as initial {mp_initial:.2f} kg "
              f"(OSCAR {run['propellant_kg']})")
    run = params.OSCAR_S3B_RUNS["constant moderate"]
    alt_m, rho = decay.read_density_table(params.ATM_DENSITY_CSV, params.ATM_MODEL)
    t_end = [decay.lifetime(run["perigee_km"] * 1e3, e * 1e3, alt_m, rho["moderate"], s3b_ballistic_coefficient())
             / YEAR_S for e in params.DECAY_END_SWEEP_KM]
    print(f"  Re-entry altitude sensitivity (constant moderate, {params.ATM_MODEL}): lifetime to "
          + " / ".join(f"{e:.0f}" for e in params.DECAY_END_SWEEP_KM) + " km = "
          + " / ".join(f"{t:.3f}" for t in t_end) + " yr")
    print("  OSCAR treats the 1130 kg as the mass AFTER the manoeuvre (rocket equation with final mass "
          "reproduces requiredPropellantMass). Altitudes measured from the equatorial radius reproduce "
          "OSCAR's delta-v.")
    oscar_input_checks()
    flattening_km = (params.R_EARTH - params.R_EARTH_POLAR) * np.sin(np.radians(params.S3B_INC_DEG)) ** 2 / 2 / 1e3
    print("  Remaining gap, plausible causes (NOT corrected): (1) Earth oblateness, HYPOTHESIS to verify "
          "(how DRAMA defines the altitude): if the altitude is measured from the equatorial radius, on a "
          f"circular orbit with i = {params.S3B_INC_DEG} deg the mean geodetic altitude is higher by "
          f"(R_eq - R_pol) <sin^2(lat)> = (R_eq - R_pol) sin^2(i) / 2 = {flattening_km:.1f} km, the same order "
          "as the altitude differences above; (2) ECSS tables equatorial and averaged over diurnal and "
          "seasonal variations vs the full NRLMSISE-00 along the sun-synchronous orbit in OSCAR; (3) daily "
          "F10.7 + 81-day mean in OSCAR vs monthly mean of the 81-day index here; (4) Ap: step of the daily "
          "F10.7 in OSCAR vs continuous in F10.7 here; (5) mean elements vs circular orbit; re-entry "
          f"altitude {params.DECAY_END_ALT_KM:.0f} km (< 0.01 yr)")
    return results


def oscar_input_checks():
    """Solar input of the OSCAR runs vs ours: ECSS run F3M vs our F81mean, Ap steps, constant run."""
    runs = params.OSCAR_S3B_RUNS
    path = oscar_output_path(runs["ECSS cycle 23"])
    if os.path.exists(path):
        act = decay.read_oscar_activity(path)
        end = np.datetime64(runs["ECSS cycle 23"]["reentry"][:10], "D")
        sel = (act["date"] >= np.datetime64(params.S3B_START[:10], "D")) & (act["date"] <= end)
        f81 = solar_scenarios()[STRONG]["f107"]
        phase = np.array([seconds_between(params.CYCLE24_START, str(d)) / MONTH_ECSS_S for d in act["date"][sel]])
        diff = act["F3M"][sel] - f81[np.floor(phase).astype(int) % len(f81)]
        steps = "; ".join(f"Ap {a:.0f} for daily F10 {act['F10'][act['Ap'] == a].min():.0f}-"
                          f"{act['F10'][act['Ap'] == a].max():.0f}" for a in np.unique(act["Ap"]))
        print(f"  Input check, ECSS run ({params.S3B_START[:10]} to re-entry): OSCAR F3M - our F81mean: mean "
              f"{diff.mean():+.1f} sfu, mean |.| {np.abs(diff).mean():.1f} sfu, max |.| {np.abs(diff).max():.0f} "
              f"sfu. OSCAR Ap is a step of the daily F10: {steps} (ours: continuous, 0/15/45 at 65/140/250)")
    path = oscar_output_path(runs["latest prediction"])
    if os.path.exists(path):
        act = decay.read_oscar_activity(path)
        i = np.argmax(act["F3M"][act["date"] <= np.datetime64("2032-12-31")])
        print(f"  Input check, latest-prediction run: observed F3M peak {act['F3M'][i]:.0f} sfu on "
              f"{act['date'][i]} (cycle 25 observed strong); daily F10 up to {act['F10'].max():.0f} sfu, "
              "monthly means of F3M used here stay inside 65-250 sfu")
    path = oscar_output_path(runs["constant moderate"])
    if os.path.exists(path):
        act = decay.read_oscar_activity(path)
        same = np.all(act["F10"] == 140) and np.all(act["F3M"] == 140) and np.all(act["Ap"] == 15)
        print(f"  Input check, constant run: F10 = F3M = 140, Ap = 15 every day: {same} (= ECSS Table G-2)")


def cycle_peak(scen):
    """Highest F10.7 [sfu] over one cycle of a scenario."""
    lo, hi = scen["range_months"]
    return scen["f107"][int(lo):int(hi)].max()


def print_cycle_warning(scen):
    """The strong cycle is not conservative for the 5-year rule: say it with every result."""
    text = f"  NOTE: strong cycle = solar cycle 23, peak F10.7 {cycle_peak(scen['strong cycle']):.0f} sfu: it " \
           "shortens the lifetimes, so it is NOT conservative for the 5-year rule"
    if WEAK in scen:
        text += (f"; {WEAK}: weak forecast cycle 25, peak {cycle_peak(scen[WEAK]):.0f} sfu (conservative). "
                 "The OBSERVED cycle 25 was strong (OSCAR latest-prediction series: F3M peak ~225 sfu "
                 "in 10/2024, about as cycle 23)")
    print(text)


def end_of_life_cycle_tables(alt_op_m, scen, grid_months, model=params.ATM_MODEL,
                             interp=params.SOLAR_INTERP_REF):
    """Residual lifetime and P_c (nominal flux) of each object class vs the phase of its end of life.

    Inputs: operational altitude [m], solar scenario (solar_scenarios), start phases [months],
            atmosphere model, density interpolation in F10.7
    Output: dict as end_of_life_tables for one level, with arrays over grid_months, plus
            "phase": {"grid_months", "period_months", "step_s", "range_months"} and
            "censored": {outcome: bool array}, True where the forecast ended before the
            re-entry (no extrapolation): lifetime, P_c and object-years are then LOWER bounds
    """
    atm_alt_m, levels = rho_levels(model)
    flux_alt_m, flux = debris.read_master_flux(params.MASTER_FLUX_CSV, params.FLUX_SIZE, YEAR_S)
    b, area_m2 = eaglet_ballistic_coefficient(), eaglet_area()
    end_m = params.DECAY_END_ALT_KM * 1e3
    residual, pc, censored = {}, {}, {}
    for outcome, h0 in ((debris.FAILED, alt_op_m), (debris.DISPOSED, params.DISPOSAL_ALT_KM * 1e3)):
        life, prob = np.empty(len(grid_months)), np.empty(len(grid_months))
        cut = np.zeros(len(grid_months), dtype=bool)
        for j, phase in enumerate(grid_months):
            traj = decay.decay_trajectory(h0, end_m, atm_alt_m, params.SOLAR_F107_NODES, levels, b,
                                          scen["f107"], scen["step_s"], phase, scen["periodic"], interp,
                                          stop_at_end=True)
            life[j], cut[j] = traj["t_s"][-1], not traj["reentered"]
            prob[j] = debris.collision_probability(debris.flux_time_integral_trajectory(
                traj, atm_alt_m, params.SOLAR_F107_NODES, levels, b, flux_alt_m, flux["nominal"], interp),
                area_m2)
        residual[outcome], pc[outcome], censored[outcome] = life, {"nominal": prob}, cut
    for d in (residual, pc, censored):
        d[debris.DISPOSAL_FAILED] = d[debris.FAILED]
    return {"residual_s": residual, "pc": pc, "censored": censored,
            "phase": {"grid_months": np.asarray(grid_months, dtype=float),
                      "period_months": scen["period_months"], "step_s": scen["step_s"],
                      "range_months": scen["range_months"]}}


def phase_grid(scen, horizon_s, step_months=params.SOLAR_PHASE_GRID_MONTHS):
    """Start phases [months] of the tables: one cycle (periodic) or one cycle + the horizon."""
    lo, hi = scen["range_months"]
    if scen["periodic"]:
        return np.arange(lo, hi, step_months)
    return np.arange(lo, hi + horizon_s / scen["step_s"] + step_months + 1.0, step_months)


def over_cycle(table, phase):
    """Values of a phase table over one full cycle (uniform phase), for statistics."""
    grid = phase["grid_months"]
    lo, hi = phase["range_months"]
    return table[(grid >= lo) & (grid < hi)]


def plot_solar_cycle(scen, cyc, alt_op_m, const, path):
    """(a) F10.7 of the two scenarios; (b) residual lifetime vs phase of the end of life."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    colors = {"strong cycle": "tab:red", WEAK: "tab:blue"}
    for name, s in scen.items():
        lo, hi = s["range_months"]
        k = np.arange(int(lo), int(hi) + 1)
        ax1.plot((k - lo) * s["step_s"] / YEAR_S, s["f107"][k % len(s["f107"])], color=colors[name],
                 label=f"{name}: {s['source']}")
    for f, lv in zip(params.SOLAR_F107_NODES, params.SOLAR_ACTIVITY_LEVELS):
        ax1.axhline(f, color="gray", linestyle=":", lw=1)
        ax1.text(11.6, f + (3 if f > 70 else -9), f"ECSS {lv} ({f:.0f})", fontsize=7, color="gray",
                 ha="right")
    ax1.set_ylim(50, 300)
    ax1.set_xlabel("Years from the start of the cycle")
    ax1.set_ylabel("F10.7 [sfu]")
    ax1.set_title("(a) Solar activity scenarios (one cycle)", fontsize=10)
    ax1.legend(fontsize=7, loc="upper left")
    styles = {debris.FAILED: ("-", f"failed at {alt_op_m / 1e3:.0f} km"),
              debris.DISPOSED: ("--", f"disposed at {params.DISPOSAL_ALT_KM:.0f} km")}
    for name, tab in cyc.items():
        ph = tab["phase"]
        lo, hi = ph["range_months"]
        sel = (ph["grid_months"] >= lo) & (ph["grid_months"] < hi)
        x = (ph["grid_months"][sel] - lo) * ph["step_s"] / YEAR_S
        for outcome, (ls, label) in styles.items():
            ax2.semilogy(x, tab["residual_s"][outcome][sel] / YEAR_S, ls, color=colors[name],
                         label=f"{name}, {label}")
    for lv, c in zip(params.SOLAR_ACTIVITY_LEVELS, ("C0", "C1", "C3")):
        ax2.axhline(const[lv]["residual_s"][debris.FAILED] / YEAR_S, color=c, lw=0.8, linestyle=":",
                    label=f"constant {lv}, failed")
    ax2.axhline(params.LEO_LIFETIME_LIMIT_YEARS, color="k", linestyle="--", label="ESSB 5-yr limit")
    ax2.set_xlabel("Phase of the end of life [years from the start of the cycle]")
    ax2.set_ylabel(f"Residual lifetime to {params.DECAY_END_ALT_KM:.0f} km [yr]")
    ax2.set_title(f"(b) Eaglet II, {params.ATM_MODEL}, ln(rho) interpolated in F10.7", fontsize=10)
    ax2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Debris indicators (v0.3): end of life, 5-year rule, object-years, collisions
# ---------------------------------------------------------------------------

def eaglet_area():
    """Mean cross-section of Eaglet II [m^2] (box surface / 4, params.py ASSUMPTION)."""
    return decay.box_mean_area(*params.EAGLET_BOX_M)


def end_of_life_tables(alt_op_m, model=params.ATM_MODEL, area_factor=1.0):
    """Residual lifetime and collision probability of each object class.

    Inputs: operational altitude [m], atmosphere model, area factor [-]
    Output: dict {level: {"residual_s": {outcome: lifetime [s]},
                          "pc": {outcome: {flux case: P_c [-]}}}}
    """
    atm_alt_m, rho = decay.read_density_table(params.ATM_DENSITY_CSV, model)
    flux_alt_m, flux = debris.read_master_flux(params.MASTER_FLUX_CSV, params.FLUX_SIZE, YEAR_S)
    area_m2 = eaglet_area() * area_factor
    b = eaglet_ballistic_coefficient() * area_factor
    end_m = params.DECAY_END_ALT_KM * 1e3
    start_m = {debris.FAILED: alt_op_m, debris.DISPOSAL_FAILED: alt_op_m,
               debris.DISPOSED: params.DISPOSAL_ALT_KM * 1e3}
    tables = {}
    for level in params.SOLAR_ACTIVITY_LEVELS:
        residual, pc = {}, {}
        for outcome, h0 in start_m.items():
            residual[outcome] = decay.lifetime(h0, end_m, atm_alt_m, rho[level], b)
            pc[outcome] = {case: debris.collision_probability(debris.flux_time_integral(
                h0, end_m, atm_alt_m, rho[level], b, flux_alt_m, flux[case]), area_m2)
                for case in debris.FLUX_CASES}
        tables[level] = {"residual_s": residual, "pc": pc}
    return tables


def operational_pc(alt_op_m):
    """P_c of one satellite over its design life at constant altitude, per flux case [-]."""
    flux_alt_m, flux = debris.read_master_flux(params.MASTER_FLUX_CSV, params.FLUX_SIZE, YEAR_S)
    t_s = params.DESIGN_LIFE_YEARS * YEAR_S
    return {case: debris.collision_probability(debris.flux_at(alt_op_m, flux_alt_m, flux[case]) * t_s,
                                               eaglet_area()) for case in debris.FLUX_CASES}


def debris_indicators(out, tables, p_manoeuvre):
    """Monte Carlo debris indicators of one constellation output, per solar activity level.

    tables: {level: end_of_life_tables entry} and/or {scenario: end_of_life_cycle_tables};
    for a solar cycle the phase at t = 0 is drawn uniform over one cycle (seed sequence
    [SEED, SOLAR_PHASE_SEED_STREAM], same draws for every scenario) unless
    tab["phase"]["start_fixed"] is given.
    The disposal draws use the seed sequence [SEED, DEBRIS_SEED_STREAM], the same
    for every level (common random numbers: the levels differ only by physics).
    Output: {level: debris.debris_summary dict + "collisions" (sum of P_c of the
    dead objects of each run, nominal flux) and "obj_years_operational"}
    """
    limit_s = params.LEO_LIFETIME_LIMIT_YEARS * YEAR_S
    horizon_s = out["t_s"][-1]
    dt_s = out["t_s"][1] - out["t_s"][0]
    live = out["n_active"].sum(axis=0) + out.get("n_spares_in_orbit", 0)
    n_runs = out["n_active"].shape[1]
    result = {}
    for level, tab in tables.items():
        rng = np.random.default_rng([params.SEED, params.DEBRIS_SEED_STREAM])
        pc = {outcome: tab["pc"][outcome]["nominal"] for outcome in tab["pc"]}
        phase = None
        if "phase" in tab:   # solar cycle: phase of the mission start, per run
            ph = tab["phase"]
            if ph.get("start_fixed") is None:   # ASSUMPTION: uniform over one cycle
                lo, hi = ph["range_months"]
                u = np.random.default_rng([params.SEED, params.SOLAR_PHASE_SEED_STREAM]).random(n_runs)
                start = lo + (hi - lo) * u
            else:                               # known launch date (IRIDE)
                start = np.full(n_runs, ph["start_fixed"])
            phase = {"start_months": start, "step_s": ph["step_s"], "grid_months": ph["grid_months"],
                     "period_months": ph["period_months"]}
        s = debris.debris_summary(out, p_manoeuvre, rng, tab["residual_s"], horizon_s, limit_s, pc, phase)
        s["obj_years_operational"] = live.sum(axis=1) * dt_s / YEAR_S
        result[level] = s
    return result


def mean_5_95(x):
    """'mean (p5-p95)' of an array, ignoring nan."""
    x = x[~np.isnan(x)]
    return f"{x.mean():.3f} ({np.percentile(x, 5):.3f}-{np.percentile(x, 95):.3f})"


def debris_study(cost):
    """End of life, 5-year rule, object-years and collision probability at the reference altitude.

    cost: output of cost_study; the lifecycle Monte Carlo of the cheapest configuration
    meeting the max-revisit criterion gives the mix and timing of the dead objects.
    """
    alt_m = params.ALT_KM * 1e3
    limit_yr = params.LEO_LIFETIME_LIMIT_YEARS
    names = {debris.FAILED: "failed (natural decay)", debris.DISPOSAL_FAILED: "disposal failed",
             debris.DISPOSED: f"disposed ({params.DISPOSAL_ALT_KM:.0f} km)"}
    print(f"Residual lifetime after end of life [yr] from {params.ALT_KM:.0f} km, limit {limit_yr:.0f} yr "
          "(ESSB 5.4.2.3a.1), compliant = <= limit:")
    tables = {}
    for model in (params.ATM_MODEL, params.ATM_MODEL_SENSITIVITY):
        tables[model] = end_of_life_tables(alt_m, model)
        for outcome in (debris.FAILED, debris.DISPOSED):
            text = []
            for level in params.SOLAR_ACTIVITY_LEVELS:
                t_yr = tables[model][level]["residual_s"][outcome] / YEAR_S
                text.append(f"{level} {t_yr:7.2f} {'OK' if t_yr <= limit_yr else 'NOT compliant'}")
            print(f"  {model:8s} {names[outcome]:24s}: " + " | ".join(text))
    for f in params.EAGLET_AREA_FACTORS:
        tab = end_of_life_tables(alt_m, params.ATM_MODEL, f)
        print(f"  area x {f}: failed " + " / ".join(
            f"{tab[lv]['residual_s'][debris.FAILED] / YEAR_S:.2f}" for lv in params.SOLAR_ACTIVITY_LEVELS)
            + f" yr; disposed " + " / ".join(
            f"{tab[lv]['residual_s'][debris.DISPOSED] / YEAR_S:.2f}" for lv in params.SOLAR_ACTIVITY_LEVELS)
            + " yr (low / moderate / high)")

    ref = tables[params.ATM_MODEL]
    print(f"Collision probability with objects > 1 cm (MASTER-8), one object, low / nominal / high "
          f"flux, limit {params.CUM_COLLISION_PROB_LIMIT:.0e} (ESSB 5.4.2.3a.2), A = {eaglet_area():.3f} m^2:")
    for outcome in (debris.FAILED, debris.DISPOSED):
        for level in params.SOLAR_ACTIVITY_LEVELS:
            pc = ref[level]["pc"][outcome]
            print(f"  {names[outcome]:24s} {level:8s}: "
                  + " / ".join(f"{pc[c]:.2e}" for c in debris.FLUX_CASES))
    op = operational_pc(alt_m)
    print(f"  operational phase ({params.DESIGN_LIFE_YEARS:.0f} yr at {params.ALT_KM:.0f} km, reported "
          "apart): " + " / ".join(f"{op[c]:.2e}" for c in debris.FLUX_CASES))

    r_dl = reliability_analytic(params.DESIGN_LIFE_YEARS * YEAR_S, **RELIABILITY)
    p_star = debris.required_manoeuvre_prob(params.DISPOSAL_PROB_REF, r_dl)
    print(f"Fraction of satellites reaching retirement R(DL) = {r_dl:.4f}: effective P_d = "
          f"p_manoeuvre x {r_dl:.4f}; P_d,eff = {params.DISPOSAL_PROB_REF} needs p_manoeuvre >= {p_star:.4f}")

    label = cost["cheapest"]["max"]["label"]
    out = cost["outputs"][label]
    ends = out["end_kind"] != LAUNCH_FAILURE
    f_ret_mc = np.mean(out["end_kind"][ends] == RETIREMENT)
    others = [r["label"] for r in cost["equivalent"]["max"] if r["label"] != label]
    if others:
        print(f"  NOTE: {label} is not a single winner (statistically equivalent to the requirement: "
              f"{', '.join(others)}); it is used here as the cheapest by point estimate")
    print(f"Monte Carlo: {label} (cheapest meeting the max-revisit criterion), {out['n_active'].shape[1]} "
          f"runs, {params.SIM_DURATION_YEARS:.0f} yr; retirements / ends in orbit = {f_ret_mc:.4f} "
          "(ends inside the horizon only)")
    print(f"  {'p_manoeuvre':>11s} {'P_d,eff':>7s}  fraction of objects compliant, mean (5-95 %) per run: "
          + "  ".join(f"{lv:>22s}" for lv in params.SOLAR_ACTIVITY_LEVELS))
    sweep = {}
    for p_m in params.DISPOSAL_PROB_SWEEP:
        ind = debris_indicators(out, ref, p_m)
        sweep[p_m] = ind
        lv0 = params.SOLAR_ACTIVITY_LEVELS[0]
        pd_eff = ind[lv0]["n_disposed"].sum() / ind[lv0]["n_objects"].sum()
        print(f"  {p_m:11.2f} {pd_eff:7.3f}  " + "  ".join(
            f"{mean_5_95(ind[lv]['frac_compliant']):>22s}" for lv in params.SOLAR_ACTIVITY_LEVELS))
    ind = sweep[params.DISPOSAL_PROB_REF]
    lv0 = params.SOLAR_ACTIVITY_LEVELS[0]
    print(f"  p_manoeuvre = {params.DISPOSAL_PROB_REF}: dead objects per run {mean_5_95(ind[lv0]['n_objects'].astype(float))}; "
          f"operational object-years {mean_5_95(ind[lv0]['obj_years_operational'])}")
    for lv in params.SOLAR_ACTIVITY_LEVELS:
        print(f"  {lv:8s}: dead object-years committed {mean_5_95(ind[lv]['obj_years_committed'] / YEAR_S)}, "
              f"inside the horizon {mean_5_95(ind[lv]['obj_years_horizon'] / YEAR_S)}; expected "
              f"collisions > 1 cm of the dead objects {ind[lv]['collisions'].mean():.2e}")
    solar_cycle_debris(out, ref)
    plot_end_of_life(os.path.join(FIG_DIR, "end_of_life.png"))
    return sweep


def lifetime_stats(values_s, limit_s):
    """'mean (p5-p95)' in years and fraction <= limit of residual lifetimes over the phases."""
    y = values_s / YEAR_S
    return f"{y.mean():6.2f} ({np.percentile(y, 5):6.2f}-{np.percentile(y, 95):6.2f})", np.mean(values_s <= limit_s)


def with_start(tab, start_months):
    """Copy of a cycle table whose Monte Carlo phase at t = 0 is fixed (known launch date)."""
    return {**tab, "phase": {**tab["phase"], "start_fixed": start_months}}


def solar_cycle_debris(out, const):
    """Residual lifetimes over the solar cycle and Monte Carlo 5-year compliance at ALT_KM.

    Inputs: lifecycle output of the reference configuration, end_of_life_tables at ALT_KM
    Output: dict with the cycle tables and the Monte Carlo indicators
    """
    alt_m = params.ALT_KM * 1e3
    limit_s = params.LEO_LIFETIME_LIMIT_YEARS * YEAR_S
    horizon_s = out["t_s"][-1]
    scen = solar_scenarios()
    cyc = {name: end_of_life_cycle_tables(alt_m, s, phase_grid(s, horizon_s)) for name, s in scen.items()}
    sens = {f"strong, rho linear in F10.7": end_of_life_cycle_tables(
                alt_m, scen["strong cycle"], phase_grid(scen["strong cycle"], horizon_s),
                interp=params.SOLAR_INTERP_SENSITIVITY),
            f"strong, {params.SOLAR_INDEX_SENSITIVITY}": end_of_life_cycle_tables(
                alt_m, solar_scenarios(params.SOLAR_INDEX_SENSITIVITY)["strong cycle"],
                phase_grid(scen["strong cycle"], horizon_s))}

    print(f"Residual lifetime after the end of life [yr], Eaglet II, {params.ATM_MODEL}; solar cycle: "
          "mean (5-95 %) over the phase of the end of life (uniform over one cycle) and fraction of "
          f"phases compliant (<= {params.LEO_LIFETIME_LIMIT_YEARS:.0f} yr):")
    classes = ((debris.FAILED, f"failed, {params.ALT_KM:.0f} km"),
               (debris.DISPOSED, f"disposed, {params.DISPOSAL_ALT_KM:.0f} km"))
    print(f"  {'scenario':34s}" + "".join(f"{label:>38s}" for _, label in classes))
    for lv in params.SOLAR_ACTIVITY_LEVELS:
        cells = []
        for outcome, _ in classes:
            t_s = const[lv]["residual_s"][outcome]
            cells.append(f"{t_s / YEAR_S:6.2f}  compliant {float(t_s <= limit_s):.2f}")
        print(f"  {'constant ' + lv:34s}" + "".join(f"{c:>38s}" for c in cells))
    for name, tab in list(cyc.items()) + list(sens.items()):
        cells = []
        for outcome, _ in classes:
            text, frac = lifetime_stats(over_cycle(tab["residual_s"][outcome], tab["phase"]), limit_s)
            cells.append(f"{text}  compliant {frac:.2f}")
        print(f"  {name:34s}" + "".join(f"{c:>38s}" for c in cells))
        if any(np.any(c) for c in tab["censored"].values()):
            print("    (some phases: forecast ended before the re-entry, lifetimes are lower bounds)")

    phases = {name: s["phase_of"](params.IRIDE_FIRST_LAUNCH) for name, s in scen.items()}
    phase_ecss = scen["strong cycle"]["phase_of"](params.IRIDE_FIRST_LAUNCH, params.CYCLE25_START_ECSS_ONLY)
    lo_weak = scen[WEAK]["range_months"][0] if WEAK in scen else 0.0
    print(f"IRIDE first launch {params.IRIDE_FIRST_LAUNCH}: phase of cycle 25 = "
          f"{phases['strong cycle']:.1f} months of 30 d (strong, cycle 25 from {params.CYCLE25_START}, "
          f"DRAMA solcycle.lst; {phase_ecss:.1f} with the ECSS-only start {params.CYCLE25_START_ECSS_ONLY})"
          + (f", {phases[WEAK] - lo_weak:.1f} calendar months ({WEAK})" if WEAK in phases else ""))

    p_m = params.DISPOSAL_PROB_REF
    cases = {f"constant {lv}": {lv: const[lv]} for lv in params.SOLAR_ACTIVITY_LEVELS}
    for name, tab in cyc.items():
        cases[f"{name}, phase uniform"] = {name: tab}
        cases[f"{name}, IRIDE date"] = {name: with_start(tab, phases[name])}
    cases["strong cycle, IRIDE date, ECSS-only start"] = {"strong cycle": with_start(cyc["strong cycle"], phase_ecss)}
    print(f"Monte Carlo, p_manoeuvre = {p_m}, mean (5-95 %) per run: fraction of dead objects compliant, "
          "dead object-years committed, expected collisions > 1 cm (nominal flux):")
    mc = {}
    for label, tables in cases.items():
        ind = list(debris_indicators(out, tables, p_m).values())[0]
        mc[label] = ind
        print(f"  {label:48s} {mean_5_95(ind['frac_compliant']):>22s} "
              f"{mean_5_95(ind['obj_years_committed'] / YEAR_S):>28s} {ind['collisions'].mean():.2e}")
    print_cycle_warning(scen)
    plot_solar_cycle(scen, cyc, alt_m, const, os.path.join(FIG_DIR, "solar_cycle.png"))
    return {"tables": cyc, "mc": mc}


def plot_end_of_life(path):
    """(a) residual lifetime vs altitude; (b) collision probability vs altitude (natural decay)."""
    alt_grid_m = np.arange(300e3, 801e3, 25e3)
    curves = [end_of_life_tables(h) for h in alt_grid_m]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for level, color in zip(params.SOLAR_ACTIVITY_LEVELS, ["tab:blue", "tab:orange", "tab:red"]):
        life = [c[level]["residual_s"][debris.FAILED] / YEAR_S for c in curves]
        ax1.semilogy(alt_grid_m / 1e3, life, color=color, label=f"{level} activity")
        pc = {case: np.array([c[level]["pc"][debris.FAILED][case] for c in curves])
              for case in debris.FLUX_CASES}
        ax2.semilogy(alt_grid_m / 1e3, pc["nominal"], color=color, label=f"{level} activity")
        ax2.fill_between(alt_grid_m / 1e3, pc["low"], pc["high"], color=color, alpha=0.2)
    ax1.axhline(params.LEO_LIFETIME_LIMIT_YEARS, color="k", linestyle="--", label="ESSB 5-yr limit")
    ax1.axvline(params.DISPOSAL_ALT_KM, color="gray", linestyle=":", label="disposal orbit")
    ax1.axvline(params.ALT_KM, color="gray", linestyle="-.", label="reference altitude")
    ax1.set_xlabel("Initial altitude [km]")
    ax1.set_ylabel(f"Residual lifetime to {params.DECAY_END_ALT_KM:.0f} km [yr]")
    ax1.set_title("(a) Natural decay, Eaglet II, NRLMSISE-00 (ECSS)")
    ax2.axhline(params.CUM_COLLISION_PROB_LIMIT, color="k", linestyle="--", label="ESSB limit 1e-3")
    ax2.axvline(params.ALT_KM, color="gray", linestyle="-.")
    ax2.set_xlabel("Initial altitude [km]")
    ax2.set_ylabel("Cumulative collision probability, objects > 1 cm [-]")
    ax2.set_title("(b) MASTER-8 flux, band = low-high (1 sigma)")
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Altitude trade-off (v0.3): launch rate vs debris indicators
# ---------------------------------------------------------------------------

def altitude_coverage(alt_km):
    """Smallest Walker and its revisit lookup table at one altitude (cached in data/).

    Same requirement, sensor and numerics as the reference (coverage.py); circular
    SSO at alt_km. Output: dict with "walker" (P, S, F) and "table_max_s", "table_mean_s"
    """
    path = os.path.join("data", f"coverage_alt{alt_km:.0f}.npz")
    key = np.append(coverage.coverage_key(), alt_km)
    if os.path.exists(path):
        cached = dict(np.load(path))
        if cached["key"].shape == key.shape and np.allclose(cached["key"], key):
            return cached
    step_s, duration_s = params.COV_STEP_S, params.COV_DURATION_DAYS * DAY_S
    eta, spread = np.radians(params.ETA_MAX_DEG), np.radians(params.RAAN_SPREAD_DEG)
    walker, _ = coverage.walker_search(eta, step_s, duration_s, params.WALKER_P_RANGE,
                                       params.WALKER_S_RANGE, spread, a_m=coverage.R_E + alt_km * 1e3)
    if walker is None:
        raise RuntimeError(f"No Walker of the search range meets the requirement at {alt_km} km")
    max_s, mean_s = coverage.revisit_table_planes(*walker, eta, step_s, duration_s, spread,
                                                  alt_m=alt_km * 1e3)
    data = {"key": key, "walker": np.array(walker), "table_max_s": max_s, "table_mean_s": mean_s}
    os.makedirs("data", exist_ok=True)
    np.savez(path, **data)
    return data


def altitude_scenario(alt_km):
    """Simulation scenario (as REFERENCE) of the smallest Walker at one altitude."""
    cov = altitude_coverage(alt_km)
    n_planes, n_slots, phasing = (int(x) for x in cov["walker"])
    plane_launch_s = np.arange(n_planes) * params.DEPLOY_INTERVAL_DAYS * DAY_S  # D1
    tables = {"max": cov["table_max_s"], "mean": cov["table_mean_s"]}
    return {"name": f"Walker {n_planes}/{n_slots}/{phasing} at {alt_km:.0f} km",
            "n_slots": n_slots, "plane_launch_s": plane_launch_s,
            "deploy_end_s": plane_launch_s[-1] + params.COMMISSIONING_DAYS * DAY_S,
            "tables": lambda: tables}


def altitude_trade(cost_ref):
    """For each altitude: smallest Walker, cheapest strategy, debris indicators.

    cost_ref: cost_study output of the reference scenario, reused at ALT_KM when the
    Walker search there gives the reference Walker (same runs, same numbers).
    ASSUMPTIONs: station-keeping propellant not modelled (relevant mainly at 400-450 km);
    reliability and design life independent of the altitude; disposal manoeuvre success
    DISPOSAL_PROB_REF at every altitude.
    """
    rows = []
    solar = solar_scenarios()
    for alt_km in params.ALT_TRADE_KM:
        scenario = altitude_scenario(alt_km)
        walker = altitude_coverage(alt_km)["walker"]
        is_ref = (alt_km == params.ALT_KM and
                  (walker[0], walker[1]) == (params.N_PLANES, params.N_SLOTS))
        print(f"--- {scenario['name']}" + (" (reference runs reused)" if is_ref else ""))
        cost = cost_ref if is_ref else cost_study(False, scenario, plot=False)
        best = cost["cheapest"]["max"]
        if best is None:
            print("  no configuration meets the max-revisit criterion")
            continue
        out = cost["outputs"][best["label"]]
        tables = end_of_life_tables(alt_km * 1e3)
        ind = debris_indicators(out, tables, params.DISPOSAL_PROB_REF)
        row = {"alt_km": alt_km, "walker": tuple(int(x) for x in walker), "label": best["label"],
               "rate": best["rate"], "n_sats": int(walker[0] * walker[1]),
               "equivalent": [(r["label"], r["rate"]) for r in cost["equivalent"]["max"]
                              if r["label"] != best["label"]]}
        for lv in params.SOLAR_ACTIVITY_LEVELS:
            row[f"life_failed_{lv}"] = tables[lv]["residual_s"][debris.FAILED] / YEAR_S
            row[f"pc_failed_{lv}"] = {c: tables[lv]["pc"][debris.FAILED][c] for c in debris.FLUX_CASES}
            row[f"compliant_{lv}"] = np.nanmean(ind[lv]["frac_compliant"])
            row[f"objyr_{lv}"] = ind[lv]["obj_years_committed"].mean() / YEAR_S
            row[f"collisions_{lv}"] = ind[lv]["collisions"].mean()
        for name, scen in solar.items():   # solar cycle: phase of the mission start uniform per run
            tab = end_of_life_cycle_tables(alt_km * 1e3, scen, phase_grid(
                scen, out["t_s"][-1], params.SOLAR_PHASE_GRID_TRADE_MONTHS))
            ind_c = debris_indicators(out, {name: tab}, params.DISPOSAL_PROB_REF)[name]
            row[f"life_failed_{name}"] = over_cycle(tab["residual_s"][debris.FAILED], tab["phase"]) / YEAR_S
            row[f"censored_{name}"] = bool(np.any(tab["censored"][debris.FAILED]))
            row[f"compliant_{name}"] = np.nanmean(ind_c["frac_compliant"])
            row[f"objyr_{name}"] = ind_c["obj_years_committed"].mean() / YEAR_S
            row[f"collisions_{name}"] = ind_c["collisions"].mean()
        row["pc_operational"] = operational_pc(alt_km * 1e3)
        rows.append(row)

    print(f"Altitude trade-off, max-revisit requirement, p_manoeuvre = {params.DISPOSAL_PROB_REF}, "
          f"{params.ATM_MODEL} (low / moderate / high activity), nominal MASTER flux:")
    print(f"{'h [km]':>6s} {'Walker':>8s} {'cheapest':>22s} {'sats/yr':>7s} "
          f"{'failed lifetime [yr]':>24s} {'objects compliant':>20s} {'dead obj-yr/run':>22s} "
          f"{'P_c failed object':>28s}")
    for r in rows:
        lv = params.SOLAR_ACTIVITY_LEVELS
        print(f"{r['alt_km']:6.0f} {'/'.join(map(str, r['walker'])):>8s} {r['label']:>22s} "
              f"{r['rate']:7.3f} " + " / ".join(f"{r[f'life_failed_{l}']:6.1f}" for l in lv) + "  "
              + " / ".join(f"{r[f'compliant_{l}']:.2f}" for l in lv) + "  "
              + " / ".join(f"{r[f'objyr_{l}']:6.1f}" for l in lv) + "  "
              + " / ".join(f"{r[f'pc_failed_{l}']['nominal']:.1e}" for l in lv))
        for label, rate in r["equivalent"]:
            print(f"{'':15s} no single winner: {label} ({rate:.3f} sats/yr) is statistically "
                  "equivalent to the requirement")
    print(f"Same trade with the solar cycle (phase of the mission start uniform per run; failed "
          f"lifetime over the phases, grid {params.SOLAR_PHASE_GRID_TRADE_MONTHS:.0f} months):")
    print(f"{'h [km]':>6s} {'scenario':>31s} {'failed lifetime mean (5-95 %) [yr]':>36s} "
          f"{'objects compliant':>18s} {'dead obj-yr/run':>16s} {'collisions/run':>15s}")
    for r in rows:
        for name in solar:
            life = r[f"life_failed_{name}"]
            bound = ">=" if r[f"censored_{name}"] else "  "
            print(f"{r['alt_km']:6.0f} {name:>31s} {bound}{life.mean():8.1f} ({np.percentile(life, 5):6.1f}-"
                  f"{np.percentile(life, 95):6.1f}){'':10s} {r[f'compliant_{name}']:18.2f} "
                  f"{bound}{r[f'objyr_{name}']:14.1f} {r[f'collisions_{name}']:15.1e}")
    if any(r[f"censored_{n}"] for r in rows for n in solar):
        print("  >= : the OSCAR forecast ends (12/2129) before the re-entry for some phases: lower bound")
    print_cycle_warning(solar)
    plot_altitude_trade(rows, os.path.join(FIG_DIR, "altitude_trade.png"), list(solar))
    return rows


def short_label(label):
    """Configuration label without the words that do not fit in a figure annotation."""
    return label.replace(" per plane", "").replace(" early ordering", "")


CYCLE_STYLE = {"strong cycle": ("s--", "tab:brown"), WEAK: ("D--", "tab:green")}


def mark_lower_bounds(ax, rows, name, values, color):
    """Open marker and '>=' where the forecast ended before the re-entry (lower bound)."""
    for r, v in zip(rows, values):
        if r[f"censored_{name}"]:
            ax.plot(r["alt_km"], v, "o", ms=12, mfc="none", color=color)
            ax.annotate(">= (forecast ends 2129)", (r["alt_km"], v), textcoords="offset points",
                        xytext=(-6, -14), ha="right", fontsize=7, color=color)


def plot_altitude_trade(rows, path, cycles=()):
    """Launch rate, failed-object lifetime, dead object-years and P_c vs altitude.

    cycles: names of the solar cycle scenarios to add to panels (b) and (c)."""
    alt = np.array([r["alt_km"] for r in rows])
    levels = params.SOLAR_ACTIVITY_LEVELS
    colors = ["tab:blue", "tab:orange", "tab:red"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(alt, [r["rate"] for r in rows], "ko-", label="cheapest meeting the criterion (point estimate)")
    tie_label = f"equivalent to the requirement ({params.MC_CI_LEVEL:.0%} binomial interval)"
    for r in rows:
        ax.annotate(f"Walker {'/'.join(map(str, r['walker']))}\n{short_label(r['label'])}",
                    (r["alt_km"], r["rate"]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7)
        for label, rate in r["equivalent"]:  # no single winner at this altitude
            ax.plot(r["alt_km"], rate, "o", mfc="none", color="k", label=tie_label)
            ax.annotate(short_label(label), (r["alt_km"], rate), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=7)
            tie_label = None  # one legend entry only
    ax.margins(x=0.12, y=0.2)  # room for the text above and below the points
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylabel("Launch rate [satellites/yr]")
    ax.set_title("(a) Smallest Walker and cheapest strategy (max revisit)")
    ax = axes[0, 1]
    for lv, c in zip(levels, colors):
        ax.semilogy(alt, [r[f"life_failed_{lv}"] for r in rows], "o-", color=c, label=f"{lv} activity")
    for name in cycles:
        marker, c = CYCLE_STYLE[name]
        ax.semilogy(alt, [r[f"life_failed_{name}"].mean() for r in rows], marker, color=c,
                    label=f"{name}{' (NOT conservative)' if name == 'strong cycle' else ''}, mean over phases")
        mark_lower_bounds(ax, rows, name, [r[f"life_failed_{name}"].mean() for r in rows], c)
    ax.axhline(params.LEO_LIFETIME_LIMIT_YEARS, color="k", linestyle="--", label="ESSB 5-yr limit")
    ax.set_ylabel("Residual lifetime of a failed satellite [yr]")
    ax.set_title("(b) Natural decay from the operational altitude")
    ax = axes[1, 0]
    for lv, c in zip(levels, colors):
        ax.semilogy(alt, [r[f"objyr_{lv}"] for r in rows], "o-", color=c, label=f"{lv} activity")
    for name in cycles:
        marker, c = CYCLE_STYLE[name]
        ax.semilogy(alt, [r[f"objyr_{name}"] for r in rows], marker, color=c, label=name)
        mark_lower_bounds(ax, rows, name, [r[f"objyr_{name}"] for r in rows], c)
    ax.set_ylabel("Dead object-years per run (committed)")
    ax.set_title(f"(c) Objects left in {params.SIM_DURATION_YEARS:.0f} yr, p_manoeuvre = "
                 f"{params.DISPOSAL_PROB_REF}")
    ax = axes[1, 1]
    for lv, c in zip(levels, colors):
        pc = {k: np.array([r[f"pc_failed_{lv}"][k] for r in rows]) for k in debris.FLUX_CASES}
        ax.semilogy(alt, pc["nominal"], "o-", color=c, label=f"{lv} activity")
        ax.fill_between(alt, pc["low"], pc["high"], color=c, alpha=0.2)
    ax.axhline(params.CUM_COLLISION_PROB_LIMIT, color="k", linestyle="--", label="ESSB limit 1e-3")
    ax.set_ylabel("Collision probability of a failed satellite, > 1 cm [-]")
    ax.set_title("(d) MASTER-8, band = low-high flux")
    for ax in axes.ravel():
        ax.set_xlabel("Operational altitude [km]")
        ax.grid(True, which="both", alpha=0.3)
        if ax is not axes[0, 0]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# On-ground casualty risk of the re-entries (v0.3): DRAMA SARA + constellation
# ---------------------------------------------------------------------------

def sci_5_95(x):
    """'mean (p5-p95)' of an array in scientific notation."""
    return f"{x.mean():.2e} ({np.percentile(x, 5):.2e}-{np.percentile(x, 95):.2e})"


def reentry_counts(out, residual_s, p_manoeuvre):
    """Re-entries per run of one constellation output.

    Inputs: lifecycle output, residual lifetime [s] of each outcome (end_of_life_tables,
            one solar level), success probability of the disposal manoeuvre [-]
    Output: dict of arrays, shape (n_runs,):
        "committed" : satellites that reached orbit within the horizon (launched - lost at
                      launch); each one re-enters uncontrolled sooner or later. A satellite
                      launched AND lost at launch in the very last time step counts as in orbit.
        "horizon"   : objects whose re-entry (end of life + residual lifetime) falls inside
                      the horizon; disposal draws as debris_indicators (stream DEBRIS_SEED_STREAM)
    """
    n_runs = out["n_launched"].shape[0]
    committed = out["n_launched"].sum(axis=1) - out["n_launch_failures"]
    rng = np.random.default_rng([params.SEED, params.DEBRIS_SEED_STREAM])
    in_orbit, outcome = debris.object_outcomes(out["end_kind"], p_manoeuvre, rng)
    run = out["end_run"][in_orbit]
    t_end_s = out["end_t_s"][in_orbit]
    t_reentry_s = t_end_s + debris.object_values(residual_s, outcome, run, t_end_s)
    horizon = np.bincount(run, weights=t_reentry_s <= out["t_s"][-1], minlength=n_runs)
    return {"committed": committed.astype(float), "horizon": horizon}


def casualty_study(results, cost):
    """Casualty risk of one re-entry (DRAMA SARA, checked) and of the constellation.

    Inputs: outputs of the four strategies (main), cost_study output (configurations d)
    Output: dict with the SARA cases, rho_eff and the per-configuration risk (None if
            no SARA run is available)
    """
    print(f"SARA runs (DRAMA 4.1.4, read at run time), Eaglet II-like, i = {params.REENTRY_INC_DEG} deg, "
          f"year {params.POPULATION_YEAR}, A_h = {params.CASUALTY_HUMAN_AREA_M2} m^2:")
    sara = {}
    for name, run in params.SARA_RUNS.items():
        path = casualty.sara_output_path(run["folder"], params.SARA_OUTPUT_FILE)
        if path is None:
            print(f"  NOTE: {run['folder']} not found: case '{name}' skipped")
            continue
        sara[name] = casualty.read_sara_results(path)
    if not sara:
        return None

    have_pop = os.path.exists(params.SARA_POPULATION_1D_PATH)
    rho_eff = None
    if have_pop:
        lat_deg, rho = casualty.read_population_1d(params.SARA_POPULATION_1D_PATH, params.POPULATION_YEAR)
        rho_eff = casualty.effective_density(lat_deg, rho, params.REENTRY_INC_DEG)
    else:
        print(f"  NOTE: {params.SARA_POPULATION_1D_PATH} not found: independent rho_eff skipped")

    for name, s in sara.items():
        print(f"  {name} ({params.SARA_RUNS[name]['text']}):")
        print(f"    fragments on ground {len(s['fragments'])}, mass {s['impact_mass_kg']:.2f} kg, "
              f"A_c SARA {s['casualty_area_m2']:.6f} m^2, E_c SARA {s['ec']:.5e} "
              f"(limit {params.CASUALTY_RISK_LIMIT:.0e}, ESSB 5.5b: "
              f"{'OK' if s['ec'] < params.CASUALTY_RISK_LIMIT else 'NOT compliant'})")
        for fr in s["fragments"]:
            ac = casualty.casualty_area(fr["cross_section_m2"])
            ours = f", E_c ours {rho_eff * ac:.5e}" if rho_eff is not None else ""
            print(f"    {fr['name']:16s} {fr['mass_kg']:.2f} kg, A_i {fr['cross_section_m2']:.6f} m^2, "
                  f"A_c SARA {fr['casualty_area_m2']:.6f} / ours {ac:.6f} m^2, E_c SARA {fr['ec']:.5e}{ours}, "
                  f"impact energy {fr['energy_j']:.0f} J")
        if s["casualty_area_m2"] > 0:
            print(f"    rho_eff = E_c / A_c: SARA {s['ec'] / s['casualty_area_m2'] * 1e6:.4f} /km^2"
                  + (f", ours {rho_eff * 1e6:.4f} /km^2" if rho_eff is not None else ""))

    years_rho = {}
    if have_pop:
        for year in params.POPULATION_YEARS_SWEEP:
            lat_deg, rho = casualty.read_population_1d(params.SARA_POPULATION_1D_PATH, year)
            years_rho[year] = casualty.effective_density(lat_deg, rho, params.REENTRY_INC_DEG)
        print("  rho_eff vs population year (sensitivity, re-entries span ~2025-2045): "
              + ", ".join(f"{y} {r * 1e6:.2f}" for y, r in years_rho.items()) + " /km^2")

    configs = dict(results)
    configs.update({label: o for label, o in cost["outputs"].items() if label.startswith("d)")})
    residual = end_of_life_tables(params.ALT_KM * 1e3)["moderate"]["residual_s"]
    risk_cases = {name: s["ec"] for name, s in sara.items()}
    n_runs = next(iter(configs.values()))["n_launched"].shape[0]
    print(f"Constellation casualty risk, {params.SIM_DURATION_YEARS:.0f} yr, {n_runs} runs, mean (5-95 %); "
          f"re-entries: committed = all satellites reaching orbit, horizon = re-entered within the "
          f"horizon (moderate activity, p_manoeuvre {params.DISPOSAL_PROB_REF}); launch failures excluded:")
    rows = {}
    for label, out in configs.items():
        counts = reentry_counts(out, residual, params.DISPOSAL_PROB_REF)
        rows[label] = {"counts": counts, "risk": {}}
        print(f"  {label}: re-entries committed {mean_5_95(counts['committed'])}, "
              f"within horizon {mean_5_95(counts['horizon'])}")
        for case, ec in risk_cases.items():
            for kind in ("committed", "horizon"):
                expected, p_one = casualty.constellation_risk(counts[kind], ec)
                rows[label]["risk"][(case, kind)] = (expected, p_one)
                if ec > 0:
                    print(f"    {case:10s} {kind:9s}: sum E_c {sci_5_95(expected)}, "
                          f"P(>= 1 casualty) {sci_5_95(p_one)}")
                elif kind == "committed":
                    print(f"    {case:10s}: E_c = 0 per re-entry (full demise) -> 0")
    n_sats = params.N_PLANES * params.N_SLOTS
    print(f"  Large constellations (>= {params.LARGE_CONST_MIN_SATS} satellites, ESSB 3.2.25, TO VERIFY): "
          f"limit {params.CASUALTY_RISK_LIMIT_LARGE_CONST:.0e} per re-entry (5.5d.1) NOT applicable to the "
          f"reference Walker ({n_sats} satellites) nor to IRIDE Eaglet II (24); for information only.")
    if "components" in sara and years_rho:
        plot_casualty(sara["components"], years_rho, rows, os.path.join(FIG_DIR, "casualty_risk.png"))
    return {"sara": sara, "rho_eff": rho_eff, "rho_years": years_rho, "rows": rows}


def plot_casualty(sara_comp, years_rho, rows, path):
    """(a) E_c of one re-entry vs population year; (b) expected casualties per configuration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    years = np.array(list(years_rho))
    ec_years = np.array(list(years_rho.values())) * sara_comp["casualty_area_m2"]
    ax1.semilogy(years, ec_years, "o-", color="C0", label="components case, our rho_eff x A_c")
    ax1.semilogy([params.POPULATION_YEAR], [sara_comp["ec"]], "x", color="k", markersize=10,
                 label="DRAMA SARA run")
    ax1.axhline(params.CASUALTY_RISK_LIMIT, color="k", linestyle="--", label="ESSB 5.5b limit 1e-4")
    ax1.axhline(params.CASUALTY_RISK_LIMIT_LARGE_CONST, color="gray", linestyle=":",
                label="large-constellation limit 1e-6 (not applicable)")
    ax1.set_xticks(years)
    ax1.set_xlabel("Population year (UN WPP 2022 medium variant)")
    ax1.set_ylabel("Casualty risk of one re-entry E_c [-]")
    ax1.set_title(f"(a) One re-entry, i = {params.REENTRY_INC_DEG} deg (base case: E_c = 0)")
    ax1.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    labels = list(rows)
    y = np.arange(len(labels))
    for kind, color, dy in (("committed", "C0", -0.15), ("horizon", "C1", 0.15)):
        expected = [rows[lb]["risk"][("components", kind)][0] for lb in labels]
        mid = np.array([np.median(e) for e in expected]) * 1e4  # median: always inside 5-95 %
        lo = mid - np.array([np.percentile(e, 5) for e in expected]) * 1e4
        hi = np.array([np.percentile(e, 95) for e in expected]) * 1e4 - mid
        text = ("committed: all satellites reaching orbit" if kind == "committed"
                else f"re-entered within {params.SIM_DURATION_YEARS:.0f} yr")
        ax2.errorbar(mid, y + dy, xerr=[lo, hi], fmt="o", color=color, capsize=3, label=text)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(left=0)
    ax2.set_xlabel(f"Expected casualties, sum of E_c of the re-entries [x 1e-4]")
    ax2.set_title("(b) Constellation, components case (median, 5-95 %)", fontsize=11)
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Constellation lifecycle simulator, nominal case")
    parser.add_argument("--full", action="store_true",
                        help="also check T* of strategy d) with the extra seeds (slower)")
    parser.add_argument("--trade", action="store_true",
                        help="also run the altitude trade-off (slower)")
    args = parser.parse_args()
    full = args.full
    print("Mode: full" if full else "Mode: fast (use --full for the T* seed check)")
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Coverage of the reference Walker constellation")
    coverage_study()

    print("Nominal case")
    results = {label: run_strategy(s, params.ORDER_TO_LAUNCH_DAYS) for label, s in STRATEGIES.items()}
    print_table(results)
    plot_active_vs_time(results, os.path.join(FIG_DIR, "active_vs_time.png"))

    print("Sweep of the order-to-launch time (constellation availability, max requirement)")
    sweeps = {}
    for label, strategy in STRATEGIES.items():
        sweeps[label] = sweep_order_time(strategy)
        print(f"  {label}: " + ", ".join(f"{a:.3f}" for a in sweeps[label]["con_mean"]))
    plot_availability_vs_order_time(sweeps, os.path.join(FIG_DIR, "availability_vs_order_time.png"))

    print("Fixed cadence d), search of the minimum cadence, cost comparison")
    cost = cost_study(full)

    print("Second scenario: IRIDE Eaglet II from public TLEs")
    iride_study()

    print("Orbital decay (v0.3)")
    decay_study()

    print("End of life and debris indicators (v0.3)")
    debris_study(cost)

    print("On-ground casualty risk of the re-entries (v0.3, DRAMA SARA)")
    casualty_study(results, cost)

    if args.trade:
        print("Altitude trade-off (v0.3): launch rate vs debris indicators")
        altitude_trade(cost)
    else:
        print("Altitude trade-off skipped, run with --trade (altitude_trade.png not updated)")
    print(f"Figures saved in {FIG_DIR}/")


if __name__ == "__main__":
    main()
