"""Run the nominal case for the four strategies and save the figures into figures/.

Strategies (lifecycle.simulate_plane):
    a)   launch on failure
    b)   in-orbit spares, (s, S) policy
    c)   early ordering of the retirement replacement
    b+c) spares and early ordering
    d)   fixed launch cadence (lifecycle.simulate_fixed_cadence), with the search of
         the minimum cadence meeting AVAIL_TARGET with probability AVAIL_CONFIDENCE

Figures:
- active_vs_time.png              : operational satellites vs time, one panel per strategy
- availability_vs_order_time.png  : availability vs ORDER_TO_LAUNCH_DAYS sweep
- cost_vs_availability.png        : strategy d) fixed cadence vs c) and b+c), launch
                                    rate vs the two alternative requirements
- revisit_vs_latitude.png         : max and mean revisit vs latitude of the reference Walker

Availabilities are averaged from the end of the nominal deployment
(last batch + commissioning) to the end of the horizon.
"""

import os

import matplotlib
matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np

import coverage
import params
from analytic import (reference_launch_rate, slot_availability_pre_order,
                      slot_availability_renewal_reward)
from lifecycle import (deployment_launch_times, plane_availability, simulate_fixed_cadence,
                       simulate_plane, time_to_full_operation)
from scipy.integrate import quad
from reliability import reliability_analytic
from search import bisect_interval, search_max_interval

FIG_DIR = "figures"
DAY_S = params.SECONDS_PER_DAY
YEAR_S = params.SECONDS_PER_YEAR
RELIABILITY = {"model": "weibull", "beta": params.WEIBULL_BETA,
               "theta_s": params.WEIBULL_THETA_YEARS * YEAR_S}
ORDER_SWEEP_DAYS = np.arange(90.0, 731.0, 80.0)  # 90, 170, ..., 730 days
DEPLOY_END_S = ((params.DEPLOY_BATCHES - 1) * params.DEPLOY_INTERVAL_DAYS
                + params.COMMISSIONING_DAYS) * DAY_S

STRATEGIES = {
    "a) Launch on failure": dict(n_spares=0, reorder_level=0, pre_order_retirement=False),
    f"b) {params.N_SPARES} in-orbit spare(s)": dict(
        n_spares=params.N_SPARES, reorder_level=params.REORDER_LEVEL, pre_order_retirement=False),
    "c) Early ordering": dict(n_spares=0, reorder_level=0, pre_order_retirement=True),
    f"b+c) {params.N_SPARES} spare(s) + early ordering": dict(
        n_spares=params.N_SPARES, reorder_level=params.REORDER_LEVEL, pre_order_retirement=True),
}


def run_strategy(strategy, order_to_launch_days, n_runs=params.N_RUNS):
    """Nominal simulation of one strategy with a given order-to-launch time.

    Inputs:
        strategy             : dict with n_spares, reorder_level, pre_order_retirement
        order_to_launch_days : order-to-launch time [days]
        n_runs               : number of Monte Carlo runs [-]
    Output: simulator output (see lifecycle.simulate_plane)
    """
    n_units = params.N_SLOTS + strategy["n_spares"]
    initial_launch_s = deployment_launch_times(n_units, params.DEPLOY_BATCHES,
                                               params.DEPLOY_INTERVAL_DAYS * DAY_S)
    return simulate_plane(
        n_slots=params.N_SLOTS, n_runs=n_runs, seed=params.SEED,
        sim_duration_s=params.SIM_DURATION_YEARS * YEAR_S, dt_s=params.DT_DAYS * DAY_S,
        initial_launch_s=initial_launch_s, order_to_launch_s=order_to_launch_days * DAY_S,
        commissioning_s=params.COMMISSIONING_DAYS * DAY_S,
        design_life_s=params.DESIGN_LIFE_YEARS * YEAR_S,
        launch_success_prob=params.LAUNCH_SUCCESS_PROB,
        sats_per_launch=params.SATS_PER_LAUNCH, reliability=RELIABILITY, **strategy)


def summary(out):
    """Key numbers of one simulation (availability from the end of deployment).

    Output: dict with mean and 5/95 percentiles of plane availability [-],
    satellites launched per run [-], time to full operation [days], and the
    number of runs that never reach full operation.
    """
    avail = plane_availability(out["n_active"], params.MIN_ACTIVE, out["t_s"], DEPLOY_END_S)
    launched = out["n_launched"].sum(axis=1)
    t_full_days = time_to_full_operation(out["n_active"], params.MIN_ACTIVE, out["t_s"]) / DAY_S
    reached = ~np.isnan(t_full_days)
    return {
        "avail": (avail.mean(), *np.percentile(avail, [5, 95])),
        "launched": (launched.mean(), *np.percentile(launched, [5, 95])),
        "t_full_days": (t_full_days[reached].mean(), *np.percentile(t_full_days[reached], [5, 95])),
        "never_full": int(np.sum(~reached)),
    }


def print_table(results):
    """Print the comparison table of the strategies."""
    print(f"{'Strategy':38s} {'Availability (5-95 %)':26s} {'Launched/run (5-95 %)':24s} "
          f"{'Time to full op. [d] (5-95 %)':30s} Never full")
    for label, out in results.items():
        s = summary(out)
        print(f"{label:38s} {s['avail'][0]:.4f} ({s['avail'][1]:.3f}-{s['avail'][2]:.3f})      "
              f"{s['launched'][0]:6.2f} ({s['launched'][1]:.0f}-{s['launched'][2]:.0f})          "
              f"{s['t_full_days'][0]:7.1f} ({s['t_full_days'][1]:.0f}-{s['t_full_days'][2]:.0f})"
              f"              {s['never_full']}")


def plot_active_vs_time(results, path):
    """2x2 panels: mean and 5-95 % band of the operational satellites vs time.

    Inputs:
        results : dict {label: simulator output}, four strategies
        path    : output file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, (label, out) in zip(axes.flat, results.items()):
        t_years = out["t_s"] / YEAR_S
        p5, p95 = np.percentile(out["n_active"], [5, 95], axis=0)
        avail = plane_availability(out["n_active"], params.MIN_ACTIVE, out["t_s"], DEPLOY_END_S)
        ax.fill_between(t_years, p5, p95, color="C0", alpha=0.25, label="5-95 % band")
        ax.plot(t_years, out["n_active"].mean(axis=0), color="C0", label="mean")
        ax.axhline(params.MIN_ACTIVE, color="k", linestyle="--", label="MIN_ACTIVE")
        ax.axvline(DEPLOY_END_S / YEAR_S, color="gray", linestyle=":", label="end of deployment")
        ax.set_title(f"{label}\nplane availability {avail.mean():.3f}", fontsize=10)
        ax.set_ylim(-0.2, params.N_SLOTS + 0.5)
        ax.grid(True, alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel("Time since first launch [years]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Operational satellites [-]")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"One plane, {params.N_SLOTS} slots, {params.N_RUNS} runs | DL = {params.DESIGN_LIFE_YEARS:.0f} yr, "
        f"order-to-launch = {params.ORDER_TO_LAUNCH_DAYS:.0f} d, commissioning = "
        f"{params.COMMISSIONING_DAYS:.0f} d, deployment {params.DEPLOY_BATCHES} batches "
        f"every {params.DEPLOY_INTERVAL_DAYS:.0f} d", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def slot_analytic(strategy, order_days):
    """Steady-state slot availability for a) and c) without spares, else NaN."""
    if strategy["n_spares"] > 0:
        return np.nan
    f = slot_availability_pre_order if strategy["pre_order_retirement"] else slot_availability_renewal_reward
    return f(RELIABILITY, params.DESIGN_LIFE_YEARS * YEAR_S, order_days * DAY_S,
             params.COMMISSIONING_DAYS * DAY_S)


def sweep_order_time(strategy):
    """Availability of one strategy for each value of ORDER_SWEEP_DAYS.

    Output: dict of arrays over the sweep
        "slot_mc"      : mean fraction of occupied slots after deployment (MC) [-]
        "slot_rr"      : steady-state slot availability, analytic [-] (a and c only)
        "plane_mean"   : mean plane availability (>= MIN_ACTIVE) over the runs [-]
        "plane_p5/p95" : 5 and 95 percentiles of the plane availability [-]
    """
    keys = ["slot_mc", "slot_rr", "plane_mean", "plane_p5", "plane_p95"]
    sweep = {k: np.zeros(len(ORDER_SWEEP_DAYS)) for k in keys}
    for i, order_days in enumerate(ORDER_SWEEP_DAYS):
        out = run_strategy(strategy, order_days, n_runs=params.N_RUNS_SWEEP)
        after = out["t_s"] >= DEPLOY_END_S
        avail = plane_availability(out["n_active"], params.MIN_ACTIVE, out["t_s"], DEPLOY_END_S)
        sweep["slot_mc"][i] = out["n_active"][:, after].mean() / params.N_SLOTS
        sweep["slot_rr"][i] = slot_analytic(strategy, order_days)
        sweep["plane_mean"][i] = avail.mean()
        sweep["plane_p5"][i], sweep["plane_p95"][i] = np.percentile(avail, [5, 95])
    return sweep


def plot_availability_vs_order_time(sweeps, path):
    """Two panels: (a) slot availability MC vs analytic, (b) plane availability MC.

    Inputs:
        sweeps : dict {label: output of sweep_order_time}
        path   : output file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for label, sw in sweeps.items():
        line, = ax1.plot(ORDER_SWEEP_DAYS, sw["slot_mc"], "o-",
                         label=f"{label}: MC, {params.SIM_DURATION_YEARS:.0f} yr")
        if not np.all(np.isnan(sw["slot_rr"])):
            ax1.plot(ORDER_SWEEP_DAYS, sw["slot_rr"], "--", color=line.get_color(),
                     label=f"{label}: analytic steady state (t -> inf)")
        ax2.plot(ORDER_SWEEP_DAYS, sw["plane_mean"], "o-", color=line.get_color(),
                 label=f"{label}: MC mean")
        ax2.fill_between(ORDER_SWEEP_DAYS, sw["plane_p5"], sw["plane_p95"],
                         color=line.get_color(), alpha=0.15)
    ax1.set_title("(a) Mean fraction of occupied slots after deployment")
    ax2.set_title(f"(b) Plane availability (>= {params.MIN_ACTIVE} operational), band 5-95 %")
    for ax in (ax1, ax2):
        ax.set_xlabel("Order-to-launch time [days]")
        ax.set_ylabel("Availability [-]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# v0.2: strategy d) fixed cadence, search of the minimum cadence, cost comparison
# ---------------------------------------------------------------------------

# Two alternative mission requirements on the per-run availability after deployment:
# "percentile": at least AVAIL_CONFIDENCE of the runs have availability >= AVAIL_TARGET
# "mean"      : the mean availability over the runs is >= AVAIL_TARGET
REQUIREMENTS = {
    "percentile": {"threshold": params.AVAIL_CONFIDENCE,
                   "text": f"{params.AVAIL_CONFIDENCE:.0%} of runs with availability >= {params.AVAIL_TARGET}"},
    "mean": {"threshold": params.AVAIL_TARGET,
             "text": f"mean availability >= {params.AVAIL_TARGET}"},
}


def run_cadence(k, interval_days, seed, n_runs):
    """Strategy d): N_SLOTS deployed in batches, then k satellites every interval_days.

    The first cadence launch is together with the LAST deployment batch, so the
    cadence satellites are commissioned at the end of deployment and act at once
    as spares. Reason: no operator would leave the plane without spares during
    the infant-mortality period right after deployment (with the Weibull fit,
    beta < 1, the failure rate is highest at the start); it is also consistent
    with strategy b), whose first batch already carries the spare. Starting the
    cadence one interval after the last batch would leave a gap of up to T days
    without spares, and the requirement would then measure that gap instead of
    the cadence (seen in the first version: T* ~ 5 % of the averaging window).
    """
    initial_launch_s = deployment_launch_times(params.N_SLOTS, params.DEPLOY_BATCHES,
                                               params.DEPLOY_INTERVAL_DAYS * DAY_S)
    return simulate_fixed_cadence(
        n_slots=params.N_SLOTS, cadence_sats=k, cadence_interval_s=interval_days * DAY_S,
        cadence_start_s=initial_launch_s.max(), n_runs=n_runs, seed=seed,
        sim_duration_s=params.SIM_DURATION_YEARS * YEAR_S, dt_s=params.DT_DAYS * DAY_S,
        initial_launch_s=initial_launch_s, commissioning_s=params.COMMISSIONING_DAYS * DAY_S,
        design_life_s=params.DESIGN_LIFE_YEARS * YEAR_S,
        launch_success_prob=params.LAUNCH_SUCCESS_PROB, reliability=RELIABILITY)


def requirement_metrics(out):
    """Values compared with the two requirements (availability after deployment).

    Output: {"percentile": fraction of runs with availability >= AVAIL_TARGET [-],
             "mean": mean availability over the runs [-]}
    """
    avail = plane_availability(out["n_active"], params.MIN_ACTIVE, out["t_s"], DEPLOY_END_S)
    return {"percentile": np.mean(avail >= params.AVAIL_TARGET), "mean": avail.mean()}


def mc_launch_rate_per_year(out):
    """Mean satellites launched per year after the end of deployment (MC) [1/yr]."""
    after = out["t_s"] >= DEPLOY_END_S
    years = (out["t_s"][-1] - DEPLOY_END_S) / YEAR_S
    return out["n_launched"][:, after].sum(axis=1).mean() / years


def steady_rate_pre_order(n_spares):
    """Steady-state launch rate of c) / b+c) with one reorder per loss [satellites/yr].

    Each of the N + S units renews independently with mean cycle
    E[cycle_c] = ORDER + integral_0^(DL - ORDER - C) R dt (analytic.slot_availability_pre_order),
    so the rate is (N + S) / E[cycle_c].
    """
    order_s = params.ORDER_TO_LAUNCH_DAYS * DAY_S
    a_s = params.DESIGN_LIFE_YEARS * YEAR_S - order_s - params.COMMISSIONING_DAYS * DAY_S
    integral_s = YEAR_S * quad(lambda ty: reliability_analytic(ty * YEAR_S, **RELIABILITY),
                               0.0, a_s / YEAR_S, limit=200)[0]
    return (params.N_SLOTS + n_spares) / (order_s + integral_s) * YEAR_S


# Simulations of strategy d) already run: {(k, seed, T_days): requirement_metrics}.
# Shared by the two requirements, so the same (k, seed, T) is never simulated twice.
CADENCE_CACHE = {}


def cadence_metrics(k, seed, t_days):
    """Requirement metrics of strategy d) for (k, seed, T), with N_RUNS_SWEEP runs (cached)."""
    key = (k, seed, int(t_days))
    if key not in CADENCE_CACHE:
        CADENCE_CACHE[key] = requirement_metrics(run_cadence(k, int(t_days), seed,
                                                             params.N_RUNS_SWEEP))
    return CADENCE_CACHE[key]


def cadence_search(k, seed, requirement, t_ref_days=None):
    """Largest interval T [days] meeting one requirement for k satellites per launch.

    Without t_ref_days: coarse sweep + bisection (search.search_max_interval).
    With t_ref_days (check of another seed): bisection between t_ref -/+ 2 grid
    steps after checking that the bracket is valid, else a full sweep.
    Output: result dict of search.search_max_interval (at least "t_star_days")
    """
    threshold = REQUIREMENTS[requirement]["threshold"]

    def evaluate(t_days):
        return cadence_metrics(k, seed, t_days)[requirement]

    start, stop, step = (np.array(params.CADENCE_GRID_DAYS_PER_SAT) * k).astype(int)
    if t_ref_days is not None:
        lo, hi = t_ref_days - 2 * step, t_ref_days + 2 * step
        if lo > 0 and evaluate(lo) >= threshold and evaluate(hi) < threshold:
            return {"t_star_days": bisect_interval(evaluate, lo, hi, threshold)}
    return search_max_interval(evaluate, np.arange(start, stop + 1, step), threshold)


def failure_breakdown(out):
    """How many unplanned losses the runs that fail the percentile requirement had.

    Unplanned loss = on-orbit failure or launch failure (retirements excluded).
    Output: (fraction of runs failing [-], {number of losses: fraction of ALL runs [-]})
    """
    avail = plane_availability(out["n_active"], params.MIN_ACTIVE, out["t_s"], DEPLOY_END_S)
    failing = avail < params.AVAIL_TARGET
    losses = out["n_failures"] + out["n_launch_failures"]
    counts = {int(n): np.sum(failing & (losses == n)) / len(avail) for n in np.unique(losses[failing])}
    return failing.mean(), counts


def cost_study():
    """Strategy d) search of T* and comparison with c) and b+c) (table + figure)."""
    ref_rate = reference_launch_rate(params.N_SLOTS, RELIABILITY,
                                     params.DESIGN_LIFE_YEARS * YEAR_S,
                                     params.COMMISSIONING_DAYS * DAY_S) * YEAR_S
    print(f"Reference rate N/E[up] = {ref_rate:.3f} satellites/yr "
          f"(every satellite gives its whole useful life, no gaps)")

    # d) search of T* for each k and each requirement (nominal seed + check seeds)
    d_best = {req: {} for req in REQUIREMENTS}
    for req, spec in REQUIREMENTS.items():
        print(f"Requirement '{req}': {spec['text']}")
        for k in params.CADENCE_SATS_OPTIONS:
            result = cadence_search(k, params.SEED, req)
            t_star = result["t_star_days"]
            if np.isnan(t_star):
                print(f"  d) k = {k}: no interval of the grid meets the requirement")
                continue
            if not result.get("bracketed", True):
                print(f"  d) k = {k}: every interval of the grid meets it (T* >= {t_star} d)")
            checks = [cadence_search(k, seed, req, t_ref_days=t_star)["t_star_days"]
                      for seed in params.SEEDS_CHECK]
            d_best[req][k] = t_star
            print(f"  d) k = {k}: T* = {t_star} d (seeds {params.SEEDS_CHECK}: {checks}) -> "
                  f"{k * 365.25 / t_star:.3f} satellites/yr; non monotone: {result.get('non_monotone')}")

    # Sweep curves for the figure (nominal seed, grid points only)
    d_curves = {}
    for k in params.CADENCE_SATS_OPTIONS:
        t_grid = np.array(sorted(t for (kk, seed, t) in CADENCE_CACHE
                                 if kk == k and seed == params.SEED))
        d_curves[k] = {"rate": k * 365.25 / t_grid,
                       **{req: np.array([CADENCE_CACHE[(k, params.SEED, t)][req] for t in t_grid])
                          for req in REQUIREMENTS}}

    # Table with N_RUNS runs: d) at each T* found, c) and b+c) with S = 0..3
    rows = []
    configs = []
    for req in REQUIREMENTS:
        for k, t_star in d_best[req].items():
            if ("d", k, t_star) not in configs:
                configs.append(("d", k, t_star))
    for n_spares in range(0, 4):
        configs.append(("c" if n_spares == 0 else "b+c", n_spares, None))

    print(f"{'Configuration':24s} {'sats/yr steady':>14s} {'sats/yr MC':>10s} "
          f"{'mean avail (5-95 %)':>22s} {'runs >= target':>14s}")
    outputs = {}
    for family, value, t_star in configs:
        if family == "d":
            out = run_cadence(value, t_star, params.SEED, params.N_RUNS)
            rate = value * 365.25 / t_star
            label = f"d) k={value}, T={t_star} d"
        else:
            strategy = dict(n_spares=value, reorder_level=value, pre_order_retirement=True)
            out = run_strategy(strategy, params.ORDER_TO_LAUNCH_DAYS)
            rate = steady_rate_pre_order(value)
            label = "c) early ordering" if value == 0 else f"b+c) S={value}"
        outputs[label] = out
        metrics = requirement_metrics(out)
        avail = plane_availability(out["n_active"], params.MIN_ACTIVE, out["t_s"], DEPLOY_END_S)
        p5, p95 = np.percentile(avail, [5, 95])
        rows.append({"family": family, "label": label, "rate": rate, **metrics})
        print(f"{label:24s} {rate:14.3f} {mc_launch_rate_per_year(out):10.3f} "
              f"{metrics['mean']:10.4f} ({p5:.3f}-{p95:.3f}) {metrics['percentile']:14.3f}")

    # Cheapest configuration of each family, for each requirement
    for req, spec in REQUIREMENTS.items():
        print(f"Cheapest configuration per family, requirement '{req}' ({spec['text']}):")
        for family in ["d", "c", "b+c"]:
            ok = [r for r in rows if r["family"] == family and r[req] >= spec["threshold"]]
            if ok:
                best = min(ok, key=lambda r: r["rate"])
                print(f"  {family:4s}: {best['label']} -> {best['rate']:.3f} satellites/yr")
            else:
                print(f"  {family:4s}: none of the configurations tested meets it")

    # c) alone: why its runs fail the percentile requirement
    failing, counts = failure_breakdown(outputs["c) early ordering"])
    print(f"c) alone: {failing:.3f} of the runs fail (availability < {params.AVAIL_TARGET}); "
          "by number of unplanned losses in the run: "
          + ", ".join(f"{n}: {f:.3f}" for n, f in counts.items()))

    plot_cost_vs_availability(d_curves, d_best, rows, ref_rate,
                              os.path.join(FIG_DIR, "cost_vs_availability.png"))


def plot_cost_vs_availability(d_curves, d_best, rows, ref_rate, path):
    """Launch rate vs requirement metric, one panel per requirement.

    Inputs:
        d_curves : {k: {"rate": [1/yr], "percentile": [-], "mean": [-]}} from the d) sweep
        d_best   : {requirement: {k: T* [days]}}
        rows     : table rows (family, label, steady rate [1/yr], metrics) from N_RUNS runs
        ref_rate : reference rate N/E[up] [1/yr]
        path     : output file path
    """
    markers = {"c) early ordering": "s", "b+c) S=1": "D", "b+c) S=2": "^", "b+c) S=3": "v"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (req, spec) in zip(axes, REQUIREMENTS.items()):
        for k, curve in d_curves.items():
            order = np.argsort(curve["rate"])
            line, = ax.plot(curve["rate"][order], curve[req][order], ".-", alpha=0.7,
                            label=f"d) fixed cadence, k = {k}")
            if k in d_best[req]:
                t_star = d_best[req][k]
                ax.plot(k * 365.25 / t_star, spec["threshold"], "*", ms=14,
                        color=line.get_color(), label=f"d) k = {k}: T* = {t_star} d")
        for row in rows:
            if row["family"] in ("c", "b+c"):
                ax.plot(row["rate"], row[req], markers[row["label"]], ms=8, color="k",
                        mfc="none" if row["family"] == "c" else "k", label=row["label"])
        ax.axhline(spec["threshold"], color="r", linestyle="--", label=f"requirement: {spec['text']}")
        ax.axvline(ref_rate, color="gray", linestyle=":",
                   label=f"reference rate N/E[up] = {ref_rate:.2f}/yr (whole useful life, no gaps)")
        ax.set_xlabel("Satellites launched per year, steady state [1/yr]")
        ax.set_xlim(0.95, 2.3)  # the comparison is between ~1 and ~2 satellites/yr
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel(f"Fraction of runs with availability >= {params.AVAIL_TARGET} [-]")
    axes[0].set_title("(a) Requirement on the 5th percentile of the runs", fontsize=10)
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Mean plane availability [-]")
    axes[1].set_title("(b) Requirement on the mean availability", fontsize=10)
    fig.suptitle(f"Cost vs availability | {params.N_SLOTS} slots, DL = {params.DESIGN_LIFE_YEARS:.0f} yr, "
                 f"order-to-launch = {params.ORDER_TO_LAUNCH_DAYS:.0f} d, commissioning = "
                 f"{params.COMMISSIONING_DAYS:.0f} d", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# v0.2: coverage of the reference Walker constellation (coverage.py)
# ---------------------------------------------------------------------------

COVERAGE_CACHE = os.path.join("data", "revisit_table.npz")


def hours(x_s):
    """Seconds -> hours as text, 'inf' when never revisited."""
    return "   inf" if not np.isfinite(x_s) else f"{x_s / 3600:6.2f}"


def coverage_study():
    """Walker choice, lookup tables and revisit vs latitude (cached in data/)."""
    data = coverage.load_or_build_coverage_data(COVERAGE_CACHE)
    a_m = coverage.R_E + params.ALT_KM * 1e3
    inc = coverage.sso_inclination(a_m)
    print(f"Reference SSO: h = {params.ALT_KM:.0f} km, i = {np.degrees(inc):.3f} deg, "
          f"nodal period = {coverage.nodal_period(a_m, inc) / 60:.3f} min")
    for eta_deg, walker in zip(data["eta_sweep_deg"], data["best_walker"]):
        lam, _ = coverage.footprint_from_sensor(params.ALT_KM * 1e3, np.radians(eta_deg))
        print(f"  eta = {eta_deg:4.0f} deg (lambda = {np.degrees(lam):.2f} deg): smallest Walker "
              f"meeting max revisit <= {params.REVISIT_MAX_HOURS:.0f} h at "
              f"{params.REVISIT_LAT_DEG:.0f} deg: {walker[0]}/{walker[1]}/{walker[2]} "
              f"(T = {walker[0] * walker[1]})")

    print(f"Tables at {params.REVISIT_LAT_DEG:.0f} deg N, eta = {params.ETA_MAX_DEG:.0f} deg [hours]")
    print(" n | T1 R1 max | T1 R1 mean | T1 R2 mean max | T1 R2 worst max | T2 R1 max | T2 R1 mean")
    for i, n in enumerate(data["table_n"]):
        print(f"{n:2d} | {hours(data['table_t1_r1_max'][i])}    | {hours(data['table_t1_r1_mean'][i])}     |"
              f" {hours(data['table_t1_r2_mean_max'][i])}         | {hours(data['table_t1_r2_worst_max'][i])}"
              f"          | {hours(data['table_t2_r1_max'][i])}    | {hours(data['table_t2_r1_mean'][i])}")
    n_min = coverage.min_active_from_table(data["table_t1_r1_max"])
    print(f"MIN_ACTIVE derived from T1-R1 (max revisit <= {params.REVISIT_MAX_HOURS:.0f} h): "
          f"{n_min} of {data['table_n'][-1]} per plane")
    plot_revisit_vs_latitude(data, os.path.join(FIG_DIR, "revisit_vs_latitude.png"))


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
    ax1.axvline(params.REVISIT_LAT_DEG, color="gray", linestyle=":")
    ax2.axvline(params.REVISIT_LAT_DEG, color="gray", linestyle=":")
    ax1.set_title("(a) Max revisit")
    ax2.set_title("(b) Mean revisit")
    for ax in (ax1, ax2):
        ax.set_xlabel("Latitude [deg]")
        ax.set_ylabel("Revisit [h]")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Walker {p}/{s}/{f} (star, RAAN spread {params.RAAN_SPREAD_DEG:.0f} deg), "
                 f"SSO {params.ALT_KM:.0f} km, geometric coverage (no illumination)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Nominal case")
    results = {label: run_strategy(s, params.ORDER_TO_LAUNCH_DAYS) for label, s in STRATEGIES.items()}
    print_table(results)
    plot_active_vs_time(results, os.path.join(FIG_DIR, "active_vs_time.png"))

    print("Sweep of the order-to-launch time")
    sweeps = {}
    for label, strategy in STRATEGIES.items():
        sweeps[label] = sweep_order_time(strategy)
        print(f"  {label}: plane availability "
              + ", ".join(f"{a:.3f}" for a in sweeps[label]["plane_mean"]))
    plot_availability_vs_order_time(sweeps, os.path.join(FIG_DIR, "availability_vs_order_time.png"))

    print("v0.2: fixed cadence d), search of the minimum cadence, cost comparison")
    cost_study()

    print("v0.2: coverage of the reference Walker constellation")
    coverage_study()
    print(f"Figures saved in {FIG_DIR}/")


if __name__ == "__main__":
    main()
