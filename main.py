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
"""

import os

import matplotlib
matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from scipy.integrate import quad

import coverage
import params
from analytic import (reference_launch_rate, slot_availability_pre_order,
                      slot_availability_renewal_reward)
from lifecycle import (constellation_availability, constellation_revisit_series,
                       simulate_constellation, simulate_fixed_cadence, simulate_plane,
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

# Revisit requirements: which lookup table and which limit
REVISIT_REQS = {
    "max": {"table": "table_planes_max_s", "limit_s": params.REVISIT_MAX_HOURS * 3600,
            "text": f"max revisit <= {params.REVISIT_MAX_HOURS:.0f} h"},
    "mean": {"table": "table_planes_mean_s", "limit_s": params.REVISIT_MEAN_HOURS * 3600,
             "text": f"mean revisit <= {params.REVISIT_MEAN_HOURS:.0f} h"},
}

_COVERAGE = {}


def coverage_data():
    """Coverage results from the cache in data/ (built once, see coverage.py)."""
    if not _COVERAGE:
        _COVERAGE.update(coverage.load_or_build_coverage_data(COVERAGE_CACHE))
    return _COVERAGE


def hours(x_s):
    """Seconds -> hours as text, 'inf' when never revisited."""
    return "   inf" if not np.isfinite(x_s) else f"{x_s / 3600:6.2f}"


# ---------------------------------------------------------------------------
# Constellation runs
# ---------------------------------------------------------------------------

def run_strategy(strategy, order_to_launch_days, n_runs=params.N_RUNS, seed=params.SEED):
    """Strategies a/b/c/b+c on every plane: one launch per plane at PLANE_LAUNCH_S (D1).

    Inputs:
        strategy             : dict with n_spares, reorder_level, pre_order_retirement
        order_to_launch_days : order-to-launch time [days]
        n_runs, seed         : Monte Carlo runs and master seed
    Output: constellation output (lifecycle.simulate_constellation)
    """
    inputs = []
    for t_launch_s in PLANE_LAUNCH_S:
        inputs.append(dict(
            n_slots=params.N_SLOTS, n_runs=n_runs,
            sim_duration_s=params.SIM_DURATION_YEARS * YEAR_S, dt_s=params.DT_DAYS * DAY_S,
            initial_launch_s=np.full(params.N_SLOTS + strategy["n_spares"], t_launch_s),
            order_to_launch_s=order_to_launch_days * DAY_S,
            commissioning_s=params.COMMISSIONING_DAYS * DAY_S,
            design_life_s=params.DESIGN_LIFE_YEARS * YEAR_S,
            launch_success_prob=params.LAUNCH_SUCCESS_PROB,
            sats_per_launch=params.SATS_PER_LAUNCH, reliability=RELIABILITY, **strategy))
    return simulate_constellation(simulate_plane, inputs, seed)


def run_cadence(k, interval_days, seed, n_runs):
    """Strategy d) on every plane: k satellites every interval_days.

    Plane p is deployed in one launch at PLANE_LAUNCH_S[p] carrying N_SLOTS + k
    satellites: the k cadence satellites of that date fly with the deployment and
    act at once as spares (no operator would leave the plane without spares in
    the infant-mortality period; consistent with b). The cadence dates of plane p
    are p * T/2 + j * T (plane 2 shifted by T/2, ASSUMPTION in params.py); the
    first cadence launch is the first of those dates after the deployment.
    """
    interval_s = interval_days * DAY_S
    inputs = []
    for p, t_launch_s in enumerate(PLANE_LAUNCH_S):
        phase_s = p * interval_s / 2.0
        j_first = int(np.floor((t_launch_s - phase_s) / interval_s)) + 1
        inputs.append(dict(
            n_slots=params.N_SLOTS, cadence_sats=k, cadence_interval_s=interval_s,
            cadence_start_s=phase_s + j_first * interval_s, n_runs=n_runs,
            sim_duration_s=params.SIM_DURATION_YEARS * YEAR_S, dt_s=params.DT_DAYS * DAY_S,
            initial_launch_s=np.full(params.N_SLOTS + k, t_launch_s),
            commissioning_s=params.COMMISSIONING_DAYS * DAY_S,
            design_life_s=params.DESIGN_LIFE_YEARS * YEAR_S,
            launch_success_prob=params.LAUNCH_SUCCESS_PROB, reliability=RELIABILITY))
    return simulate_constellation(simulate_fixed_cadence, inputs, seed)


def availability(out, req):
    """Availability of each run for one revisit requirement ("max" or "mean") [-]."""
    spec = REVISIT_REQS[req]
    revisit_s = constellation_revisit_series(out["n_active"], coverage_data()[spec["table"]])
    return constellation_availability(revisit_s, spec["limit_s"], out["t_s"], DEPLOY_END_S)


def summary(out):
    """Key numbers of one constellation simulation.

    Output: dict with, for each requirement, mean and 5/95 percentiles of the
    availability; satellites launched per run; time to full operation [days]
    (first time the max-revisit requirement is met) and runs never reaching it.
    """
    result = {}
    for req in REVISIT_REQS:
        avail = availability(out, req)
        result[req] = (avail.mean(), *np.percentile(avail, [5, 95]))
    launched = out["n_launched"].sum(axis=1)
    revisit_s = constellation_revisit_series(out["n_active"], coverage_data()["table_planes_max_s"])
    t_full_days = time_to_requirement(revisit_s, REVISIT_REQS["max"]["limit_s"], out["t_s"]) / DAY_S
    reached = ~np.isnan(t_full_days)
    result["launched"] = (launched.mean(), *np.percentile(launched, [5, 95]))
    result["t_full_days"] = (t_full_days[reached].mean(), *np.percentile(t_full_days[reached], [5, 95]))
    result["never_full"] = int(np.sum(~reached))
    return result


def print_table(results):
    """Print the comparison table of the strategies (both revisit requirements)."""
    print(f"{'Strategy':46s} {'Avail. max<=12h (5-95 %)':26s} {'Avail. mean<=6h (5-95 %)':26s} "
          f"{'Launched/run':14s} {'Full op. [d] (5-95 %)':22s} Never")
    for label, out in results.items():
        s = summary(out)
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
CADENCE_CACHE = {}  # {(k, seed, T_days): {req: fraction of runs meeting the criterion}}


def criterion_fractions(out):
    """Fraction of runs with availability >= AVAIL_TARGET, for each revisit requirement."""
    return {req: np.mean(availability(out, req) >= params.AVAIL_TARGET) for req in REVISIT_REQS}


def cadence_metrics(k, seed, t_days):
    """Criterion fractions of strategy d) for (k, seed, T), N_RUNS_SWEEP runs (cached)."""
    key = (k, seed, int(t_days))
    if key not in CADENCE_CACHE:
        CADENCE_CACHE[key] = criterion_fractions(run_cadence(k, int(t_days), seed,
                                                             params.N_RUNS_SWEEP))
    return CADENCE_CACHE[key]


def cadence_search(k, seed, req, t_ref_days=None):
    """Largest interval T [days] per plane meeting the criterion for one revisit requirement.

    Coarse sweep + bisection (search.py); with t_ref_days (other seeds) a bisection
    between t_ref -/+ 2 grid steps after checking the bracket, else a full sweep.
    """
    def evaluate(t_days):
        return cadence_metrics(k, seed, t_days)[req]

    start, stop, step = (np.array(params.CADENCE_GRID_DAYS_PER_SAT) * k).astype(int)
    if t_ref_days is not None:
        lo, hi = t_ref_days - 2 * step, t_ref_days + 2 * step
        if lo > 0 and evaluate(lo) >= params.AVAIL_CONFIDENCE and evaluate(hi) < params.AVAIL_CONFIDENCE:
            return {"t_star_days": bisect_interval(evaluate, lo, hi, params.AVAIL_CONFIDENCE)}
    return search_max_interval(evaluate, np.arange(start, stop + 1, step), params.AVAIL_CONFIDENCE)


def mc_launch_rate_per_year(out):
    """Mean satellites launched per year by the constellation after deployment (MC) [1/yr]."""
    after = out["t_s"] >= DEPLOY_END_S
    years = (out["t_s"][-1] - DEPLOY_END_S) / YEAR_S
    return out["n_launched"][:, after].sum(axis=1).mean() / years


def steady_rate_pre_order(n_spares):
    """Steady-state launch rate of c) / b+c), one reorder per loss [satellites/yr].

    Each of the N + S units of a plane renews independently with mean cycle
    E[cycle_c] = ORDER + integral_0^(DL - ORDER - C) R dt; constellation rate
    P (N + S) / E[cycle_c].
    """
    order_s = params.ORDER_TO_LAUNCH_DAYS * DAY_S
    a_s = params.DESIGN_LIFE_YEARS * YEAR_S - order_s - params.COMMISSIONING_DAYS * DAY_S
    integral_s = YEAR_S * quad(lambda ty: reliability_analytic(ty * YEAR_S, **RELIABILITY),
                               0.0, a_s / YEAR_S, limit=200)[0]
    return params.N_PLANES * (params.N_SLOTS + n_spares) / (order_s + integral_s) * YEAR_S


def failure_breakdown(out, req="max"):
    """Runs failing availability >= AVAIL_TARGET, by number of unplanned losses.

    Output: (fraction of runs failing [-], {number of losses: fraction of ALL runs [-]})
    """
    failing = availability(out, req) < params.AVAIL_TARGET
    losses = out["n_failures"] + out["n_launch_failures"]
    counts = {int(n): np.sum(failing & (losses == n)) / len(failing)
              for n in np.unique(losses[failing])}
    return failing.mean(), counts


def cost_study():
    """Search of T* for d) and comparison with c) and b+c), for both revisit requirements."""
    ref_rate = params.N_PLANES * reference_launch_rate(
        params.N_SLOTS, RELIABILITY, params.DESIGN_LIFE_YEARS * YEAR_S,
        params.COMMISSIONING_DAYS * DAY_S) * YEAR_S
    print(f"Reference rate P N / E[up] = {ref_rate:.3f} satellites/yr "
          "(every satellite gives its whole useful life, no gaps)")
    print(f"Mission criterion: {params.AVAIL_CONFIDENCE:.0%} of the runs with availability "
          f">= {params.AVAIL_TARGET}")

    d_best = {req: {} for req in REVISIT_REQS}
    for req, spec in REVISIT_REQS.items():
        print(f"Requirement '{spec['text']}':")
        for k in params.CADENCE_SATS_OPTIONS:
            result = cadence_search(k, params.SEED, req)
            t_star = result["t_star_days"]
            if np.isnan(t_star):
                print(f"  d) k = {k}: no interval of the grid meets the criterion")
                continue
            if not result.get("bracketed", True):
                print(f"  d) k = {k}: every interval of the grid meets it (T* >= {t_star} d)")
            checks = [cadence_search(k, seed, req, t_ref_days=t_star)["t_star_days"]
                      for seed in params.SEEDS_CHECK]
            d_best[req][k] = t_star
            print(f"  d) k = {k}: T* = {t_star} d (seeds {params.SEEDS_CHECK}: {checks}) -> "
                  f"{params.N_PLANES * k * 365.25 / t_star:.3f} satellites/yr; "
                  f"non monotone: {result.get('non_monotone')}")

    d_curves = {}
    for k in params.CADENCE_SATS_OPTIONS:
        t_grid = np.array(sorted(t for (kk, seed, t) in CADENCE_CACHE
                                 if kk == k and seed == params.SEED))
        d_curves[k] = {"rate": params.N_PLANES * k * 365.25 / t_grid,
                       **{req: np.array([CADENCE_CACHE[(k, params.SEED, t)][req] for t in t_grid])
                          for req in REVISIT_REQS}}

    configs = []
    for req in REVISIT_REQS:
        for k, t_star in d_best[req].items():
            if ("d", k, t_star) not in configs:
                configs.append(("d", k, t_star))
    for n_spares in range(0, 4):
        configs.append(("c" if n_spares == 0 else "b+c", n_spares, None))

    print(f"{'Configuration':26s} {'sats/yr steady':>14s} {'sats/yr MC':>10s} "
          f"{'avail max<=12h (5-95 %) | runs ok':>36s} {'avail mean<=6h (5-95 %) | runs ok':>36s}")
    rows, outputs = [], {}
    for family, value, t_star in configs:
        if family == "d":
            out = run_cadence(value, t_star, params.SEED, params.N_RUNS)
            rate = params.N_PLANES * value * 365.25 / t_star
            label = f"d) k={value}, T={t_star} d"
        else:
            strategy = dict(n_spares=value, reorder_level=value, pre_order_retirement=True)
            out = run_strategy(strategy, params.ORDER_TO_LAUNCH_DAYS)
            rate = steady_rate_pre_order(value)
            label = "c) early ordering" if value == 0 else f"b+c) S={value} per plane"
        outputs[label] = out
        fractions = criterion_fractions(out)
        rows.append({"family": family, "label": label, "rate": rate, **fractions})
        text = []
        for req in REVISIT_REQS:
            avail = availability(out, req)
            p5, p95 = np.percentile(avail, [5, 95])
            text.append(f"{avail.mean():.4f} ({p5:.3f}-{p95:.3f}) | {fractions[req]:.3f}")
        print(f"{label:26s} {rate:14.3f} {mc_launch_rate_per_year(out):10.3f}   "
              f"{text[0]:>34s}   {text[1]:>34s}")

    for req, spec in REVISIT_REQS.items():
        print(f"Cheapest configuration per family, requirement '{spec['text']}':")
        for family in ["d", "c", "b+c"]:
            ok = [r for r in rows if r["family"] == family and r[req] >= params.AVAIL_CONFIDENCE]
            if ok:
                best = min(ok, key=lambda r: r["rate"])
                print(f"  {family:4s}: {best['label']} -> {best['rate']:.3f} satellites/yr")
            else:
                print(f"  {family:4s}: none of the configurations tested meets it")

    failing, counts = failure_breakdown(outputs["c) early ordering"])
    print(f"c) alone, requirement 'max': {failing:.3f} of the runs have availability < "
          f"{params.AVAIL_TARGET}; by number of unplanned losses in the run: "
          + ", ".join(f"{n}: {f:.3f}" for n, f in counts.items()))

    plot_cost_vs_availability(d_curves, d_best, rows, ref_rate,
                              os.path.join(FIG_DIR, "cost_vs_availability.png"))


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
                text = "inf" if not np.isfinite(table_h[i, j]) else f"{table_h[i, j]:.1f}"
                ok = np.isfinite(table_h[i, j]) and table_h[i, j] <= limit_h
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="w" if not ok else "k", fontweight="bold" if ok else "normal")
        ax.set_xlabel("n_2 operational in plane 2 [-]")
        ax.set_ylabel("n_1 operational in plane 1 [-]")
        ax.set_title(f"{title}; bold = meets <= {limit_h:.0f} h")
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"Revisit at {params.REVISIT_LAT_DEG:.0f} deg N vs operational satellites per plane "
                 "(R1 rephasing, quasi-stationary)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
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
    cost_study()
    print(f"Figures saved in {FIG_DIR}/")


if __name__ == "__main__":
    main()
