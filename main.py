"""Run the nominal case for the four strategies and save the figures into figures/.

Strategies (lifecycle.simulate_plane):
    a)   launch on failure
    b)   in-orbit spares, (s, S) policy
    c)   early ordering of the retirement replacement
    b+c) spares and early ordering

Figures:
- active_vs_time.png              : operational satellites vs time, one panel per strategy
- availability_vs_order_time.png  : availability vs ORDER_TO_LAUNCH_DAYS sweep

Availabilities are averaged from the end of the nominal deployment
(last batch + commissioning) to the end of the horizon.
"""

import os

import matplotlib
matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np

import params
from analytic import slot_availability_pre_order, slot_availability_renewal_reward
from lifecycle import (deployment_launch_times, plane_availability, simulate_plane,
                       time_to_full_operation)

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


def run_strategy(strategy, order_to_launch_days):
    """Nominal simulation of one strategy with a given order-to-launch time.

    Inputs:
        strategy             : dict with n_spares, reorder_level, pre_order_retirement
        order_to_launch_days : order-to-launch time [days]
    Output: simulator output (see lifecycle.simulate_plane)
    """
    n_units = params.N_SLOTS + strategy["n_spares"]
    initial_launch_s = deployment_launch_times(n_units, params.DEPLOY_BATCHES,
                                               params.DEPLOY_INTERVAL_DAYS * DAY_S)
    return simulate_plane(
        n_slots=params.N_SLOTS, n_runs=params.N_RUNS, seed=params.SEED,
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
        out = run_strategy(strategy, order_days)
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
    print(f"Figures saved in {FIG_DIR}/")


if __name__ == "__main__":
    main()
