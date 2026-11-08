"""Run the nominal case and save the figures into figures/.

Figures:
- active_vs_time.png              : operational satellites vs time (mean, 5-95 % band)
- availability_vs_order_time.png  : availability vs ORDER_TO_LAUNCH_DAYS sweep
Both figures compare strategy a) launch on failure and b) in-orbit spares.
"""

import os

import matplotlib
matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np

import params
from analytic import active_distribution_binomial, slot_availability_renewal_reward
from lifecycle import plane_availability, simulate_in_orbit_spares, simulate_launch_on_failure

FIG_DIR = "figures"
DAY_S = params.SECONDS_PER_DAY
YEAR_S = params.SECONDS_PER_YEAR
RELIABILITY = {"model": "weibull", "beta": params.WEIBULL_BETA,
               "theta_s": params.WEIBULL_THETA_YEARS * YEAR_S}
ORDER_SWEEP_DAYS = np.arange(90.0, 731.0, 80.0)  # 90, 170, ..., 730 days


def common_inputs(order_to_launch_days):
    """Simulator inputs shared by both strategies, from params.py."""
    return dict(
        n_slots=params.N_SLOTS, n_runs=params.N_RUNS, seed=params.SEED,
        sim_duration_s=params.SIM_DURATION_YEARS * YEAR_S, dt_s=params.DT_DAYS * DAY_S,
        order_to_launch_s=order_to_launch_days * DAY_S,
        commissioning_s=params.COMMISSIONING_DAYS * DAY_S,
        design_life_s=params.DESIGN_LIFE_YEARS * YEAR_S,
        launch_success_prob=params.LAUNCH_SUCCESS_PROB,
        sats_per_launch=params.SATS_PER_LAUNCH, reliability=RELIABILITY)


def run_launch_on_failure(order_to_launch_days):
    """Strategy a): launch on failure, with a given order-to-launch time [days]."""
    return simulate_launch_on_failure(**common_inputs(order_to_launch_days))


def run_in_orbit_spares(order_to_launch_days):
    """Strategy b): in-orbit spares, with a given order-to-launch time [days]."""
    return simulate_in_orbit_spares(n_spares=params.N_SPARES, reorder_level=params.REORDER_LEVEL,
                                    **common_inputs(order_to_launch_days))


def slot_fraction_analytic(n_spares, order_to_launch_days):
    """Steady-state mean fraction of occupied slots (t -> inf).

    Each unit of the pool is usable with the renewal-reward probability p (life
    from launch, usable after commissioning). With n_spares = 0 this is p itself;
    with spares and one reorder per loss the units are independent and
    n_active = min(N, Binomial(N + S, p)) (see tests/test_spares.py).
    """
    p_up = slot_availability_renewal_reward(
        RELIABILITY, params.DESIGN_LIFE_YEARS * YEAR_S, order_to_launch_days * DAY_S,
        params.COMMISSIONING_DAYS * DAY_S)
    p_active = active_distribution_binomial(params.N_SLOTS, n_spares, p_up)
    return np.sum(np.arange(params.N_SLOTS + 1) * p_active) / params.N_SLOTS


def plot_active_vs_time(results, path):
    """Mean and 5-95 % band of the operational satellites vs time.

    Inputs:
        results : dict {label: simulator output}
        path    : output file path
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, out in results.items():
        t_years = out["t_s"] / YEAR_S
        mean = out["n_active"].mean(axis=0)
        p5, p95 = np.percentile(out["n_active"], [5, 95], axis=0)
        avail = plane_availability(out["n_active"], params.MIN_ACTIVE).mean()
        line, = ax.plot(t_years, mean, label=f"{label}: mean (plane availability {avail:.3f})")
        ax.fill_between(t_years, p5, p95, color=line.get_color(), alpha=0.2,
                        label=f"{label}: 5-95 % band")
    ax.axhline(params.MIN_ACTIVE, color="k", linestyle="--", label=f"MIN_ACTIVE = {params.MIN_ACTIVE}")
    ax.set_xlabel("Time since first launch [years]")
    ax.set_ylabel("Operational satellites [-]")
    ax.set_title(f"One plane, {params.N_SLOTS} slots, {params.N_RUNS} Monte Carlo runs")
    ax.set_ylim(-0.2, params.N_SLOTS + 0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sweep_order_time(run_strategy, n_spares):
    """Availability of one strategy for each value of ORDER_SWEEP_DAYS.

    Inputs:
        run_strategy : run_launch_on_failure or run_in_orbit_spares
        n_spares     : spares of the strategy (0 for launch on failure) [-]
    Output: dict of arrays over the sweep
        "slot_mc"      : mean fraction of occupied slots over the horizon (MC) [-]
        "slot_rr"      : steady-state fraction of occupied slots, analytic [-]
                         (NaN if the reorder policy is not one-for-one)
        "plane_mean"   : mean plane availability (>= MIN_ACTIVE) over the runs [-]
        "plane_p5/p95" : 5 and 95 percentiles of the plane availability [-]
    """
    keys = ["slot_mc", "slot_rr", "plane_mean", "plane_p5", "plane_p95"]
    sweep = {k: np.zeros(len(ORDER_SWEEP_DAYS)) for k in keys}
    for i, order_days in enumerate(ORDER_SWEEP_DAYS):
        out = run_strategy(order_days)
        avail = plane_availability(out["n_active"], params.MIN_ACTIVE)
        sweep["slot_mc"][i] = out["n_active"].mean() / params.N_SLOTS
        one_for_one = n_spares == 0 or params.REORDER_LEVEL == n_spares
        sweep["slot_rr"][i] = slot_fraction_analytic(n_spares, order_days) if one_for_one else np.nan
        sweep["plane_mean"][i] = avail.mean()
        sweep["plane_p5"][i], sweep["plane_p95"][i] = np.percentile(avail, [5, 95])
        print(f"  order-to-launch {order_days:5.0f} d: slot MC {sweep['slot_mc'][i]:.4f}, "
              f"slot RR {sweep['slot_rr'][i]:.4f}, plane {sweep['plane_mean'][i]:.4f}")
    return sweep


def plot_availability_vs_order_time(sweeps, path):
    """Two panels: (a) slot availability MC vs analytic, (b) plane availability MC.

    Inputs:
        sweeps : dict {label: output of the sweep function}
        path   : output file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, sw in sweeps.items():
        line, = ax1.plot(ORDER_SWEEP_DAYS, sw["slot_mc"], "o-", label=f"{label}: MC, {params.SIM_DURATION_YEARS:.0f} yr")
        ax1.plot(ORDER_SWEEP_DAYS, sw["slot_rr"], "--", color=line.get_color(),
                 label=f"{label}: analytic steady state (t -> inf)")
        ax2.plot(ORDER_SWEEP_DAYS, sw["plane_mean"], "o-", color=line.get_color(), label=f"{label}: MC mean")
        ax2.fill_between(ORDER_SWEEP_DAYS, sw["plane_p5"], sw["plane_p95"],
                         color=line.get_color(), alpha=0.2, label=f"{label}: MC 5-95 %")
    ax1.set_title("(a) Mean fraction of occupied slots")
    ax2.set_title(f"(b) Plane availability (>= {params.MIN_ACTIVE} operational)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Order-to-launch time [days]")
        ax.set_ylabel("Availability [-]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    label_a = "a) Launch on failure"
    label_b = f"b) {params.N_SPARES} in-orbit spare(s)"

    print("Nominal case")
    nominal = {label_a: run_launch_on_failure(params.ORDER_TO_LAUNCH_DAYS),
               label_b: run_in_orbit_spares(params.ORDER_TO_LAUNCH_DAYS)}
    for label, out in nominal.items():
        avail = plane_availability(out["n_active"], params.MIN_ACTIVE)
        print(f"  {label}: plane availability {avail.mean():.4f} "
              f"(5-95 %: {np.percentile(avail, 5):.4f}-{np.percentile(avail, 95):.4f}), "
              f"satellites launched per run {out['n_launched'].sum(axis=1).mean():.2f}")
    plot_active_vs_time(nominal, os.path.join(FIG_DIR, "active_vs_time.png"))

    print("Sweep of the order-to-launch time")
    sweeps = {label_a: sweep_order_time(run_launch_on_failure, 0),
              label_b: sweep_order_time(run_in_orbit_spares, params.N_SPARES)}
    plot_availability_vs_order_time(sweeps, os.path.join(FIG_DIR, "availability_vs_order_time.png"))
    print(f"Figures saved in {FIG_DIR}/")


if __name__ == "__main__":
    main()
