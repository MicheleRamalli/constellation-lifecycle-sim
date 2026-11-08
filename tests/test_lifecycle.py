"""Validation of lifecycle.py (launch on failure, one plane) against analytical cases.

Discretization: an end of life at continuous time t_end is detected at the first
grid step >= t_end, i.e. with a delay between 0 and dt. The tolerances below
include this delay as a deterministic band [0, dt], plus a 4-sigma statistical
tolerance (false-failure probability ~6e-5, normal approximation).
"""

import numpy as np

import params
from analytic import slot_availability_renewal_reward
from lifecycle import simulate_launch_on_failure
from reliability import reliability_analytic

DAY_S = params.SECONDS_PER_DAY
YEAR_S = params.SECONDS_PER_YEAR
DT_S = params.DT_DAYS * DAY_S
ORDER_S = params.ORDER_TO_LAUNCH_DAYS * DAY_S
COMM_S = params.COMMISSIONING_DAYS * DAY_S
DESIGN_LIFE_S = params.DESIGN_LIFE_YEARS * YEAR_S
N_SIGMA = 4.0

WEIBULL = {"model": "weibull", "beta": params.WEIBULL_BETA,
           "theta_s": params.WEIBULL_THETA_YEARS * YEAR_S}
# Test-only value: a short MTTF gives many failure/replacement cycles
EXPONENTIAL_SHORT = {"model": "exponential", "mttf_s": 1.0 * YEAR_S}


def run(n_slots, n_runs, duration_s, order_s, comm_s, design_life_s, reliability, seed=1):
    """Shortcut to the simulator with launch success 1 and one satellite per launch."""
    return simulate_launch_on_failure(
        n_slots=n_slots, n_runs=n_runs, seed=seed, sim_duration_s=duration_s, dt_s=DT_S,
        order_to_launch_s=order_s, commissioning_s=comm_s, design_life_s=design_life_s,
        launch_success_prob=1.0, sats_per_launch=1, reliability=reliability)


def test_active_fraction_before_first_replacement():
    """Before the first replacement can arrive, the active fraction is R(t).

    No replacement can be launched before ORDER_TO_LAUNCH, so for
    COMMISSIONING <= t < ORDER_TO_LAUNCH a satellite is active iff it has not
    failed yet. Here the activity is evaluated on the grid against the exact
    failure time, so there is no discretization delay.
    """
    n_slots, n_runs = 100, 5000          # 5e5 satellites
    duration_s = ORDER_S - DT_S          # stop before the first possible replacement
    out = run(n_slots, n_runs, duration_s, ORDER_S, COMM_S, DESIGN_LIFE_S, WEIBULL)
    n_sats = n_slots * n_runs

    assert out["n_launched"][:, 1:].sum() == 0               # no replacement launched
    k_before_comm = int(COMM_S // DT_S) - 1
    assert out["n_active"][:, k_before_comm].sum() == 0      # still in commissioning

    for t_days in [60, 120, 240, 360]:
        k = int(round(t_days * DAY_S / DT_S))
        fraction = out["n_active"][:, k].sum() / n_sats
        r = reliability_analytic(out["t_s"][k], **WEIBULL)
        sigma = np.sqrt(r * (1.0 - r) / n_sats)
        assert abs(fraction - r) < N_SIGMA * sigma, (
            f"t = {t_days} d: MC {fraction:.5f} vs R(t) {r:.5f}")


def check_availability(reliability, design_life_s, duration_s, n_runs, n_cycles):
    """Compare the MC slot availability with the renewal-reward value.

    One slot per run. Each run uses exactly its first n_cycles complete cycles
    (launch to next launch). n_cycles is fixed in advance: cutting each run at
    its last launch before the horizon would select cycles that fit in a fixed
    time, which are shorter than average (inspection paradox) and bias A.
    A_mc = sum(up time) / sum(cycle time) over all runs (ratio estimator).
    Its standard error comes from the delta method:
    se^2 = sum((U_r - A L_r)^2) / (n (n - 1) mean(L)^2).
    Discretization: on the grid both the up time and the cycle length grow by
    a delay in [0, dt), so A_mc must lie in [A(0) .. A(dt)] (+- 4 se).
    """
    out = run(1, n_runs, duration_s, ORDER_S, COMM_S, design_life_s, reliability)
    up = np.zeros(n_runs)
    length = np.zeros(n_runs)
    for r in range(n_runs):
        k_launches = np.nonzero(out["n_launched"][r])[0]
        assert len(k_launches) > n_cycles, "horizon too short for n_cycles"
        k_end = k_launches[n_cycles]  # launch that closes the n_cycles-th cycle
        up[r] = out["n_active"][r, :k_end].sum()
        length[r] = k_end
    a_mc = up.sum() / length.sum()
    se = np.sqrt(np.sum((up - a_mc * length) ** 2) / (n_runs * (n_runs - 1))) / length.mean()

    a0 = slot_availability_renewal_reward(reliability, design_life_s, ORDER_S, COMM_S, 0.0)
    a1 = slot_availability_renewal_reward(reliability, design_life_s, ORDER_S, COMM_S, DT_S)
    a_lo, a_hi = min(a0, a1), max(a0, a1)
    assert a_lo - N_SIGMA * se < a_mc < a_hi + N_SIGMA * se, (
        f"MC {a_mc:.5f} vs renewal-reward [{a_lo:.5f}, {a_hi:.5f}], 4 se = {N_SIGMA * se:.5f}")


def test_availability_exponential():
    """Short MTTF (1 yr), no retirement.

    Mean cycle ~2 yr, so 15 cycles take ~30 yr on average; the 60 yr horizon
    leaves a wide margin (the assert in check_availability guards it).
    """
    check_availability(EXPONENTIAL_SHORT, np.inf, 60.0 * YEAR_S, n_runs=500, n_cycles=15)


def test_availability_weibull_with_design_life():
    """Fitted Weibull with retirement at design life.

    A cycle is at most design life + order + dt ~ 6.0 yr, so a 60 yr horizon
    always contains 9 complete cycles.
    """
    check_availability(WEIBULL, DESIGN_LIFE_S, 60.0 * YEAR_S, n_runs=500, n_cycles=9)


def test_failures_poisson_with_zero_lead_time():
    """Zero order and commissioning time, exponential, no retirement.

    The failures of each slot are a Poisson process: E[N(T)] = T / MTTF.
    Each failure is detected with a delay in [0, dt), so the mean cycle lies in
    [MTTF, MTTF + dt] and E[N] in [T / (MTTF + dt), T / MTTF] per slot.
    Statistical tolerance: Poisson count, sigma = sqrt(E[N]).
    """
    n_slots, n_runs = 6, 1000
    duration_s = 15.0 * YEAR_S
    out = run(n_slots, n_runs, duration_s, 0.0, 0.0, np.inf, EXPONENTIAL_SHORT)
    t_s = out["t_s"][-1]
    mttf_s = EXPONENTIAL_SHORT["mttf_s"]

    n_mc = out["n_failures"].sum()
    n_hi = n_slots * n_runs * t_s / mttf_s
    n_lo = n_slots * n_runs * t_s / (mttf_s + DT_S)
    tol = N_SIGMA * np.sqrt(n_hi)
    assert n_lo - tol < n_mc < n_hi + tol, (
        f"MC {n_mc} vs [{n_lo:.0f}, {n_hi:.0f}], 4 sigma = {tol:.0f}")
    assert out["n_retirements"].sum() == 0
