"""Validation of the in-orbit spares strategy (lifecycle.simulate_in_orbit_spares).

Why an analytical solution exists (insensitivity)
------------------------------------------------
Test case: hot spares (H1, H2: same R(t) and design life from launch, whether
active or dormant), one reorder per lost satellite (REORDER_LEVEL = N_SPARES),
commissioning 0. Look at the pool of N + S units. When a satellite is lost the
inventory position drops below S and exactly one replacement is ordered at that
moment; it arrives after the lead time and is usable at once. So every unit
alternates "life" (a satellite, active or spare) and "lead time" (replacement on
its way), and what happens to one unit never depends on the others: its life
does not depend on its role (H1, H2) and its replacement is ordered as soon as
it is lost, whatever the other units do. The units are independent alternating
renewal processes. By the renewal-reward theorem each unit is usable in the
long run with probability p = E[life] / (E[life] + lead time), which depends
only on the means (not on the lead-time distribution). Independence then gives
units down ~ Binomial(N + S, 1 - p), and operational satellites = min(N, usable).
With exponential lives and exponential lead times the same system is the
birth-death Markov chain of analytic.birth_death_stationary.

Tolerances: each run is independent, so the standard error of a time average is
std(run averages) / sqrt(n_runs); we accept 4 standard errors. Discretization:
a loss is detected with a delay in [0, dt), which lengthens each life on the
grid; the analytical value is computed with delay 0 and dt and the MC value
must lie in that band (+- 4 se).
"""

import numpy as np
from scipy.stats import binom

import params
from analytic import (active_distribution_binomial, active_distribution_from_down,
                      birth_death_stationary, slot_availability_renewal_reward)
from lifecycle import simulate_in_orbit_spares, simulate_launch_on_failure

DAY_S = params.SECONDS_PER_DAY
YEAR_S = params.SECONDS_PER_YEAR
DT_S = params.DT_DAYS * DAY_S
ORDER_S = params.ORDER_TO_LAUNCH_DAYS * DAY_S
N_SLOTS, N_SPARES, MIN_ACTIVE = 6, 1, 6
N_SIGMA = 4.0

WEIBULL = {"model": "weibull", "beta": params.WEIBULL_BETA,
           "theta_s": params.WEIBULL_THETA_YEARS * YEAR_S}
# Test-only value: MTTF of 4 yr gives p ~ 0.8, so P(active >= 6) ~ 0.58 is not trivial
EXPONENTIAL = {"model": "exponential", "mttf_s": 4.0 * YEAR_S}


def run_spares(n_runs, duration_s, design_life_s, reliability, seed=1):
    """Test configuration: one reorder per loss, commissioning 0, launch success 1."""
    return simulate_in_orbit_spares(
        n_slots=N_SLOTS, n_spares=N_SPARES, reorder_level=N_SPARES, n_runs=n_runs, seed=seed,
        sim_duration_s=duration_s, dt_s=DT_S, order_to_launch_s=ORDER_S, commissioning_s=0.0,
        design_life_s=design_life_s, launch_success_prob=1.0, sats_per_launch=1,
        reliability=reliability)


def grouped(p_active):
    """Group P(n_active = j) into bins {6}, {5}, {4}, {<= 3} (rare states merged)."""
    return np.array([p_active[6], p_active[5], p_active[4], np.sum(p_active[:4])])


def check_against_analytic(out, warmup_s, p_active_lo, p_active_hi):
    """Compare MC time averages after warm-up with an analytical band.

    Inputs:
        out                      : simulator output
        warmup_s                 : time discarded at the start [s]
        p_active_lo, p_active_hi : analytical P(n_active = j) with delay 0 and dt
    """
    keep = out["t_s"] >= warmup_s
    n_active = out["n_active"][:, keep]
    assert out["n_spares_in_orbit"].max() <= N_SPARES

    # Plane availability P(n_active >= MIN_ACTIVE)
    avail_runs = np.mean(n_active >= MIN_ACTIVE, axis=1)
    a_mc = avail_runs.mean()
    se = avail_runs.std(ddof=1) / np.sqrt(len(avail_runs))
    a_lo, a_hi = sorted([p_active_lo[MIN_ACTIVE:].sum(), p_active_hi[MIN_ACTIVE:].sum()])
    assert a_lo - N_SIGMA * se < a_mc < a_hi + N_SIGMA * se, (
        f"availability MC {a_mc:.4f} vs [{a_lo:.4f}, {a_hi:.4f}], 4 se = {N_SIGMA * se:.4f}")

    # Distribution of n_active
    bins_runs = np.array([grouped(np.bincount(row, minlength=N_SLOTS + 1) / len(row))
                          for row in n_active])
    bins_mc = bins_runs.mean(axis=0)
    bins_se = bins_runs.std(axis=0, ddof=1) / np.sqrt(len(bins_runs))
    lo = np.minimum(grouped(p_active_lo), grouped(p_active_hi))
    hi = np.maximum(grouped(p_active_lo), grouped(p_active_hi))
    assert np.all(lo - N_SIGMA * bins_se < bins_mc), f"MC {bins_mc} vs low {lo}"
    assert np.all(bins_mc < hi + N_SIGMA * bins_se), f"MC {bins_mc} vs high {hi}"


def test_markov_chain_equals_binomial():
    """Birth-death chain solved with numpy = closed-form Binomial(N + S, 1 - p)."""
    for n_units, mttf_years, lead_years in [(7, 4.0, 1.0), (3, 1.0, 2.0), (12, 10.0, 0.5)]:
        fail_rate = 1.0 / (mttf_years * YEAR_S)
        repair_rate = 1.0 / (lead_years * YEAR_S)
        pi = birth_death_stationary(n_units, fail_rate, repair_rate)
        p_up = mttf_years / (mttf_years + lead_years)
        expected = binom.pmf(np.arange(n_units + 1), n_units, 1.0 - p_up)
        np.testing.assert_allclose(pi, expected, rtol=1e-10, atol=1e-14)


def test_spares_exponential_vs_markov_chain():
    """Exponential lives, DETERMINISTIC lead time, no design life, vs the chain.

    The chain assumes an exponential lead time; by insensitivity (see module
    docstring) the stationary distribution only needs its mean, so mu = 1/lead time.
    The time to stationarity is a few MTTF: warm-up 20 yr.
    """
    out = run_spares(n_runs=200, duration_s=100.0 * YEAR_S, design_life_s=np.inf,
                     reliability=EXPONENTIAL)
    mttf_s = EXPONENTIAL["mttf_s"]
    bands = []
    for delay_s in [0.0, DT_S]:
        pi = birth_death_stationary(N_SLOTS + N_SPARES, 1.0 / (mttf_s + delay_s), 1.0 / ORDER_S)
        bands.append(active_distribution_from_down(pi, N_SLOTS))
    check_against_analytic(out, 20.0 * YEAR_S, bands[0], bands[1])


def test_spares_weibull_design_life_vs_binomial():
    """Fitted Weibull with design life, p from renewal-reward (commissioning 0).

    All units start together at t = 0 and the cycle (design life + lead time) is
    almost deterministic, so the units stay in phase until early failures (~6 %
    per cycle) shift them. Measured convergence: the plane availability is still
    ~0.76 between 0 and 90 yr and settles to the stationary ~0.654 after ~180 yr.
    Warm-up 200 yr, average over 200-500 yr.
    """
    design_life_s = params.DESIGN_LIFE_YEARS * YEAR_S
    out = run_spares(n_runs=60, duration_s=500.0 * YEAR_S, design_life_s=design_life_s,
                     reliability=WEIBULL)
    bands = []
    for delay_s in [0.0, DT_S]:
        p_up = slot_availability_renewal_reward(WEIBULL, design_life_s, ORDER_S, 0.0, delay_s)
        bands.append(active_distribution_binomial(N_SLOTS, N_SPARES, p_up))
    check_against_analytic(out, 200.0 * YEAR_S, bands[0], bands[1])


def test_zero_spares_equals_launch_on_failure():
    """N_SPARES = REORDER_LEVEL = 0 gives identical results to launch on failure.

    Nominal timing, with launch failures switched on to also cover that branch.
    """
    common = dict(n_slots=N_SLOTS, n_runs=200, seed=7, sim_duration_s=15.0 * YEAR_S, dt_s=DT_S,
                  order_to_launch_s=ORDER_S, commissioning_s=params.COMMISSIONING_DAYS * DAY_S,
                  design_life_s=params.DESIGN_LIFE_YEARS * YEAR_S, launch_success_prob=0.9,
                  sats_per_launch=1, reliability=WEIBULL)
    a = simulate_launch_on_failure(**common)
    b = simulate_in_orbit_spares(n_spares=0, reorder_level=0, **common)
    for key in ["n_active", "n_launched", "n_failures", "n_retirements", "n_launch_failures"]:
        assert np.array_equal(a[key], b[key]), key
    assert b["n_launch_failures"].sum() > 0  # the launch-failure branch was exercised
