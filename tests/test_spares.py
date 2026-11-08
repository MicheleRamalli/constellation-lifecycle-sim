"""Validation of lifecycle.simulate_plane: in-orbit spares (b), early ordering (c), b + c.

Why an analytical solution exists (insensitivity)
------------------------------------------------
Test case: hot spares (H1, H2: same R(t) and design life from launch, whether
active or dormant), one reorder per lost satellite (REORDER_LEVEL = N_SPARES).
Look at the pool of N + S units. When a satellite is lost the inventory
position drops below S and exactly one replacement is ordered at that moment
(with early ordering, the successor of a retiring satellite is ordered by the
satellite itself at a fixed age). So every unit goes through its own sequence of
satellites, and what happens to one unit never depends on the others: a
satellite's life does not depend on its role (H1, H2) and its replacement is
ordered by rules that only look at that satellite. The units are independent
renewal processes. By the renewal-reward theorem each unit is usable in the
long run with probability p = E[up] / E[cycle], which depends only on means
(not on the lead-time distribution). Independence then gives
units down ~ Binomial(N + S, 1 - p), and operational satellites = min(N, usable).
With exponential lives and exponential lead times the same system is the
birth-death Markov chain of analytic.birth_death_stationary.

Random initial phases
---------------------
With a Weibull life and a design life, the cycle of a unit is almost
deterministic (~94 % of the cycles last exactly c0). If all units start
together they stay in phase for hundreds of years. Instead, each unit gets its
first launch at an independent uniform time in [0, c0). A periodic process with
a uniform random phase is stationary, so after one cycle (when every unit has
been launched) each unit is already close to its stationary regime, and the
units are independent from the start. The cycles ended by early failures make
the period imperfect; measured residual effect: none visible within the
statistical error (+- 0.004) after the first cycle. Warm-up: one cycle.

Tolerances: each run is independent, so the standard error of a time average is
std(run averages) / sqrt(n_runs); we accept 4 standard errors. Discretization:
a loss is detected with a delay in [0, dt), and planned launches happen at the
next grid step; the analytical value is computed with delay 0 and dt and the
MC value must lie in that band (+- 4 se).
"""

import numpy as np
import pytest
from scipy.stats import binom

import params
from analytic import (active_distribution_binomial, active_distribution_from_down,
                      birth_death_stationary, slot_availability_pre_order,
                      slot_availability_renewal_reward)
from lifecycle import deployment_launch_times, simulate_launch_on_failure, simulate_plane

DAY_S = params.SECONDS_PER_DAY
YEAR_S = params.SECONDS_PER_YEAR
DT_S = params.DT_DAYS * DAY_S
ORDER_S = params.ORDER_TO_LAUNCH_DAYS * DAY_S
COMM_S = params.COMMISSIONING_DAYS * DAY_S
DESIGN_LIFE_S = params.DESIGN_LIFE_YEARS * YEAR_S
N_SLOTS, MIN_ACTIVE = 6, 6
N_SIGMA = 4.0
MIN_BIN_PROB = 1e-3  # bins rarer than this are not compared (se not estimable)

WEIBULL = {"model": "weibull", "beta": params.WEIBULL_BETA,
           "theta_s": params.WEIBULL_THETA_YEARS * YEAR_S}
# Test-only value: MTTF of 4 yr gives p ~ 0.8, so P(active >= 6) ~ 0.58 is not trivial
EXPONENTIAL = {"model": "exponential", "mttf_s": 4.0 * YEAR_S}
NO_FAILURES = {"model": "exponential", "mttf_s": np.inf}


def run_plane(n_spares, pre_order, n_runs, duration_s, comm_s, design_life_s, reliability,
              initial_launch_s, seed=1):
    """Test configuration: one reorder per loss, launch success 1."""
    return simulate_plane(
        n_slots=N_SLOTS, n_spares=n_spares, reorder_level=n_spares,
        pre_order_retirement=pre_order, n_runs=n_runs, seed=seed, sim_duration_s=duration_s,
        dt_s=DT_S, initial_launch_s=initial_launch_s, order_to_launch_s=ORDER_S,
        commissioning_s=comm_s, design_life_s=design_life_s, launch_success_prob=1.0,
        sats_per_launch=1, reliability=reliability)


def random_phases(n_runs, n_units, cycle_s, seed=123):
    """Independent uniform first launch times in [0, cycle), rounded down to the grid."""
    rng = np.random.default_rng(seed)
    return np.floor(rng.uniform(0.0, cycle_s, size=(n_runs, n_units)) / DT_S) * DT_S


def grouped(p_active):
    """Group P(n_active = j) into bins {6}, {5}, {<= 4}."""
    return np.array([p_active[6], p_active[5], np.sum(p_active[:5])])


def check_against_analytic(out, n_spares, warmup_s, p_active_lo, p_active_hi):
    """Compare MC time averages after warm-up with an analytical band.

    Inputs:
        out                      : simulator output
        n_spares                 : spares of the configuration [-]
        warmup_s                 : time discarded at the start [s]
        p_active_lo, p_active_hi : analytical P(n_active = j) with delay 0 and dt
    """
    keep = out["t_s"] >= warmup_s
    n_active = out["n_active"][:, keep]
    assert out["n_spares_in_orbit"].max() <= n_spares

    # Plane availability P(n_active >= MIN_ACTIVE)
    avail_runs = np.mean(n_active >= MIN_ACTIVE, axis=1)
    a_mc = avail_runs.mean()
    se = avail_runs.std(ddof=1) / np.sqrt(len(avail_runs))
    a_lo, a_hi = sorted([p_active_lo[MIN_ACTIVE:].sum(), p_active_hi[MIN_ACTIVE:].sum()])
    assert a_lo - N_SIGMA * se < a_mc < a_hi + N_SIGMA * se, (
        f"availability MC {a_mc:.4f} vs [{a_lo:.4f}, {a_hi:.4f}], 4 se = {N_SIGMA * se:.4f}")

    # Distribution of n_active (bins with analytical probability >= MIN_BIN_PROB)
    bins_runs = np.array([grouped(np.bincount(row, minlength=N_SLOTS + 1) / len(row))
                          for row in n_active])
    bins_mc = bins_runs.mean(axis=0)
    bins_se = bins_runs.std(axis=0, ddof=1) / np.sqrt(len(bins_runs))
    lo = np.minimum(grouped(p_active_lo), grouped(p_active_hi))
    hi = np.maximum(grouped(p_active_lo), grouped(p_active_hi))
    used = hi >= MIN_BIN_PROB
    assert np.all((lo - N_SIGMA * bins_se < bins_mc)[used]), f"MC {bins_mc} vs low {lo}"
    assert np.all((bins_mc < hi + N_SIGMA * bins_se)[used]), f"MC {bins_mc} vs high {hi}"


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
    """b) Exponential lives, DETERMINISTIC lead time, no design life, vs the chain.

    The chain assumes an exponential lead time; by insensitivity (see module
    docstring) the stationary distribution only needs its mean, so mu = 1/lead time.
    Exponential lives mix fast: all units start at t = 0, warm-up 20 yr.
    """
    n_spares = 1
    out = run_plane(n_spares, False, n_runs=200, duration_s=100.0 * YEAR_S, comm_s=0.0,
                    design_life_s=np.inf, reliability=EXPONENTIAL,
                    initial_launch_s=np.zeros(N_SLOTS + n_spares))
    mttf_s = EXPONENTIAL["mttf_s"]
    bands = []
    for delay_s in [0.0, DT_S]:
        pi = birth_death_stationary(N_SLOTS + n_spares, 1.0 / (mttf_s + delay_s), 1.0 / ORDER_S)
        bands.append(active_distribution_from_down(pi, N_SLOTS))
    check_against_analytic(out, n_spares, 20.0 * YEAR_S, bands[0], bands[1])


def test_spares_weibull_design_life_vs_binomial():
    """b) Fitted Weibull with design life, p from renewal-reward (commissioning 0).

    Random initial phases over one cycle c0 = DL + ORDER (see module docstring);
    warm-up one cycle, average up to 40 yr.
    """
    n_spares, n_runs = 1, 300
    cycle_s = DESIGN_LIFE_S + ORDER_S
    out = run_plane(n_spares, False, n_runs, 40.0 * YEAR_S, 0.0, DESIGN_LIFE_S, WEIBULL,
                    random_phases(n_runs, N_SLOTS + n_spares, cycle_s))
    bands = []
    for delay_s in [0.0, DT_S]:
        p_up = slot_availability_renewal_reward(WEIBULL, DESIGN_LIFE_S, ORDER_S, 0.0, delay_s)
        bands.append(active_distribution_binomial(N_SLOTS, n_spares, p_up))
    check_against_analytic(out, n_spares, cycle_s, bands[0], bands[1])


def test_zero_spares_equals_launch_on_failure():
    """a) N_SPARES = REORDER_LEVEL = 0, no early ordering: identical to launch on failure.

    Nominal timing and deployment in batches, with launch failures switched on
    to also cover that branch.
    """
    t_launch_s = deployment_launch_times(N_SLOTS, params.DEPLOY_BATCHES,
                                         params.DEPLOY_INTERVAL_DAYS * DAY_S)
    common = dict(n_slots=N_SLOTS, n_runs=200, seed=7, sim_duration_s=15.0 * YEAR_S, dt_s=DT_S,
                  initial_launch_s=t_launch_s, order_to_launch_s=ORDER_S, commissioning_s=COMM_S,
                  design_life_s=DESIGN_LIFE_S, launch_success_prob=0.9, sats_per_launch=1,
                  reliability=WEIBULL)
    a = simulate_launch_on_failure(**common)
    b = simulate_plane(n_spares=0, reorder_level=0, pre_order_retirement=False, **common)
    for key in ["n_active", "n_launched", "n_failures", "n_retirements", "n_launch_failures"]:
        assert np.array_equal(a[key], b[key]), key
    assert b["n_launch_failures"].sum() > 0  # the launch-failure branch was exercised


def test_early_ordering_without_failures_is_always_available():
    """c) No random failures: after deployment every slot is operational at every step.

    The successor is ordered at age DL - ORDER - COMM, launched at DL - COMM and
    commissioned exactly at DL, when the old satellite retires. On the grid the
    order happens at the first step >= launch + a, and the handover happens at
    the first step >= launch + DL, for both satellites: no gap, no overlap.
    The same case with launch on failure must lose availability at each retirement.
    """
    t_launch_s = deployment_launch_times(N_SLOTS, params.DEPLOY_BATCHES,
                                         params.DEPLOY_INTERVAL_DAYS * DAY_S)
    t_deployed_s = t_launch_s.max() + COMM_S
    for pre_order in [True, False]:
        out = run_plane(0, pre_order, n_runs=20, duration_s=30.0 * YEAR_S, comm_s=COMM_S,
                        design_life_s=DESIGN_LIFE_S, reliability=NO_FAILURES,
                        initial_launch_s=t_launch_s)
        after = out["t_s"] >= t_deployed_s
        assert out["n_failures"].sum() == 0
        if pre_order:
            assert np.all(out["n_active"][:, after] == N_SLOTS)
            assert out["n_retirements"].min() > 0  # retirements did happen
        else:
            assert out["n_active"][:, after].min() < N_SLOTS


@pytest.mark.parametrize("n_spares", [0, 1])
def test_early_ordering_weibull_vs_binomial(n_spares):
    """c) and b + c: Weibull + design life + commissioning, vs Binomial(N + S, A_c).

    A_c from analytic.slot_availability_pre_order (renewal-reward with early
    ordering, derivation in its docstring). Random initial phases over one
    cycle c0 = DL - COMM (the planned launch-to-launch time); warm-up one cycle.
    """
    n_runs = 300
    cycle_s = DESIGN_LIFE_S - COMM_S
    out = run_plane(n_spares, True, n_runs, 40.0 * YEAR_S, COMM_S, DESIGN_LIFE_S, WEIBULL,
                    random_phases(n_runs, N_SLOTS + n_spares, cycle_s))
    bands = []
    for delay_s in [0.0, DT_S]:
        p_up = slot_availability_pre_order(WEIBULL, DESIGN_LIFE_S, ORDER_S, COMM_S, delay_s)
        bands.append(active_distribution_binomial(N_SLOTS, n_spares, p_up))
    check_against_analytic(out, n_spares, cycle_s, bands[0], bands[1])
