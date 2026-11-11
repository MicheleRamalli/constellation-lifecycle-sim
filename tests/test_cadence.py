"""Validation of lifecycle.simulate_fixed_cadence (strategy d) against exact results.

With a fixed cadence no satellite is ever reordered, and every satellite is
fails independently (H1, H2: same R(t) and design life from launch, active or
spare); each launch has one success draw for all its satellites. At any time t
the usable satellites are U(t) = sum over launches of a mixture: 0 with prob.
1 - p_launch, Binomial(k_j, R(t - t_j)) with prob. p_launch, for the launches
with C <= t - t_j < DL, and
the operational satellites are min(N, U(t)) (spares fill every slot, H4).
analytic.usable_distribution_cadence computes this exactly.

The simulator evaluates "usable" on the grid against the exact end-of-life
time (t < t_end) and the exact commissioning time, so there is no
discretization error: the comparison is exact up to Monte Carlo noise.
Tolerance: 4 standard errors.
"""

import numpy as np
import pytest

import params
from analytic import (active_distribution_from_usable, reference_launch_rate,
                      usable_distribution_cadence)
from lifecycle import cadence_launch_times, deployment_launch_times, simulate_fixed_cadence

DAY_S = params.SECONDS_PER_DAY
YEAR_S = params.SECONDS_PER_YEAR
DT_S = params.DT_DAYS * DAY_S
COMM_S = params.COMMISSIONING_DAYS * DAY_S
DESIGN_LIFE_S = params.DESIGN_LIFE_YEARS * YEAR_S
N_SIGMA = 4.0
MIN_BIN_PROB = 1e-3  # bins rarer than this are not compared (se not estimable)

WEIBULL = {"model": "weibull", "beta": params.WEIBULL_BETA,
           "theta_s": params.WEIBULL_THETA_YEARS * YEAR_S}

# Test configuration: 6 slots deployed in 3 batches, then k = 2 every 200 days.
# Launch success 0.9 (test-only value) to also exercise the launch failures.
N_SLOTS, MIN_ACTIVE = 6, 6
K, INTERVAL_S = 2, 200.0 * DAY_S
P_LAUNCH = 0.9
DURATION_S = 10.0 * YEAR_S
N_RUNS = 2000
INITIAL_S = deployment_launch_times(N_SLOTS, 3, params.DEPLOY_INTERVAL_DAYS * DAY_S)
START_S = INITIAL_S.max() + INTERVAL_S


@pytest.fixture(scope="module")
def out():
    """One simulation shared by the tests of this module."""
    return simulate_fixed_cadence(
        n_slots=N_SLOTS, cadence_sats=K, cadence_interval_s=INTERVAL_S, cadence_start_s=START_S,
        n_runs=N_RUNS, seed=11, sim_duration_s=DURATION_S, dt_s=DT_S, initial_launch_s=INITIAL_S,
        commissioning_s=COMM_S, design_life_s=DESIGN_LIFE_S, launch_success_prob=P_LAUNCH,
        reliability=WEIBULL)


def cohorts():
    """Launch dates and sizes of all launches: initial deployment + cadence.

    The deployment units with the same launch time fly on one launch (batch).
    """
    t_batch_s, batch_counts = np.unique(INITIAL_S, return_counts=True)
    t_cadence_s = cadence_launch_times(START_S, INTERVAL_S, DURATION_S)
    times_s = np.concatenate([t_batch_s, t_cadence_s])
    counts = np.concatenate([batch_counts, np.full(len(t_cadence_s), K)])
    return times_s, counts


def exact_p_active(t_s):
    """Exact P(n_active = j), j = 0..N, at time t."""
    times_s, counts = cohorts()
    p_usable = usable_distribution_cadence(t_s, times_s, counts, WEIBULL, DESIGN_LIFE_S,
                                           COMM_S, P_LAUNCH)
    return active_distribution_from_usable(p_usable, N_SLOTS)


def test_launch_count(out):
    """Satellites launched = initial deployment + k per cadence date."""
    times_s, counts = cohorts()
    assert np.all(out["n_launched"].sum(axis=1) == counts.sum())


def test_distribution_at_fixed_times(out):
    """Distribution of n_active over the runs = exact min(N, Poisson-binomial).

    Times during deployment, before and after the initial satellites retire.
    The frequency of each value over the runs is binomial: sigma = sqrt(p (1 - p) / n_runs).
    """
    for t_days in [200, 1000, 2000, 2500, 3500]:
        k = int(round(t_days * DAY_S / DT_S))
        p_exact = exact_p_active(out["t_s"][k])
        p_mc = np.bincount(out["n_active"][:, k], minlength=N_SLOTS + 1) / N_RUNS
        sigma = np.sqrt(p_exact * (1.0 - p_exact) / N_RUNS)
        used = p_exact >= MIN_BIN_PROB
        assert np.all(np.abs(p_mc - p_exact)[used] < N_SIGMA * sigma[used]), (
            f"t = {t_days} d: MC {np.round(p_mc, 4)} vs exact {np.round(p_exact, 4)}")


def test_time_averaged_availability(out):
    """Mean availability after deployment = mean of the exact P(n_active >= MIN) on the grid."""
    keep = out["t_s"] >= INITIAL_S.max() + COMM_S
    avail_runs = np.mean(out["n_active"][:, keep] >= MIN_ACTIVE, axis=1)
    se = avail_runs.std(ddof=1) / np.sqrt(N_RUNS)
    exact = np.mean([exact_p_active(t)[MIN_ACTIVE:].sum() for t in out["t_s"][keep]])
    assert abs(avail_runs.mean() - exact) < N_SIGMA * se, (
        f"MC {avail_runs.mean():.4f} vs exact {exact:.4f}, 4 se = {N_SIGMA * se:.4f}")


def test_little_law_usable_satellites(out):
    """Mean usable satellites = launch rate x mean useful life (Little's law).

    Steady state: after cadence start + DL only cadence cohorts are alive. Over
    a whole number of periods the mean of U is (k / T) p_launch E[up], with
    E[up] = integral_C^DL R dt = N / reference_launch_rate. Grid band: the daily
    sum differs from the integral by at most ~2 dt (k / T) p_launch.
    """
    t0_s = START_S + DESIGN_LIFE_S
    n_periods = int((DURATION_S - t0_s) // INTERVAL_S)
    window = (out["t_s"] >= t0_s) & (out["t_s"] < t0_s + n_periods * INTERVAL_S)
    usable = out["n_active"][:, window] + out["n_spares_in_orbit"][:, window]
    usable_runs = usable.mean(axis=1)
    se = usable_runs.std(ddof=1) / np.sqrt(N_RUNS)

    e_up_s = N_SLOTS / reference_launch_rate(N_SLOTS, WEIBULL, DESIGN_LIFE_S, COMM_S)
    rate = K / INTERVAL_S * P_LAUNCH
    expected = rate * e_up_s
    band = 2.0 * DT_S * rate
    assert abs(usable_runs.mean() - expected) < band + N_SIGMA * se, (
        f"MC {usable_runs.mean():.4f} vs Little {expected:.4f}")


def test_infinite_design_life_is_rejected():
    with pytest.raises(ValueError):
        simulate_fixed_cadence(N_SLOTS, K, INTERVAL_S, START_S, 2, 1, YEAR_S, DT_S, INITIAL_S,
                               COMM_S, np.inf, 1.0, WEIBULL)
