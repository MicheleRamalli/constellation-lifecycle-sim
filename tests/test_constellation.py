"""Validation of the constellation layer (lifecycle.simulate_constellation and helpers).

Analytical reference: the planes are independent, so the distribution of the
configuration (n_1, n_2) is the product of the per-plane distributions, and the
constellation availability for a table-defined requirement is
    sum over (n_1, n_2) of P_1(n_1) P_2(n_2) 1{table[n_1, n_2] <= requirement}.
The per-plane distributions are the ones already validated in test_spares.py
(birth-death chain; Binomial with renewal-reward p). Tolerances: 4 standard
errors over the independent runs, plus the discretization band (delay 0 and dt).
"""

import numpy as np

import params
from analytic import (active_distribution_binomial, active_distribution_from_down,
                      birth_death_stationary, slot_availability_pre_order)
from lifecycle import (constellation_availability, constellation_revisit_series,
                       simulate_constellation, simulate_plane)

DAY_S = params.SECONDS_PER_DAY
YEAR_S = params.SECONDS_PER_YEAR
DT_S = params.DT_DAYS * DAY_S
ORDER_S = params.ORDER_TO_LAUNCH_DAYS * DAY_S
COMM_S = params.COMMISSIONING_DAYS * DAY_S
DESIGN_LIFE_S = params.DESIGN_LIFE_YEARS * YEAR_S
N_SLOTS, N_SPARES, N_PLANES = 4, 1, 2
N_SIGMA = 4.0
WEIBULL = {"model": "weibull", "beta": params.WEIBULL_BETA,
           "theta_s": params.WEIBULL_THETA_YEARS * YEAR_S}
EXPONENTIAL = {"model": "exponential", "mttf_s": 4.0 * YEAR_S}  # test-only value

# Synthetic lookup table [h]: 10 h if at least 7 of the 8 slots are operational, else 13 h
N = np.arange(N_SLOTS + 1)
TABLE_H = np.where(N[:, None] + N[None, :] >= 7, 10.0, 13.0)
REQUIREMENT_H = 12.0


def plane_inputs(n_runs, duration_s, comm_s, design_life_s, reliability, pre_order,
                 initial_launch_s, n_spares=N_SPARES):
    """Inputs of simulate_plane for one plane, one reorder per loss."""
    return dict(n_slots=N_SLOTS, n_spares=n_spares, reorder_level=n_spares,
                pre_order_retirement=pre_order, n_runs=n_runs, sim_duration_s=duration_s,
                dt_s=DT_S, initial_launch_s=initial_launch_s, order_to_launch_s=ORDER_S,
                commissioning_s=comm_s, design_life_s=design_life_s, launch_success_prob=1.0,
                sats_per_launch=1, reliability=reliability)


def analytic_availability(p_active_1, p_active_2, table_h=TABLE_H):
    """sum P1(n1) P2(n2) 1{table[n1, n2] <= requirement}."""
    joint = p_active_1[:, None] * p_active_2[None, :]
    return np.sum(joint[table_h <= REQUIREMENT_H])


def check(out, warmup_s, a_lo, a_hi, table_h=TABLE_H):
    """MC constellation availability after warm-up vs the analytical band."""
    revisit_h = constellation_revisit_series(out["n_active"], table_h)
    avail = constellation_availability(revisit_h, REQUIREMENT_H, out["t_s"], warmup_s)
    se = avail.std(ddof=1) / np.sqrt(len(avail))
    lo, hi = min(a_lo, a_hi), max(a_lo, a_hi)
    assert lo - N_SIGMA * se < avail.mean() < hi + N_SIGMA * se, (
        f"MC {avail.mean():.4f} vs [{lo:.4f}, {hi:.4f}], 4 se = {N_SIGMA * se:.4f}")
    return avail.mean(), se


def test_one_plane_equals_plane_simulator():
    """A one-plane constellation is the plane simulator with seed [seed, 0]."""
    inputs = plane_inputs(50, 10 * YEAR_S, COMM_S, DESIGN_LIFE_S, WEIBULL, True,
                          np.zeros(N_SLOTS + N_SPARES))
    con = simulate_constellation(simulate_plane, [inputs], seed=7)
    plane = simulate_plane(seed=[7, 0], **inputs)
    np.testing.assert_array_equal(con["n_active"][0], plane["n_active"])
    np.testing.assert_array_equal(con["n_launched"], plane["n_launched"])


def test_planes_are_independent():
    """Same inputs, different seeds: run averages of n_1 and n_2 are uncorrelated.

    For independent runs the sample correlation of n_runs pairs has standard
    deviation ~ 1 / sqrt(n_runs).
    """
    n_runs = 400
    inputs = plane_inputs(n_runs, 15 * YEAR_S, COMM_S, DESIGN_LIFE_S, EXPONENTIAL, False,
                          np.zeros(N_SLOTS + N_SPARES))
    out = simulate_constellation(simulate_plane, [inputs, inputs], seed=3)
    mean_1, mean_2 = out["n_active"][0].mean(axis=1), out["n_active"][1].mean(axis=1)
    assert not np.array_equal(mean_1, mean_2)
    assert abs(np.corrcoef(mean_1, mean_2)[0, 1]) < N_SIGMA / np.sqrt(n_runs)


def test_revisit_series_lookup():
    """table[n_1, n_2] on a small known case."""
    n_active = np.array([[[4, 3, 0]], [[4, 4, 2]]])        # (planes, runs, steps)
    np.testing.assert_array_equal(constellation_revisit_series(n_active, TABLE_H),
                                  [[10.0, 10.0, 13.0]])


def test_constellation_exponential_vs_markov_chain():
    """b) per plane, exponential lives, deterministic lead time: product of the chains."""
    out = simulate_constellation(
        simulate_plane,
        [plane_inputs(200, 100 * YEAR_S, 0.0, np.inf, EXPONENTIAL, False,
                      np.zeros(N_SLOTS + N_SPARES))] * N_PLANES, seed=5)
    mttf_s = EXPONENTIAL["mttf_s"]
    band = []
    for delay_s in [0.0, DT_S]:
        pi = birth_death_stationary(N_SLOTS + N_SPARES, 1.0 / (mttf_s + delay_s), 1.0 / ORDER_S)
        p_active = active_distribution_from_down(pi, N_SLOTS)
        band.append(analytic_availability(p_active, p_active))
    check(out, 20 * YEAR_S, *band)


def test_constellation_weibull_pre_order_vs_binomial():
    """c) per plane (no spares), Weibull + design life + commissioning, random phases.

    Each plane: n_active ~ Binomial(N, A_c) (test_spares.py). With a spare the
    availability would be ~0.99996 and the MC would always see 1 (se not
    estimable), so this test uses no spares and a requirement met only with
    both planes full: expected A_c^(2N) ~ 0.906. Warm-up one cycle DL - C.
    """
    n_runs = 300
    cycle_s = DESIGN_LIFE_S - COMM_S
    table_full_h = np.where(N[:, None] + N[None, :] == 2 * N_SLOTS, 10.0, 13.0)
    rng = np.random.default_rng(321)
    inputs = [plane_inputs(n_runs, 40 * YEAR_S, COMM_S, DESIGN_LIFE_S, WEIBULL, True,
                           np.floor(rng.uniform(0, cycle_s, (n_runs, N_SLOTS)) / DT_S) * DT_S,
                           n_spares=0)
              for _ in range(N_PLANES)]
    out = simulate_constellation(simulate_plane, inputs, seed=9)
    band = []
    for delay_s in [0.0, DT_S]:
        p_up = slot_availability_pre_order(WEIBULL, DESIGN_LIFE_S, ORDER_S, COMM_S, delay_s)
        p_active = active_distribution_binomial(N_SLOTS, 0, p_up)
        band.append(analytic_availability(p_active, p_active, table_full_h))
    check(out, cycle_s, *band, table_h=table_full_h)
