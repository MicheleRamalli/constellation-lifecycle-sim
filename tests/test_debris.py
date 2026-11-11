"""Validation of debris.py against analytical cases."""

import numpy as np
import pytest

import debris
import decay
import params
from lifecycle import LAUNCH_FAILURE, ON_ORBIT_FAILURE, RETIREMENT, simulate_constellation, simulate_plane
from reliability import reliability_analytic

YEAR_S = params.SECONDS_PER_YEAR
DAY_S = params.SECONDS_PER_DAY
ATM_ALT_M, ATM_RHO = decay.read_density_table(params.ATM_DENSITY_CSV, params.ATM_MODEL)
FLUX_ALT_M, FLUX = debris.read_master_flux(params.MASTER_FLUX_CSV, "1cm", YEAR_S)
B = decay.ballistic_coefficient(params.DRAG_COEFFICIENT, decay.box_mean_area(*params.EAGLET_BOX_M),
                                params.EAGLET_MASS_KG)
WEIBULL = {"model": "weibull", "beta": params.WEIBULL_BETA,
           "theta_s": params.WEIBULL_THETA_YEARS * YEAR_S}


def test_flux_interpolation():
    """Exact at the nodes, log-linear in between (geometric mean at mid-node), 0 below 200 km."""
    f = FLUX["nominal"]
    assert np.allclose(debris.flux_at(FLUX_ALT_M, FLUX_ALT_M, f), f)
    mid = 0.5 * (FLUX_ALT_M[4] + FLUX_ALT_M[5])
    assert debris.flux_at(mid, FLUX_ALT_M, f) == pytest.approx(np.sqrt(f[4] * f[5]))
    assert debris.flux_at(150e3, FLUX_ALT_M, f) == 0.0
    with pytest.raises(ValueError):
        debris.flux_at(850e3, FLUX_ALT_M, f)


def test_flux_cases_ordered():
    """low <= nominal <= high and units 1/(m^2 s) (file value / seconds per year)."""
    assert np.all(FLUX["low"] <= FLUX["nominal"]) and np.all(FLUX["nominal"] <= FLUX["high"])
    assert FLUX["nominal"][FLUX_ALT_M == 500e3][0] * YEAR_S == pytest.approx(3.358e-5)


@pytest.mark.parametrize("level", params.SOLAR_ACTIVITY_LEVELS)
def test_constant_flux_gives_flux_times_lifetime(level):
    """Constant flux F0 (nodes from 200 km): Phi = F0 x lifetime (closed form of decay.py)."""
    f0 = np.full(len(FLUX_ALT_M), 1e-12)
    start, end = 500e3, 200e3
    phi = debris.flux_time_integral(start, end, ATM_ALT_M, ATM_RHO[level], B, FLUX_ALT_M, f0)
    expected = 1e-12 * decay.lifetime(start, end, ATM_ALT_M, ATM_RHO[level], B)
    assert phi == pytest.approx(expected, rel=1e-4)


def test_collision_probability_small_and_large():
    """P = A Phi for small A Phi; P = 1 - e^-1 for A Phi = 1."""
    assert debris.collision_probability(1e-6, 0.5) == pytest.approx(5e-7, rel=1e-6)
    assert debris.collision_probability(2.0, 0.5) == pytest.approx(1.0 - np.exp(-1.0))


def fake_out(end_kind, end_run, n_runs):
    return {"end_kind": np.asarray(end_kind), "end_run": np.asarray(end_run),
            "end_t_s": np.zeros(len(end_kind)), "n_active": np.zeros((1, n_runs, 1))}


def test_all_disposed_all_compliant():
    """P_manoeuvre = 1, only retirements, disposed lifetime < limit: all compliant, P_d,eff = 1."""
    out = fake_out([RETIREMENT] * 6 + [LAUNCH_FAILURE] * 2, [0, 0, 0, 1, 1, 1, 0, 1], 2)
    residual = {debris.FAILED: 10 * YEAR_S, debris.DISPOSAL_FAILED: 10 * YEAR_S,
                debris.DISPOSED: 1 * YEAR_S}
    pc = {debris.FAILED: 1e-3, debris.DISPOSAL_FAILED: 1e-3, debris.DISPOSED: 1e-6}
    s = debris.debris_summary(out, 1.0, np.random.default_rng(0), residual, 20 * YEAR_S, 5 * YEAR_S, pc)
    assert np.all(s["n_objects"] == 3)              # launch failures leave no object
    assert np.allclose(s["collisions"], 3e-6)       # sum of P_c of the 3 disposed objects
    assert np.all(s["pd_effective"] == 1.0) and np.all(s["frac_compliant"] == 1.0)
    assert np.allclose(s["obj_years_committed"], 3 * YEAR_S)


def test_failures_never_disposed():
    """On-orbit failures always decay naturally, whatever the manoeuvre probability."""
    _, outcome = debris.object_outcomes(np.full(100, ON_ORBIT_FAILURE), 1.0, np.random.default_rng(0))
    assert np.all(outcome == debris.FAILED)


def test_disposed_fraction_is_binomial():
    """Disposed / retirements ~ p within 4 sigma."""
    n, p = 20000, 0.7
    _, outcome = debris.object_outcomes(np.full(n, RETIREMENT), p, np.random.default_rng(1))
    assert abs(np.mean(outcome == debris.DISPOSED) - p) < 4 * np.sqrt(p * (1 - p) / n)


def test_effective_pd_from_lifecycle_matches_reliability():
    """Lifecycle MC, launch on failure, long horizon: P_d,eff -> p_m x R(DL).

    Each satellite ends once, by failure before DL (prob. 1 - R(DL)) or by
    retirement (prob. R(DL)). Satellites still alive at the horizon are not
    counted; with a horizon of many design lives this censoring is small but
    biases towards failures (short lives end first), so the tolerance adds a
    band of 1 / (ends per run). A launch failure leaves no object.
    """
    dl_s, n_runs, p_m = 2.0 * YEAR_S, 200, 0.9
    inputs = [dict(n_slots=4, n_spares=0, reorder_level=0, pre_order_retirement=False,
                   n_runs=n_runs, sim_duration_s=40 * YEAR_S, dt_s=DAY_S,
                   initial_launch_s=np.zeros(4), order_to_launch_s=30 * DAY_S,
                   commissioning_s=10 * DAY_S, design_life_s=dl_s, launch_success_prob=0.95,
                   sats_per_launch=1, reliability=WEIBULL)]
    out = simulate_constellation(simulate_plane, inputs, seed=3)
    residual = {k: YEAR_S for k in (debris.FAILED, debris.DISPOSAL_FAILED, debris.DISPOSED)}
    s = debris.debris_summary(out, p_m, np.random.default_rng(4), residual, 40 * YEAR_S, 5 * YEAR_S)
    expected = p_m * reliability_analytic(dl_s, **WEIBULL)
    pooled = s["n_disposed"].sum() / s["n_objects"].sum()
    se = np.sqrt(expected * (1 - expected) / s["n_objects"].sum())
    assert abs(pooled - expected) < 4 * se + 1.0 / s["n_objects"].mean()


def test_required_manoeuvre_prob():
    assert debris.required_manoeuvre_prob(0.9, 0.95) == pytest.approx(0.9 / 0.95)
