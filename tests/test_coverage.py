"""Validation of coverage.py against geometric, analytical and numerical references."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import coverage as cv
import params

ALT_M = params.ALT_KM * 1e3
A_M = cv.R_E + ALT_M


# ---------------------------------------------------------------------------
# Footprint geometry
# ---------------------------------------------------------------------------

def elevation_and_nadir_from_vectors(alt_m, lambda_rad):
    """Independent vector geometry: satellite above the pole, ground point at central angle lambda.

    Outputs: elevation of the satellite seen from the point [rad], nadir angle of the point [rad]
    """
    r_sat = np.array([0.0, 0.0, cv.R_E + alt_m])
    r_gnd = cv.R_E * np.array([np.sin(lambda_rad), 0.0, np.cos(lambda_rad)])
    los = r_sat - r_gnd
    elevation = np.arcsin(los @ r_gnd / (np.linalg.norm(los) * cv.R_E))
    nadir = np.arccos(-r_sat @ -los / (np.linalg.norm(r_sat) * np.linalg.norm(los)))
    return elevation, nadir


def test_footprint_horizon():
    """eps = 0: lambda = arccos(R / (R + h)) and eta = rho."""
    lam, eta = cv.footprint_from_elevation(ALT_M, 0.0)
    assert np.isclose(lam, np.arccos(cv.R_E / (cv.R_E + ALT_M)), rtol=1e-12)
    assert np.isclose(eta, cv.earth_angular_radius(ALT_M), rtol=1e-12)


@pytest.mark.parametrize("eps_deg", [5.0, 20.0, 45.0, 70.0])
def test_footprint_from_elevation_vs_vectors(eps_deg):
    """Given eps -> lambda, the vector geometry must give back eps and eta."""
    lam, eta = cv.footprint_from_elevation(ALT_M, np.radians(eps_deg))
    elev, nadir = elevation_and_nadir_from_vectors(ALT_M, lam)
    assert np.isclose(elev, np.radians(eps_deg), atol=1e-10)
    assert np.isclose(nadir, eta, atol=1e-10)


@pytest.mark.parametrize("eta_deg", [15.0, 30.0, 45.0, 60.0])
def test_footprint_from_sensor_vs_vectors(eta_deg):
    """Given eta -> eps -> lambda (cos eps = sin eta / sin rho), the vectors give back eta and eps."""
    lam, eps = cv.footprint_from_sensor(ALT_M, np.radians(eta_deg))
    elev, nadir = elevation_and_nadir_from_vectors(ALT_M, lam)
    assert np.isclose(nadir, np.radians(eta_deg), atol=1e-10)
    assert np.isclose(elev, eps, atol=1e-10)


def test_sensor_beyond_horizon_is_rejected():
    with pytest.raises(ValueError):
        cv.footprint_from_sensor(ALT_M, np.radians(80.0))  # rho = 68 deg at 500 km


# ---------------------------------------------------------------------------
# J2 secular theory
# ---------------------------------------------------------------------------

def test_sso_condition():
    """At the SSO inclination the RAAN turns by exactly 2 pi in one tropical year."""
    inc = cv.sso_inclination(A_M)
    raan_dot, _ = cv.j2_secular_rates(A_M, inc)
    year_s = params.TROPICAL_YEAR_DAYS * params.SECONDS_PER_DAY
    assert np.isclose(raan_dot * year_s, 2.0 * np.pi, rtol=1e-12)
    assert 90.0 < np.degrees(inc) < 100.0  # retrograde, near polar


def j2_equations(t, y):
    """Two-body + J2 equations of motion in ECI (full, not averaged)."""
    r = y[:3]
    rn = np.linalg.norm(r)
    z2 = (r[2] / rn)**2
    k = -1.5 * cv.J2 * cv.MU * cv.R_E**2 / rn**5
    a_j2 = k * np.array([r[0] * (1 - 5 * z2), r[1] * (1 - 5 * z2), r[2] * (3 - 5 * z2)])
    return np.concatenate([y[3:], -cv.MU * r / rn**3 + a_j2])


def test_secular_rates_vs_full_j2_integration():
    """Full J2 numerical integration (10 days) vs the secular formulas.

    The formulas hold for MEAN elements, so they are evaluated at the mean of the
    osculating a and i over a whole number of nodal periods (with the initial
    osculating a the difference would be ~2e-3, as large as the J2 effect).
    Tolerances: the J2 effect is ~1.3e-3 on the period and 100 % of the RAAN
    drift. Nodal period: first-order theory, residual O(J2^2) ~ 1e-6 -> tol 1e-5.
    RAAN rate: mean-vs-osculating definitions give O(J2) relative errors -> tol 2e-3.
    """
    inc = cv.sso_inclination(A_M)
    v = np.sqrt(cv.MU / A_M)
    y0 = [A_M, 0.0, 0.0, 0.0, v * np.cos(inc), v * np.sin(inc)]
    t = np.linspace(0.0, 10 * 86400.0, 200001)
    sol = solve_ivp(j2_equations, (t[0], t[-1]), y0, t_eval=t, rtol=1e-11, atol=1e-6,
                    method="DOP853")
    r, vel = sol.y[:3].T, sol.y[3:].T

    h = np.cross(r, vel)
    i_osc = np.arccos(h[:, 2] / np.linalg.norm(h, axis=1))
    raan = np.unwrap(np.arctan2(h[:, 0], -h[:, 1]))
    a_osc = 1.0 / (2.0 / np.linalg.norm(r, axis=1) - np.sum(vel**2, axis=1) / cv.MU)

    z = r[:, 2]
    k = np.nonzero((z[:-1] < 0) & (z[1:] >= 0))[0]           # ascending node crossings
    t_node = t[k] - z[k] * (t[k + 1] - t[k]) / (z[k + 1] - z[k])
    period_meas = np.mean(np.diff(t_node))

    window = (t >= t_node[0]) & (t <= t_node[-1])
    a_mean, i_mean = a_osc[window].mean(), i_osc[window].mean()
    raan_rate_meas = np.polyfit(t[window], raan[window], 1)[0]
    raan_dot, _ = cv.j2_secular_rates(a_mean, i_mean)

    assert abs(period_meas / cv.nodal_period(a_mean, i_mean) - 1) < 1e-5
    assert abs(raan_rate_meas / raan_dot - 1) < 2e-3


# ---------------------------------------------------------------------------
# Constellation and revisit
# ---------------------------------------------------------------------------

def test_walker_elements_small_case():
    """2/2/1 Walker delta: RAAN 0, 180 deg; u offset F*360/T = 90 deg between planes."""
    raan0, u0 = cv.walker_elements(2, 2, 1, 2 * np.pi)
    np.testing.assert_allclose(np.degrees(raan0), [[0, 0], [180, 180]], atol=1e-12)
    np.testing.assert_allclose(np.degrees(u0), [[0, 180], [90, 270]], atol=1e-12)


def test_gaps_on_synthetic_series():
    """Boolean series: accesses at steps 2-3, 7, 12-13 -> gaps 3 and 4 steps.

    For a boolean series the crossings are at mid-step, so the gaps keep their
    length in steps and the starts move back by half a step.
    """
    vis = np.zeros(16, dtype=bool)
    vis[[2, 3, 7, 12, 13]] = True
    starts, gaps = cv.access_starts_and_gaps(vis, 10.0)
    np.testing.assert_array_equal(starts, [15.0, 65.0, 115.0])
    np.testing.assert_array_equal(gaps, [30.0, 40.0])


def test_interpolated_entry_and_exit_times():
    """Piecewise-linear margin with known zero crossings (dt = 10 s).

    m(t) = 1 - |t - t_c| / h around each access centre t_c (half-width h), so the
    access goes exactly from t_c - h to t_c + h. Accesses: centre 103 s, h = 37 s
    and centre 412 s, h = 21 s -> entries 66 and 391 s, gap 391 - 140 = 251 s.
    The interpolation recovers them to rounding error; counting steps would not.
    """
    t = np.arange(0.0, 600.0, 10.0)
    m = np.maximum(1 - np.abs(t - 103.0) / 37.0, 1 - np.abs(t - 412.0) / 21.0)
    starts, gaps = cv.access_starts_and_gaps(m, 10.0)
    np.testing.assert_allclose(starts, [66.0, 391.0], atol=1e-9)
    np.testing.assert_allclose(gaps, [251.0], atol=1e-9)
    # Second access lasts 42 s: dropped with a 45 s minimum, kept with 40 s
    assert len(cv.access_starts_and_gaps(m, 10.0, min_access_s=45.0)[0]) == 1
    assert len(cv.access_starts_and_gaps(m, 10.0, min_access_s=40.0)[0]) == 2


def access_intervals(inc_rad, n_sats, point_lat_deg, step_s=10.0, days=3.0):
    """Mean interval between access starts of one ground point (lon 0)."""
    t = np.arange(0.0, days * 86400.0, step_s)
    u0 = 2 * np.pi * np.arange(n_sats) / n_sats
    lam, _ = cv.footprint_from_sensor(ALT_M, np.radians(params.ETA_MAX_DEG))
    points, _ = cv.grid_directions(np.array([point_lat_deg]), np.array([0.0]))
    margin = cv.access_margin(t, np.zeros(n_sats), u0, A_M, inc_rad, lam, points)
    starts, _ = cv.access_starts_and_gaps(margin[:, 0], step_s)
    return np.diff(starts), step_s


@pytest.mark.parametrize("n_sats", [1, 3])
def test_equatorial_revisit(n_sats):
    """i = 0, point on the equator: interval = 2 pi / (n_sats (u_dot + Omega_dot - omega_E)).

    With i = 0 the RAAN also drifts by J2, so the inertial longitude rate of the
    satellite is u_dot + Omega_dot. Each start is known within one step dt, so
    every interval is within dt and the MEAN interval (last - first start over n
    intervals) within 2 dt / n. The mean check resolves the Omega_dot term (~9 s
    for one satellite at 500 km, smaller than dt = 10 s).
    """
    intervals, dt = access_intervals(0.0, n_sats, 0.0)
    raan_dot, u_dot = cv.j2_secular_rates(A_M, 0.0)
    expected = 2 * np.pi / (n_sats * (u_dot + raan_dot - cv.OMEGA_E))
    assert len(intervals) > 10
    assert np.all(np.abs(intervals - expected) <= dt)
    assert abs(intervals.mean() - expected) <= 2 * dt / len(intervals)


def test_polar_revisit_at_pole():
    """i = 90 deg, point at the pole: one access per orbit, interval = nodal period."""
    intervals, dt = access_intervals(np.pi / 2, 1, 90.0)
    expected = cv.nodal_period(A_M, np.pi / 2)
    assert len(intervals) > 10
    assert np.all(np.abs(intervals - expected) <= dt)
    assert abs(intervals.mean() - expected) <= 2 * dt / len(intervals)


def test_walker_search_picks_smallest_feasible_total():
    """Loose requirement -> smallest T (1 plane x 2 sats); impossible one -> None."""
    kwargs = dict(eta_rad=np.radians(params.ETA_MAX_DEG), step_s=30.0, duration_s=2 * 86400.0,
                  p_range=(1, 2), s_range=(2, 3), raan_spread_rad=np.pi,
                  min_access_s=0.0)  # search logic only: no minimum access
    best, rows = cv.walker_search(max_revisit_s=1e9, **kwargs)
    assert best == (1, 2, 0)
    assert all(r[0] * r[1] == 2 for r in rows)  # stopped at the first total T
    best, rows = cv.walker_search(max_revisit_s=1.0, **kwargs)
    assert best is None
    assert len(rows) == 2 + 2 * 2  # P=1: S=2,3; P=2: S=2,3 with F=0,1


def test_quadratic_crossing_on_parabolic_margin():
    """Parabolic margin m(t) = 1 - ((t - 100) / 23)^2 (dt = 10 s): access 77 to 123 s.

    The parabola through three samples is exact for a parabolic margin, while a
    straight line between two samples would give a shorter access.
    """
    t = np.arange(0.0, 300.0, 10.0)
    m = 1.0 - ((t - 100.0) / 23.0)**2
    _, _ = cv.access_starts_and_gaps(m, 10.0)
    k_in, k_out = 7, 12   # m[7] < 0 <= m[8], m[12] >= 0 > m[13]
    assert np.isclose(cv.crossing_time(m, k_in) * 10.0, 77.0, atol=1e-9)
    assert np.isclose(cv.crossing_time(m, k_out) * 10.0, 123.0, atol=1e-9)
    linear_entry = (k_in + m[k_in] / (m[k_in] - m[k_in + 1])) * 10.0
    assert abs(linear_entry - 77.0) > 0.3   # the linear estimate is off by 0.42 s


def test_min_access_merges_gaps():
    """Boolean series: accesses at steps 2-4 (30 s), 8 (10 s), 13-15 (30 s), dt = 10 s.

    min 0 s : gaps 3 and 4 steps; min 30 s: the 10 s access is dropped and the
    gaps merge into one of 8 steps (from the exit at 4.5 to the entry at 12.5).
    """
    vis = np.zeros(20, dtype=bool)
    vis[[2, 3, 4, 8, 13, 14, 15]] = True
    _, gaps = cv.access_starts_and_gaps(vis, 10.0)
    np.testing.assert_array_equal(gaps, [30.0, 40.0])
    starts, gaps = cv.access_starts_and_gaps(vis, 10.0, min_access_s=30.0)
    np.testing.assert_array_equal(starts, [15.0, 125.0])
    np.testing.assert_array_equal(gaps, [80.0])


def test_integer_time_step_gives_same_result():
    """Regression: an integer dt must not truncate the interpolated times."""
    t = np.arange(0.0, 600.0, 10.0)
    m = np.maximum(1 - np.abs(t - 103.0) / 37.0, 1 - np.abs(t - 412.0) / 21.0)
    starts_f, gaps_f = cv.access_starts_and_gaps(m, 10.0)
    starts_i, gaps_i = cv.access_starts_and_gaps(m, 10)
    np.testing.assert_array_equal(starts_f, starts_i)
    np.testing.assert_array_equal(gaps_f, gaps_i)


def test_per_satellite_orbits_equal_scalar_call():
    """Per-satellite a and i equal to the reference give exactly the default result."""
    raan0, u0 = cv.walker_elements(2, 2, 1, np.pi)
    a = cv.R_E + ALT_M
    kwargs = dict(lat_deg=45.0, eta_rad=np.radians(30), step_s=30.0, duration_s=2 * 86400.0,
                  min_access_s=0.0)
    ref = cv.constellation_revisit(raan0, u0, **kwargs)
    per_sat = cv.constellation_revisit(raan0, u0, a_m=np.full(4, a),
                                       inc_rad=np.full(4, cv.sso_inclination(a)), **kwargs)
    np.testing.assert_array_equal(ref[1], per_sat[1])
    np.testing.assert_array_equal(ref[2], per_sat[2])
