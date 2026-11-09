"""Tests of decay.py against analytical cases.

The atmospheres here are SYNTHETIC (round numbers chosen only to exercise the
formulas), not physical data: the real densities come from ECSS-E-ST-10-04C.
"""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import erfi

import decay

R_E = decay.R_E
B = 0.02                  # ballistic coefficient [m^2/kg], synthetic
RHO_REF = 1e-12           # density at the reference altitude [kg/m^3], synthetic
ALT_REF = 500e3           # reference altitude [m]
H = 60e3                  # scale height [m], synthetic


def test_closed_form_matches_numerical_ode():
    """Single scale height: closed form (Dawson) vs numerical integration of da/dt in time."""
    a_start, a_end = R_E + 500e3, R_E + 300e3

    def rhs(t, y):
        rho = RHO_REF * np.exp(-(y[0] - (R_E + ALT_REF)) / H)
        return [decay.decay_rate(y[0], rho, B)]

    def reached_end(t, y):
        return y[0] - a_end
    reached_end.terminal = True

    sol = solve_ivp(rhs, [0.0, 1e10], [a_start], events=reached_end, rtol=1e-10, atol=1e-6)
    t_ode = sol.t_events[0][0]
    t_closed = decay.time_single_scale(a_start, a_end, RHO_REF, R_E + ALT_REF, H, B)
    assert np.isclose(t_closed, t_ode, rtol=1e-6)


def test_dawson_form_equals_erfi_form():
    """sqrt(pi H) erfi(sqrt(a/H)) written with the Dawson function (large H: no overflow)."""
    h_big = 1000e3
    a_start, a_end = R_E + 500e3, R_E + 200e3
    integral = np.sqrt(np.pi * h_big) * (erfi(np.sqrt(a_start / h_big)) - erfi(np.sqrt(a_end / h_big)))
    t_erfi = (np.exp(-(R_E + ALT_REF) / h_big) * integral / (B * np.sqrt(decay.MU) * RHO_REF))
    t_closed = decay.time_single_scale(a_start, a_end, RHO_REF, R_E + ALT_REF, h_big, B)
    t_quad = quad(lambda a: 1.0 / -decay.decay_rate(a, RHO_REF * np.exp(-(a - R_E - ALT_REF) / h_big), B),
                  a_end, a_start)[0]
    assert np.isclose(t_closed, t_erfi, rtol=1e-9)
    assert np.isclose(t_closed, t_quad, rtol=1e-9)


def test_small_decay_limit():
    """For a small drop the time is delta_a / |da/dt| at the start."""
    a_start, delta_a = R_E + 500e3, 10.0
    t = decay.time_single_scale(a_start, a_start - delta_a, RHO_REF, R_E + ALT_REF, H, B)
    assert np.isclose(t, delta_a / -decay.decay_rate(a_start, RHO_REF, B), rtol=1e-3)


def test_decay_rate_is_drag_power():
    """mu / (2 a^2) da/dt equals the drag power per unit mass -(1/2) rho B v^3."""
    a, rho = R_E + 500e3, 1e-12
    v = np.sqrt(decay.MU / a)
    assert np.isclose(decay.MU / (2 * a**2) * decay.decay_rate(a, rho, B), -0.5 * rho * B * v**3)


def test_piecewise_with_one_scale_height_equals_single_band():
    """Table built from one exponential: same density and same lifetime as one band."""
    nodes = np.arange(200e3, 801e3, 50e3)
    rho = RHO_REF * np.exp(-(nodes - ALT_REF) / H)
    assert np.allclose(decay.scale_heights(nodes, rho), H)
    alt = np.array([200e3, 333e3, 500e3, 777e3, 800e3])
    assert np.allclose(decay.density(alt, nodes, rho), RHO_REF * np.exp(-(alt - ALT_REF) / H))
    t_piece = decay.lifetime(620e3, 230e3, nodes, rho, B)
    t_single = decay.time_single_scale(R_E + 620e3, R_E + 230e3, RHO_REF, R_E + ALT_REF, H, B)
    assert np.isclose(t_piece, t_single, rtol=1e-9)


def test_piecewise_density_passes_through_nodes_and_is_continuous():
    nodes = np.array([300e3, 400e3, 500e3, 600e3])
    rho = np.array([2e-11, 3e-12, 6e-13, 1.5e-13])   # synthetic, different H per band
    assert np.allclose(decay.density(nodes, nodes, rho), rho)
    eps = 1e-3
    for h in nodes[1:-1]:
        assert np.isclose(decay.density(h - eps, nodes, rho), decay.density(h + eps, nodes, rho),
                          rtol=1e-6)
    # lifetime is additive over bands
    total = decay.lifetime(600e3, 300e3, nodes, rho, B)
    parts = sum(decay.lifetime(nodes[i + 1], nodes[i], nodes, rho, B) for i in range(3))
    assert np.isclose(total, parts, rtol=1e-12)


def test_no_extrapolation():
    nodes, rho = np.array([300e3, 400e3]), np.array([2e-11, 3e-12])
    for bad in (250e3, 450e3):
        try:
            decay.density(bad, nodes, rho)
        except ValueError:
            continue
        raise AssertionError("density outside the table must raise")


def test_box_mean_area_cauchy():
    """Mean projected area over random directions = surface / 4 (Monte Carlo check)."""
    lx, ly, lz = 0.5, 0.4, 0.3
    rng = np.random.default_rng(1)
    n = rng.normal(size=(400_000, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)    # uniform directions on the sphere
    projected = np.abs(n[:, 0]) * ly * lz + np.abs(n[:, 1]) * lx * lz + np.abs(n[:, 2]) * lx * ly
    assert np.isclose(projected.mean(), decay.box_mean_area(lx, ly, lz), rtol=3e-3)
    assert np.isclose(decay.box_mean_area(1.0, 1.0, 1.0), 1.5)


def test_sma_rate_from_ndot_matches_kepler():
    """da/dt from ndot vs finite difference of a(n) = (mu / n^2)^(1/3)."""
    n = 2 * np.pi * 15.2 / 86400.0      # rad/s
    ndot = 1e-15                        # rad/s^2
    dt = 1000.0
    a = lambda nn: (decay.MU / nn**2) ** (1.0 / 3.0)
    fd = (a(n + ndot * dt) - a(n - ndot * dt)) / (2 * dt)
    assert np.isclose(decay.sma_rate_from_ndot(a(n), n, ndot), fd, rtol=1e-6)


def test_ecss_table_is_physical_and_lifetime_ordered():
    """ECSS-E-ST-10-04C Rev.1 table: density decreasing with altitude (H > 0) in every column;
    lifetime longer for lower solar activity and for higher initial altitude."""
    import params
    for model in ("nrlmsise", "jb2006"):
        alt_m, rho = decay.read_density_table(params.ATM_DENSITY_CSV, model)
        assert np.all(np.diff(alt_m) > 0)
        for level in params.SOLAR_ACTIVITY_LEVELS:
            assert np.all(decay.scale_heights(alt_m, rho[level]) > 0)
        t = {lv: decay.lifetime(500e3, 150e3, alt_m, rho[lv], B) for lv in params.SOLAR_ACTIVITY_LEVELS}
        assert t["high"] < t["moderate"] < t["low"]
        assert decay.lifetime(600e3, 150e3, alt_m, rho["moderate"], B) > t["moderate"]
    alt_m, _ = decay.read_density_table(params.ATM_DENSITY_CSV, "jb2006")
    assert alt_m[0] == 100e3          # JB-2006 has no values below 100 km: rows skipped
