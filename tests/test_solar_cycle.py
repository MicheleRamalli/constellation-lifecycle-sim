"""Tests of the time-varying solar activity (decay.py, debris.py).

Reference cases: the constant-activity closed form (decay.lifetime), an independent
numerical integration (scipy solve_ivp), and values read on the ECSS PDF.
"""

import os

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import debris
import decay
import params

YEAR_S = params.SECONDS_PER_YEAR
MONTH_S = params.SOLAR_MONTH_DAYS * params.SECONDS_PER_DAY
F_NODES = params.SOLAR_F107_NODES
ALT_M, RHO = decay.read_density_table(params.ATM_DENSITY_CSV, params.ATM_MODEL)
RHO_LEVELS = np.vstack([RHO[lv] for lv in params.SOLAR_ACTIVITY_LEVELS])
B = decay.ballistic_coefficient(params.DRAG_COEFFICIENT, decay.box_mean_area(*params.EAGLET_BOX_M),
                                params.EAGLET_MASS_KG)
END_M = params.DECAY_END_ALT_KM * 1e3


def test_table_a1_csv():
    """140 months, min <= mean <= max for every index, rows checked by hand on the PDF."""
    f81 = decay.read_solar_cycle(params.SOLAR_CYCLE_CSV, "F81mean")
    assert len(f81) == 140
    for name in ("F10", "F81", "S10", "S81", "M10", "M81"):
        lo, mid, hi = (decay.read_solar_cycle(params.SOLAR_CYCLE_CSV, name + s)
                       for s in ("min", "mean", "max"))
        assert np.all(lo <= mid) and np.all(mid <= hi)
    f10 = decay.read_solar_cycle(params.SOLAR_CYCLE_CSV, "F10mean")
    # ECSS-E-ST-10-04C Rev.1 Table A-1, pages 85, 86, 88 (F10mean, F81mean)
    for month, f10_pdf, f81_pdf in ((0, 74, 76), (60, 234, 224), (114, 78, 80), (139, 66, 67)):
        assert (f10[month], f81[month]) == (f10_pdf, f81_pdf)
    assert f81.min() >= F_NODES[0] and f81.max() <= F_NODES[-1]      # no extrapolation needed
    assert f10.min() >= F_NODES[0] and f10.max() <= F_NODES[-1]


def test_density_at_f107_nodes_and_range():
    """Exact ECSS table at F10.7 = 65, 140, 250; ValueError outside; log <= linear."""
    for f, lv in zip(F_NODES, params.SOLAR_ACTIVITY_LEVELS):
        for kind in ("log", "linear"):
            assert np.allclose(decay.density_at_f107(f, F_NODES, RHO_LEVELS, kind), RHO[lv], rtol=1e-12)
    for bad in (64.9, 250.1):
        with pytest.raises(ValueError):
            decay.density_at_f107(bad, F_NODES, RHO_LEVELS)
    # weighted geometric mean <= weighted arithmetic mean
    for f in (80.0, 100.0, 200.0):
        assert np.all(decay.density_at_f107(f, F_NODES, RHO_LEVELS, "log")
                      <= decay.density_at_f107(f, F_NODES, RHO_LEVELS, "linear") * (1 + 1e-12))
    # log interpolation: ln(rho) at the mid point is the mean of the two nodes
    mid = decay.density_at_f107(102.5, F_NODES, RHO_LEVELS, "log")
    assert np.allclose(mid, np.sqrt(RHO["low"] * RHO["moderate"]))


@pytest.mark.parametrize("model", ["nrlmsise", "jb2006"])
def test_log_density_concave_in_f107(model):
    """ln(rho) concave in F10.7 at every node 120-900 km: slope 65-140 > slope 140-250.
    Then (if the true curve keeps this curvature) log interpolation underestimates rho."""
    alt_m, rho = decay.read_density_table(params.ATM_DENSITY_CSV, model)
    ln = np.log(np.vstack([rho[lv] for lv in params.SOLAR_ACTIVITY_LEVELS]))
    s1 = (ln[1] - ln[0]) / (F_NODES[1] - F_NODES[0])
    s2 = (ln[2] - ln[1]) / (F_NODES[2] - F_NODES[1])
    used = (alt_m >= 120e3) & (alt_m <= 900e3)
    assert np.all(s1[used] > s2[used])


@pytest.mark.parametrize("f107, level", list(zip(F_NODES, params.SOLAR_ACTIVITY_LEVELS)))
def test_constant_series_gives_constant_lifetime(f107, level):
    """Constant F10.7: month-by-month decay = closed-form lifetime of that ECSS table."""
    series = np.full(140, f107)
    expected = decay.lifetime(500e3, END_M, ALT_M, RHO[level], B)
    for start in (0.0, 37.6):
        t = decay.lifetime_variable(500e3, END_M, ALT_M, F_NODES, RHO_LEVELS, B, series, MONTH_S,
                                    start, periodic=True)
        assert t == pytest.approx(expected, rel=1e-9)


def test_variable_series_matches_ode():
    """Varying F10.7: month-by-month closed form vs solve_ivp of da/dt month after month."""
    series = 65.0 + 90.0 * (1.0 + np.sin(np.arange(140) * 2 * np.pi / 140.0))   # synthetic
    start = 10.3
    t_closed = decay.lifetime_variable(450e3, END_M, ALT_M, F_NODES, RHO_LEVELS, B, series, MONTH_S,
                                       start, periodic=True)

    def reached_end(t, y):
        return y[0] - (decay.R_E + END_M)
    reached_end.terminal = True

    a, t0, k, dt = decay.R_E + 450e3, 0.0, int(start), (1 - (start % 1)) * MONTH_S
    while True:
        rho_nodes = decay.density_at_f107(series[k % 140], F_NODES, RHO_LEVELS)

        def rhs(t, y):
            return [decay.decay_rate(y[0], decay.density(y[0] - decay.R_E, ALT_M, rho_nodes), B)]
        sol = solve_ivp(rhs, [0.0, dt], [a], events=reached_end, rtol=1e-11, atol=1e-6)
        if sol.t_events[0].size:
            t_ode = t0 + sol.t_events[0][0]
            break
        a, t0, k, dt = sol.y[0, -1], t0 + dt, k + 1, MONTH_S
    assert t_closed == pytest.approx(t_ode, rel=1e-6)


def test_periodicity_monotonicity_and_end_of_series():
    """Phase phi and phi + 140 give the same lifetime; stronger activity, shorter lifetime;
    a non-periodic series that ends before the re-entry raises ValueError."""
    f81 = decay.read_solar_cycle(params.SOLAR_CYCLE_CSV, params.SOLAR_INDEX)
    args = (500e3, END_M, ALT_M, F_NODES, RHO_LEVELS, B)
    t1 = decay.lifetime_variable(*args, f81, MONTH_S, 20.5, periodic=True)
    t2 = decay.lifetime_variable(*args, f81, MONTH_S, 160.5, periodic=True)
    assert t1 == pytest.approx(t2, rel=1e-12)
    t_strong = decay.lifetime_variable(*args, np.minimum(f81 * 1.1, 250.0), MONTH_S, 20.5, periodic=True)
    assert t_strong < t1
    with pytest.raises(ValueError):
        decay.lifetime_variable(*args, f81[:10], MONTH_S, 0.0, periodic=False)
    # stop at the end of the series: lower bound, 10 full months, not re-entered
    cut = decay.decay_trajectory(*args, f81[:10], MONTH_S, 0.0, periodic=False, stop_at_end=True)
    assert not cut["reentered"] and cut["t_s"][-1] == pytest.approx(10 * MONTH_S)
    assert cut["alt_m"][-1] > END_M


def test_trajectory_is_continuous_and_monotone():
    f81 = decay.read_solar_cycle(params.SOLAR_CYCLE_CSV, params.SOLAR_INDEX)
    traj = decay.decay_trajectory(500e3, END_M, ALT_M, F_NODES, RHO_LEVELS, B, f81, MONTH_S, 3.25, True)
    assert traj["t_s"][1] == pytest.approx(0.75 * MONTH_S)            # partial first month
    assert np.allclose(np.diff(traj["t_s"])[1:-1], MONTH_S)
    assert np.all(np.diff(traj["alt_m"]) < 0) and traj["alt_m"][-1] == END_M
    assert len(traj["f107"]) == len(traj["t_s"]) - 1


def test_flux_integral_constant_activity():
    """Constant F10.7: Phi along the month segments = Phi of the single closed integral."""
    flux_alt_m, flux = debris.read_master_flux(params.MASTER_FLUX_CSV, "1cm", YEAR_S)
    traj = decay.decay_trajectory(500e3, END_M, ALT_M, F_NODES, RHO_LEVELS, B, np.full(140, 140.0),
                                  MONTH_S, 0.0, True)
    phi_traj = debris.flux_time_integral_trajectory(traj, ALT_M, F_NODES, RHO_LEVELS, B, flux_alt_m,
                                                    flux["nominal"])
    phi = debris.flux_time_integral(500e3, END_M, ALT_M, RHO["moderate"], B, flux_alt_m, flux["nominal"])
    assert phi_traj == pytest.approx(phi, rel=1e-3)


def test_values_at_phase():
    grid = np.arange(140.0)
    table = np.sin(2 * np.pi * grid / 140.0)
    assert debris.values_at_phase(table, 141.0, grid, 140.0) == pytest.approx(table[1])
    assert debris.values_at_phase(table, 139.5, grid, 140.0) == pytest.approx(0.5 * table[139])
    with pytest.raises(ValueError):
        debris.values_at_phase(table, 141.0, grid, None)


def test_phase_tables_constant_equal_scalar_case():
    """debris_summary with phase tables that do not depend on the phase = scalar case."""
    from lifecycle import ON_ORBIT_FAILURE, RETIREMENT
    rng_end = np.random.default_rng(3)
    n_runs, n_obj = 5, 60
    out = {"end_kind": rng_end.choice([ON_ORBIT_FAILURE, RETIREMENT], n_obj),
           "end_run": rng_end.integers(0, n_runs, n_obj),
           "end_t_s": rng_end.uniform(0, 10 * YEAR_S, n_obj), "n_active": np.zeros((1, n_runs, 1))}
    life = {debris.FAILED: 6 * YEAR_S, debris.DISPOSAL_FAILED: 6 * YEAR_S, debris.DISPOSED: 0.5 * YEAR_S}
    grid = np.arange(140.0)
    tables = {k: np.full(140, v) for k, v in life.items()}
    phase = {"start_months": np.linspace(0, 100, n_runs), "step_s": MONTH_S, "grid_months": grid,
             "period_months": 140.0}
    a = debris.debris_summary(out, 0.9, np.random.default_rng(1), life, 15 * YEAR_S, 5 * YEAR_S)
    b = debris.debris_summary(out, 0.9, np.random.default_rng(1), tables, 15 * YEAR_S, 5 * YEAR_S,
                              phase=phase)
    for key in a:
        assert np.allclose(a[key], b[key], equal_nan=True)


@pytest.mark.skipif(not os.path.exists(params.OSCAR_FORECAST_PATH), reason="DRAMA not installed")
def test_oscar_forecast_file():
    """fap_mon.dat: consecutive months, F10.7 inside the ECSS range from cycle 24 onwards."""
    year, month, f107 = decay.read_oscar_forecast(params.OSCAR_FORECAST_PATH)
    after = year >= 2008
    assert f107[after].min() >= F_NODES[0] and f107[after].max() <= F_NODES[-1]
    assert (year[-1], month[-1]) == (2129, 12)


# ---------------------------------------------------------------------------
# OSCAR reference runs on Sentinel-3B (read at run time; skipped if not on this machine)
# ---------------------------------------------------------------------------

def oscar_path(name):
    run = params.OSCAR_S3B_RUNS[name]
    return os.path.join(os.path.expanduser(run["folder"]), params.OSCAR_OUTPUT_FILE)


def oscar_json(name):
    import json
    with open(oscar_path(name)) as f:
        return json.load(f)


def test_monthly_mean_partial_months():
    dates = np.arange(np.datetime64("2018-04-17"), np.datetime64("2018-06-03"))
    values = np.arange(len(dates), dtype=float)
    first, mean = decay.monthly_mean(dates, values)
    assert first == np.datetime64("2018-04") and len(mean) == 3
    assert mean[0] == pytest.approx(values[:14].mean())          # 17-30 April
    assert mean[2] == pytest.approx(values[-2:].mean())          # 1-2 June


def test_hohmann_and_rocket_equation():
    """Hohmann: zero for r1 = r2, small transfer = |dv| of the vis-viva difference; rocket
    equation: final-mass and initial-mass forms consistent (m0 = mf + mp)."""
    r = decay.R_E + 700e3
    assert decay.hohmann_delta_v(r, r) == 0.0
    dr = 1.0
    assert decay.hohmann_delta_v(r, r - dr) == pytest.approx(np.sqrt(decay.MU / r) * dr / (2 * r), rel=1e-4)
    mp_f = decay.propellant_mass(100.0, 200.0, 1000.0, mass_is_final=True)
    mp_i = decay.propellant_mass(100.0, 200.0, 1000.0 + mp_f, mass_is_final=False)
    assert mp_i == pytest.approx(mp_f, rel=1e-12)


@pytest.mark.parametrize("name", list(params.OSCAR_S3B_RUNS))
def test_oscar_reference_values(name):
    """params.OSCAR_S3B_RUNS = PyOscar.oop; propagated lifetime = re-entry - start; our Hohmann
    delta-v and rocket equation (1130 kg = mass AFTER the manoeuvre) reproduce OSCAR."""
    if not os.path.exists(oscar_path(name)):
        pytest.skip("OSCAR run not on this machine")
    run = params.OSCAR_S3B_RUNS[name]
    oop = decay.find_key(oscar_json(name), "PyOscar.oop")
    assert oop["perigeeAltitudeRequiredMatchUserDefinedLifetimeLimit"] == run["perigee_km"]
    assert oop["lifetime"] == run["lifetime_years"] and oop["finalDate"] == run["reentry"]
    assert oop["requiredDeltaV"] == run["delta_v_m_s"]
    assert oop["requiredPropellantMass"] == pytest.approx(run["propellant_kg"])
    t_s = (np.datetime64(run["reentry"].replace(" ", "T"), "s")
           - np.datetime64(params.S3B_START, "s")).astype(int)
    assert t_s / YEAR_S == pytest.approx(run["lifetime_years"], abs=0.01)
    dv = decay.hohmann_delta_v(params.S3B_SMA_INITIAL_KM * 1e3, decay.R_E + run["perigee_km"] * 1e3)
    assert dv == pytest.approx(run["delta_v_m_s"], abs=0.05)
    mp = decay.propellant_mass(run["delta_v_m_s"], params.S3B_ISP_S, params.S3B_MASS_KG, mass_is_final=True)
    assert mp == pytest.approx(run["propellant_kg"], abs=0.05)
    assert abs(decay.propellant_mass(run["delta_v_m_s"], params.S3B_ISP_S, params.S3B_MASS_KG, False)
               - run["propellant_kg"]) > 2.0                     # initial-mass form does not match


def test_oscar_constant_series_gives_constant_lifetime():
    """The daily series of the OSCAR constant run (F3M = 140 every day), monthly means, through
    our month-by-month decay = closed-form lifetime with the ECSS moderate table."""
    if not os.path.exists(oscar_path("constant moderate")):
        pytest.skip("OSCAR run not on this machine")
    act = decay.read_oscar_activity(oscar_path("constant moderate"))
    assert np.all(act["F3M"] == 140.0) and np.all(act["Ap"] == 15.0)
    _, f3m = decay.monthly_mean(act["date"], act["F3M"])
    b = decay.ballistic_coefficient(params.DRAG_COEFFICIENT, params.S3B_AREA_M2, params.S3B_MASS_KG)
    h = params.OSCAR_S3B_RUNS["constant moderate"]["perigee_km"] * 1e3
    t_var = decay.lifetime_variable(h, END_M, ALT_M, F_NODES, RHO_LEVELS, b, f3m, YEAR_S / 12, 0.83,
                                    periodic=False)
    assert t_var == pytest.approx(decay.lifetime(h, END_M, ALT_M, RHO["moderate"], b), rel=1e-9)


def test_equivalent_altitude_returns_target():
    """brentq altitude: our lifetime there equals the target lifetime."""
    import main
    b = decay.ballistic_coefficient(params.DRAG_COEFFICIENT, params.S3B_AREA_M2, params.S3B_MASS_KG)
    life = lambda h: decay.lifetime(h, END_M, ALT_M, RHO["moderate"], b)
    target = 14.259 * YEAR_S
    h_eq = main.equivalent_altitude(life, target, 500e3, 800e3)
    assert life(h_eq) == pytest.approx(target, rel=1e-4)
