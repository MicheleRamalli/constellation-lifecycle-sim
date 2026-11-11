"""Debris indicators of the constellation: end of life, 5-year rule, object-years, collisions.

Post-processing of the lifecycle Monte Carlo (lifecycle.simulate_constellation):
every satellite that ends in orbit becomes a dead object (model in params.py, next to
EAGLET_PROPULSION):
- on-orbit failure (propulsion lost)  -> natural decay from the operational altitude
- retirement, manoeuvre succeeds      -> circular orbit at DISPOSAL_ALT_KM, natural decay
- retirement, manoeuvre fails         -> natural decay from the operational altitude
- launch failure                      -> no object in orbit
With a constant solar activity the residual lifetime of each class is deterministic
(decay.lifetime), so the Monte Carlo only gives the mix and the timing of the objects.
With a time-varying activity (solar cycle) the residual lifetime depends on the phase
of the cycle at the end of life: it is tabulated on a grid of start phases
(decay.lifetime_variable) and each object reads the table at its own phase
(values_at_phase).

Collision probability of one dead object with debris larger than a size (Poisson):
    P_c = 1 - exp(-A Phi),  Phi = integral F(h(t)) dt = integral F(h) / |dh/dt| dh
with F the MASTER-8 flux [1/(m^2 s)] and dh/dt = -rho B sqrt(mu a) (decay.py).
"""

import csv

import numpy as np

import decay
from lifecycle import LAUNCH_FAILURE, RETIREMENT

# Outcome of a dead object
FAILED = 0           # on-orbit failure, natural decay
DISPOSAL_FAILED = 1  # retirement, manoeuvre failed, natural decay
DISPOSED = 2         # retirement, manoeuvre succeeded

FLUX_CASES = ("low", "nominal", "high")


def read_master_flux(path, size, seconds_per_year):
    """MASTER-8 flux vs altitude for one size threshold, three uncertainty cases.

    Inputs: CSV path (params.MASTER_FLUX_CSV), size "1mm" / "1cm" / "10cm",
            seconds in one year (the file is in 1/(m^2 yr))
    Output: altitude nodes [m], dict {"low", "nominal", "high": flux [1/(m^2 s)]};
            low = nominal - sigma_minus, high = nominal + sigma_plus
    """
    with open(path) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    alt_m = np.array([float(r["alt_km"]) for r in rows]) * 1e3
    nominal = np.array([float(r[f"flux_gt_{size}"]) for r in rows])
    minus = np.array([float(r[f"sigma_minus_gt_{size}"]) for r in rows])
    plus = np.array([float(r[f"sigma_plus_gt_{size}"]) for r in rows])
    flux = {"low": nominal - minus, "nominal": nominal, "high": nominal + plus}
    return alt_m, {case: f / seconds_per_year for case, f in flux.items()}


def flux_at(alt_m, alt_nodes_m, flux_nodes):
    """Flux at altitude alt_m [m], log-linear between the nodes (scalar or array).

    Zero below the first node (ASSUMPTION, params.py); above the last node -> ValueError.
    """
    alt = np.asarray(alt_m, dtype=float)
    if np.any(alt > alt_nodes_m[-1] + 1e-6):
        raise ValueError("Altitude above the flux table (no extrapolation)")
    log_f = np.interp(alt, alt_nodes_m, np.log(flux_nodes))
    return np.where(alt < alt_nodes_m[0], 0.0, np.exp(log_f))


def flux_time_integral(alt_start_m, alt_end_m, atm_nodes_m, rho_nodes, b_m2_kg,
                       flux_alt_m, flux_nodes, step_m=500.0):
    """Phi = integral of the flux over the decay from alt_start to alt_end [1/m^2].

    Change of variable t -> h: dt = dh / |dh/dt|, |dh/dt| = rho B sqrt(mu a).
    Trapezoid rule on a grid of step step_m that also contains the flux and
    density nodes (the integrand has kinks there).
    Inputs: altitudes [m], density table (decay.read_density_table), B [m^2/kg],
            flux table (read_master_flux), grid step [m]
    """
    nodes = np.concatenate([atm_nodes_m, flux_alt_m])
    nodes = nodes[(nodes > alt_end_m) & (nodes < alt_start_m)]
    h = np.unique(np.concatenate([np.arange(alt_end_m, alt_start_m, step_m), nodes, [alt_start_m]]))
    rate = -decay.decay_rate(decay.R_E + h, decay.density(h, atm_nodes_m, rho_nodes), b_m2_kg)
    return np.trapezoid(flux_at(h, flux_alt_m, flux_nodes) / rate, h)


def flux_time_integral_trajectory(traj, atm_nodes_m, f107_nodes, rho_levels, b_m2_kg,
                                  flux_alt_m, flux_nodes, interp="log"):
    """Phi [1/m^2] along a decay with time-varying solar activity (decay.decay_trajectory).

    Inside each segment of the trajectory the density table is fixed, so Phi is the sum of
    flux_time_integral over the segments, each with the density of its F10.7.
    """
    phi = 0.0
    for k in range(len(traj["f107"])):
        rho = decay.density_at_f107(traj["f107"][k], f107_nodes, rho_levels, interp)
        phi += flux_time_integral(traj["alt_m"][k], traj["alt_m"][k + 1], atm_nodes_m, rho, b_m2_kg,
                                  flux_alt_m, flux_nodes)
    return phi


def values_at_phase(table, phase_months, grid_months, period_months=None):
    """Value of a quantity tabulated vs the start phase, at the phase of each object.

    Linear interpolation (np.interp, like interp1); with period_months the phase is
    taken modulo the period (repeated cycle). Without it, a phase outside the grid
    raises ValueError (no extrapolation).
    """
    phase = np.asarray(phase_months, dtype=float)
    if period_months is None and (np.any(phase < grid_months[0]) or np.any(phase > grid_months[-1])):
        raise ValueError("Phase outside the tabulated grid")
    return np.interp(phase, grid_months, table, period=period_months)


def collision_probability(phi_per_m2, area_m2):
    """Poisson probability of at least one collision: 1 - exp(-A Phi) [-]."""
    return -np.expm1(-area_m2 * phi_per_m2)


def object_outcomes(end_kind, p_manoeuvre, rng):
    """Outcome of every satellite that ended (launch failures removed).

    Inputs: end_kind array (lifecycle), manoeuvre success probability at
            retirement [-], numpy Generator (one Bernoulli draw per retirement)
    Outputs: (mask of the ends that leave an object in orbit, outcome of those objects)
    """
    in_orbit = end_kind != LAUNCH_FAILURE
    kind = end_kind[in_orbit]
    success = rng.random(len(kind)) < p_manoeuvre
    outcome = np.where(kind == RETIREMENT, np.where(success, DISPOSED, DISPOSAL_FAILED), FAILED)
    return in_orbit, outcome


def debris_summary(out, p_manoeuvre, rng, residual_s, horizon_s, limit_s, pc=None, phase=None):
    """Debris indicators of a constellation Monte Carlo, per run.

    Inputs:
        out         : output of lifecycle.simulate_constellation (end_run, end_t_s, end_kind)
        p_manoeuvre : success probability of the disposal manoeuvre at retirement [-]
        rng         : numpy Generator for the manoeuvre draws
        residual_s  : residual lifetime [s] of each outcome, dict {FAILED, DISPOSAL_FAILED, DISPOSED}
        horizon_s   : end of the simulated time span [s]
        limit_s     : lifetime limit [s] (ESSB 5.4.2.3a.1)
        pc          : optional collision probability [-] of each outcome, dict as residual_s
        phase       : optional, time-varying solar activity. dict with
                      "start_months" (phase at t = 0 of each run, shape (n_runs,)),
                      "step_s" (length of one month [s]), "grid_months" (start phases of
                      the tables), "period_months" (None if not periodic). Then each value
                      of residual_s and pc is an array over grid_months, read at the phase
                      of the end of life: start_months[run] + t_end / step_s.
    Output: dict of arrays, shape (n_runs,):
        "n_objects"         : dead objects left in orbit
        "n_disposed"        : of which successfully disposed
        "pd_effective"      : n_disposed / n_objects (nan if no object)
        "frac_compliant"    : fraction of the objects with residual lifetime <= limit
        "obj_years_committed" : sum of the residual lifetimes [object-s]
        "obj_years_horizon" : part of it inside the simulated time span [object-s]
        (divide by the seconds in a year for object-years)
        "collisions"        : sum of the P_c of the dead objects (only if pc is given) [-]
    """
    n_runs = out["n_active"].shape[1]
    in_orbit, outcome = object_outcomes(out["end_kind"], p_manoeuvre, rng)
    run = out["end_run"][in_orbit]
    t_end_s = out["end_t_s"][in_orbit]
    life_s = object_values(residual_s, outcome, run, t_end_s, phase)

    n_objects = np.bincount(run, minlength=n_runs)
    n_disposed = np.bincount(run, weights=outcome == DISPOSED, minlength=n_runs)
    n_ok = np.bincount(run, weights=life_s <= limit_s, minlength=n_runs)
    with np.errstate(invalid="ignore", divide="ignore"):
        pd_eff = n_disposed / n_objects
        frac_ok = n_ok / n_objects
    result = {"n_objects": n_objects, "n_disposed": n_disposed, "pd_effective": pd_eff,
              "frac_compliant": frac_ok,
              "obj_years_committed": np.bincount(run, weights=life_s, minlength=n_runs),
              "obj_years_horizon": np.bincount(run, weights=np.minimum(life_s, horizon_s - t_end_s),
                                               minlength=n_runs)}
    if pc is not None:
        pc_obj = object_values(pc, outcome, run, t_end_s, phase)
        result["collisions"] = np.bincount(run, weights=pc_obj, minlength=n_runs)
    return result


def object_values(values, outcome, run, t_end_s, phase=None):
    """Value (residual lifetime or P_c) of each dead object from the table of its outcome.

    Inputs: dict {outcome: scalar} (constant activity) or {outcome: array over the phase
            grid} (with phase, see debris_summary); outcome, run and end time [s] of each object
    Output: array, one value per object
    """
    if phase is None:
        return np.array([values[FAILED], values[DISPOSAL_FAILED], values[DISPOSED]])[outcome]
    obj_phase = phase["start_months"][run] + t_end_s / phase["step_s"]
    result = np.empty(len(outcome))
    for kind in (FAILED, DISPOSAL_FAILED, DISPOSED):
        mask = outcome == kind
        result[mask] = values_at_phase(values[kind], obj_phase[mask], phase["grid_months"],
                                       phase["period_months"])
    return result


def required_manoeuvre_prob(pd_target, retirement_fraction):
    """Manoeuvre success probability giving an effective P_d = pd_target [-].

    P_d,eff = p_manoeuvre x f_ret, f_ret = fraction of the satellites that reach
    retirement (= R(design life) for independent satellites). Values > 1 mean the
    target cannot be met by the manoeuvre alone.
    """
    return pd_target / retirement_fraction
