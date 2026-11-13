"""Orbital decay by atmospheric drag of a circular orbit.

Model (physical principle, energy balance):
    specific energy  eps = -mu / (2 a)
    drag power       d(eps)/dt = -D v = -(1/2) rho B v^3,   B = C_D A / m,  v = sqrt(mu / a)
    => da/dt = (2 a^2 / mu) d(eps)/dt = -rho B sqrt(mu a)
The orbit stays circular (drag acts along the velocity, eccentricity neglected).

Atmosphere: piecewise exponential (exponential atmosphere model of Vallado, Fundamentals
of Astrodynamics and Applications, 4th ed., 2013), rho(h) = rho_i exp(-(h - h_i) / H_i) for
h_i <= h < h_(i+1). Here the scale height of each band comes from the densities at its
two nodes, H_i = (h_(i+1) - h_i) / ln(rho_i / rho_(i+1)), so rho is continuous and
passes exactly through the table values. The table values (low / moderate / high
solar activity, constant in time) go into params.py.

Inside one band the decay time has a closed form, so the lifetime is a sum over
bands, with no numerical integration:
    t = integral da / (rho B sqrt(mu a)),  with  integral e^(a/H) a^(-1/2) da
      = sqrt(pi H) erfi(sqrt(a/H)) = 2 sqrt(H) e^(a/H) D(sqrt(a/H))
where D is the Dawson function (scipy.special.dawsn). The form with D avoids the
overflow of erfi for large a/H (a/H ~ 100 in LEO).

Time-varying solar activity (decay_trajectory): F10.7(t) is constant inside each
month (Table A-1 of ECSS-E-ST-10-04C Rev.1 gives 30-day averages), the density
table of the month is interpolated between the three ECSS tables in ln(rho) vs
F10.7 (density_at_f107), and the closed form above is applied month by month.
With a constant F10.7 this is exactly the constant-activity lifetime.
"""

import csv
import json

import numpy as np
from scipy.optimize import brentq
from scipy.special import dawsn

import params

MU = params.MU_EARTH
R_E = params.R_EARTH


def ballistic_coefficient(drag_coefficient, area_m2, mass_kg):
    """B = C_D A / m [m^2/kg]."""
    return drag_coefficient * area_m2 / mass_kg


def box_mean_area(lx_m, ly_m, lz_m):
    """Mean cross-section of a randomly tumbling box [m^2].

    Cauchy's theorem: for a convex body, the cross-section averaged over all
    directions is one quarter of its surface.
    """
    return 2.0 * (lx_m * ly_m + ly_m * lz_m + lx_m * lz_m) / 4.0


def decay_rate(a_m, rho_kg_m3, b_m2_kg):
    """da/dt of a circular orbit [m/s] (negative): -rho B sqrt(mu a)."""
    return -rho_kg_m3 * b_m2_kg * np.sqrt(MU * a_m)


def scale_heights(alt_nodes_m, rho_nodes_kg_m3):
    """Scale height of each band between consecutive nodes [m], shape (n_nodes - 1,)."""
    alt = np.asarray(alt_nodes_m, dtype=float)
    rho = np.asarray(rho_nodes_kg_m3, dtype=float)
    return np.diff(alt) / np.log(rho[:-1] / rho[1:])


def density(alt_m, alt_nodes_m, rho_nodes_kg_m3):
    """Piecewise exponential density [kg/m^3] at altitude alt_m [m] (scalar or array).

    No extrapolation: an altitude outside [first node, last node] raises ValueError.
    """
    alt = np.asarray(alt_m, dtype=float)
    nodes = np.asarray(alt_nodes_m, dtype=float)
    rho = np.asarray(rho_nodes_kg_m3, dtype=float)
    if np.any(alt < nodes[0]) or np.any(alt > nodes[-1]):
        raise ValueError("Altitude outside the density table (no extrapolation)")
    i = np.clip(np.searchsorted(nodes, alt, side="right") - 1, 0, len(nodes) - 2)
    return rho[i] * np.exp(-(alt - nodes[i]) / scale_heights(nodes, rho)[i])


def time_single_scale(a_start_m, a_end_m, rho_ref_kg_m3, a_ref_m, h_scale_m, b_m2_kg):
    """Time to decay from a_start to a_end (a_end < a_start) in one exponential band [s].

    rho(a) = rho_ref exp(-(a - a_ref) / H). Closed form (module docstring):
        t = 2 sqrt(H) / (B sqrt(mu) rho_ref) [g(a_start) - g(a_end)],
        g(a) = e^((a - a_ref)/H) D(sqrt(a/H))
    """
    def g(a_m):
        return np.exp((a_m - a_ref_m) / h_scale_m) * dawsn(np.sqrt(a_m / h_scale_m))

    return (2.0 * np.sqrt(h_scale_m) / (b_m2_kg * np.sqrt(MU) * rho_ref_kg_m3)
            * (g(a_start_m) - g(a_end_m)))


def lifetime(alt_start_m, alt_end_m, alt_nodes_m, rho_nodes_kg_m3, b_m2_kg):
    """Time to decay from alt_start to alt_end through the piecewise atmosphere [s].

    Sum of the closed-form time of each band crossed. Both altitudes must be
    inside the table (no extrapolation).
    """
    nodes = np.asarray(alt_nodes_m, dtype=float)
    rho = np.asarray(rho_nodes_kg_m3, dtype=float)
    if not nodes[0] <= alt_end_m <= alt_start_m <= nodes[-1]:
        raise ValueError("Need first node <= alt_end <= alt_start <= last node")
    h = scale_heights(nodes, rho)
    total_s = 0.0
    for i in range(len(nodes) - 1):
        top = min(alt_start_m, nodes[i + 1])     # part of band i that is crossed
        bottom = max(alt_end_m, nodes[i])
        if top > bottom:
            total_s += time_single_scale(R_E + top, R_E + bottom, rho[i], R_E + nodes[i],
                                         h[i], b_m2_kg)
    return total_s


def read_density_table(path, model):
    """Tabulated densities of one model (params.ATM_DENSITY_CSV, ECSS-E-ST-10-04C Rev.1 Annex G).

    Inputs: CSV path; model "nrlmsise" or "jb2006" (column prefix rho_<model>_<level>)
    Output: altitude nodes [m] (increasing), dict {level: densities [kg/m^3]} for
            params.SOLAR_ACTIVITY_LEVELS; rows where the model has no value are dropped
            (JB-2006 starts at 100 km).
    """
    with open(path) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    columns = [f"rho_{model}_{level}" for level in params.SOLAR_ACTIVITY_LEVELS]
    rows = [r for r in rows if all(r[c].strip() for c in columns)]   # skip empty cells
    alt_m = np.array([float(r["alt_km"]) for r in rows]) * 1e3
    return alt_m, {level: np.array([float(r[c]) for r in rows])
                   for level, c in zip(params.SOLAR_ACTIVITY_LEVELS, columns)}


# ---------------------------------------------------------------------------
# Time-varying solar activity (F10.7 series, one value per month)
# ---------------------------------------------------------------------------

def read_solar_cycle(path, column):
    """One column of ECSS-E-ST-10-04C Rev.1 Table A-1 (params.SOLAR_CYCLE_CSV).

    Inputs: CSV path; column name, e.g. "F81mean" or "F10mean"
    Output: index value of each 30-day month of solar cycle 23 [sfu], shape (140,)
    """
    with open(path) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    months = [int(r["month"]) for r in rows]
    if months != list(range(len(rows))):
        raise ValueError("Months of Table A-1 not in sequence 0, 1, 2, ...")
    return np.array([float(r[column]) for r in rows])


def read_oscar_forecast(path):
    """Monthly F10.7 of the DRAMA/OSCAR long-term forecast file fap_mon.dat.

    Column 6 of the file = predicted 13-month centred mean of F10.7 (equal to the
    observed smoothed value before the forecast starts); see the file header.
    Inputs: path of fap_mon.dat
    Output: (year, month, F10.7 [sfu]) arrays, one entry per calendar month, in order
    """
    data = np.loadtxt(path, comments="#")
    month, year, f107 = data[:, 0].astype(int), data[:, 1].astype(int), data[:, 5]
    if np.any(np.diff(year * 12 + month) != 1):
        raise ValueError("fap_mon.dat: months not consecutive")
    return year, month, f107


def read_oscar_activity(json_path):
    """Daily solar activity used by an OSCAR run (PyOscar.osg in output/dmf_output.json).

    Inputs: path of dmf_output.json
    Output: dict "date" (numpy datetime64[D]), "F10", "F3M" (81-day mean), "Ap" arrays
    """
    with open(json_path) as f:
        data = json.load(f)
    osg = find_key(data, "PyOscar.osg")
    if osg is None:
        raise ValueError(f"No PyOscar.osg in {json_path}")
    day, month, year = np.array([r["date"].split("/") for r in osg]).T
    dates = np.array([f"{y}-{m}-{d}" for d, m, y in zip(day, month, year)], dtype="datetime64[D]")
    if np.any(np.diff(dates) != np.timedelta64(1, "D")):
        raise ValueError("PyOscar.osg: days not consecutive")
    return {"date": dates, **{k: np.array([float(r[k]) for r in osg]) for k in ("F10", "F3M", "Ap")}}


def find_key(obj, key):
    """First value of `key` in a nested dict/list (JSON), or None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        children = obj.values()
    elif isinstance(obj, list):
        children = obj
    else:
        return None
    for child in children:
        found = find_key(child, key)
        if found is not None:
            return found
    return None


def monthly_mean(dates, values):
    """Mean of a daily series over each calendar month.

    Inputs: dates (datetime64[D], consecutive), values
    Output: (first month as datetime64[M], array of monthly means); the first and last
            months may be partial (mean of the days available)
    """
    months = dates.astype("datetime64[M]")
    keys = np.arange(months[0], months[-1] + np.timedelta64(1, "M"))
    return months[0], np.array([values[months == k].mean() for k in keys])


def density_at_f107(f107, f107_nodes, rho_levels, interp="log"):
    """Density table [kg/m^3] at the altitude nodes for one F10.7 value.

    Interpolation between the three ECSS tables (low / moderate / high activity):
        "log"    : ln(rho) linear in F10.7 (reference)
        "linear" : rho linear in F10.7 (sensitivity)
    At the nodes F10.7 = 65, 140, 250 the table itself is returned. No extrapolation:
    F10.7 outside [f107_nodes[0], f107_nodes[-1]] raises ValueError.
    Inputs: F10.7 [sfu], node values of F10.7 [sfu] (increasing), rho_levels array
            (n_levels, n_alt) [kg/m^3], interpolation kind
    Output: densities at the altitude nodes, shape (n_alt,)
    """
    f_nodes = np.asarray(f107_nodes, dtype=float)
    rho = np.asarray(rho_levels, dtype=float)
    if not f_nodes[0] <= f107 <= f_nodes[-1]:
        raise ValueError(f"F10.7 = {f107} outside [{f_nodes[0]}, {f_nodes[-1]}] (no extrapolation)")
    i = min(np.searchsorted(f_nodes, f107, side="right") - 1, len(f_nodes) - 2)
    w = (f107 - f_nodes[i]) / (f_nodes[i + 1] - f_nodes[i])     # weight of the upper node, 0..1
    if interp == "log":
        return np.exp((1.0 - w) * np.log(rho[i]) + w * np.log(rho[i + 1]))
    if interp == "linear":
        return (1.0 - w) * rho[i] + w * rho[i + 1]
    raise ValueError("interp must be 'log' or 'linear'")


def decay_trajectory(alt_start_m, alt_end_m, alt_nodes_m, f107_nodes, rho_levels, b_m2_kg,
                     f107_series, step_s, start_index, periodic, interp="log", stop_at_end=False):
    """Decay with a time-varying solar activity, constant inside each step (one month).

    Inside step k the activity F10.7 = f107_series[k] is constant, so the closed form
    of lifetime() holds: the altitude at the end of the step is the root h of
        lifetime(h_k, h, rho(F_k)) = step length     (scipy brentq, like fzero)
    The decay ends inside the first step whose remaining lifetime is shorter than
    the step. With a constant series this gives lifetime() exactly (time is additive).

    Inputs:
        alt_start_m, alt_end_m : initial and final altitude [m]
        alt_nodes_m, rho_levels: density table (n_levels, n_alt), levels at f107_nodes
        b_m2_kg                : ballistic coefficient [m^2/kg]
        f107_series            : F10.7 of each step [sfu]
        step_s                 : length of one step [s]
        start_index            : index (may be fractional) of the series at the start
        periodic               : True -> the series repeats (index modulo its length);
                                 False -> running past its end raises ValueError, or
                                 stops there if stop_at_end (no extrapolation)
        interp                 : "log" or "linear" (density_at_f107)
    Output: dict "t_s" (times of the step boundaries crossed + final time, starting at 0),
            "alt_m" (altitude at those times), "f107" (activity of each segment);
            len(f107) = len(t_s) - 1; "reentered" False if stopped at the end of the
            series (then t_s[-1] is only a lower bound of the lifetime)
    """
    series = np.asarray(f107_series, dtype=float)
    k = int(np.floor(start_index))
    dt_s = (k + 1 - start_index) * step_s       # first step may be partial
    t_list, alt_list, f_list = [0.0], [alt_start_m], []
    alt_m = alt_start_m
    while True:
        if periodic:
            f107 = series[k % len(series)]
        elif k < len(series):
            f107 = series[k]
        elif stop_at_end:
            return {"t_s": np.array(t_list), "alt_m": np.array(alt_list), "f107": np.array(f_list),
                    "reentered": False}
        else:
            raise ValueError("Solar activity series ended before the re-entry")
        rho = density_at_f107(f107, f107_nodes, rho_levels, interp)
        t_rest_s = lifetime(alt_m, alt_end_m, alt_nodes_m, rho, b_m2_kg)
        f_list.append(f107)
        if t_rest_s <= dt_s:
            t_list.append(t_list[-1] + t_rest_s)
            alt_list.append(alt_end_m)
            break
        alt_m = brentq(lambda h: lifetime(alt_m, h, alt_nodes_m, rho, b_m2_kg) - dt_s,
                       alt_end_m, alt_m, xtol=1e-6, rtol=1e-14)
        t_list.append(t_list[-1] + dt_s)
        alt_list.append(alt_m)
        k += 1
        dt_s = step_s
    return {"t_s": np.array(t_list), "alt_m": np.array(alt_list), "f107": np.array(f_list),
            "reentered": True}


def lifetime_variable(alt_start_m, alt_end_m, alt_nodes_m, f107_nodes, rho_levels, b_m2_kg,
                      f107_series, step_s, start_index, periodic, interp="log"):
    """Lifetime [s] with time-varying solar activity: last time of decay_trajectory."""
    return decay_trajectory(alt_start_m, alt_end_m, alt_nodes_m, f107_nodes, rho_levels, b_m2_kg,
                            f107_series, step_s, start_index, periodic, interp)["t_s"][-1]


# ---------------------------------------------------------------------------
# Disposal manoeuvre (only to check the OSCAR reference runs)
# ---------------------------------------------------------------------------

def hohmann_delta_v(r1_m, r2_m):
    """Total delta-v [m/s] of a Hohmann transfer between circular orbits of radius r1, r2 [m].

    Two impulses: v_circ -> v_transfer at r1, v_transfer -> v_circ at r2 (vis-viva).
    """
    a_t = 0.5 * (r1_m + r2_m)
    dv1 = abs(np.sqrt(MU * (2.0 / r1_m - 1.0 / a_t)) - np.sqrt(MU / r1_m))
    dv2 = abs(np.sqrt(MU / r2_m) - np.sqrt(MU * (2.0 / r2_m - 1.0 / a_t)))
    return dv1 + dv2


def propellant_mass(delta_v_m_s, isp_s, mass_kg, mass_is_final=True):
    """Propellant [kg] for a delta-v (Tsiolkovsky rocket equation), g0 = params.G0.

    mass_is_final=True : mass_kg is the mass AFTER the manoeuvre: m_p = m (e^(dv/(Isp g0)) - 1)
    mass_is_final=False: mass_kg is the mass BEFORE:             m_p = m (1 - e^(-dv/(Isp g0)))
    """
    x = delta_v_m_s / (isp_s * params.G0)
    return mass_kg * np.expm1(x) if mass_is_final else -mass_kg * np.expm1(-x)


def sma_rate_from_ndot(a_m, n_rad_s, ndot_rad_s2):
    """da/dt [m/s] from the mean-motion derivative (Kepler: n^2 a^3 = mu).

    dn/n = -(3/2) da/a  =>  da/dt = -(2/3) a ndot / n
    """
    return -2.0 / 3.0 * a_m * ndot_rad_s2 / n_rad_s
