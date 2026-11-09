"""Orbital decay by atmospheric drag of a circular orbit.

Model (physical principle, energy balance):
    specific energy  eps = -mu / (2 a)
    drag power       d(eps)/dt = -D v = -(1/2) rho B v^3,   B = C_D A / m,  v = sqrt(mu / a)
    => da/dt = (2 a^2 / mu) d(eps)/dt = -rho B sqrt(mu a)
The orbit stays circular (drag acts along the velocity, eccentricity neglected).

Atmosphere: piecewise exponential (exponential atmosphere model of Vallado, Fundamentals
of Astrodynamics and Applications, 4th ed., 2013; section to be checked), rho(h) = rho_i exp(-(h - h_i) / H_i) for
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
"""

import csv

import numpy as np
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


def sma_rate_from_ndot(a_m, n_rad_s, ndot_rad_s2):
    """da/dt [m/s] from the mean-motion derivative (Kepler: n^2 a^3 = mu).

    dn/n = -(3/2) da/a  =>  da/dt = -(2/3) a ndot / n
    """
    return -2.0 / 3.0 * a_m * ndot_rad_s2 / n_rad_s
