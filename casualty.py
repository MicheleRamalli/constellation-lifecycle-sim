"""On-ground casualty risk of uncontrolled re-entries (v0.3).

Casualty area of one surviving fragment (NASA-STD-8719.14, debris casualty area;
same model in DRAMA SARA, reproduced to machine precision in tests/test_casualty.py):
    A_c = (sqrt(A_h) + sqrt(A_i))^2,   A_h = 0.36 m^2 (cross section of a standing person)
A_i = mean cross section of the fragment; for a tumbling convex body A_i = surface / 4
(Cauchy).

Expected casualties of one re-entry (1D projection, as SARA "circular" uncontrolled):
    E_c = A_c,total * rho_eff,   rho_eff = sum_j w_j rho_j
with rho_j the population density of latitude band j and w_j the fraction of time
the orbit spends in band j. For a circular orbit sin(phi) = sin(i) sin(u), with the
argument of latitude u uniform in time, so the fraction of time below latitude phi is
    F(phi) = 1/2 + arcsin(sin(phi) / sin(i)) / pi       (|phi| <= i, or 180 - i if i > 90)
and w_j = F(upper edge) - F(lower edge): no singularity at the turning latitude.
Longitude, Earth rotation and the ground dispersion of the fragments are not modelled
(as in SARA's 1D projection). The eccentricity is neglected (e = 0.001 in the SARA runs).

Constellation: N re-entries with the same E_c (independent events):
    expected casualties  = N E_c
    P(at least one)      = 1 - (1 - E_c)^N   (<= N E_c)

DRAMA files (SARA output, 1D population density) are read at run time from the user's
installation and projects, never copied into this repository.
"""

import json
import os

import numpy as np

import decay
import params


def casualty_area(cross_section_m2, human_area_m2=params.CASUALTY_HUMAN_AREA_M2):
    """Casualty area of one fragment [m^2]: (sqrt(A_h) + sqrt(A_i))^2.

    Inputs: mean cross section of the fragment A_i [m^2] (scalar or array),
            cross section of a person A_h [m^2]
    Output: casualty area [m^2]
    """
    return (np.sqrt(human_area_m2) + np.sqrt(cross_section_m2)) ** 2


def tumbling_cross_section(surface_m2):
    """Mean cross section of a tumbling convex body [m^2] = surface / 4 (Cauchy)."""
    return surface_m2 / 4.0


def read_sara_results(json_path):
    """Risk results of a DRAMA SARA run ("Reentry-risk" mode) from output/dmf_output.json.

    The SARA files (PySara.RiskResults.dat, PySara.Risk-Total.dat) are keys inside the
    JSON, not separate files.
    Inputs: path of dmf_output.json
    Output: dict with
        "casualty_area_m2" : total casualty area [m^2]
        "impact_mass_kg"   : total mass reaching the ground [kg]
        "ec"               : casualty probability, 1D projection [-]
        "fragments"        : list of dicts (name, mass_kg, cross_section_m2,
                             casualty_area_m2, ec, energy_j), one per fragment
        "lat_deg", "impact_prob" : latitude bins and impact probability of Risk-Total
                             (empty arrays if no fragment reaches the ground)
    """
    with open(json_path) as f:
        data = json.load(f)
    risk = decay.find_key(data, "PySara.RiskResults.dat")
    if risk is None:
        raise ValueError(f"No PySara.RiskResults.dat in {json_path}")
    fragments = [{"name": fr["Name"], "mass_kg": fr["Mass"],
                  "cross_section_m2": fr["Average geometric cross section"],
                  "casualty_area_m2": fr["Average casualty area"],
                  "ec": fr["Casualty probability 1D"], "energy_j": fr["Impact energy"]}
                 for fr in risk["Individual fragment risk of the run configuration"]]
    bins = decay.find_key(data, "PySara.Risk-Total.dat") or []
    return {"casualty_area_m2": risk["Total casualty area / m^2"],
            "impact_mass_kg": risk["Total impact mass / kg"],
            "ec": risk["Total casualty probability under the 1D projection"],
            "fragments": fragments,
            "lat_deg": np.array([b["latitude"] for b in bins]),
            "impact_prob": np.array([b["imp.probability"] for b in bins])}


def read_population_1d(path, year):
    """Population density vs latitude from a DRAMA SARA 1D file (UN WPP 2022).

    File: comment lines '#', a 'Latitude' line, a line with the years, then one row
    per latitude (step 1/120 deg, from +90 down), densities in people/km^2.
    Inputs: path of Population_Growth_1D_<SCENARIO>.csv, year (one of the columns)
    Output: (latitude [deg] = UPPER edge of each band, density [people/m^2]) arrays
    """
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#") and not line.startswith("Latitude"):
                rows.append([float(x) for x in line.split(",")])
    years = np.array(rows[0][1:])
    data = np.array(rows[1:])
    col = np.nonzero(years == year)[0]
    if len(col) == 0:
        raise ValueError(f"Year {year} not in {path}")
    return data[:, 0], data[:, 1 + col[0]] / 1e6   # people/km^2 -> people/m^2


def latitude_weights(lat_upper_deg, step_deg, incl_deg):
    """Fraction of time of a circular orbit in each latitude band [lat - step, lat].

    The band convention [lat - step, lat] is the one of SARA's Risk-Total output
    (checked bin by bin in the tests).
    Inputs: upper edges of the bands [deg], band width [deg], inclination [deg]
    Output: weights [-], same shape as lat_upper_deg (sum = 1 over all latitudes)
    """
    s = np.sin(np.radians(incl_deg))

    def frac_below(lat_deg):
        x = np.clip(np.sin(np.radians(lat_deg)) / s, -1.0, 1.0)
        return 0.5 + np.arcsin(x) / np.pi

    return frac_below(lat_upper_deg) - frac_below(lat_upper_deg - step_deg)


def effective_density(lat_upper_deg, density_m2, incl_deg):
    """Population density seen by an uncontrolled re-entry from a circular orbit [1/m^2].

    Inputs: latitude grid of read_population_1d (uniform step), density [1/m^2],
            inclination [deg]
    Output: rho_eff = sum of weight x density [1/m^2]
    """
    step_deg = abs(lat_upper_deg[0] - lat_upper_deg[1])
    return np.sum(latitude_weights(lat_upper_deg, step_deg, incl_deg) * density_m2)


def constellation_risk(n_reentries, ec):
    """Cumulative casualty risk of N re-entries with the same E_c.

    Inputs: number of re-entries (scalar or array, e.g. one per Monte Carlo run),
            E_c of one re-entry [-]
    Outputs: (expected casualties N E_c [-], probability of at least one casualty
              1 - (1 - E_c)^N [-])
    """
    n = np.asarray(n_reentries, dtype=float)
    return n * ec, -np.expm1(n * np.log1p(-ec))


def sara_output_path(folder, output_file):
    """Path of a SARA run output (read at run time, never copied); None if missing."""
    path = os.path.join(os.path.expanduser(folder), output_file)
    return path if os.path.exists(path) else None
