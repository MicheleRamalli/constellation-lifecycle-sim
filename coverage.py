"""Coverage of a Walker constellation in sun-synchronous orbit: revisit time.

Model:
- Circular orbits, two-body + J2 secular rates of RAAN and argument of
  latitude u = omega + M (Vallado, Fundamentals of Astrodynamics and
  Applications, 4th ed. 2013; to be checked on the reference).
- Spherical Earth rotating at OMEGA_EARTH; Greenwich angle 0 at t = 0.
- A ground point is accessed when its angular distance from the sub-satellite
  point is <= lambda, the Earth central angle of the sensor footprint
  (Wertz & Larson, Space Mission Analysis and Design, 3rd ed., space mission
  geometry; to be checked on the reference).
- Revisit (gap) = time from the end of one useful access to the start of the
  next. Entry/exit times are interpolated where the angular distance crosses
  lambda (parabola through three samples). A useful access is a continuous
  access of ONE satellite lasting at least MIN_ACCESS_S; the useful accesses of
  all satellites are then merged.
- Geometric coverage only: solar illumination not modelled (ASSUMPTION).

Walker notation (Walker J.G. 1984, "Satellite constellations", JBIS 37; to be
checked on the reference): T = P * S satellites, P planes with RAAN spaced by
raan_spread / P, S satellites per plane spaced by 360/S deg, phase offset
F * 360 / T deg between adjacent planes.

All angles in rad and times in s inside the functions.
"""

import itertools
import os

import numpy as np

import params

MU = params.MU_EARTH
R_E = params.R_EARTH
OMEGA_E = params.OMEGA_EARTH
J2 = params.J2_EARTH


# ---------------------------------------------------------------------------
# Orbit
# ---------------------------------------------------------------------------

def mean_motion(a_m):
    """Keplerian mean motion n = sqrt(mu / a^3) [rad/s] for semi-major axis a [m]."""
    return np.sqrt(MU / a_m**3)


def j2_secular_rates(a_m, inc_rad):
    """Secular J2 rates of a circular orbit.

    Omega_dot = -(3/2) n J2 (R/a)^2 cos i
    u_dot     = M_dot + omega_dot = n [1 + (3/2) J2 (R/a)^2 (4 cos^2 i - 1)]
    (sum of omega_dot = (3/4) n J2 (R/a)^2 (5 cos^2 i - 1) and
     M_dot = n [1 + (3/4) J2 (R/a)^2 (3 cos^2 i - 1)] for e = 0).

    Inputs:
        a_m     : semi-major axis [m]
        inc_rad : inclination [rad]
    Outputs:
        raan_dot : RAAN rate [rad/s]
        u_dot    : argument-of-latitude rate [rad/s]
    """
    n = mean_motion(a_m)
    k = 1.5 * J2 * (R_E / a_m)**2
    raan_dot = -k * n * np.cos(inc_rad)
    u_dot = n * (1.0 + k * (4.0 * np.cos(inc_rad)**2 - 1.0))
    return raan_dot, u_dot


def sso_inclination(a_m):
    """Sun-synchronous inclination [rad] of a circular orbit of semi-major axis a [m].

    Condition: Omega_dot = 2 pi / tropical year, solved for cos i.
    """
    raan_dot_sso = 2.0 * np.pi / (params.TROPICAL_YEAR_DAYS * params.SECONDS_PER_DAY)
    cos_i = -raan_dot_sso / (1.5 * mean_motion(a_m) * J2 * (R_E / a_m)**2)
    return np.arccos(cos_i)


def nodal_period(a_m, inc_rad):
    """Nodal period 2 pi / u_dot [s] (time between two ascending node crossings)."""
    return 2.0 * np.pi / j2_secular_rates(a_m, inc_rad)[1]


# ---------------------------------------------------------------------------
# Footprint geometry (spherical Earth)
# ---------------------------------------------------------------------------

def earth_angular_radius(alt_m):
    """Angular radius rho of the Earth seen from altitude h: sin rho = R / (R + h) [rad]."""
    return np.arcsin(R_E / (R_E + alt_m))


def footprint_from_elevation(alt_m, eps_rad):
    """Footprint for a minimum elevation angle eps.

    sin eta = cos eps sin rho (nadir angle), lambda = pi/2 - eta - eps.
    Outputs: lambda (Earth central angle) [rad], eta (nadir angle) [rad]
    """
    eta = np.arcsin(np.cos(eps_rad) * np.sin(earth_angular_radius(alt_m)))
    return np.pi / 2 - eta - eps_rad, eta


def footprint_from_sensor(alt_m, eta_rad):
    """Footprint for a sensor half-angle eta (max off-nadir).

    Same relation solved for eps: cos eps = sin eta / sin rho; lambda = pi/2 - eta - eps.
    Outputs: lambda (Earth central angle) [rad], eps (elevation at the edge) [rad]
    """
    sin_rho = np.sin(earth_angular_radius(alt_m))
    if np.sin(eta_rad) > sin_rho:
        raise ValueError("Sensor half-angle beyond the Earth horizon")
    eps = np.arccos(np.sin(eta_rad) / sin_rho)
    return np.pi / 2 - eta_rad - eps, eps


# ---------------------------------------------------------------------------
# Constellation and propagation
# ---------------------------------------------------------------------------

def walker_elements(n_planes, sats_per_plane, phasing, raan_spread_rad):
    """Initial RAAN and argument of latitude of a Walker P/S/F constellation.

    Inputs:
        n_planes        : P [-]
        sats_per_plane  : S [-]
        phasing         : F [-], 0 <= F < P
        raan_spread_rad : RAAN spread of the planes [rad] (2 pi delta, pi star)
    Outputs:
        raan0, u0 : [rad], shape (P, S)
    """
    p = np.arange(n_planes)[:, None]
    s = np.arange(sats_per_plane)[None, :]
    n_total = n_planes * sats_per_plane
    raan0 = np.broadcast_to(p * raan_spread_rad / n_planes, (n_planes, sats_per_plane)).copy()
    u0 = 2.0 * np.pi * s / sats_per_plane + 2.0 * np.pi * phasing * p / n_total
    return raan0, u0


def satellite_directions(t_s, raan0, u0, a_m, inc_rad):
    """Unit vectors of the satellites in the Earth-fixed frame.

    r_ECI = [cos W cos u - sin W sin u cos i, sin W cos u + cos W sin u cos i, sin u sin i],
    then rotation by -OMEGA_E t about z (Greenwich angle 0 at t = 0).

    Inputs:
        t_s           : times [s], shape (n_t,)
        raan0, u0     : initial RAAN and argument of latitude [rad], shape (n_sat,)
        a_m, inc_rad  : semi-major axis [m], inclination [rad] (same for all)
    Output:
        unit vectors, shape (n_t, n_sat, 3)
    """
    raan_dot, u_dot = j2_secular_rates(a_m, inc_rad)
    t = np.asarray(t_s)[:, None]
    raan = raan0[None, :] + raan_dot * t - OMEGA_E * t  # RAAN measured from Greenwich
    u = u0[None, :] + u_dot * t
    ci, si = np.cos(inc_rad), np.sin(inc_rad)
    x = np.cos(raan) * np.cos(u) - np.sin(raan) * np.sin(u) * ci
    y = np.sin(raan) * np.cos(u) + np.cos(raan) * np.sin(u) * ci
    z = np.sin(u) * si * np.ones_like(raan)
    return np.stack([x, y, z], axis=-1)


def grid_directions(lat_deg, lon_deg):
    """Unit vectors of ground points on a lat x lon grid.

    Inputs: latitudes and longitudes [deg], 1-D arrays
    Outputs: unit vectors shape (n_lat * n_lon, 3), and latitude of each point [deg]
    """
    lat, lon = np.meshgrid(np.radians(lat_deg), np.radians(lon_deg), indexing="ij")
    vectors = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
                       axis=-1).reshape(-1, 3)
    return vectors, np.degrees(lat).ravel()


def access_margin(t_s, raan0, u0, a_m, inc_rad, lambda_rad, points, chunk=500):
    """Access margin of each ground point at each time.

    m(t) = max over the satellites of [cos(angular distance to the sub-satellite
    point) - cos(lambda)]. The point is accessed when m >= 0. m is continuous in
    time, so the entry and exit times can be interpolated where it crosses zero.

    Inputs:
        t_s, raan0, u0, a_m, inc_rad : as satellite_directions
        lambda_rad : Earth central angle of the footprint [rad]
        points     : ground unit vectors, shape (n_p, 3)
        chunk      : time steps per block (memory control)
    Output:
        margin [-], float32 array, shape (n_t, n_p)
    """
    cos_lambda = np.cos(lambda_rad)
    margin = np.zeros((len(t_s), len(points)), dtype=np.float32)
    for start in range(0, len(t_s), chunk):
        sat = satellite_directions(t_s[start:start + chunk], raan0, u0, a_m, inc_rad)
        cos_angle = sat @ points.T                          # (n_c, n_sat, n_p)
        margin[start:start + chunk] = np.max(cos_angle, axis=1) - cos_lambda
    return margin


# ---------------------------------------------------------------------------
# Revisit
# ---------------------------------------------------------------------------

def crossing_time(m, k):
    """Zero crossing of the series m between steps k and k + 1, in units of steps.

    m[k] and m[k + 1] have opposite signs. A parabola is fitted through three
    samples (k - 1, k, k + 1, or k, k + 1, k + 2 at the start of the series) and
    its root inside [k, k + 1] is returned: near a short access the margin is
    curved (peak of the pass) and a straight line would underestimate the access
    duration. Falls back to linear interpolation if fewer than three samples
    exist or the parabola has no root in the interval.
    Inputs:
        m : series, shape (n,)
        k : step index (int or integer array)
    Output:
        crossing position [steps], same shape as k
    """
    scalar = np.isscalar(k)
    k = np.atleast_1d(k)
    linear = k + m[k] / (m[k] - m[k + 1])
    x0 = np.where(k >= 1, k - 1, k)
    valid = x0 + 2 < len(m)
    x0 = np.minimum(x0, len(m) - 3)
    y0, y1, y2 = m[x0], m[x0 + 1], m[x0 + 2]
    c = 0.5 * (y2 - 2.0 * y1 + y0)          # p(x) = y0 + b x + c x^2, x = step - x0
    b = y1 - y0 - c
    disc = b * b - 4.0 * c * y0
    usable = valid & (np.abs(c) > 1e-15) & (disc >= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = np.sqrt(np.maximum(disc, 0.0))
        x1 = x0 + (-b - sq) / (2.0 * c)
        x2 = x0 + (-b + sq) / (2.0 * c)
    ok1 = usable & (x1 >= k) & (x1 <= k + 1)
    ok2 = usable & (x2 >= k) & (x2 <= k + 1)
    result = np.where(ok1, x1, np.where(ok2, x2, linear))
    return result[0] if scalar else result


def access_intervals(margin_column, dt_s, min_access_s=0.0):
    """Access intervals of one ground point by ONE satellite (or one margin series).

    An access is a run of consecutive steps with margin >= 0; entry and exit
    times are interpolated where the margin crosses zero (crossing_time).
    Accesses shorter than min_access_s are discarded. An access in progress at
    the first (last) step is truncated: its entry (exit) is set to that step.
    For a boolean series (no margin values) the crossings are at mid-step.
    Inputs:
        margin_column : margin series (see access_margin) or boolean series, shape (n_t,)
        dt_s          : time step [s]
        min_access_s  : minimum useful access duration [s]
    Outputs:
        entry_s, exit_s : interval bounds [s]
        truncated       : True if the entry is the first step (true entry unknown)
    """
    dt_s = float(dt_s)  # an integer dt would make the time arrays integer and truncate them
    is_bool = margin_column.dtype == bool
    m = np.where(margin_column, 1.0, -1.0) if is_bool else np.asarray(margin_column, float)
    n = len(m)
    inside = np.concatenate([[0], (m >= 0).astype(np.int8), [0]])
    d = np.diff(inside)
    first_in = np.nonzero(d == 1)[0]    # first step of each access
    first_out = np.nonzero(d == -1)[0]  # first step after each access

    def crossings(k):
        if is_bool:
            return k + 0.5
        return crossing_time(m, k)

    entry_s = first_in * dt_s
    has_entry = first_in > 0
    entry_s[has_entry] = crossings(first_in[has_entry] - 1) * dt_s
    exit_s = (first_out - 1) * dt_s
    has_exit = first_out < n
    exit_s[has_exit] = crossings(first_out[has_exit] - 1) * dt_s

    keep = exit_s - entry_s >= min_access_s
    return entry_s[keep], exit_s[keep], ~has_entry[keep]


def merge_intervals(entry_s, exit_s, truncated):
    """Union of access intervals (e.g. of several satellites) of one ground point.

    Overlapping or touching intervals are merged. A merged interval is
    truncated if its first interval is.
    Outputs: merged entry_s, exit_s, truncated (sorted by entry)
    """
    if len(entry_s) == 0:
        return entry_s, exit_s, truncated
    order = np.argsort(entry_s, kind="stable")
    e, x, tr = entry_s[order], exit_s[order], truncated[order]
    running_exit = np.maximum.accumulate(x)
    new_group = np.concatenate([[True], e[1:] > running_exit[:-1]])
    first = np.nonzero(new_group)[0]
    return e[first], np.maximum.reduceat(x, first), tr[first]


def starts_and_gaps(entry_s, exit_s, truncated):
    """Start times (truncated excluded) and complete gaps of sorted, disjoint intervals [s]."""
    return entry_s[~truncated], entry_s[1:] - exit_s[:-1]


def access_starts_and_gaps(margin_column, dt_s, min_access_s=0.0):
    """Access start times and complete gaps of one margin (or boolean) series.

    Convenience for one series: access_intervals, then starts_and_gaps.
    A gap goes from the exit of one useful access to the entry of the next;
    gaps before the first and after the last access are ignored.
    Outputs: starts_s [s], gaps_s [s]
    """
    return starts_and_gaps(*access_intervals(margin_column, dt_s, min_access_s))


def aggregate_revisit(starts_per_point, gaps_per_point):
    """Max and mean revisit over the points of one latitude.

    max  = largest complete gap over the points (inf if a point has < 2 accesses)
    mean = total gap time / number of gaps, pooled over the points
    Outputs: max_s, mean_s [s]
    """
    worst = 0.0
    for starts, gaps in zip(starts_per_point, gaps_per_point):
        if len(starts) < 2:
            worst = np.inf
        elif len(gaps) > 0:
            worst = max(worst, gaps.max())
    pooled = np.concatenate(gaps_per_point)
    return worst, (pooled.mean() if len(pooled) > 0 else np.inf)


def constellation_revisit(raan0, u0, lat_deg, eta_rad, step_s, duration_s,
                          lon_step_deg=params.COV_LON_STEP_DEG,
                          min_access_s=params.MIN_ACCESS_S, a_m=None, inc_rad=None):
    """Max and mean revisit vs latitude for a set of satellites.

    By default all satellites are on the reference SSO (ALT_KM, J2 sun-synchronous
    inclination). a_m and inc_rad can be given per satellite (real constellation):
    each satellite then has its own J2 rates and its own footprint (lambda depends
    on the altitude).

    Useful access = a continuous access of ONE satellite lasting at least
    min_access_s (the sensor of one satellite must point and acquire). The
    accesses of each satellite are filtered first, then merged over the
    satellites (merge_intervals); the gaps are measured on the merged intervals.

    Inputs:
        raan0, u0    : initial RAAN and argument of latitude [rad], shape (n_sat,)
        lat_deg      : latitudes of the grid [deg]
        eta_rad      : sensor half-angle [rad]
        step_s       : time step [s]
        duration_s   : propagation duration [s]
        lon_step_deg : longitude step [deg]
        min_access_s : minimum useful access duration [s]
        a_m, inc_rad : semi-major axis [m] and inclination [rad], scalars or shape (n_sat,);
                       default: reference SSO
    Outputs: lats [deg], max_revisit_s, mean_revisit_s
    """
    t_s = np.arange(0.0, duration_s + step_s / 2, step_s)
    lats = np.atleast_1d(lat_deg).astype(float)
    max_s = np.full(len(lats), np.inf)
    mean_s = np.full(len(lats), np.inf)
    raan0, u0 = np.ravel(raan0), np.ravel(u0)
    if len(raan0) == 0:
        return lats, max_s, mean_s
    if a_m is None:
        a_m = R_E + params.ALT_KM * 1e3
    if inc_rad is None:
        inc_rad = sso_inclination(a_m)
    a_sat = np.broadcast_to(a_m, raan0.shape)
    inc_sat = np.broadcast_to(inc_rad, raan0.shape)
    lam_sat = [footprint_from_sensor(a - R_E, eta_rad)[0] for a in a_sat]
    lons = np.arange(0.0, 360.0, lon_step_deg)
    block = max(1, int(2e7 // len(t_s)))  # longitudes per block (memory control)
    for i, lat in enumerate(lats):
        starts_list, gaps_list = [], []
        for b0 in range(0, len(lons), block):
            points, _ = grid_directions(np.array([lat]), lons[b0:b0 + block])
            intervals = [[] for _ in range(len(points))]
            for j in range(len(raan0)):   # one satellite at a time
                margin = access_margin(t_s, raan0[j:j + 1], u0[j:j + 1], a_sat[j], inc_sat[j],
                                       lam_sat[j], points)
                for p in range(len(points)):
                    intervals[p].append(access_intervals(margin[:, p], step_s, min_access_s))
            for p in range(len(points)):
                entry = np.concatenate([iv[0] for iv in intervals[p]])
                exit_ = np.concatenate([iv[1] for iv in intervals[p]])
                trunc = np.concatenate([iv[2] for iv in intervals[p]])
                starts, gaps = starts_and_gaps(*merge_intervals(entry, exit_, trunc))
                starts_list.append(starts)
                gaps_list.append(gaps)
        max_s[i], mean_s[i] = aggregate_revisit(starts_list, gaps_list)
    return lats, max_s, mean_s


# ---------------------------------------------------------------------------
# Walker search and lookup table
# ---------------------------------------------------------------------------

def walker_search(eta_rad, step_s, duration_s, p_range, s_range, raan_spread_rad,
                  lat_deg=params.REVISIT_LAT_DEG, max_revisit_s=params.REVISIT_MAX_HOURS * 3600,
                  min_access_s=params.MIN_ACCESS_S):
    """Smallest Walker P/S/F meeting the max-revisit requirement at one latitude.

    Explores P in p_range, S in s_range, F in 0..P-1. Picks the smallest total
    T = P*S; ties are broken by the smallest max revisit (then fewer planes).
    The configurations are evaluated by increasing T, and the search stops at the
    first T with a feasible configuration (same result, much faster).
    Output: (best (P, S, F) or None, list of rows (P, S, F, max_s, mean_s) evaluated)
    """
    configs = [(n_planes, sats, phasing)
               for n_planes in range(p_range[0], p_range[1] + 1)
               for sats in range(s_range[0], s_range[1] + 1)
               for phasing in range(n_planes)]
    rows = []
    for total in sorted({c[0] * c[1] for c in configs}):
        group = []
        for n_planes, sats, phasing in [c for c in configs if c[0] * c[1] == total]:
            raan0, u0 = walker_elements(n_planes, sats, phasing, raan_spread_rad)
            _, max_s, mean_s = constellation_revisit(raan0, u0, lat_deg, eta_rad, step_s,
                                                     duration_s, min_access_s=min_access_s)
            group.append((n_planes, sats, phasing, max_s[0], mean_s[0]))
        rows.extend(group)
        feasible = [r for r in group if r[3] <= max_revisit_s]
        if feasible:
            best = min(feasible, key=lambda r: (r[3], r[0]))
            return best[:3], rows
    return None, rows


def rephased_constellation(raan0, u0, n_per_plane):
    """Elements with n_per_plane[p] satellites in plane p, re-spaced evenly (R1).

    ASSUMPTION: rephasing is instantaneous (in reality an in-plane drift takes weeks).
    Inputs:
        raan0, u0   : Walker elements, shape (P, S)
        n_per_plane : satellites left in each plane, length P
    Outputs: raan0, u0 as flat arrays of the remaining satellites
    """
    raan_list, u_list = [], []
    for p, n in enumerate(n_per_plane):
        raan_list.append(np.full(n, raan0[p, 0]))
        u_list.append(u0[p, 0] + 2.0 * np.pi * np.arange(n) / max(n, 1))
    return np.concatenate(raan_list), np.concatenate(u_list)


def revisit_table_planes(n_planes, sats_per_plane, phasing, eta_rad, step_s, duration_s,
                         raan_spread_rad, lat_deg=params.REVISIT_LAT_DEG):
    """Full lookup table revisit(n_1, ..., n_P) of the reference Walker, R1 rephasing.

    One entry for every combination of operational satellites per plane,
    n_p = 0..S: (S + 1)^P configurations (25 for 2/4).
    Outputs: max_s, mean_s [s], arrays of shape (S + 1,) * P (inf if never revisited)
    """
    raan0, u0 = walker_elements(n_planes, sats_per_plane, phasing, raan_spread_rad)
    a_m = R_E + params.ALT_KM * 1e3
    return revisit_table_from_planes(raan0[:, 0], u0[:, 0], np.full(n_planes, a_m),
                                     np.full(n_planes, sso_inclination(a_m)), sats_per_plane,
                                     eta_rad, step_s, duration_s, lat_deg)


def revisit_table_from_planes(plane_raan, plane_u0, plane_a, plane_inc, sats_per_plane,
                              eta_rad, step_s, duration_s, lat_deg=params.REVISIT_LAT_DEG):
    """Lookup table revisit(n_1, ..., n_P) for planes given by their mean elements, R1 rephasing.

    Plane p has RAAN plane_raan[p], semi-major axis plane_a[p], inclination
    plane_inc[p]; its n_p satellites are spaced evenly in u starting at plane_u0[p]
    (R1, instantaneous rephasing ASSUMPTION).
    Inputs: per-plane arrays [rad], [m], shape (P,); S [-]; as constellation_revisit
    Outputs: max_s, mean_s [s], arrays of shape (S + 1,) * P
    """
    n_planes = len(plane_raan)
    shape = (sats_per_plane + 1,) * n_planes
    max_s, mean_s = np.full(shape, np.inf), np.full(shape, np.inf)
    raan0 = np.repeat(np.asarray(plane_raan)[:, None], sats_per_plane, axis=1)
    u0 = np.repeat(np.asarray(plane_u0)[:, None], sats_per_plane, axis=1)
    for n_per_plane in itertools.product(range(sats_per_plane + 1), repeat=n_planes):
        r0, uu = rephased_constellation(raan0, u0, n_per_plane)
        a_sat = np.repeat(plane_a, n_per_plane)
        inc_sat = np.repeat(plane_inc, n_per_plane)
        _, mx, mn = constellation_revisit(r0, uu, lat_deg, eta_rad, step_s, duration_s,
                                          a_m=a_sat, inc_rad=inc_sat)
        max_s[n_per_plane], mean_s[n_per_plane] = mx[0], mn[0]
    return max_s, mean_s


# ---------------------------------------------------------------------------
# Cached coverage results (computed once, reused by main.py and the Monte Carlo)
# ---------------------------------------------------------------------------

def coverage_key():
    """Parameters that determine the coverage results (to detect a stale cache)."""
    return np.array([params.ALT_KM, params.ECCENTRICITY, params.RAAN_SPREAD_DEG, params.ETA_MAX_DEG,
                     *params.ETA_SWEEP_DEG, params.REVISIT_MAX_HOURS, params.REVISIT_LAT_DEG,
                     *params.WALKER_P_RANGE, *params.WALKER_S_RANGE, params.COV_STEP_S,
                     params.COV_DURATION_DAYS, *params.COV_LAT_GRID_DEG, params.COV_LON_STEP_DEG,
                     params.MIN_ACCESS_S, *params.MIN_ACCESS_SWEEP_S,
                     MU, R_E, OMEGA_E, J2, params.TROPICAL_YEAR_DAYS])


def build_coverage_data():
    """Walker search for each eta of ETA_SWEEP_DEG, tables and latitude profiles at ETA_MAX_DEG.

    Output: dict of numpy arrays (saved as is in the .npz cache)
    """
    step_s, duration_s = params.COV_STEP_S, params.COV_DURATION_DAYS * params.SECONDS_PER_DAY
    spread = np.radians(params.RAAN_SPREAD_DEG)
    lat_grid = np.arange(params.COV_LAT_GRID_DEG[0], params.COV_LAT_GRID_DEG[1] + 1e-9,
                         params.COV_LAT_GRID_DEG[2])
    data = {"key": coverage_key(), "eta_sweep_deg": np.array(params.ETA_SWEEP_DEG),
            "lat_grid_deg": lat_grid}
    best = []
    for eta_deg in params.ETA_SWEEP_DEG:
        walker, _ = walker_search(np.radians(eta_deg), step_s, duration_s, params.WALKER_P_RANGE,
                                  params.WALKER_S_RANGE, spread)
        best.append(walker if walker is not None else (0, 0, 0))
    data["best_walker"] = np.array(best)  # (P, S, F) per eta, (0, 0, 0) if none

    # Reference eta: lookup table revisit(n_1, ..., n_P) and latitude profiles of the chosen Walker
    i_ref = list(params.ETA_SWEEP_DEG).index(params.ETA_MAX_DEG)
    n_planes, sats, phasing = data["best_walker"][i_ref]
    eta = np.radians(params.ETA_MAX_DEG)
    raan0, u0 = walker_elements(n_planes, sats, phasing, spread)
    data["table_planes_max_s"], data["table_planes_mean_s"] = revisit_table_planes(
        n_planes, sats, phasing, eta, step_s, duration_s, spread)

    # Sensitivity to the minimum access duration: chosen Walker, and smallest
    # Walker meeting the requirement with the largest minimum access of the sweep
    sweep_max, sweep_mean = [], []
    for min_access in params.MIN_ACCESS_SWEEP_S:
        _, mx, mn = constellation_revisit(raan0, u0, params.REVISIT_LAT_DEG, eta, step_s,
                                          duration_s, min_access_s=min_access)
        sweep_max.append(mx[0])
        sweep_mean.append(mn[0])
    data["min_access_sweep_s"] = np.array(params.MIN_ACCESS_SWEEP_S)
    data["min_access_sweep_max_s"] = np.array(sweep_max)
    data["min_access_sweep_mean_s"] = np.array(sweep_mean)
    walker_long, _ = walker_search(eta, step_s, duration_s, params.WALKER_P_RANGE,
                                   params.WALKER_S_RANGE, spread,
                                   min_access_s=max(params.MIN_ACCESS_SWEEP_S))
    data["best_walker_longest_access"] = np.array(walker_long if walker_long else (0, 0, 0))

    for eta_deg in params.ETA_SWEEP_DEG:
        _, mx, mn = constellation_revisit(raan0, u0, lat_grid, np.radians(eta_deg), step_s, duration_s)
        data[f"profile_max_s_eta{eta_deg:.0f}"] = mx
        data[f"profile_mean_s_eta{eta_deg:.0f}"] = mn
    return data


def load_or_build_coverage_data(path):
    """Load the coverage cache if it matches the current parameters, else rebuild it.

    Inputs: path of the .npz file (its folder is created if needed)
    Output: dict of numpy arrays (see build_coverage_data)
    """
    if os.path.exists(path):
        cached = dict(np.load(path))
        if cached["key"].shape == coverage_key().shape and np.allclose(cached["key"], coverage_key()):
            return cached
    data = build_coverage_data()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **data)
    return data
