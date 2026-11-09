"""Reconstruction of the orbital planes of a real constellation from its mean elements.

Steps:
1. Common epoch: the TLE epochs differ by hours (and are often taken at the
   ascending node, so u ~ 0 at each epoch). RAAN and argument of latitude u are
   propagated to a common epoch with the J2 secular rates (coverage.j2_secular_rates)
   of each satellite (its own a and i).
2. Grouping: satellites are sorted by RAAN on the circle and a new plane starts
   where the gap between consecutive RAANs exceeds a threshold (ASSUMPTION,
   params.PLANE_GROUP_THRESHOLD_DEG). Motivation: inside one launch the RAAN
   dispersion is small, and the differential J2 drift from delta-a ~ 5 km and
   delta-i ~ 0.05 deg is ~1e-3 deg/day (< 0.5 deg/yr), far below the threshold.
3. Per plane: number of satellites, circular mean of RAAN, mean inclination,
   mean altitude, sorted u and in-plane spacing.
"""

import os
import zlib

import numpy as np

import coverage
import params
import tle


def propagate_to_epoch(elements, t_target_s):
    """RAAN and u of each satellite at t_target (J2 secular rates of each satellite).

    Inputs:
        elements   : dict from tle.read_elements_csv (epoch_s, a_m, inc_rad, raan_rad, u_rad)
        t_target_s : common epoch [s], same origin as elements["epoch_s"]
    Outputs: raan_rad, u_rad at the common epoch, wrapped to [0, 2 pi)
    """
    raan_dot, u_dot = coverage.j2_secular_rates(elements["a_m"], elements["inc_rad"])
    dt_s = t_target_s - elements["epoch_s"]
    raan = (elements["raan_rad"] + raan_dot * dt_s) % (2 * np.pi)
    u = (elements["u_rad"] + u_dot * dt_s) % (2 * np.pi)
    return raan, u


def group_planes(raan_rad, threshold_rad):
    """Plane label of each satellite: circular gap clustering of the RAAN.

    The RAANs are sorted on the circle starting after the largest gap, and a new
    plane starts at every gap larger than threshold_rad.
    Inputs: RAAN [rad], shape (n_sat,); threshold [rad]
    Output: integer labels 0..P-1 (ordered by RAAN from the largest gap), shape (n_sat,)
    """
    order = np.argsort(raan_rad)
    sorted_raan = raan_rad[order]
    gaps = np.diff(np.concatenate([sorted_raan, [sorted_raan[0] + 2 * np.pi]]))
    start = (np.argmax(gaps) + 1) % len(raan_rad)        # first satellite after the largest gap
    order = np.roll(order, -start)
    gaps = np.roll(gaps, -start)[:-1]                    # gaps between consecutive in the new order
    labels_sorted = np.concatenate([[0], np.cumsum(gaps > threshold_rad)])
    labels = np.empty(len(raan_rad), dtype=int)
    labels[order] = labels_sorted
    return labels


def circular_mean(angles_rad):
    """Mean direction of angles [rad], in [0, 2 pi)."""
    return np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad))) % (2 * np.pi)


def plane_summary(elements, raan_rad, u_rad, labels):
    """Per-plane description.

    Output: list of dicts (one per plane) with n, raan_rad (circular mean),
    inc_rad (mean), alt_m (mean a - R_E), a_m, u_sorted_rad, spacing_rad (gaps
    between consecutive satellites along the orbit), names
    """
    planes = []
    for p in range(labels.max() + 1):
        idx = np.nonzero(labels == p)[0]
        u_sorted = np.sort(u_rad[idx])
        spacing = np.diff(np.concatenate([u_sorted, [u_sorted[0] + 2 * np.pi]]))
        planes.append({
            "n": len(idx), "raan_rad": circular_mean(raan_rad[idx]),
            "inc_rad": elements["inc_rad"][idx].mean(), "a_m": elements["a_m"][idx].mean(),
            "alt_m": elements["a_m"][idx].mean() - coverage.R_E,
            "u_sorted_rad": u_sorted, "spacing_rad": spacing,
            "names": elements["name"][idx], "intdes": elements["intdes"][idx], "index": idx,
        })
    return planes


# ---------------------------------------------------------------------------
# IRIDE Eaglet II scenario (derived CSV from fetch_tle.py)
# ---------------------------------------------------------------------------

def iride_scenario(csv_path, threshold_deg):
    """Planes of the real constellation from the derived CSV.

    The common epoch is the latest TLE epoch of the table.
    Output: dict with elements, t_common_s, raan_rad, u_rad (at the common
    epoch), labels and planes (plane_summary)
    """
    elements = tle.read_elements_csv(csv_path)
    t_common_s = elements["epoch_s"].max()
    raan, u = propagate_to_epoch(elements, t_common_s)
    labels = group_planes(raan, np.radians(threshold_deg))
    return {"elements": elements, "t_common_s": t_common_s, "raan_rad": raan, "u_rad": u,
            "labels": labels, "planes": plane_summary(elements, raan, u, labels)}


def iride_coverage_key(csv_path):
    """Coverage parameters + checksum of the CSV content (to detect a stale cache)."""
    with open(csv_path, "rb") as f:
        crc = zlib.crc32(f.read())
    return np.concatenate([coverage.coverage_key(), [crc]])


def build_iride_coverage(scenario):
    """Revisit of the real constellation and its lookup table (eta = ETA_MAX_DEG).

    - "observed": the 16 satellites with their real RAAN, u, a, i at the common epoch.
    - "ideal": the same planes (mean RAAN, a, i) with S satellites evenly spaced (R1).
    - table revisit(n_1, ..., n_P) at REVISIT_LAT_DEG, R1 from the first satellite
      of each plane.
    Output: dict of numpy arrays
    """
    el, planes = scenario["elements"], scenario["planes"]
    sizes = {p["n"] for p in planes}
    if len(sizes) != 1:
        raise ValueError(f"Planes with different numbers of satellites: {sorted(sizes)}")
    sats = sizes.pop()
    eta = np.radians(params.ETA_MAX_DEG)
    step_s, duration_s = params.COV_STEP_S, params.COV_DURATION_DAYS * params.SECONDS_PER_DAY
    lat_grid = np.arange(params.COV_LAT_GRID_DEG[0], params.COV_LAT_GRID_DEG[1] + 1e-9,
                         params.COV_LAT_GRID_DEG[2])
    plane_raan = np.array([p["raan_rad"] for p in planes])
    plane_u0 = np.array([p["u_sorted_rad"][0] for p in planes])
    plane_a = np.array([p["a_m"] for p in planes])
    plane_inc = np.array([p["inc_rad"] for p in planes])

    data = {"key": iride_coverage_key(params.IRIDE_ELEMENTS_CSV), "lat_grid_deg": lat_grid,
            "sats_per_plane": np.array(sats)}
    _, data["observed_max_s"], data["observed_mean_s"] = coverage.constellation_revisit(
        scenario["raan_rad"], scenario["u_rad"], lat_grid, eta, step_s, duration_s,
        a_m=el["a_m"], inc_rad=el["inc_rad"])
    ideal_raan, ideal_u = coverage.rephased_constellation(
        np.repeat(plane_raan[:, None], sats, axis=1), np.repeat(plane_u0[:, None], sats, axis=1),
        [sats] * len(planes))
    _, data["ideal_max_s"], data["ideal_mean_s"] = coverage.constellation_revisit(
        ideal_raan, ideal_u, lat_grid, eta, step_s, duration_s,
        a_m=np.repeat(plane_a, sats), inc_rad=np.repeat(plane_inc, sats))
    data["table_max_s"], data["table_mean_s"] = coverage.revisit_table_from_planes(
        plane_raan, plane_u0, plane_a, plane_inc, sats, eta, step_s, duration_s)
    return data


def load_or_build_iride_coverage(path, scenario):
    """Load the IRIDE coverage cache if it matches parameters and CSV, else rebuild it."""
    key = iride_coverage_key(params.IRIDE_ELEMENTS_CSV)
    if os.path.exists(path):
        cached = dict(np.load(path))
        if np.array_equal(cached["key"], key):  # exact: the key holds a CRC checksum
            return cached
    data = build_iride_coverage(scenario)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **data)
    return data
