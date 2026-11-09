"""Validation of planes.py (common epoch, circular grouping, IRIDE data check)."""

import numpy as np

import coverage
import params
import planes


def test_group_planes_with_wraparound():
    """RAANs 359.5, 0.3, 1.0 deg are one plane across 0/360; 90, 91 deg another."""
    raan = np.radians([90.0, 359.5, 91.0, 0.3, 1.0])
    labels = planes.group_planes(raan, np.radians(5.0))
    assert labels[1] == labels[3] == labels[4]
    assert labels[0] == labels[2] != labels[1]
    assert labels.max() == 1
    # A threshold below the 0.7-0.8 deg gaps splits every satellite apart
    assert planes.group_planes(raan, np.radians(0.5)).max() == 4


def test_circular_mean_across_zero():
    assert np.isclose(np.degrees(planes.circular_mean(np.radians([359.0, 1.0]))) % 360, 0.0,
                      atol=1e-9)


def test_propagation_one_tropical_year_sso():
    """At the SSO inclination the RAAN comes back to itself after one tropical year."""
    a = coverage.R_E + 500e3
    el = {"a_m": np.array([a]), "inc_rad": np.array([coverage.sso_inclination(a)]),
          "raan_rad": np.array([1.0]), "u_rad": np.array([0.0]), "epoch_s": np.array([0.0])}
    year_s = params.TROPICAL_YEAR_DAYS * params.SECONDS_PER_DAY
    raan, _ = planes.propagate_to_epoch(el, year_s)
    assert np.isclose(raan[0], 1.0, atol=1e-9)
    raan0, u0 = planes.propagate_to_epoch(el, 0.0)
    assert raan0[0] == 1.0 and u0[0] == 0.0


def test_iride_planes_are_robust():
    """Data check on the derived CSV: same 2 planes x 8 for every threshold of the sweep,
    and each plane is one launch (international designator prefix)."""
    for threshold in params.PLANE_GROUP_SWEEP_DEG:
        sc = planes.iride_scenario(params.IRIDE_ELEMENTS_CSV, threshold)
        assert [p["n"] for p in sc["planes"]] == [8, 8]
        for p in sc["planes"]:
            assert len({d[:5] for d in p["intdes"]}) == 1
