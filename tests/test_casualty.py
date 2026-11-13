"""Validation of casualty.py against analytical cases and the DRAMA 4.1.4 SARA runs."""

import os

import numpy as np
import pytest

import casualty
import params

STEP_DEG = 1.0 / 120.0                              # band width of the SARA 1D population file
LAT_UPPER_DEG = 90.0 - np.arange(21600) * STEP_DEG  # upper edges, +90 ... -89.99 (as the file)

SARA_BASE = casualty.sara_output_path(params.SARA_RUNS["base"]["folder"], params.SARA_OUTPUT_FILE)
SARA_COMP = casualty.sara_output_path(params.SARA_RUNS["components"]["folder"], params.SARA_OUTPUT_FILE)
HAS_POP = os.path.exists(params.SARA_POPULATION_1D_PATH)


def test_casualty_area_limits():
    """A_i = 0 -> A_c = A_h; A_i = A_h -> A_c = 4 A_h."""
    assert casualty.casualty_area(0.0) == pytest.approx(0.36)
    assert casualty.casualty_area(0.36) == pytest.approx(4 * 0.36)


def test_casualty_area_sara_fragments():
    """Same casualty area as SARA for its two fragments (values from PySara.RiskResults.dat
    of the components run): tank 0.5503 m^2, optics 0.5708 m^2."""
    a_i = np.array([0.02010619298297468, 0.02419026343264141])
    a_c_sara = np.array([0.5502617626699041, 0.5708288964725444])
    assert np.allclose(casualty.casualty_area(a_i), a_c_sara, rtol=1e-12, atol=0)


def test_tumbling_cross_section_sara():
    """Surface / 4 gives SARA's mean cross sections: sphere r = 0.08 m (tank) and
    cylinder r = 0.07 m, h = 0.15 m (optics)."""
    sphere = casualty.tumbling_cross_section(4 * np.pi * 0.08**2)
    cylinder = casualty.tumbling_cross_section(2 * np.pi * 0.07**2 + 2 * np.pi * 0.07 * 0.15)
    assert sphere == pytest.approx(0.02010619298297468, rel=1e-12)
    assert cylinder == pytest.approx(0.02419026343264141, rel=1e-12)


def test_latitude_weights_sum_and_symmetry():
    """Weights sum to 1, are symmetric about the equator and vanish beyond 180 - i."""
    w = casualty.latitude_weights(LAT_UPPER_DEG, STEP_DEG, 97.4)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    # band [l - step, l] mirrors [-l, -l + step]: upper edge -l + step
    north = LAT_UPPER_DEG > 0
    mirror = casualty.latitude_weights(-LAT_UPPER_DEG[north] + STEP_DEG, STEP_DEG, 97.4)
    assert np.allclose(w[north], mirror, rtol=0, atol=1e-14)
    assert np.all(w[LAT_UPPER_DEG - STEP_DEG >= 82.6 + 1e-9] == 0.0)


def test_latitude_weights_polar_orbit():
    """i = 90 deg: latitude = argument of latitude, uniform in time -> equal weights
    step / 180, and 1/3 of the time above 60 deg (both hemispheres)."""
    w = casualty.latitude_weights(LAT_UPPER_DEG, STEP_DEG, 90.0)
    # rtol 1e-8: arcsin(sin(phi)) is ill-conditioned next to the pole (|x| -> 1), ~1e-9 there
    assert np.allclose(w, STEP_DEG / 180.0, rtol=1e-8, atol=0)
    high = np.abs(LAT_UPPER_DEG - 0.5 * STEP_DEG) > 60.0
    assert w[high].sum() == pytest.approx(1.0 / 3.0, rel=1e-9)


def test_effective_density_uniform():
    """Uniform population density -> rho_eff = rho for any inclination."""
    rho = np.full(len(LAT_UPPER_DEG), 2.5e-5)
    for incl in (30.0, 53.0, 97.4):
        assert casualty.effective_density(LAT_UPPER_DEG, rho, incl) == pytest.approx(2.5e-5, rel=1e-12)


def test_constellation_risk():
    """N E_c and 1 - (1 - E_c)^N; the probability never exceeds the expected value."""
    n = np.array([0, 1, 24, 500])
    ec = 1.4e-5
    expected, p_one = casualty.constellation_risk(n, ec)
    assert np.allclose(expected, n * ec)
    # rtol 1e-9: the plain formula loses digits by cancellation (1 - (1 - 1e-5)^N), which is
    # why casualty.py uses expm1 / log1p
    assert np.allclose(p_one, 1 - (1 - ec) ** n, rtol=1e-9, atol=0)
    assert np.all(p_one <= expected)


@pytest.mark.skipif(not HAS_POP, reason="DRAMA not installed")
def test_population_file_units():
    """Density in people/km^2: total over the sphere in 2030 ~ 8.5 billion (UN WPP 2022 medium
    variant). 2 % tolerance: spherical Earth of radius R_EARTH, gridding of the land."""
    lat_deg, rho_m2 = casualty.read_population_1d(params.SARA_POPULATION_1D_PATH, 2030)
    band_m2 = 2 * np.pi * params.R_EARTH**2 * np.cos(np.radians(lat_deg - STEP_DEG / 2)) * np.radians(STEP_DEG)
    assert np.sum(rho_m2 * band_m2) == pytest.approx(8.5e9, rel=0.02)


@pytest.mark.skipif(SARA_COMP is None or not HAS_POP, reason="SARA run or DRAMA not on this machine")
def test_weights_and_density_vs_sara():
    """Independent estimate vs SARA (components run, i = 97.4 deg, year 2030).

    1. Our analytic weights equal SARA's impact probability bin by bin (Risk-Total).
    2. rho_eff x A_c,total = SARA E_c. Tolerance rtol 1e-8: same bands and same weights,
       so only rounding (~1e-14) is left; a band centred on the file latitude instead of
       [lat - step, lat] would differ by 1.2e-6 and fail.
    """
    sara = casualty.read_sara_results(SARA_COMP)
    w = casualty.latitude_weights(sara["lat_deg"], STEP_DEG, params.REENTRY_INC_DEG)
    assert np.allclose(w, sara["impact_prob"], rtol=0, atol=1e-12)
    lat_deg, rho_m2 = casualty.read_population_1d(params.SARA_POPULATION_1D_PATH, params.POPULATION_YEAR)
    rho_eff = casualty.effective_density(lat_deg, rho_m2, params.REENTRY_INC_DEG)
    assert rho_eff * sara["casualty_area_m2"] == pytest.approx(sara["ec"], rel=1e-8)
    # per fragment, with our casualty area
    for fr in sara["fragments"]:
        ec = rho_eff * casualty.casualty_area(fr["cross_section_m2"])
        assert ec == pytest.approx(fr["ec"], rel=1e-8)
        assert fr["energy_j"] > 15.0   # above the casualty threshold of the runs


@pytest.mark.skipif(SARA_BASE is None, reason="SARA run not on this machine")
def test_sara_base_case_full_demise():
    """Base case (all AA7075): no fragment reaches the ground, E_c = 0."""
    sara = casualty.read_sara_results(SARA_BASE)
    assert sara["fragments"] == []
    assert sara["casualty_area_m2"] == 0 and sara["ec"] == 0
